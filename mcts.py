# mcts.py
import math
import numpy as np
import torch
from typing import Dict, Tuple, List, Optional

class Node:
    """
    A tree node storing PUCT statistics.
    - to_play: player to move at this node (1 or -1)
    - P: prior probability from the policy head
    - N: visit count
    - W: total value from this node's current player's perspective (sum of backed-up values)
    - Q: mean value = W / N
    - children: dict[move_tuple] -> Node
    - move: the move taken from parent to reach this node (None for root)
    - parent: parent node (None for root)
    """
    __slots__ = ("to_play", "P", "N", "W", "Q", "children", "move", "parent")

    def __init__(self, to_play: Optional[int] = None, prior: float = 0.0, parent: 'Node' = None, move: Tuple[int,int,int,int] = None):
        self.to_play: Optional[int] = to_play
        self.P: float = float(prior)
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.children: Dict[Tuple[int,int,int,int], 'Node'] = {}
        self.move: Optional[Tuple[int,int,int,int]] = move
        self.parent: Optional['Node'] = parent

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """
    AlphaZero-style MCTS with:
    - PUCT selection
    - Neural network evaluation (policy logits + value)
    - Root Dirichlet noise
    - Visit-count policy at root
    - Tree reuse

    Expected game API (as in provided draughts_game.py):
    - get_observation(board, player) -> np.ndarray[C,H,W]
    - get_legal_actions(board, player) -> List[move_tuple]
    - get_next_state(board, player, move) -> (board', player')
    - get_game_ended(board, player) -> 0 if not ended, else +1 if P1 wins, -1 if P2 wins
    - encode_move(move) -> int in [0, action_space_size)
    - action_space_size: int
    """
    def __init__(self, game, nnet: torch.nn.Module, args: dict, device: Optional[torch.device] = None):
        self.game = game
        self.nnet = nnet
        self.args = args
        

        self.num_sims: int = int(args.get("num_mcts_sims", 40))
        self.c_puct: float = float(args.get("c_puct", 1.5))
        self.dirichlet_alpha: float = float(args.get("dirichlet_alpha", 0.3))
        self.dirichlet_epsilon: float = float(args.get("dirichlet_epsilon", 0.25))
        self.eval_batch_size: int = int(args.get("eval_batch_size", 1))

        if device is None:
            try:
                device = next(nnet.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        # Ensure eval mode
        self.nnet.eval()

    def getActionProb(self, board, player, game_state, temperature=1):
        """
        Return root visit-count policy; temperature is intentionally ignored here
        because the caller (self-play) already applies temperature during action selection.
        """
        root = Node(to_play=player)
        return self.search((board, player, game_state), root)

    def search(self, state, root: Node) -> np.ndarray:
        # Unpack the updated state tuple: (board, player, game_state)
        board, player, game_state = state
        root.to_play = player

        applied_root_noise = False
        sims_done = 0

        while sims_done < self.num_sims:
            batch_obs = []
            # (path_nodes, leaf_node, cur_board, cur_player, cur_game_state, is_root_leaf)
            batch_meta = []
            batch_target = min(self.eval_batch_size, self.num_sims - sims_done)

            # 1) Selection to collect a mini-batch of leaves
            while len(batch_obs) < batch_target and sims_done + len(batch_obs) < self.num_sims:
                path_nodes = [root]
                node = root
                cur_board = board
                cur_player = player
                cur_game_state = game_state

                # Traverse
                while node.is_expanded:
                    winner = self.game.get_game_ended(cur_board, cur_player, cur_game_state)
                    if winner != 0:
                        # Handle draw (=2) as zero value; otherwise ±1 from current player's perspective
                        if winner == 2:
                            leaf_value = 0.0
                        else:
                            leaf_value = 1.0 if winner == cur_player else -1.0
                        self._backpropagate(path_nodes, leaf_value)
                        sims_done += 1
                        break

                    move, node = self._select_child(node)
                    cur_board, cur_player, cur_game_state = self.game.get_next_state(
                        cur_board, cur_player, move, cur_game_state
                    )
                    path_nodes.append(node)
                else:
                    # Leaf reached and not terminal -> enqueue for batched eval
                    winner = self.game.get_game_ended(cur_board, cur_player, cur_game_state)
                    if winner != 0:
                        if winner == 2:
                            leaf_value = 0.0
                        else:
                            leaf_value = 1.0 if winner == cur_player else -1.0
                        self._backpropagate(path_nodes, leaf_value)
                        sims_done += 1
                    else:
                        obs = self.game.get_observation(cur_board, cur_player)
                        batch_obs.append(obs)
                        batch_meta.append((path_nodes, node, cur_board, cur_player, cur_game_state, node is root))

            # 2) One batched forward on all collected leaves
            if batch_obs:
                obs_t = torch.from_numpy(np.stack(batch_obs)).to(self.device, non_blocking=True)
                with torch.no_grad(), torch.inference_mode():
                    policy_logits_t, values_t = self.nnet(obs_t)
                policy_logits_np = policy_logits_t.detach().cpu().numpy()
                values_np = values_t.detach().cpu().numpy().reshape(-1)

                # 3) Expand each leaf and backprop values
                for i, (path_nodes, node, cur_board, cur_player, cur_game_state, is_root_leaf) in enumerate(batch_meta):
                    policy_logits = policy_logits_np[i]
                    v = float(values_np[i])

                    legal_moves = self.game.get_legal_actions(cur_board, cur_player)
                    priors = self._masked_softmax(policy_logits, legal_moves, cur_player)

                    node.to_play = cur_player
                    for mv in legal_moves:
                        if mv not in node.children:
                            canon_idx = self.game.encode_move(self._to_canonical_move(mv, cur_player))
                            node.children[mv] = Node(
                                to_play=-cur_player,
                                prior=float(priors[canon_idx]),
                                parent=node,
                                move=mv,
                            )

                    if is_root_leaf and not applied_root_noise:
                        self._apply_dirichlet_noise(root)
                        applied_root_noise = True

                    self._backpropagate(path_nodes, v)

                sims_done += len(batch_obs)

        # 4) Build visit-count policy
        policy = np.zeros(self.game.action_space_size, dtype=np.float32)
        total_N = sum(ch.N for ch in root.children.values())
        if total_N > 0:
            for mv, ch in root.children.items():
                idx = self.game.encode_move(self._to_canonical_move(mv, root.to_play))
                policy[idx] = ch.N / total_N
        else:
            prior_sum = sum(max(ch.P, 0.0) for ch in root.children.values())
            if prior_sum > 0:
                for mv, ch in root.children.items():
                    idx = self.game.encode_move(self._to_canonical_move(mv, root.to_play))
                    policy[idx] = ch.P / prior_sum
        return policy

    def _select_child(self, node: Node) -> Tuple[Tuple[int,int,int,int], Node]:
        """
        PUCT selection: choose child maximizing -Q(child) + U, because child's Q is
        from child's to_play perspective, which is the opponent relative to node.to_play.
        """
        parent_N = max(1, node.N)
        sqrt_parent_N = math.sqrt(parent_N)
        best_score = -1e30
        best_move = None
        best_child = None

        for mv, ch in node.children.items():
            # Exploitation term uses -Q(child) to flip perspective back to node.to_play
            q = -ch.Q
            u = self.c_puct * ch.P * (sqrt_parent_N / (1 + ch.N))
            score = q + u
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = ch

        return best_move, best_child

    def _evaluate(self, board: np.ndarray, player: int) -> Tuple[np.ndarray, float]:
        """
        Run the network on the canonical observation to get policy logits and value.
        Returns:
          - policy_logits: np.ndarray of shape [action_space_size]
          - value: float in [-1, 1] from current player's perspective
        """
        obs = self.game.get_observation(board, player)  # [C,H,W], canonical
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # [1,C,H,W]
        with torch.no_grad():
            policy_logits_t, value_t = self.nnet(obs_t)
        policy_logits = policy_logits_t.squeeze(0).detach().cpu().numpy()  # [A]
        value = float(value_t.squeeze(0).detach().cpu().numpy().item())
        return policy_logits, value

    def _masked_softmax(self, logits, legal_moves, player):
        # Use canonical indices to align with logits from canonical observation
        legal_indices = [
            self.game.encode_move(self._to_canonical_move(mv, player))
            for mv in legal_moves
        ]
        if not legal_indices:
            return np.zeros_like(logits, dtype=np.float32)

        probs = np.full_like(logits, fill_value=-1e9, dtype=np.float32)
        probs[legal_indices] = logits[legal_indices].astype(np.float32)

        max_logit = np.max(probs)
        exp = np.exp(probs - max_logit)
        exp_sum = np.sum(exp)
        if exp_sum <= 0.0 or not np.isfinite(exp_sum):
            out = np.zeros_like(logits, dtype=np.float32)
            out[legal_indices] = 1.0 / len(legal_indices)
            return out

        softmax = exp / exp_sum
        s_legal = np.sum(softmax[legal_indices])
        if s_legal > 0:
            softmax[legal_indices] /= s_legal
        else:
            softmax[:] = 0.0
            softmax[legal_indices] = 1.0 / len(legal_indices)
        return softmax

    def _apply_dirichlet_noise(self, root: Node):
        """
        Inject Dirichlet noise into priors at the root:
          P' = (1 - epsilon)*P + epsilon*Dir(alpha)
        """
        if self.dirichlet_epsilon <= 0 or self.dirichlet_alpha <= 0:
            return
        if not root.children:
            return

        moves = list(root.children.keys())
        alphas = [self.dirichlet_alpha] * len(moves)
        noise = np.random.dirichlet(alphas).astype(np.float32)

        # Mix noise into existing priors
        new_ps = []
        for i, mv in enumerate(moves):
            child = root.children[mv]
            p = (1.0 - self.dirichlet_epsilon) * max(child.P, 0.0) + self.dirichlet_epsilon * float(noise[i])
            child.P = float(p)
            new_ps.append(child.P)

        # Re-normalize priors to sum to 1 across root children
        s = sum(new_ps)
        if s > 0:
            for mv in moves:
                root.children[mv].P /= s

    def _backpropagate(self, path_nodes: List[Node], leaf_value: float):
        """
        Backup the leaf_value up the path, flipping sign at each step
        because players alternate turns between nodes.
        path_nodes is root; path_nodes[-1] is leaf.
        """
        v = leaf_value
        for node in reversed(path_nodes):
            node.N += 1
            node.W += v
            node.Q = node.W / node.N
            v = -v  # flip perspective for the parent

    def _to_canonical_move(self, move, player):
        # Map original-board move to the canonical frame used for the network
        if player == 1:
            return move
        bs = self.game.board_size
        # --- FIX 2: Use slicing to safely unpack the first 4 elements ---
        fr, fc, tr, tc = move[:4]
        
        # We only need to convert the start and end points for encoding.
        # The MCTS doesn't need to understand the sequence, just the game engine does.
        # If it's a capture move, we must pass the sequence along unmodified.
        canon_move_start = (bs - 1 - fr, fc, bs - 1 - tr, tc)
        
        if len(move) == 5: # It's a capture move, append the sequence
            return canon_move_start + (move[4],)
        else: # It's a simple move
            return canon_move_start


    
    

    

 
