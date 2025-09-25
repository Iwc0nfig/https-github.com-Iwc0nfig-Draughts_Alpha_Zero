import numpy as np
from draughts_game import DraughtsGame
from mcts import MCTS, Node
import random
from collections import defaultdict

class SelfPlay:
    def __init__(self, game: DraughtsGame, mcts: MCTS, args: dict ,current_iteration:int,total_iterations:int):
        self.game = game
        self.mcts = mcts
        self.args = args
        self.warmup = True
        self.current_iteration = current_iteration
        self.total_iterations = total_iterations
        self.custom_start_positions = self._create_custom_positions()



    def _create_custom_positions(self):
        """
        Creates a curriculum of starting positions to teach the agent about kings.
        
        Returns:
            A list of tuples, where each tuple is (board_array, player_to_move).
        """
        positions = []
        bs = self.game.board_size # Board size, e.g., 8

        # --- SCENARIO 1: Player 1 is one move away from promotion ---
        # Goal: Teach the immediate value of reaching the back rank.
        board1 = np.zeros((bs, bs), dtype=np.int8)
        board1[bs - 2, 1] = self.game.P1_REGULAR  # P1 piece close to promotion
        board1[bs - 1, 0] = self.game.P2_REGULAR  # A P2 piece to create a realistic scenario
        board1[bs - 3, 4] = self.game.P2_REGULAR
        positions.append((board1, 1)) # Player 1 to move

        # --- SCENARIO 2: Player 2 is one move away from promotion ---
        # Goal: The symmetrical case for Player 2.
        board2 = np.zeros((bs, bs), dtype=np.int8)
        board2[1, 4] = self.game.P2_REGULAR # P2 piece close to promotion
        board2[0, 5] = self.game.P1_REGULAR # A P1 piece
        board2[2, 1] = self.game.P1_REGULAR
        positions.append((board2, -1)) # Player 2 to move

        # --- SCENARIO 3: Mid-game with existing kings ---
        # Goal: Teach how to use a king's power in a more complex board state.
        board3 = np.zeros((bs, bs), dtype=np.int8)
        board3[bs - 3, 2] = self.game.P1_KING      # P1 King
        board3[2, 5] = self.game.P2_KING          # P2 King
        board3[bs - 1, 0] = self.game.P2_REGULAR
        board3[bs - 2, 1] = self.game.P2_REGULAR
        board3[1, 6] = self.game.P1_REGULAR
        board3[0, 7] = self.game.P1_REGULAR
        positions.append((board3, 1)) # Player 1 to move

        # --- SCENARIO 4: Classic King vs. King endgame ---
        # Goal: Teach fundamental king movement and opposition.
        board4 = np.zeros((bs, bs), dtype=np.int8)
        board4[bs - 2, 1] = self.game.P1_KING
        board4[1, 4] = self.game.P2_KING
        positions.append((board4, -1)) # Player 2 to move

        # --- SCENARIO 5: King vs. Pawns advantage ---
        # Goal: Teach the strategic advantage of a king against multiple regular pieces.
        board5 = np.zeros((bs, bs), dtype=np.int8)
        board5[4, 3] = self.game.P1_KING          # P1 has the king
        board5[2, 1] = self.game.P2_REGULAR       # P2 has two pawns
        board5[2, 5] = self.game.P2_REGULAR
        positions.append((board5, 1)) # Player 1 to move

        #print(f"Initialized curriculum with {len(positions)} custom starting positions.")
        return positions

    def play_game(self) -> list:
        """Play a full self-play game and return the training data."""
        game_history = []
        initial_prob = 0.3
        min_prob = 0.15
        turn = 0

        


        decay_factor = self.current_iteration / self.total_iterations
        curriculum_prob = max(min_prob, initial_prob * (1 - decay_factor))

        if curriculum_prob > min_prob:
            self.warmup = False

        if self.warmup and random.random() < curriculum_prob:
            board, current_player = random.choice(self.custom_start_positions)
            board , current_player = self._augment_position(board, current_player)
            board.copy()

            game_state = {
                'moves_no_capture': 0,
                'position_history': defaultdict(int)
            }
            board_key = self.game._board_to_key(board)
            game_state['position_history'][board_key] = 1
        else:
            board, current_player, game_state = self.game.get_initial_state()
            

        while True:
            # Get MCTS policy
            # temp = self.args['temperature'] if turn < self.args['temp_threshold'] else 0
            # if temp:
            #     temp = temp*decay_factor

            if turn < self.args['temp_threshold']:
                temp = self.args['temperature'] * (1 - decay_factor)
            else:
                temp = 0 
                   
            mcts_policy = self.mcts.getActionProb(board, current_player, game_state, temp)

            # Create observation for neural network
            observation = self.game.get_observation(board, current_player)

            # Store (observation, policy, current_player) for training
            game_history.append((observation, mcts_policy, current_player))

            # Select action based on policy
            action_index = self.select_action(mcts_policy, temp)
            move = self.game._decode_action(action_index)

            # Convert move from canonical to actual if needed
            if current_player == -1:
                move = self.game.get_action_from_canonical(move, current_player)

            # Make the move
            board, current_player, game_state = self.game.get_next_state(board, current_player, move, game_state)
            turn += 1

            # Check if game is over
            winner = self.game.get_game_ended(board, current_player, game_state)
            if winner != 0:
                # Assign outcomes based on game result
                training_examples = []
                for obs, policy, player in game_history:
                    if winner == 2:  # Draw
                        outcome = 0.0  # Draw outcome is 0
                    elif winner == player:
                        outcome = 1.0  # Win
                    else:
                        outcome = -1.0  # Loss

                    training_examples.append((obs, policy, outcome))

                return training_examples

    def select_action(self, policy, temperature):
        """Select action based on policy and temperature."""
        if temperature == 0:
            # Greedy selection
            best_action_indices = np.where(policy == np.max(policy))[0]
            return np.random.choice(best_action_indices)
        else:
            # Stochastic selection
            policy = np.array(policy) + 1e-8  # avoid zero probabilities
            policy = policy ** (1. / temperature)
            policy = policy / np.sum(policy)
            return np.random.choice(len(policy), p=policy)
        

    def _augment_position(self, board: np.ndarray, player: int):
        """
        Applies random augmentations to a board position to prevent overfitting.
        Currently implements a horizontal flip.
        """
        if random.random() < 0.5:
            augmented_board = np.fliplr(board)
            return augmented_board, player
        else:
            return board, player