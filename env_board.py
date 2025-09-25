import gymnasium as gym
from gymnasium import spaces
import numpy as np

# We import our new professional game engine
from draughts_game import DraughtsGame # Assuming you save it in draughts_game.py

class DraughtsEnv(gym.Env):
    """
    A Gymnasium environment for Draughts, acting as a wrapper around the
    stateless DraughtsGame engine. This class handles the RL-specific aspects
    like managing an episode and returning rewards.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, board_size=8):
        super().__init__()
        self.game = DraughtsGame(board_size)
        self.board_size = board_size
        
        # The state of the environment is the board and the current player
        self.board = None
        self.current_player = None

        # --- Gymnasium Interface ---
        self.action_space = spaces.Discrete(self.game.action_space_size)
        
        # The observation space is the multi-channel representation
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(5, self.board_size, self.board_size),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning of a new episode."""
        super().reset(seed=seed)
        self.board, self.current_player = self.game.get_initial_state()
        
        observation = self.game.get_observation(self.board, self.current_player)
        info = {'player_to_move': self.current_player}
        
        return observation, info

    def step(self, action_index):
        """
        Executes one step in the environment.
        The action is an integer index from the policy network.
        """
        # 1. Decode the action into a move
        move = self.game._decode_action(action_index) # Using the helper from the game engine

        # 2. Check if the move is legal
        legal_actions = self.game.get_legal_actions(self.board, self.current_player)
        if move not in legal_actions:
            # An illegal move results in an immediate loss for the current player.
            # This should ideally not happen if the MCTS is masking correctly.
            reward = -1.0
            terminated = True
            winner = -self.current_player
            info = {'error': 'Illegal move', 'winner': winner}
            # The observation doesn't matter much as the game is over.
            obs = self.game.get_observation(self.board, self.current_player)
            return obs, reward, terminated, False, info

        # 3. Apply the move to get the new state
        self.board, self.current_player = self.game.get_next_state(
            self.board, self.current_player, move
        )

        # 4. Check for game termination and determine reward
        winner = self.game.get_game_ended(self.board, self.current_player)
        
        if winner != 0:
            # Game has ended
            terminated = True
            reward = winner # +1 if player 1 wins, -1 if player 2 wins
            info = {'winner': winner}
        else:
            # Game is ongoing
            terminated = False
            reward = 0.0
            info = {'player_to_move': self.current_player}

        observation = self.game.get_observation(self.board, self.current_player)
        
        # The truncated flag is always False for a game like Draughts
        return observation, reward, terminated, False, info

    def render(self):
        """Renders the board to the console."""
        # We can create a simple helper for this
        piece_map = {
            1: 'b', 2: 'B',
            -1: 'w', -2: 'W',
            0: '.'
        }
        board_str = "  " + " ".join([str(i) for i in range(self.board_size)]) + "\n"
        for r in range(self.board_size):
            board_str += str(r) + " "
            for c in range(self.board_size):
                board_str += piece_map[self.board[r,c]] + " "
            board_str += "\n"
        print(board_str)
        print(f"Player to move: {'Player 1 (b)' if self.current_player == 1 else 'Player 2 (w)'}")