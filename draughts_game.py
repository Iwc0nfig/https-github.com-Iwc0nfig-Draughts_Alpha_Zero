# draughts_game.py

import numpy as np
from itertools import product
from collections import defaultdict

class DraughtsGame:
    """
    A stateless, professional game engine for Draughts, updated to correctly
    handle multi-jump captures according to American Draughts rules.
    """

    def __init__(self, board_size=8):
        self.board_size = board_size
        self.action_space_size = board_size**4

        # Piece values (positive for P1, negative for P2)
        self.P1_REGULAR, self.P1_KING = 1, 2
        self.P2_REGULAR, self.P2_KING = -1, -2

        # Draw detection parameters
        self.max_moves_no_capture = 50
        self.max_repetitions = 3

    def get_initial_state(self):
        """Returns the starting board state, starting player, and game state."""
        board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        num_rows = (self.board_size - 2) // 2

        for r in range(num_rows):
            for c in range(self.board_size):
                if (r + c) % 2 == 1:
                    board[r, c] = self.P1_REGULAR

        for r in range(self.board_size - num_rows, self.board_size):
            for c in range(self.board_size):
                if (r + c) % 2 == 1:
                    board[r, c] = self.P2_REGULAR
        
        game_state = {
            'moves_no_capture': 0,
            'position_history': defaultdict(int)
        }
        board_key = self._board_to_key(board)
        game_state['position_history'][board_key] = 1
        return board, 1, game_state

    def _board_to_key(self, board):
        return board.tobytes()

    def _find_capture_sequences(self, board, player, r, c, current_path, captured_on_path, all_sequences):
        """
        Recursively finds all possible multi-jump sequences from a given position.
        """
        piece = board[r, c]
        is_king = (piece == self.P1_KING or piece == self.P2_KING)
        
        move_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else \
                    [(-1, -1), (-1, 1)] if player == -1 else [(1, -1), (1, 1)]
        
        found_next_jump = False
        for dr, dc in move_dirs:
            nr, nc = r + dr, c + dc  # Neighbor (opponent piece)
            jr, jc = r + 2 * dr, c + 2 * dc  # Jump to square

            captured_pos = (nr, nc)

            if (0 <= jr < self.board_size and 0 <= jc < self.board_size and
                    board[jr, jc] == 0 and board[nr, nc] * player < 0 and
                    captured_pos not in captured_on_path):
                
                found_next_jump = True
                
                temp_board = board.copy()
                temp_board[r, c] = 0
                temp_board[nr, nc] = 0
                temp_board[jr, jc] = piece
                
                new_path = current_path + [(r, c, jr, jc)]
                new_captured = captured_on_path | {captured_pos}
                
                self._find_capture_sequences(temp_board, player, jr, jc, new_path, new_captured, all_sequences)

        if not found_next_jump and current_path:
            # --- FIX 1: Convert the path (a list) to a tuple to make it hashable ---
            all_sequences.append(tuple(current_path))

    def get_legal_actions(self, board, player):
        capture_sequences = []
        
        for r, c in product(range(self.board_size), range(self.board_size)):
            if board[r, c] * player > 0:
                self._find_capture_sequences(board, player, r, c, [], set(), capture_sequences)
        
        if capture_sequences:
            # A move is now (start_r, start_c, final_r, final_c, sequence_tuple)
            return [ (seq[0][0], seq[0][1], seq[-1][2], seq[-1][3], seq) for seq in capture_sequences]

        legal_moves = []
        for r, c in product(range(self.board_size), range(self.board_size)):
            piece = board[r, c]
            if piece * player > 0:
                is_king = (piece == self.P1_KING or piece == self.P2_KING)
                move_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)] if is_king else \
                            [(-1, -1), (-1, 1)] if player == -1 else [(1, -1), (1, 1)]

                for dr, dc in move_dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == 0:
                        legal_moves.append((r, c, nr, nc))
        return legal_moves

    def get_next_state(self, board, player, move, game_state):
        b = board.copy()
        is_capture = len(move) == 5 
        
        if is_capture:
            from_r, from_c, to_r, to_c, sequence = move
            piece = b[from_r, from_c]
            b[from_r, from_c] = 0
            for single_jump in sequence:
                jump_from_r, jump_from_c, jump_to_r, jump_to_c = single_jump
                captured_r, captured_c = (jump_from_r + jump_to_r) // 2, (jump_from_c + jump_to_c) // 2
                b[captured_r, captured_c] = 0
            b[to_r, to_c] = piece
        else:
            from_r, from_c, to_r, to_c = move
            piece = b[from_r, from_c]
            b[from_r, from_c] = 0
            b[to_r, to_c] = piece

        if piece == self.P1_REGULAR and to_r == self.board_size - 1:
            b[to_r, to_c] = self.P1_KING
        elif piece == self.P2_REGULAR and to_r == 0:
            b[to_r, to_c] = self.P2_KING

        new_game_state = {
            'moves_no_capture': 0 if is_capture else game_state['moves_no_capture'] + 1,
            'position_history': game_state['position_history'].copy()
        }
        board_key = self._board_to_key(b)
        new_game_state['position_history'][board_key] += 1
        return b, -player, new_game_state
    
    def get_game_ended(self, board, player, game_state):
        if not self.get_legal_actions(board, player):
            return -player
        if game_state['moves_no_capture'] >= self.max_moves_no_capture:
            return 2
        board_key = self._board_to_key(board)
        if game_state['position_history'][board_key] >= self.max_repetitions:
            return 2
        p1_pieces = np.sum(board > 0)
        p2_pieces = np.sum(board < 0)
        if p1_pieces == 0: return -1
        if p2_pieces == 0: return 1
        p1_kings = np.sum(board == self.P1_KING)
        p2_kings = np.sum(board == self.P2_KING)
        if p1_pieces == 1 and p2_pieces == 1 and p1_kings == 1 and p2_kings == 1:
            return 2
        return 0

    def get_canonical_form(self, board, player):
        if player == 1:
            return board
        else:
            return np.flip(board * -1, axis=0)

    def get_observation(self, board, player):
        canonical_board = self.get_canonical_form(board, player)
        obs = np.zeros((5, self.board_size, self.board_size), dtype=np.float32)
        obs[0, :, :] = (canonical_board == self.P1_REGULAR).astype(np.float32)
        obs[1, :, :] = (canonical_board == self.P1_KING).astype(np.float32)
        obs[2, :, :] = (canonical_board == self.P2_REGULAR).astype(np.float32)
        obs[3, :, :] = (canonical_board == self.P2_KING).astype(np.float32)
        obs[4, :, :] = np.ones((self.board_size, self.board_size), dtype=np.float32)
        return obs

    def encode_move(self, move):
        # --- FIX 2: Use slicing to safely unpack the first 4 elements ---
        from_r, from_c, to_r, to_c = move[:4]
        return (from_r * self.board_size**3 + from_c * self.board_size**2 +
                to_r * self.board_size**1 + to_c)

    def _decode_action(self, index):
        to_c = index % self.board_size
        index //= self.board_size
        to_r = index % self.board_size
        index //= self.board_size
        from_c = index % self.board_size
        index //= self.board_size
        from_r = index
        return (from_r, from_c, to_r, to_c)

    def get_action_from_canonical(self, move, player):
        if player == 1:
            return move
        else:
            bs = self.board_size
            # --- FIX 2: Use slicing to safely unpack the first 4 elements ---
            from_r, from_c, to_r, to_c = move[:4]
            canon_from_r, canon_to_r = bs - 1 - from_r, bs - 1 - to_r
            if len(move) == 5:
                # Also convert the sequence back
                canon_seq = []
                for step in move[4]:
                    s_fr, s_fc, s_tr, s_tc = step
                    canon_seq.append((bs - 1 - s_fr, s_fc, bs - 1 - s_tr, s_tc))
                # Rebuild the move with the converted sequence
                return (canon_from_r, from_c, canon_to_r, to_c, tuple(canon_seq))
            else:
                return (canon_from_r, from_c, canon_to_r, to_c)

    def render(self, board, player):
        piece_map = {
            self.P1_REGULAR: 'b', self.P1_KING: 'B',
            self.P2_REGULAR: 'w', self.P2_KING: 'W',
            0: '.'
        }
        board_str = "  " + " ".join([str(i) for i in range(self.board_size)]) + "\n"
        for r in range(self.board_size):
            board_str += str(r) + " "
            for c in range(self.board_size):
                board_str += piece_map[board[r,c]] + " "
            board_str += "\n"
        print(board_str)
        print(f"Player to move: {'Player 1 (b)' if player == 1 else 'Player 2 (w)'}\n")