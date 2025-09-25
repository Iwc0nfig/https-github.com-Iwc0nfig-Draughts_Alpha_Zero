import torch
import numpy as np
import argparse
import time

from draughts_game import DraughtsGame
from alpha_policy import AlphaZeroDraughtsNet
from mcts import MCTS
from cuda_optim import setup_cuda_optimizations

def play_game(game, model_first, model_second, args):
    """
    Plays one game where model_first is the starting side (Player 1 at the initial position).
    This version correctly handles multi-jump captures.
    Returns:
        +1 -> model_first wins
        -1 -> model_second wins
         0 -> draw
    """
    board, current_player, game_state = game.get_initial_state()

    mcts_args = dict(args)
    mcts_args['dirichlet_epsilon'] = 1.15 #We must keep the same value for training and testing
    mcts_args['temperature'] = 0.0

    mcts_first = MCTS(game, model_first, mcts_args)
    mcts_second = MCTS(game, model_second, mcts_args)

    while True:
        if current_player == 1:
            policy = mcts_first.getActionProb(board, current_player, game_state, temperature=0.0)
        else:
            policy = mcts_second.getActionProb(board, current_player, game_state, temperature=0.0)

        def to_canon(mv, player):
            if player == 1: return mv
            bs = game.board_size
            fr, fc, tr, tc = mv[:4]
            canon_start = (bs - 1 - fr, fc, bs - 1 - tr, tc)
            if len(mv) == 5:
                # We need to flip the sequence as well for the canonical representation
                canon_seq = tuple([(bs - 1 - s_fr, s_fc, bs - 1 - s_tr, s_tc) for s_fr, s_fc, s_tr, s_tc in mv[4]])
                return canon_start + (canon_seq,)
            return canon_start
        
        # --- START: CORRECTED MOVE SELECTION LOGIC ---

        # 1. Get all legal moves. These are the "true" move objects, which may
        #    contain the jump sequence for captures.
        legal_moves = game.get_legal_actions(board, current_player)

        # 2. Create a dictionary mapping the policy index to the actual move object.
        #    This allows us to find the best move based on the policy and then
        #    retrieve the full move object associated with it.
        move_map = {}
        for move in legal_moves:
            # The neural network policy is based on the canonical board state.
            canonical_move = to_canon(move, current_player)
            index = game.encode_move(canonical_move)
            # Map the policy index to the ORIGINAL, non-canonical move.
            move_map[index] = move

        # 3. Find the best index from the policy output, but only consider the
        #    keys (indices) of our legal moves.
        if not move_map:
            # This should not happen if the game is not over, but as a safeguard:
            winner = game.get_game_ended(board, current_player, game_state)
            if winner == 2: return 0
            return -1 if current_player == 1 else 1
            
        best_index = max(move_map.keys(), key=lambda idx: policy[idx])

        # 4. Retrieve the complete, original move object using the best index.
        #    This 'move' will be a 5-element tuple if it's a capture.
        move = move_map[best_index]

        # --- END: CORRECTED MOVE SELECTION LOGIC ---

        board, current_player, game_state = game.get_next_state(board, current_player, move, game_state)
        
        if args['render']:
            game.render(board, current_player)
            time.sleep(0.5)

        winner = game.get_game_ended(board, current_player, game_state)
        if winner != 0:
            if winner == 2:
                return 0
            elif winner == 1:
                return 1   # model_first is always Player 1 at the start
            else: # winner == -1
                return -1  # model_second is always Player 2 at the start

def main(args):
    game = DraughtsGame(board_size=args['board_size'])
    device = setup_cuda_optimizations()

    # Load models
    ckpt1 = torch.load(args['m1'], map_location=device)
    model1 = AlphaZeroDraughtsNet(
        board_size=args['board_size'],
        num_res_blocks=args['num_res_blocks'],
        num_channels=args['num_channels']
    ).to(device)
    model1.load_state_dict(ckpt1['model_state'])
    model1.eval()

    ckpt2 = torch.load(args['m2'], map_location=device)
    model2 = AlphaZeroDraughtsNet(
        board_size=args['board_size'],
        num_res_blocks=args['num_res_blocks'],
        num_channels=args['num_channels']
    ).to(device)
    model2.load_state_dict(ckpt2['model_state'])
    model2.eval()

    print(f"Pitting {args['m1']} vs {args['m2']}")

    n = args['num_games']
    assert n % 2 == 0, "Use an even number of games to alternate starts evenly."

    m1_as_first_w = 0
    m1_as_second_w = 0
    m2_as_first_w = 0
    m2_as_second_w = 0
    draws = 0

    # First half: model1 starts
    for _ in range(n // 2):
        r = play_game(game, model1, model2, args)
        if r == 1:
            m1_as_first_w += 1
        elif r == -1:
            m2_as_second_w += 1
        else:
            draws += 1
        print(f"Score so far: M1 {m1_as_first_w + m1_as_second_w}, M2 {m2_as_first_w + m2_as_second_w}, D {draws}")

    # Second half: model2 starts
    for _ in range(n // 2):
        r = play_game(game, model2, model1, args)
        if r == 1:
            m2_as_first_w += 1
        elif r == -1:
            m1_as_second_w += 1
        else:
            draws += 1
        print(f"Score so far: M1 {m1_as_first_w + m1_as_second_w}, M2 {m2_as_first_w + m2_as_second_w}, D {draws}")

    m1_total = m1_as_first_w + m1_as_second_w
    m2_total = m2_as_first_w + m2_as_second_w

    print("\n--- FINAL RESULTS ---")
    print(f"Total Games: {n}")
    print(f"Wins for {args['m1']}: {m1_total} ({m1_total/n:.2%})")
    print(f"  - as first:  {m1_as_first_w}")
    print(f"  - as second: {m1_as_second_w}")
    print(f"Wins for {args['m2']}: {m2_total} ({m2_total/n:.2%})")
    print(f"  - as first:  {m2_as_first_w}")
    print(f"  - as second: {m2_as_second_w}")
    print(f"Draws: {draws} ({draws/n:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m1", type=str, required=True)
    parser.add_argument("--m2", type=str, required=True)
    parser.add_argument("--num_games", type=int, default=20)
    parser.add_argument("--render", action='store_true', help="Render the games to console")
    # Game/Model/MCTS params
    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--num_res_blocks", type=int, default=5)
    parser.add_argument("--num_channels", type=int, default=128)
    parser.add_argument("--num_mcts_sims", type=int, default=80)
    parser.add_argument("--c_puct", type=float, default=1.5)
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet_epsilon", type=float, default=0.25)
    parser.add_argument("--eval_batch_size", type=int, default=8192)

    cli_args = parser.parse_args()
    main(vars(cli_args))