# train.py
import os
import json
import time
import math
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR

# Project imports
from alpha_policy import AlphaZeroDraughtsNet
from draughts_game import DraughtsGame
from self_play import SelfPlay
from replay_buffer import ReplayBuffer
from mcts import MCTS, Node
from cuda_optim import setup_cuda_optimizations


import logging

#import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


logging.basicConfig(
    filename='draughts.log',
    level= logging.INFO,
    format='%(message)s'
)

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config.json") -> dict:
    with open(path, "r") as f:
        return json.load(f)


def linear_epochs(iter_idx: int, total_iters: int, start_ep: int, end_ep: int) -> int:
    if total_iters <= 1:
        return int(start_ep)
    alpha = iter_idx / (total_iters - 1)
    return int(round((1 - alpha) * start_ep + alpha * end_ep))


def policy_value_loss(policy_logits: torch.Tensor,
                      value_pred: torch.Tensor,
                      policy_targets: torch.Tensor,
                      value_targets: torch.Tensor,
                      value_weight: float = 1.0) -> tuple:
    # Cross-entropy with soft targets: -sum(pi * log_softmax(logits))
    log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_targets * log_probs).sum(dim=1).mean()

    # Ensure value shapes match
    if value_pred.dim() == 2 and value_pred.size(1) == 1:
        value_pred = value_pred.squeeze(1)
    value_loss = torch.mean((value_pred - value_targets) ** 2)

    total = policy_loss + value_weight * value_loss
    return total, policy_loss, value_loss

# Corrected to_tensor_batch
import numpy as np
import torch

def collate_fn(batch):
    """Custom collate function that returns raw tuples instead of stacked tensors"""
    return batch

def to_tensor_batch(batch, device, action_space_size, board_size, pin_memory=False):
    A = int(action_space_size)
    BS = int(board_size)

    obs_list, pol_list, val_list = [], [], []

    for i, exp in enumerate(batch):
        # Unwrap one-level nested triples [(obs,pol,val)] -> (obs,pol,val)
        if isinstance(exp, (list, tuple)) and len(exp) == 1:
            exp = exp[0]
        if not (isinstance(exp, (list, tuple)) and len(exp) == 3):
            raise ValueError(f"Bad sample {i}: expected (obs, policy, outcome), got {type(exp)} with length {len(exp) if hasattr(exp, '__len__') else 'unknown'}")

        obs, pol, val = exp

        obs = np.asarray(obs, dtype=np.float32)
        pol = np.asarray(pol, dtype=np.float32).reshape(-1)  # flatten safely
        val = np.float32(val)

        if obs.ndim != 3 or tuple(obs.shape) != (5, BS, BS):
            raise ValueError(f"Obs idx {i} bad shape {tuple(obs.shape)}; expected (5,{BS},{BS})")
        if pol.ndim != 1 or pol.size != A:
            raise ValueError(f"Policy idx {i} bad shape {tuple(pol.shape)}; expected ({A},)")

        obs_list.append(obs)
        pol_list.append(pol)
        val_list.append(val)

    obs_t = torch.from_numpy(np.stack(obs_list, 0)).to(device, non_blocking=True)
    pol_t = torch.from_numpy(np.stack(pol_list, 0)).to(device, non_blocking=True)
    val_t = torch.from_numpy(np.asarray(val_list, np.float32)).to(device, non_blocking=True)
    return obs_t, pol_t, val_t


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer.buffer[idx]


def self_play_single_game_worker(shared_data):
    """Worker function that plays a single game and returns the data"""
    net_state_dict, cfg, mcts_args, seed ,current_iteration = shared_data
    
    # Set unique seed for this worker
    set_global_seed(seed)
    
    # Setup device and models for this process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = DraughtsGame(board_size=int(cfg.get("board_size", 8)))
    
    net = AlphaZeroDraughtsNet(
        board_size=int(cfg.get("board_size", 8)),
        num_res_blocks=int(cfg.get("num_res_blocks", 5)),
        num_channels=int(cfg.get("num_channels", 128))
    ).to(device)
    net.load_state_dict(net_state_dict)
    net.eval()
    
    mcts = MCTS(game, net, mcts_args, device=device)
    self_play = SelfPlay(game, mcts, cfg , current_iteration,int(cfg.get("num_iterations", 1000)))
    
    # Play single game
    game_data = self_play.play_game()
    return game_data

def dynamic_self_play_generation(net, num_games_per_iter, num_workers, cfg, mcts_args, seed_base,current_iteration):
    """Generate self-play data using dynamic task distribution"""
    print(f"Starting dynamic self-play with {num_workers} workers for {num_games_per_iter} games")
    
    # Move network to CPU and get state dict
    net.to('cpu').share_memory()
    net_state_dict = net.state_dict()
    
    all_game_data = []
    completed_games = 0
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch of tasks (one per worker)
        futures = set()
        games_submitted = 0
        
        # Submit initial batch
        for i in range(min(num_workers, num_games_per_iter)):
            future = executor.submit(self_play_single_game_worker, 
                                   (net_state_dict, cfg, mcts_args, seed_base + games_submitted ,current_iteration))
            futures.add(future)
            games_submitted += 1
        
        # Process completed tasks and submit new ones dynamically
        with tqdm(total=num_games_per_iter, desc="Self-play games") as pbar:
            while futures:
                # Wait for next game to complete
                done_futures, futures = concurrent.futures.wait(
                    futures, return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done_futures:
                    try:
                        game_data = future.result()
                        all_game_data.extend(game_data)
                        completed_games += 1
                        pbar.update(1)
                        
                        # Submit new game if we haven't reached the target
                        if games_submitted < num_games_per_iter:
                            new_future = executor.submit(self_play_single_game_worker,
                                                       (net_state_dict, cfg, mcts_args, seed_base + games_submitted, current_iteration))
                            futures.add(new_future)
                            games_submitted += 1
                            
                    except Exception as e:
                        print(f"Game failed with error: {e}")
                        # Still submit a replacement game
                        if games_submitted < num_games_per_iter:
                            new_future = executor.submit(self_play_single_game_worker,
                                                       (net_state_dict, cfg, mcts_args, seed_base + games_submitted,current_iteration))
                            futures.add(new_future)
                            games_submitted += 1
    
    print(f"Completed {completed_games} games, generated {len(all_game_data)} positions")
    return all_game_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile",action="store_true", help="Enable torch.compile()")
    parser.add_argument("--value_weight", type=float, default=1.0, help="Weight for value loss")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count() or 1, help="Number of parallel workers for self-play")
    args_cli = parser.parse_args()

    cfg = load_config(args_cli.config)

    for i ,v in cfg.items():
        print(f"  {i}: {v}")

    # CPU/GPU setup
    set_global_seed(args_cli.seed)

    try:
        mp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass # Already set
    
    # The recommendation from the todo.md is to set these environment variables.
    os.environ['OMP_NUM_THREADS'] = '10'
    os.environ['MKL_NUM_THREADS'] = '10'

    torch.set_num_threads(10) # As per the todo file
    device_type = setup_cuda_optimizations()  
    device = torch.device(device_type)

    # Instantiate game and network
    board_size = int(cfg.get("board_size", 8))
    game = DraughtsGame(board_size=board_size)

    net = AlphaZeroDraughtsNet(
        board_size=board_size,
        num_res_blocks=int(cfg.get("num_res_blocks", 5)),
        num_channels=int(cfg.get("num_channels", 128))
    ).to(device)

    if args_cli.compile and hasattr(torch, "compile"):
        print("Compiling the model...")
        net = torch.compile(net, mode="max-autotune") # As per the todo file

    optimizer = optim.AdamW(net.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
    scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
    scaler = amp.GradScaler() if device_type == 'cuda' else None

    buffer = ReplayBuffer(max_size=int(cfg.get("buffer_size", 200_000)))

    mcts_args = {
        "num_mcts_sims": int(cfg.get("num_mcts_sims", 40)),
        "c_puct": float(cfg.get("c_puct", 1.5)),
        "dirichlet_alpha": float(cfg.get("dirichlet_alpha", 0.3)),
        "dirichlet_epsilon": float(cfg.get("dirichlet_epsilon", 0.25)),
        "eval_batch_size": int(cfg.get("eval_batch_size", 2048)),
    }

    # Training settings
    batch_size = int(cfg.get("batch_size", 4096))
    num_iterations = int(cfg.get("num_iterations", 1000))
    num_games_per_iter = int(cfg.get("num_games_per_iter", 80))
    checkpoint_dir = Path(cfg.get("checkpoint_path", "./models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Use multiple workers for asynchronous data loading
    num_dataloader_workers = 4 

    start_time = time.time()
    for it in range(num_iterations):
        iter_start = time.time()

        # 1) Self-play to generate training data in parallel
        net.eval()
        
        all_game_data = dynamic_self_play_generation(
            net, 
            num_games_per_iter, 
            args_cli.num_workers, 
            cfg, 
            mcts_args, 
            args_cli.seed + it * 10000 , # Different seed per iteration,
            it
        )
        
        buffer.add_game_data(all_game_data)
        generated_positions = len(all_game_data)
        
        # Move network back to the training device
        net.to(device)


        # 2) Train for a number of epochs
        epochs = linear_epochs(it, num_iterations,
                               int(cfg.get("num_epochs_start", 4)),
                               int(cfg.get("num_epochs_end", 8)))

        net.train()
        
        # Create a Dataset and DataLoader for optimized batch loading
        
        
        dataset = BufferDataset(buffer)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_dataloader_workers,
            pin_memory=True,
            persistent_workers=True if num_dataloader_workers > 0 else False,
            collate_fn=collate_fn
        )
        
        for ep in range(epochs):
            if len(buffer) == 0:
                continue

            ep_policy_loss = 0.0
            ep_value_loss = 0.0
            ep_total_loss = 0.0
            

            for batch in dataloader:
                obs_t, pol_t, val_t = to_tensor_batch(
                    batch,
                    device=device,
                    action_space_size=int(game.action_space_size),
                    board_size=int(game.board_size),
                )

                optimizer.zero_grad(set_to_none=True)
                
                if scaler:
                    with amp.autocast("cuda"):
                        policy_logits, value_pred = net(obs_t)
                        total_loss, pol_loss, val_loss = policy_value_loss(
                            policy_logits, value_pred, pol_t, val_t, value_weight=args_cli.value_weight
                        )
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else: # CPU training
                    policy_logits, value_pred = net(obs_t)
                    total_loss, pol_loss, val_loss = policy_value_loss(
                        policy_logits, value_pred, pol_t, val_t, value_weight=args_cli.value_weight
                    )
                    total_loss.backward()
                    optimizer.step()


                ep_policy_loss += pol_loss.item()
                ep_value_loss += val_loss.item()
                ep_total_loss += total_loss.item()
                
                
                

            denom = len(dataloader)
            if denom > 0:
                print(f"[Iter {it+1}/{num_iterations}] Epoch {ep+1}/{epochs} | "
                      f"avg_policy_loss={ep_policy_loss/denom:.4f} | avg_value_loss={ep_value_loss/denom:.4f} | "
                      f"avg_total={ep_total_loss/denom:.4f}")
                
                step_logs = {
                    'iteration': it + 1,
                    'epoch': ep + 1,
                    'policy_loss': ep_policy_loss / denom,
                    'value_loss': ep_value_loss / denom,
                    'total_loss': ep_total_loss / denom,                
                }

                logging.info(json.dumps(step_logs))
                
        if (it+1) %5 == 0:
            # 3) Save checkpoint at end of iteration
            ckpt_path = checkpoint_dir / f"alphazero_draughts_iter_{it+1}.pt"
            torch.save({
                "iter": it + 1,
                "model_state": net.state_dict(),
                #"optimizer_state": optimizer.state_dict(),
            }, ckpt_path)

        elapsed_iter = time.time() - iter_start
        total_elapsed = time.time() - start_time
        print(f"Iter {it+1} finished | self-play positions={generated_positions} | "
              f"buffer size={len(buffer)} | elapsed={elapsed_iter:.1f}s | total={total_elapsed/3600:.2f}h")
        scheduler.step()


if __name__ == "__main__":
    # This is important for CUDA multiprocessing
    #mp.set_start_method('spawn')
    mp.set_start_method('spawn', force=True)
    main()