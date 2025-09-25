# AlphaZero for American Draughts

This repository implements the AlphaZero algorithm for American draughts (checkers) using a stateless engine, batched neural network evaluations inside MCTS, and self-play training with a large replay buffer.
The design follows DeepMind’s AlphaZero framework with root Dirichlet noise, visit-count policies, and a shared CNN trunk with separate policy and value heads, adapted here to an 8×8 draughts board and legal multi-jump captures.

### Highlights

- American-draughts engine with correct multi-jump captures, promotion, draw rules, and canonicalization for learning.
- AlphaZero-style MCTS with PUCT, root Dirichlet noise, and batched leaf evaluation for throughput on GPUs.
- Residual policy–value network in PyTorch with 5 input planes and heads sized to board_size^4 action space.
- Self-play pipeline with dynamic parallelization and a 200,000-sample FIFO replay buffer for on-policy training.
- Evaluation via pit.py to pit models against themselves or each other with alternating starts and temperature 0.
- Optimized defaults for RTX Ada 4000 SFF 20GB and a 10‑core Xeon with TF32 paths, cudnn autotune, and thread pinning.
- I have upload on .pt file for you to try 


### Acknowledgments

This project is inspired by AlphaZero’s general self-play RL algorithm and its application to chess, shogi, and Go.
Root Dirichlet noise and its α/ε hyperparameters follow the AlphaZero approach, with evidence for game-dependent α choices in prior work.

## Project overview

The engine exposes a clean API for observation construction, legal move generation including capture sequences, and next-state transitions; MCTS consumes this API and returns a visit-count policy; the network predicts policy logits over board_size^4 moves and a scalar value in [-1, 1].
Training iterates between parallel self-play to grow the buffer and multi-epoch updates over large batches with mixed-precision on CUDA when available.

## Key files

- draughts_game.py: Stateless American-draughts rules with multi-jump capture enumeration, promotions, draw detection, and canonical forms for learning.
- mcts.py: PUCT MCTS with batched leaf evaluation, root Dirichlet noise, and visit-count policy extraction.
- alpha_policy.py: Residual CNN trunk with 1×1 policy/value heads, policy to board_size^4 logits, and Tanh value head.
- self_play.py: Self-play driver with temperature scheduling, optional curated starting positions for king understanding, and outcome labeling.
- replay_buffer.py: FIFO buffer with maxlen for training tuples (observation, policy, outcome).
- train_optim.py: Orchestrates self-play using ProcessPoolExecutor, training with DataLoader, AMP, AdamW, and periodic checkpoints.
- pit.py: Model-vs-model arena with alternating starts, temperature 0, and result summaries for quick regression testing.
- cuda_optim.py: Hardware optimizations including TF32, cuDNN autotune, and optional CUDA cache management.
- env_board.py: Gymnasium wrapper for the engine exposing action/observation spaces and a step/reset interface.
- config.json: Canonical hyperparameters for board size, network width/depth, MCTS, replay buffer, epochs, and optimizer.


## Dependencies

The codebase uses Python with PyTorch for the model, Gymnasium for an optional environment interface, NumPy for array ops, tqdm for progress, and standard multiprocessing/concurrency utilities.
CUDA mixed precision and TF32 paths are enabled automatically when a compatible NVIDIA GPU is available.

## Install

- Create a virtual environment, install PyTorch with CUDA build suited to the system, and install numpy, gymnasium, and tqdm as used in the code.
- Confirm nvidia-smi and torch.cuda.is_available() report the expected RTX Ada 4000 SFF 20GB and CUDA runtime on the host.


## Configuration

Hyperparameters are defined in config.json and read at runtime, including board size, model width/depth, MCTS settings, replay buffer size, batch size, epoch schedule, and optimizer parameters.
Defaults include board_size=8, num_res_blocks=5, num_channels=128, num_mcts_sims=200, c_puct=1.5, temperature=0.8, temp_threshold=50, dirichlet_alpha=1.15, dirichlet_epsilon=0.25, buffer_size=200000, batch_size=8192, lr=3e-4, and num_games_per_iter=200.

## Training

Run the main training script to alternate self-play generation and supervised updates with large-batch training on GPU where available; the script prints epoch losses and saves checkpoints periodically.
Self-play uses dynamic task distribution across processes, transferring a CPU-shared network state dict and reseeding each worker for diversity and reproducibility.

Example:

- Configure config.json to desired values for MCTS sims, buffer size, batch size, and iterations as provided in the sample config.
- Use the provided launcher which sets OMP/MKL threads to 10 and torch.set_num_threads(10) for the intended Xeon 10‑core machine profile.


## Evaluation (pit)

Use pit.py to pit two checkpoints over an even number of games with alternating starts; evaluation runs with temperature 0 and prints cumulative scores including draws.
The script loads models with the same architecture parameters, constructs MCTS instances, and selects legal moves under the corrected capture sequence handling.

Example:

- Provide two checkpoint files saved by the training script and set num_games to an even value to balance starting sides, with optional console rendering.
- Expect printed summaries of wins as Player 1, wins as Player 2, and draws, including percentages over the match.


## Model

The network takes a 5×8×8 observation with separate planes for each side’s men and kings plus a bias plane, then passes through an initial conv, a stack of residual blocks, and 1×1 heads for policy and value.
The policy head flattens to board_size^4 logits representing from/to indices, while the value head compresses to 4 channels before a Tanh scalar in [-1, 1].

## Environment

An optional Gymnasium env wrapper exposes Discrete action space sized to the engine action encoding and a 5-channel Box observation, with illegal moves treated as immediate terminal losses to surface masking bugs early.
The wrapper relays to the stateless engine for step/termination logic and provides a minimal console renderer for debugging.

## MCTS details

Selection uses PUCT with exploitation term derived from child Q flipped back to the parent’s perspective and exploration U scaled by c_puct and priors, with tree reuse during search.
Root Dirichlet noise is applied once per search to encourage exploration at the root, mixing α-parameterized Dirichlet samples with the policy prior, consistent with AlphaZero practice; this code sets α via config (1.15 here) and uses ε mixing at the root as in the original method.

## Hardware notes

CUDA is auto-detected; on GPUs the script enables TF32 matmul paths and cuDNN autotuning, which are well-suited to Ada architecture and large batched convolutions.
Thread affinity is tuned via OMP/MKL environment variables and torch.set_num_threads(10) to match a 10‑core Xeon profile, improving data-loader and host-side throughput.

## Hyperparameters of interest

- Replay buffer: 200,000 samples (FIFO) to balance freshness and coverage for large-batch updates.
- Batch and eval sizes: batch_size=8192 and eval_batch_size=8192 to exploit GPU memory and throughput on 20GB VRAM.
- MCTS: num_mcts_sims=200 and c_puct=1.5 provide a practical exploration–exploitation balance for 8×8 draughts.
- Temperature schedule: temperature=0.8 for the first 50 plies, then 0 for deterministic move selection later in self-play.
- Dirichlet parameters: dirichlet_alpha=1.15 with ε mixing at the root, consistent with AlphaZero’s exploration mechanism and adapted here to draughts branching.


## Reproducibility

Global seeding is applied across Python, NumPy, and PyTorch, with per-worker reseeding in self-play to ensure diversity across matches while remaining deterministic per seed.
Checkpoints store model state and training logs include per-epoch loss summaries to facilitate regression testing against prior runs.

## How this differs from AlphaZero

This implementation targets 8×8 American draughts with explicit multi-jump capture sequences, distinct observation planes, and move encoding designed for the action-space factorization board_size^4.
Root exploration and PUCT follow AlphaZero, while the α value is chosen at 1.15 for this domain per configuration and testing, acknowledging that DeepMind varied α across games and that α materially affects search bias.

## References

- AlphaZero general algorithm for chess, shogi, and Go, including root exploration and self-play training.
- Hyperparameter choice and Dirichlet α’s role as search bias with game-dependent recommendations.
