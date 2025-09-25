
import numpy as np
import torch
import torch.nn as nn



def setup_cuda_optimizations():
    if torch.cuda.is_available():
        # Prefer TF32 / fast matmuls on Ada
        try:
            torch.set_float32_matmul_precision("high")  # enables TF32 or similar fast paths
        except Exception:
            pass

        # cuDNN autotune & non-deterministic fast kernels
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Allow TF32 paths explicitly for matmul and cuDNN convolutions
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Optional: free cached blocks before large allocations (use sparingly)
        torch.cuda.empty_cache()

        # Optional: limit process VRAM usage if sharing GPU (leave commented by default)
        # torch.cuda.set_per_process_memory_fraction(0.9)

        return "cuda"
    else:
        print("CUDA not available, using CPU")
        return "cpu"
