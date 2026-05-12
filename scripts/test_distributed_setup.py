#!/usr/bin/env python3
"""
Quick test to verify distributed environment is correctly set up on the cluster.

Run this before the full training to diagnose env var issues:

    srun python scripts/test_distributed_setup.py

or (if platform already injects RANK/WORLD_SIZE):

    python scripts/test_distributed_setup.py

Expected output on a 2-node × 8-GPU setup:
    RANK=0 -> LOCAL_RANK=0 -> WORLD_SIZE=16 (node 0)
    RANK=8 -> LOCAL_RANK=0 -> WORLD_SIZE=16 (node 1)
    ...
"""

import os
import sys

import torch
import torch.distributed as dist


def main():
    print("=" * 60)
    print("Distributed Environment Diagnostic")
    print("=" * 60)

    env_vars = [
        "RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "SLURM_PROCID",
        "SLURM_LOCALID",
        "SLURM_NNODES",
        "SLURM_NTASKS_PER_NODE",
        "CUDA_VISIBLE_DEVICES",
        "NCCL_SOCKET_IFNAME",
        "PET_NNODES",
        "PET_NPROC_PER_NODE",
        "PET_NODE_RANK",
    ]

    print("\nEnvironment variables:")
    for var in env_vars:
        value = os.environ.get(var, "<not set>")
        print(f"  {var:30s} = {value}")

    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        print(f"  Device name: {torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'}")

    # Try to initialize process group (same logic as OLMo-core)
    backend = "cpu:gloo,cuda:nccl"
    try:
        if dist.is_initialized():
            print("\n[torch.distributed] Already initialized!")
        else:
            print("\n[torch.distributed] Initializing process group...")
            dist.init_process_group(backend=backend)
            print("[torch.distributed] Initialization successful!")

        print(f"\nDistributed status:")
        print(f"  is_distributed: True")
        print(f"  backend: {dist.get_backend()}")
        print(f"  rank: {dist.get_rank()}")
        print(f"  world_size: {dist.get_world_size()}")

        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(local_rank)
            print(f"  CUDA device set to: {local_rank}")

        # Test all-reduce
        tensor = torch.tensor([1.0], device="cuda" if torch.cuda.is_available() else "cpu")
        dist.all_reduce(tensor)
        expected = dist.get_world_size()
        print(f"\n[Test] All-reduce result: {tensor.item():.0f} (expected: {expected})")
        if abs(tensor.item() - expected) < 0.1:
            print("[Test] PASSED: Distributed communication works!")
        else:
            print("[Test] FAILED: All-reduce produced unexpected result!")

    except Exception as e:
        print(f"\n[ERROR] Distributed initialization failed:")
        print(f"  {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
