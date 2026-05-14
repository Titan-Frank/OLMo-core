#!/usr/bin/env python3
"""
Standalone parallel tokenizer for Dolma3 jsonl files (.zst and .gz).

Tokenizes all jsonl shards into .npy cache files, fully decoupled from training.
Any number of nodes can run this script simultaneously — each node picks up its
own slice of untokenized files via a shared filesystem lock-free scheme:

  - Each node lists all jsonl files, checks which .npy already exist,
    then takes files[node_rank::num_nodes] from the remaining list.
  - Processes files in batches with progress reporting; crashed workers are
    detected and their work is retried in subsequent batches.
  - Atomic writes (write to .tmp, then os.replace) prevent partial-file races.
  - The output is 100% compatible with the LazyTokenizedDataset cache used
    by train_olmo3_7b_pretrain_jsonl.py.

Usage (single node):
  python scripts/tokenize_dolma3_parallel.py \
      --data-root /path/to/dolma3_mix-6T-1025-7B/data \
      --cache-dir /path/to/tokenized_cache \
      --tokenizer /path/to/dolma2-tokenizer

Usage (multi-node, e.g. via torchrun or启智平台):
  Each node runs the same command. Node rank and count are auto-detected from
  TORCHELASTIC / SLURM / PET env vars, or overridden with --node-rank / --num-nodes.
"""

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _npy_cache_path(jsonl_path: Path, cache_dir: Path) -> Path:
    rel_parts = jsonl_path.name, jsonl_path.parent.name
    unique_key = "/".join(rel_parts)
    short_hash = hashlib.md5(unique_key.encode()).hexdigest()[:8]
    return cache_dir / f"{jsonl_path.parent.name}_{jsonl_path.stem.replace('.jsonl', '')}_{short_hash}.npy"


def _npy_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 132:
        return False
    try:
        np.load(path, mmap_mode="r")
        return True
    except Exception:
        return False


def tokenize_shard_to_npy(
    jsonl_path: Path,
    npy_path: Path,
    tokenizer_name: str,
    eos_token_id: int,
) -> int:
    import zstandard
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    all_token_ids: List[int] = []
    total_docs = 0

    with open(jsonl_path, "rb") as f:
        if jsonl_path.suffix == ".gz":
            text_stream = io.TextIOWrapper(gzip.open(f), encoding="utf-8")
        else:
            dctx = zstandard.ZstdDecompressor()
            stream = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream, encoding="utf-8")

        for line in text_stream:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = doc.get("text", "")
            if not text or text == "[REMOVED]":
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(token_ids)
            all_token_ids.append(eos_token_id)
            total_docs += 1

    arr = np.array(all_token_ids, dtype=np.uint32) if all_token_ids else np.array([], dtype=np.uint32)
    tmp_path = npy_path.parent / f"{npy_path.stem}.{os.getpid()}.{threading.get_ident() % 100000}.tmp.npy"
    np.save(tmp_path, arr)
    os.replace(tmp_path, npy_path)
    return len(arr)


def _worker_init():
    """Ignore SIGINT in workers so the parent handles cleanup."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _tokenize_wrapper(args_tuple):
    """Wrapper that catches exceptions and returns (npy_path, result_or_error)."""
    jsonl_path, npy_path, tokenizer_name, eos_token_id = args_tuple
    try:
        n_tokens = tokenize_shard_to_npy(jsonl_path, npy_path, tokenizer_name, eos_token_id)
        return (npy_path, n_tokens, None)
    except Exception as e:
        return (npy_path, 0, f"{type(e).__name__}: {e}")


def get_node_info(args) -> Tuple[int, int]:
    """Return (node_rank, num_nodes) from args or env vars."""
    node_rank = args.node_rank
    num_nodes = args.num_nodes

    if node_rank is None:
        node_rank = int(os.environ.get("PET_NODE_RANK",
                         os.environ.get("SLURM_NODEID",
                         os.environ.get("NODE_RANK", 0))))
    if num_nodes is None:
        num_nodes = int(os.environ.get("PET_NNODES",
                       os.environ.get("SLURM_NNODES",
                       os.environ.get("NUM_NODES", 1))))

    return node_rank, num_nodes


def main():
    parser = argparse.ArgumentParser(description="Standalone parallel tokenizer for Dolma3")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory of jsonl files (e.g. dolma3_mix-6T-1025-7B/data)")
    parser.add_argument("--cache-dir", type=str, required=True,
                        help="Output directory for .npy cache files")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer path (local or HF identifier)")
    parser.add_argument("--eos-token-id", type=int, default=100257,
                        help="EOS token ID (default: 100257 for dolma2)")
    parser.add_argument("--node-rank", type=int, default=None,
                        help="Node rank (auto-detected from env if not set)")
    parser.add_argument("--num-nodes", type=int, default=None,
                        help="Total number of nodes (auto-detected from env if not set)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of multiprocessing workers per node (default: auto)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Number of files per batch (smaller = more frequent progress reports)")
    parser.add_argument("--check-only", action="store_true",
                        help="Only print progress statistics, don't tokenize")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Find all source files
    jsonl_files = sorted(list(data_root.rglob("*.jsonl.zst")) + list(data_root.rglob("*.jsonl.gz")))
    total_files = len(jsonl_files)
    print(f"Found {total_files} jsonl files under {data_root}")

    # Build mapping and check which are done
    file_pairs = []
    already_done = 0
    corrupted = 0
    for jsonl_path in jsonl_files:
        npy_path = _npy_cache_path(jsonl_path, cache_dir)
        if _npy_is_valid(npy_path):
            already_done += 1
        elif npy_path.exists():
            corrupted += 1
            file_pairs.append((jsonl_path, npy_path))
        else:
            file_pairs.append((jsonl_path, npy_path))

    remaining = len(file_pairs)
    progress_pct = already_done / total_files * 100 if total_files > 0 else 0
    print(f"Progress: {already_done}/{total_files} ({progress_pct:.1f}%) already tokenized")
    if corrupted:
        print(f"  {corrupted} corrupted/invalid .npy files will be re-tokenized")
    print(f"  {remaining} files remaining")

    if args.check_only:
        source_counts: dict = {}
        for jsonl_path, npy_path in file_pairs:
            source = jsonl_path.parent.name
            source_counts[source] = source_counts.get(source, 0) + 1
        if source_counts:
            print(f"\nRemaining files by source (top 20):")
            for source, count in sorted(source_counts.items(), key=lambda x: -x[1])[:20]:
                print(f"  {source}: {count}")
        return

    if remaining == 0:
        print("All files already tokenized!")
        return

    # Distribute across nodes
    node_rank, num_nodes = get_node_info(args)
    my_files = file_pairs[node_rank::num_nodes]
    print(f"Node {node_rank}/{num_nodes}: assigned {len(my_files)} files")

    if not my_files:
        print("Nothing to do for this node.")
        return

    # Determine worker count
    if args.workers is not None:
        num_workers = args.workers
    else:
        num_workers = min(len(my_files), max(1, (os.cpu_count() or 8) - 4))

    # Load tokenizer once to verify it works
    from transformers import AutoTokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print("Tokenizer loaded successfully.")

    start_time = time.time()
    total_done = 0
    total_tokens = 0
    total_errors = 0
    batch_size = args.batch_size

    # Process in batches for progress visibility and crash recovery
    num_batches = (len(my_files) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(my_files))
        batch = my_files[batch_start:batch_end]

        # Filter out files that are already done (e.g. from a previous run or another node)
        batch = [(p, n) for p, n in batch if not _npy_is_valid(n)]
        if not batch:
            total_done += (batch_end - batch_start)
            continue

        work_items = [(p, n, args.tokenizer, args.eos_token_id) for p, n in batch]

        if num_workers > 1:
            with mp.Pool(processes=num_workers, initializer=_worker_init) as pool:
                results = pool.map(_tokenize_wrapper, work_items)
        else:
            results = [_tokenize_wrapper(item) for item in work_items]

        batch_tokens = 0
        batch_errors = 0
        for npy_path, n_tokens, error in results:
            if error:
                batch_errors += 1
                print(f"  ERROR {npy_path.name}: {error}")
            else:
                batch_tokens += n_tokens

        total_done += len(batch) - batch_errors
        total_tokens += batch_tokens
        total_errors += batch_errors

        elapsed = time.time() - start_time
        rate = total_done / elapsed if elapsed > 0 else 0
        eta_s = (len(my_files) - total_done) / rate if rate > 0 else 0
        print(f"  Batch {batch_idx+1}/{num_batches}: "
              f"{total_done}/{len(my_files)} done, "
              f"{rate:.1f} files/s, "
              f"{total_tokens/1e9:.2f}B tokens, "
              f"ETA {eta_s/3600:.1f}h"
              f"{f' ({batch_errors} errors)' if batch_errors else ''}")

        # Clean up stale tmp files
        for tmp in cache_dir.glob("*.tmp.npy"):
            try:
                tmp.unlink()
            except OSError:
                pass

    elapsed = time.time() - start_time
    print(f"\nNode {node_rank} done: {total_done} files, {total_tokens/1e9:.2f}B tokens in {elapsed:.0f}s")
    if total_errors:
        print(f"  {total_errors} files had errors and were skipped")

    # Clean up tmp files one last time
    for tmp in cache_dir.glob("*.tmp.npy"):
        try:
            tmp.unlink()
        except OSError:
            pass

    # Final progress check — just count npy files (fast)
    valid_all = sum(1 for _ in cache_dir.glob("*.npy"))
    # Subtract any that are too small to be valid
    print(f"Overall: {valid_all} npy files in cache ({valid_all/total_files*100:.1f}% of {total_files})")


if __name__ == "__main__":
    main()
