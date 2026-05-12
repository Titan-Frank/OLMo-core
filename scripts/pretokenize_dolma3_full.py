#!/usr/bin/env python3
"""
分布式并行预处理 Dolma3 全量 jsonl.zst 数据为 OLMo-core 所需的 .npy 格式。

特性：
    - **跨节点分布式分片**：自动识别 SLURM / MPI 环境变量，按 rank 划分文件，
      支持多节点同时写入同一输出目录（需共享文件系统）。
    - **节点内多进程并行**：每个节点内部使用 ``--workers`` 进程处理本地分到的文件。
    - **断点续传**：跳过已存在的 .npy，失败后清理并允许重试。
    - **处理完成后输出每个 source 的 token 统计报告**。

用法示例：

单机（单个SLURM task）：
    python scripts/pretokenize_dolma3_full.py \
        --data-root /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data \
        --output-root /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized \
        --workers 120

多节点分布式（每个节点享有一个SBATCH task，通过 srun 启动）：
    srun python scripts/pretokenize_dolma3_full.py \
        --data-root /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/dolma3_mix-6T-1025-7B/data \
        --output-root /inspire/qb-ilm/project/ai4education/public/wwb/datasets/dolma3/pretrain/tokenized

只处理一个子目录（快速测试）：
    python scripts/pretokenize_dolma3_full.py ... --source-filter "dolma1_7-wiki-en"

只扫描并打印分片结果：
    python scripts/pretokenize_dolma3_full.py ... --dry-run
"""

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zstandard
from tqdm import tqdm
from transformers import AutoTokenizer

EOS_TOKEN_ID = 100257
BATCH_SIZE_LINES = 10000


def _get_global_rank() -> int:
    """从各种分布式环境变量中推断全局 rank。"""
    for env in (
        "RANK",
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "SLURM_PROCID",
        "MV2_COMM_WORLD_RANK",
    ):
        if env in os.environ:
            return int(os.environ[env])
    return 0


def _get_global_world_size() -> int:
    """从各种分布式环境变量中推断全局 world size。"""
    for env in (
        "WORLD_SIZE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "SLURM_NTASKS",
        "MV2_COMM_WORLD_SIZE",
    ):
        if env in os.environ:
            return int(os.environ[env])
    return 1


def _get_local_rank() -> int:
    """获取节点内 local rank（用于设置 CUDA，此处纯 CPU 任务也可用于进度条前缀）。"""
    for env in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"):
        if env in os.environ:
            return int(os.environ[env])
    return 0


def find_sources(data_root: str) -> List[str]:
    """列出数据根目录下所有 source 子目录名。"""
    root = Path(data_root)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def find_all_shards(source_dir: str) -> List[Path]:
    """递归查找某个 source 下的所有 *.jsonl.zst 文件。"""
    return sorted(Path(source_dir).rglob("*.jsonl.zst"))


def get_output_path(input_path: Path, data_root: str, output_root: str) -> Path:
    """将输入路径映射为输出路径：保持相对目录结构，仅替换后缀。"""
    rel = input_path.relative_to(data_root)
    out_name = rel.name.replace(".jsonl.zst", ".npy")
    return Path(output_root) / rel.parent / out_name


def process_shard(input_path: Path, output_path: Path, tokenizer_name: str) -> Tuple[str, int, bool, str]:
    """
    处理单个 shard。
    返回 (input_path_str, token_count, is_processed_now, error_msg)。
    如果输出已存在则直接跳过，token_count 返回 0。
    """
    if output_path.exists():
        return str(input_path), 0, False, ""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    all_token_ids: List[int] = []
    token_count = 0

    try:
        with open(input_path, "rb") as f:
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
                all_token_ids.append(EOS_TOKEN_ID)
                token_count += len(token_ids) + 1

        if token_count == 0:
            np.save(output_path, np.array([], dtype=np.uint32))
            return str(input_path), 0, True, ""

        arr = np.array(all_token_ids, dtype=np.uint32)
        np.save(output_path, arr)
        return str(input_path), token_count, True, ""

    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        return str(input_path), -1, True, str(e)


def process_with_progress(task):
    """Pool worker: 处理单个 shard 并提取 source 名称。"""
    shard, out, tok, data_root = task
    path_str, tokens, processed, error_msg = process_shard(shard, out, tok)
    src_name = Path(path_str).relative_to(data_root).parts[0]
    return path_str, tokens, processed, src_name, error_msg


def main():
    parser = argparse.ArgumentParser(description="Tokenize Dolma3 full dataset to .npy")
    parser.add_argument("--data-root", type=str, required=True, help="原始 data/ 目录")
    parser.add_argument("--output-root", type=str, required=True, help="输出 tokenized/ 目录")
    parser.add_argument("--tokenizer", type=str, default="allenai/dolma2-tokenizer")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="每个节点内的并行进程数。0 表示自动使用 CPU 核心数。"
    )
    parser.add_argument(
        "--source-filter", type=str, default=None,
        help="只处理匹配的 source 目录名（支持子字符串匹配）"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="只扫描并打印分片结果，不实际处理")
    parser.add_argument("--rank", type=int, default=None,
                        help="手动指定全局 rank（覆盖自动检测）")
    parser.add_argument("--world-size", type=int, default=None,
                        help="手动指定全局 world size（覆盖自动检测）")
    opts = parser.parse_args()

    data_root = opts.data_root.rstrip("/")
    output_root = opts.output_root.rstrip("/")

    # 分布式环境（手动参数优先于环境变量）
    global_rank = opts.rank if opts.rank is not None else _get_global_rank()
    global_world_size = opts.world_size if opts.world_size is not None else _get_global_world_size()
    local_rank = _get_local_rank()
    workers = opts.workers if opts.workers > 0 else mp.cpu_count()

    os.makedirs(output_root, exist_ok=True)

    # 1. 查找所有 source
    sources = find_sources(data_root)
    if opts.source_filter:
        sources = [s for s in sources if opts.source_filter in s]
        if not sources:
            print(f"没有 source 匹配过滤条件: {opts.source_filter}")
            sys.exit(1)

    # 2. 收集所有 shard 文件
    all_tasks: List[Tuple[Path, Path, str, str]] = []
    for src in sources:
        src_dir = Path(data_root) / src
        shards = find_all_shards(str(src_dir))
        for shard in shards:
            out_path = get_output_path(shard, data_root, output_root)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            all_tasks.append((shard, out_path, opts.tokenizer, data_root))

    # 3. 全局分片：每个 rank 只处理自己分到的文件
    # 按 shard 路径哈希排序后均匀分片，保证结果稳定可复现
    all_tasks.sort(key=lambda t: str(t[0]))
    my_tasks = [t for i, t in enumerate(all_tasks) if i % global_world_size == global_rank]

    print("=" * 60)
    print(f"全局 rank: {global_rank} / {global_world_size}")
    print(f"本节点 local_rank: {local_rank}, workers: {workers}")
    print(f"总 shards: {len(all_tasks)}")
    print(f"本 rank 分到: {len(my_tasks)}")
    print("=" * 60)

    # 统计已存在的文件（只在本地统计，避免跨节点竞争）
    existing = sum(1 for _, out, _, _ in my_tasks if out.exists())
    print(f"其中已有 {existing} 个 .npy 存在（将被跳过），本次需处理 {len(my_tasks) - existing} 个")

    if opts.dry_run:
        print("\n[DRY RUN] 本 rank 待处理文件示例（前 10 个未处理的）:")
        shown = 0
        for shard, out, _, _ in my_tasks:
            if not out.exists() and shown < 10:
                print(f"  {shard} -> {out}")
                shown += 1
        sys.exit(0)

    if len(my_tasks) - existing == 0:
        print("本 rank 所有 shard 已处理完毕！")
        sys.exit(0)

    # 4. 并行处理（节点内多进程）
    start_time = time.time()
    results = {"ok": 0, "skipped": 0, "error": 0}
    source_stats: dict[str, int] = {}

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            iterator = pool.imap_unordered(process_with_progress, my_tasks, chunksize=max(1, len(my_tasks) // workers // 4))
            for path_str, tokens, processed, src_name, error_msg in tqdm(
                iterator, total=len(my_tasks), desc=f"Rank {global_rank} tokenizing", unit="shard",
                position=local_rank, leave=True,
            ):
                if not processed:
                    results["skipped"] += 1
                elif tokens < 0:
                    results["error"] += 1
                    print(f"\n[ERROR][rank {global_rank}] {path_str}: {error_msg}")
                else:
                    results["ok"] += 1
                    source_stats[src_name] = source_stats.get(src_name, 0) + tokens
    else:
        for task in tqdm(my_tasks, desc=f"Rank {global_rank} tokenizing", unit="shard"):
            path_str, tokens, processed, src_name, error_msg = process_with_progress(task)
            if not processed:
                results["skipped"] += 1
            elif tokens < 0:
                results["error"] += 1
                print(f"\n[ERROR][rank {global_rank}] {path_str}: {error_msg}")
            else:
                results["ok"] += 1
                source_stats[src_name] = source_stats.get(src_name, 0) + tokens

    elapsed = time.time() - start_time
    total_tokens = sum(source_stats.values())

    print("\n" + "=" * 60)
    print(f"[Rank {global_rank} / {global_world_size}] 处理完成！耗时 {elapsed / 60:.1f} 分钟")
    print(f"  成功:   {results['ok']}")
    print(f"  跳过:   {results['skipped']}")
    print(f"  失败:   {results['error']}")
    print(f"  总 tokens: {total_tokens:,} ({total_tokens / 1e12:.2f}T)")
    print("\n各 source token 统计:")
    for src, count in sorted(source_stats.items(), key=lambda x: -x[1]):
        pct = count / total_tokens * 100 if total_tokens else 0
        print(f"  {src:60s} {count:>18,} ({pct:5.2f}%)")

    if results["error"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
