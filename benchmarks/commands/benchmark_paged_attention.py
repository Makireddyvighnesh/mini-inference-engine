"""Microbenchmark the project-owned one-token paged-attention kernels."""

from __future__ import annotations

import argparse
import math
from statistics import median

import torch

from minillm_l4.engine.kv_cache.triton_paged_attention import (
    triton_paged_decode_attention,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=3072)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--split-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    return parser.parse_args()


def _time_variant(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    split_count: int,
    use_gqa_reuse: bool,
    block_tokens: int,
    max_sequence_length: int,
    warmups: int,
    repetitions: int,
) -> tuple[torch.Tensor, float]:
    kwargs = {
        "scale": 1.0 / math.sqrt(query.shape[-1]),
        "split_count": split_count,
        "max_sequence_length": max_sequence_length,
        "use_gqa_reuse": use_gqa_reuse,
        "block_tokens": block_tokens,
    }
    for _ in range(warmups):
        output = triton_paged_decode_attention(
            query,
            key_blocks,
            value_blocks,
            block_tables,
            sequence_lengths,
            **kwargs,
        )
    torch.cuda.synchronize(query.device)

    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = triton_paged_decode_attention(
            query,
            key_blocks,
            value_blocks,
            block_tables,
            sequence_lengths,
            **kwargs,
        )
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return output, median(samples)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if args.batch_size < 1 or args.sequence_length < 1:
        raise ValueError("batch size and sequence length must be positive")
    if args.query_heads % args.kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")

    torch.manual_seed(41)
    device = torch.device("cuda:0")
    blocks_per_request = math.ceil(args.sequence_length / args.block_size)
    total_blocks = args.batch_size * blocks_per_request
    query = torch.randn(
        args.batch_size,
        args.query_heads,
        1,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key_blocks = torch.randn(
        total_blocks,
        args.kv_heads,
        args.block_size,
        args.head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value_blocks = torch.randn_like(key_blocks)
    block_tables = torch.arange(
        total_blocks,
        dtype=torch.int32,
        device=device,
    ).reshape(args.batch_size, blocks_per_request)
    sequence_lengths = torch.full(
        (args.batch_size,),
        args.sequence_length,
        dtype=torch.int32,
        device=device,
    )

    reference, reference_ms = _time_variant(
        query,
        key_blocks,
        value_blocks,
        block_tables,
        sequence_lengths,
        split_count=1,
        use_gqa_reuse=False,
        block_tokens=16,
        max_sequence_length=args.sequence_length,
        warmups=args.warmups,
        repetitions=args.repetitions,
    )
    print(
        f"batch={args.batch_size} context={args.sequence_length} "
        f"q_heads={args.query_heads} kv_heads={args.kv_heads}"
    )
    for use_gqa_reuse in (False, True):
        for split_count in args.split_counts:
            for block_tokens in args.tile_sizes:
                output, elapsed_ms = _time_variant(
                    query,
                    key_blocks,
                    value_blocks,
                    block_tables,
                    sequence_lengths,
                    split_count=split_count,
                    use_gqa_reuse=use_gqa_reuse,
                    block_tokens=block_tokens,
                    max_sequence_length=args.sequence_length,
                    warmups=args.warmups,
                    repetitions=args.repetitions,
                )
                max_error = float(
                    (output.float() - reference.float()).abs().max().item()
                )
                print(
                    f"kernel={'gqa' if use_gqa_reuse else 'per_head':8s} "
                    f"splits={split_count:<2d} tile={block_tokens:<2d} "
                    f"median={elapsed_ms:.4f} ms "
                    f"speedup={reference_ms / elapsed_ms:.3f}x "
                    f"max_abs_error={max_error:.6f}"
                )


if __name__ == "__main__":
    main()
