"""Fused Triton kernels for one-token paged-KV decode attention.

The default kernel is a readable one-pass online-softmax implementation.  For
long contexts, the optional split-KV path divides the logical sequence into
independent chunks, computes partial softmax reductions in parallel, and
combines those partials in a second kernel.  Both paths read physical pages
through the block table and never materialize a dense KV tensor.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only without Triton installed
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _paged_decode_gqa_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        block_table_ptr,
        sequence_length_ptr,
        output_ptr,
        scale,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        query_stride_token: tl.constexpr,
        query_stride_dim: tl.constexpr,
        key_stride_block: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_token: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_block: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_dim: tl.constexpr,
        table_stride_batch: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_head: tl.constexpr,
        output_stride_token: tl.constexpr,
        output_stride_dim: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        """Decode one KV head and all of its grouped query heads together.

        Qwen3 has four query heads for every KV head.  The older per-head
        kernel loaded the same K/V page four times.  This program keeps the
        shared K/V tile live while computing all four query heads, reducing
        global-memory traffic without changing the page-table representation.
        """

        program_id = tl.program_id(0)
        batch_index = program_id // NUM_KV_HEADS
        kv_head = program_id % NUM_KV_HEADS

        group_offsets = tl.arange(0, GROUP_SIZE)
        query_heads = kv_head * GROUP_SIZE + group_offsets
        dim_offsets = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < HEAD_DIM
        query_offsets = (
            batch_index * query_stride_batch
            + query_heads[:, None] * query_stride_head
            + dim_offsets[None, :] * query_stride_dim
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=dim_mask[None, :],
            other=0.0,
        )

        sequence_length = tl.load(sequence_length_ptr + batch_index)
        tile_count = tl.cdiv(sequence_length, BLOCK_TOKENS)
        token_offsets = tl.arange(0, BLOCK_TOKENS)
        running_max = tl.full((GROUP_SIZE,), -float("inf"), tl.float32)
        running_sum = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        accumulator = tl.zeros((GROUP_SIZE, BLOCK_DIM), dtype=tl.float32)

        for tile_index in tl.range(0, tile_count):
            logical_tokens = tile_index * BLOCK_TOKENS + token_offsets
            token_mask = logical_tokens < sequence_length
            logical_blocks = logical_tokens // BLOCK_SIZE
            offsets_in_block = logical_tokens % BLOCK_SIZE
            physical_blocks = tl.load(
                block_table_ptr
                + batch_index * table_stride_batch
                + logical_blocks,
                mask=token_mask,
                other=0,
            )

            key_offsets = (
                physical_blocks[:, None] * key_stride_block
                + kv_head * key_stride_head
                + offsets_in_block[:, None] * key_stride_token
                + dim_offsets[None, :] * key_stride_dim
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(
                query[:, None, :] * keys[None, :, :],
                axis=2,
            ) * scale
            scores = tl.where(token_mask[None, :], scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            previous_weight = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(token_mask[None, :], probabilities, 0.0)

            value_offsets = (
                physical_blocks[:, None] * value_stride_block
                + kv_head * value_stride_head
                + offsets_in_block[:, None] * value_stride_token
                + dim_offsets[None, :] * value_stride_dim
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator = (
                accumulator * previous_weight[:, None]
                + tl.sum(
                    probabilities[:, :, None] * values[None, :, :],
                    axis=1,
                )
            )
            running_sum = (
                running_sum * previous_weight + tl.sum(probabilities, axis=1)
            )
            running_max = new_max

        output = accumulator / running_sum[:, None]
        output_offsets = (
            batch_index * output_stride_batch
            + query_heads[:, None] * output_stride_head
            + dim_offsets[None, :] * output_stride_dim
        )
        tl.store(
            output_ptr + output_offsets,
            output,
            mask=dim_mask[None, :],
        )

    @triton.jit
    def _paged_decode_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        block_table_ptr,
        sequence_length_ptr,
        output_ptr,
        scale,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        query_stride_token: tl.constexpr,
        query_stride_dim: tl.constexpr,
        key_stride_block: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_token: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_block: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_dim: tl.constexpr,
        table_stride_batch: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_head: tl.constexpr,
        output_stride_token: tl.constexpr,
        output_stride_dim: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        program_id = tl.program_id(0)
        batch_index = program_id // NUM_HEADS
        query_head = program_id % NUM_HEADS
        kv_head = query_head // (NUM_HEADS // NUM_KV_HEADS)

        dim_offsets = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < HEAD_DIM
        query_offsets = (
            batch_index * query_stride_batch
            + query_head * query_stride_head
            + dim_offsets * query_stride_dim
        )
        query = tl.load(query_ptr + query_offsets, mask=dim_mask, other=0.0)

        sequence_length = tl.load(sequence_length_ptr + batch_index)
        tile_count = tl.cdiv(sequence_length, BLOCK_TOKENS)
        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        token_offsets = tl.arange(0, BLOCK_TOKENS)

        for tile_index in tl.range(0, tile_count):
            logical_tokens = tile_index * BLOCK_TOKENS + token_offsets
            token_mask = logical_tokens < sequence_length
            logical_blocks = logical_tokens // BLOCK_SIZE
            offsets_in_block = logical_tokens % BLOCK_SIZE
            physical_blocks = tl.load(
                block_table_ptr
                + batch_index * table_stride_batch
                + logical_blocks,
                mask=token_mask,
                other=0,
            )

            key_offsets = (
                physical_blocks[:, None] * key_stride_block
                + kv_head * key_stride_head
                + offsets_in_block[:, None] * key_stride_token
                + dim_offsets[None, :] * key_stride_dim
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(keys * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            previous_weight = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            probabilities = tl.where(token_mask, probabilities, 0.0)

            value_offsets = (
                physical_blocks[:, None] * value_stride_block
                + kv_head * value_stride_head
                + offsets_in_block[:, None] * value_stride_token
                + dim_offsets[None, :] * value_stride_dim
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator = accumulator * previous_weight + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_sum = (
                running_sum * previous_weight + tl.sum(probabilities, axis=0)
            )
            running_max = new_max

        output = accumulator / running_sum
        output_offsets = (
            batch_index * output_stride_batch
            + query_head * output_stride_head
            + dim_offsets * output_stride_dim
        )
        tl.store(output_ptr + output_offsets, output, mask=dim_mask)


    @triton.jit
    def _paged_decode_gqa_split_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        block_table_ptr,
        sequence_length_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        partial_output_ptr,
        scale,
        split_span,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        query_stride_token: tl.constexpr,
        query_stride_dim: tl.constexpr,
        key_stride_block: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_token: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_block: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_dim: tl.constexpr,
        table_stride_batch: tl.constexpr,
        partial_max_stride_batch: tl.constexpr,
        partial_max_stride_head: tl.constexpr,
        partial_max_stride_split: tl.constexpr,
        partial_sum_stride_batch: tl.constexpr,
        partial_sum_stride_head: tl.constexpr,
        partial_sum_stride_split: tl.constexpr,
        partial_output_stride_batch: tl.constexpr,
        partial_output_stride_head: tl.constexpr,
        partial_output_stride_split: tl.constexpr,
        partial_output_stride_dim: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        SPLIT_COUNT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        """Compute one split-KV partial while sharing K/V across GQA heads."""

        program_id = tl.program_id(0)
        split_index = program_id % SPLIT_COUNT
        kv_program = program_id // SPLIT_COUNT
        batch_index = kv_program // NUM_KV_HEADS
        kv_head = kv_program % NUM_KV_HEADS

        group_offsets = tl.arange(0, GROUP_SIZE)
        query_heads = kv_head * GROUP_SIZE + group_offsets
        dim_offsets = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < HEAD_DIM
        query_offsets = (
            batch_index * query_stride_batch
            + query_heads[:, None] * query_stride_head
            + dim_offsets[None, :] * query_stride_dim
        )
        query = tl.load(
            query_ptr + query_offsets,
            mask=dim_mask[None, :],
            other=0.0,
        )

        sequence_length = tl.load(sequence_length_ptr + batch_index)
        split_start = split_index * split_span
        split_end = tl.minimum(sequence_length, split_start + split_span)
        tokens_in_split = tl.maximum(split_end - split_start, 0)
        tile_count = tl.cdiv(tokens_in_split, BLOCK_TOKENS)
        token_offsets = tl.arange(0, BLOCK_TOKENS)
        running_max = tl.full((GROUP_SIZE,), -float("inf"), tl.float32)
        running_sum = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
        accumulator = tl.zeros((GROUP_SIZE, BLOCK_DIM), dtype=tl.float32)

        for tile_index in tl.range(0, tile_count):
            logical_tokens = split_start + tile_index * BLOCK_TOKENS + token_offsets
            token_mask = logical_tokens < split_end
            logical_blocks = logical_tokens // BLOCK_SIZE
            offsets_in_block = logical_tokens % BLOCK_SIZE
            physical_blocks = tl.load(
                block_table_ptr
                + batch_index * table_stride_batch
                + logical_blocks,
                mask=token_mask,
                other=0,
            )

            key_offsets = (
                physical_blocks[:, None] * key_stride_block
                + kv_head * key_stride_head
                + offsets_in_block[:, None] * key_stride_token
                + dim_offsets[None, :] * key_stride_dim
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(
                query[:, None, :] * keys[None, :, :],
                axis=2,
            ) * scale
            scores = tl.where(token_mask[None, :], scores, -float("inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            previous_weight = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(token_mask[None, :], probabilities, 0.0)

            value_offsets = (
                physical_blocks[:, None] * value_stride_block
                + kv_head * value_stride_head
                + offsets_in_block[:, None] * value_stride_token
                + dim_offsets[None, :] * value_stride_dim
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator = (
                accumulator * previous_weight[:, None]
                + tl.sum(
                    probabilities[:, :, None] * values[None, :, :],
                    axis=1,
                )
            )
            running_sum = (
                running_sum * previous_weight + tl.sum(probabilities, axis=1)
            )
            running_max = new_max

        partial_max_offsets = (
            batch_index * partial_max_stride_batch
            + query_heads * partial_max_stride_head
            + split_index * partial_max_stride_split
        )
        partial_sum_offsets = (
            batch_index * partial_sum_stride_batch
            + query_heads * partial_sum_stride_head
            + split_index * partial_sum_stride_split
        )
        partial_output_offsets = (
            batch_index * partial_output_stride_batch
            + query_heads[:, None] * partial_output_stride_head
            + split_index * partial_output_stride_split
            + dim_offsets[None, :] * partial_output_stride_dim
        )
        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        tl.store(partial_max_ptr + partial_max_offsets, running_max)
        tl.store(partial_sum_ptr + partial_sum_offsets, running_sum)
        tl.store(
            partial_output_ptr + partial_output_offsets,
            accumulator / denominator[:, None],
            mask=dim_mask[None, :],
        )


    @triton.jit
    def _paged_decode_split_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        block_table_ptr,
        sequence_length_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        partial_output_ptr,
        scale,
        split_span,
        query_stride_batch: tl.constexpr,
        query_stride_head: tl.constexpr,
        query_stride_token: tl.constexpr,
        query_stride_dim: tl.constexpr,
        key_stride_block: tl.constexpr,
        key_stride_head: tl.constexpr,
        key_stride_token: tl.constexpr,
        key_stride_dim: tl.constexpr,
        value_stride_block: tl.constexpr,
        value_stride_head: tl.constexpr,
        value_stride_token: tl.constexpr,
        value_stride_dim: tl.constexpr,
        table_stride_batch: tl.constexpr,
        partial_max_stride_batch: tl.constexpr,
        partial_max_stride_head: tl.constexpr,
        partial_max_stride_split: tl.constexpr,
        partial_sum_stride_batch: tl.constexpr,
        partial_sum_stride_head: tl.constexpr,
        partial_sum_stride_split: tl.constexpr,
        partial_output_stride_batch: tl.constexpr,
        partial_output_stride_head: tl.constexpr,
        partial_output_stride_split: tl.constexpr,
        partial_output_stride_dim: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        SPLIT_COUNT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        """Compute one online-softmax partial for one request/head/chunk."""

        program_id = tl.program_id(0)
        split_index = program_id % SPLIT_COUNT
        head_program = program_id // SPLIT_COUNT
        batch_index = head_program // NUM_HEADS
        query_head = head_program % NUM_HEADS
        kv_head = query_head // (NUM_HEADS // NUM_KV_HEADS)

        dim_offsets = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < HEAD_DIM
        query_offsets = (
            batch_index * query_stride_batch
            + query_head * query_stride_head
            + dim_offsets * query_stride_dim
        )
        query = tl.load(query_ptr + query_offsets, mask=dim_mask, other=0.0)

        sequence_length = tl.load(sequence_length_ptr + batch_index)
        split_start = split_index * split_span
        split_end = tl.minimum(sequence_length, split_start + split_span)
        tokens_in_split = tl.maximum(split_end - split_start, 0)
        tile_count = tl.cdiv(tokens_in_split, BLOCK_TOKENS)
        token_offsets = tl.arange(0, BLOCK_TOKENS)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

        for tile_index in tl.range(0, tile_count):
            logical_tokens = (
                split_start + tile_index * BLOCK_TOKENS + token_offsets
            )
            token_mask = logical_tokens < split_end
            logical_blocks = logical_tokens // BLOCK_SIZE
            offsets_in_block = logical_tokens % BLOCK_SIZE
            physical_blocks = tl.load(
                block_table_ptr
                + batch_index * table_stride_batch
                + logical_blocks,
                mask=token_mask,
                other=0,
            )

            key_offsets = (
                physical_blocks[:, None] * key_stride_block
                + kv_head * key_stride_head
                + offsets_in_block[:, None] * key_stride_token
                + dim_offsets[None, :] * key_stride_dim
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            scores = tl.sum(keys * query[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)
            previous_weight = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max)
            probabilities = tl.where(token_mask, probabilities, 0.0)

            value_offsets = (
                physical_blocks[:, None] * value_stride_block
                + kv_head * value_stride_head
                + offsets_in_block[:, None] * value_stride_token
                + dim_offsets[None, :] * value_stride_dim
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator = accumulator * previous_weight + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_sum = (
                running_sum * previous_weight + tl.sum(probabilities, axis=0)
            )
            running_max = new_max

        partial_max_offset = (
            batch_index * partial_max_stride_batch
            + query_head * partial_max_stride_head
            + split_index * partial_max_stride_split
        )
        partial_sum_offset = (
            batch_index * partial_sum_stride_batch
            + query_head * partial_sum_stride_head
            + split_index * partial_sum_stride_split
        )
        partial_output_offset = (
            batch_index * partial_output_stride_batch
            + query_head * partial_output_stride_head
            + split_index * partial_output_stride_split
            + dim_offsets * partial_output_stride_dim
        )
        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        partial_output = accumulator / denominator
        tl.store(partial_max_ptr + partial_max_offset, running_max)
        tl.store(partial_sum_ptr + partial_sum_offset, running_sum)
        tl.store(
            partial_output_ptr + partial_output_offset,
            partial_output,
            mask=dim_mask,
        )


    @triton.jit
    def _paged_decode_split_reduce_kernel(
        partial_max_ptr,
        partial_sum_ptr,
        partial_output_ptr,
        output_ptr,
        partial_max_stride_batch: tl.constexpr,
        partial_max_stride_head: tl.constexpr,
        partial_max_stride_split: tl.constexpr,
        partial_sum_stride_batch: tl.constexpr,
        partial_sum_stride_head: tl.constexpr,
        partial_sum_stride_split: tl.constexpr,
        partial_output_stride_batch: tl.constexpr,
        partial_output_stride_head: tl.constexpr,
        partial_output_stride_split: tl.constexpr,
        partial_output_stride_dim: tl.constexpr,
        output_stride_batch: tl.constexpr,
        output_stride_head: tl.constexpr,
        output_stride_token: tl.constexpr,
        output_stride_dim: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        SPLIT_COUNT: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_DIM: tl.constexpr,
    ):
        """Combine chunk-local online-softmax states for one request/head."""

        program_id = tl.program_id(0)
        batch_index = program_id // NUM_HEADS
        query_head = program_id % NUM_HEADS
        dim_offsets = tl.arange(0, BLOCK_DIM)
        dim_mask = dim_offsets < HEAD_DIM

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

        for split_index in range(0, SPLIT_COUNT):
            max_offset = (
                batch_index * partial_max_stride_batch
                + query_head * partial_max_stride_head
                + split_index * partial_max_stride_split
            )
            sum_offset = (
                batch_index * partial_sum_stride_batch
                + query_head * partial_sum_stride_head
                + split_index * partial_sum_stride_split
            )
            output_offset = (
                batch_index * partial_output_stride_batch
                + query_head * partial_output_stride_head
                + split_index * partial_output_stride_split
                + dim_offsets * partial_output_stride_dim
            )
            local_max = tl.load(partial_max_ptr + max_offset)
            local_sum = tl.load(partial_sum_ptr + sum_offset)
            local_output = tl.load(
                partial_output_ptr + output_offset,
                mask=dim_mask,
                other=0.0,
            )

            has_running = running_sum > 0.0
            has_local = local_sum > 0.0
            any_values = has_running | has_local
            new_max = tl.maximum(running_max, local_max)
            safe_max = tl.where(any_values, new_max, 0.0)
            previous_weight = tl.where(
                has_running,
                tl.exp(running_max - safe_max),
                0.0,
            )
            local_weight = tl.where(
                has_local,
                tl.exp(local_max - safe_max),
                0.0,
            )
            running_sum = (
                running_sum * previous_weight
                + local_sum * local_weight
            )
            accumulator = (
                accumulator * previous_weight
                + local_output * (local_sum * local_weight)
            )
            running_max = tl.where(any_values, new_max, running_max)

        output_offset = (
            batch_index * output_stride_batch
            + query_head * output_stride_head
            + dim_offsets * output_stride_dim
        )
        denominator = tl.where(running_sum > 0.0, running_sum, 1.0)
        output = accumulator / denominator
        tl.store(output_ptr + output_offset, output, mask=dim_mask)


def triton_is_available() -> bool:
    """Return whether Triton and a CUDA runtime are available."""

    return triton is not None and torch.cuda.is_available()


def select_decode_split_count(
    max_sequence_length: int,
    batch_size: int | None = None,
) -> int:
    """Choose the measured L4 split-KV shape for a fixed decode graph.

    Small batches need more sequence splits to expose enough parallel work;
    batch eight already supplies enough independent heads at medium context.
    Long contexts benefit from eight splits at every tested batch size.
    ``batch_size=None`` preserves the conservative legacy buckets for callers
    that do not know their final graph shape.
    """

    length = int(max_sequence_length)
    if length < 1024:
        return 1
    if length <= 2048:
        if batch_size is not None:
            batch = int(batch_size)
            if batch < 1:
                raise ValueError("batch_size must be positive")
            if batch <= 2:
                return 8
            if batch <= 4:
                return 4
            return 1
        return 2
    if length <= 4096 and batch_size is None:
        return 4
    return 8


def select_decode_block_tokens(max_sequence_length: int) -> int:
    """Select the profiled token-reduction tile for NVIDIA L4 decode."""

    return 64 if int(max_sequence_length) >= 1024 else 16


def can_use_triton_paged_decode(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor | None,
    sequence_lengths: torch.Tensor | None,
) -> bool:
    """Check the narrow shape/device contract supported by the fused kernel."""

    return bool(
        triton_is_available()
        and query.is_cuda
        and key_blocks.is_cuda
        and value_blocks.is_cuda
        and query.ndim == 4
        and int(query.shape[2]) == 1
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and key_blocks.dtype == query.dtype
        and value_blocks.dtype == query.dtype
        and block_tables is not None
        and block_tables.is_cuda
        and block_tables.ndim == 2
        and sequence_lengths is not None
        and sequence_lengths.is_cuda
        and sequence_lengths.ndim == 1
    )


def triton_paged_decode_attention(
    query: torch.Tensor,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float,
    split_count: int = 1,
    max_sequence_length: int | None = None,
    use_gqa_reuse: bool | None = False,
    block_tokens: int = 16,
) -> torch.Tensor:
    """Decode one token per request directly from physical KV pages.

    ``query`` is ``[batch, query_heads, 1, head_dim]`` and each cache tensor is
    ``[physical_blocks, kv_heads, block_size, head_dim]``. The result uses the
    Qwen attention-interface layout ``[batch, 1, query_heads, head_dim]``.
    """

    if not can_use_triton_paged_decode(
        query,
        key_blocks,
        value_blocks,
        block_tables,
        sequence_lengths,
    ):
        raise ValueError("inputs do not satisfy the Triton paged-decode contract")
    batch_size, num_heads, _, head_dim = (int(value) for value in query.shape)
    num_kv_heads = int(key_blocks.shape[1])
    block_size = int(key_blocks.shape[2])
    if value_blocks.shape != key_blocks.shape:
        raise ValueError("key and value page tensors must have identical shapes")
    if int(block_tables.shape[0]) != batch_size:
        raise ValueError("block-table batch does not match query batch")
    if int(sequence_lengths.shape[0]) != batch_size:
        raise ValueError("sequence lengths do not match query batch")
    if num_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by KV heads")
    group_size = num_heads // num_kv_heads
    gqa_supported = group_size in {2, 4}
    if use_gqa_reuse is True and not gqa_supported:
        raise ValueError(
            "GQA-reuse decode currently supports 2 or 4 query heads per KV head"
        )
    use_grouped_kernel = (
        gqa_supported if use_gqa_reuse is None else bool(use_gqa_reuse)
    )
    if block_size < 1 or block_size > 128:
        raise ValueError("Triton paged decode supports block sizes from 1 to 128")
    block_tokens = int(block_tokens)
    if block_tokens not in {8, 16, 32, 64}:
        raise ValueError("block_tokens must be one of 8, 16, 32, or 64")

    split_count = int(split_count)
    if split_count not in {1, 2, 4, 8}:
        raise ValueError("split_count must be one of 1, 2, 4, or 8")
    if split_count > 1:
        if max_sequence_length is None or int(max_sequence_length) < 1:
            raise ValueError(
                "max_sequence_length is required when split-KV is enabled"
            )
        max_sequence_length = int(max_sequence_length)
        if max_sequence_length < split_count:
            raise ValueError(
                "max_sequence_length must cover every requested split"
            )

    output = torch.empty_like(query)
    if split_count == 1:
        block_dim = triton.next_power_of_2(head_dim)
        # Storage page size and compute tile size are separate concerns.  A fixed
        # 16-token reduction tile keeps the online-softmax accumulation order
        # stable across physical page sizes and avoids changing greedy argmax
        # results when a page size is swept.
        if use_grouped_kernel:
            grid = (batch_size * num_kv_heads,)
            _paged_decode_gqa_kernel[grid](
                query,
                key_blocks,
                value_blocks,
                block_tables,
                sequence_lengths,
                output,
                float(scale),
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),
                key_blocks.stride(0),
                key_blocks.stride(1),
                key_blocks.stride(2),
                key_blocks.stride(3),
                value_blocks.stride(0),
                value_blocks.stride(1),
                value_blocks.stride(2),
                value_blocks.stride(3),
                block_tables.stride(0),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                NUM_KV_HEADS=num_kv_heads,
                GROUP_SIZE=group_size,
                BLOCK_SIZE=block_size,
                HEAD_DIM=head_dim,
                BLOCK_DIM=block_dim,
                BLOCK_TOKENS=block_tokens,
                num_warps=4,
            )
            return output.transpose(1, 2)
        grid = (batch_size * num_heads,)
        _paged_decode_kernel[grid](
            query,
            key_blocks,
            value_blocks,
            block_tables,
            sequence_lengths,
            output,
            float(scale),
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key_blocks.stride(0),
            key_blocks.stride(1),
            key_blocks.stride(2),
            key_blocks.stride(3),
            value_blocks.stride(0),
            value_blocks.stride(1),
            value_blocks.stride(2),
            value_blocks.stride(3),
            block_tables.stride(0),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            NUM_HEADS=num_heads,
            NUM_KV_HEADS=num_kv_heads,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            BLOCK_DIM=block_dim,
            BLOCK_TOKENS=block_tokens,
            num_warps=4,
        )
        return output.transpose(1, 2)

    split_span = (max_sequence_length + split_count - 1) // split_count
    block_dim = triton.next_power_of_2(head_dim)
    partial_max = torch.empty(
        (batch_size, num_heads, split_count),
        dtype=torch.float32,
        device=query.device,
    )
    partial_sum = torch.empty_like(partial_max)
    partial_output = torch.empty(
        (batch_size, num_heads, split_count, head_dim),
        dtype=torch.float32,
        device=query.device,
    )
    split_grid = (
        batch_size
        * (num_kv_heads if use_grouped_kernel else num_heads)
        * split_count,
    )
    split_kernel = (
        _paged_decode_gqa_split_kernel
        if use_grouped_kernel
        else _paged_decode_split_kernel
    )
    split_kernel[split_grid](
        query,
        key_blocks,
        value_blocks,
        block_tables,
        sequence_lengths,
        partial_max,
        partial_sum,
        partial_output,
        float(scale),
        split_span,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query.stride(3),
        key_blocks.stride(0),
        key_blocks.stride(1),
        key_blocks.stride(2),
        key_blocks.stride(3),
        value_blocks.stride(0),
        value_blocks.stride(1),
        value_blocks.stride(2),
        value_blocks.stride(3),
        block_tables.stride(0),
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_sum.stride(0),
        partial_sum.stride(1),
        partial_sum.stride(2),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        **(
            {
                "NUM_KV_HEADS": num_kv_heads,
                "GROUP_SIZE": group_size,
            }
            if use_grouped_kernel
            else {
                "NUM_HEADS": num_heads,
                "NUM_KV_HEADS": num_kv_heads,
            }
        ),
        SPLIT_COUNT=split_count,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
        BLOCK_TOKENS=block_tokens,
        num_warps=4,
    )
    reduce_grid = (batch_size * num_heads,)
    _paged_decode_split_reduce_kernel[reduce_grid](
        partial_max,
        partial_sum,
        partial_output,
        output,
        partial_max.stride(0),
        partial_max.stride(1),
        partial_max.stride(2),
        partial_sum.stride(0),
        partial_sum.stride(1),
        partial_sum.stride(2),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_output.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        NUM_HEADS=num_heads,
        SPLIT_COUNT=split_count,
        HEAD_DIM=head_dim,
        BLOCK_DIM=block_dim,
        num_warps=4,
    )
    return output.transpose(1, 2)


__all__ = [
    "can_use_triton_paged_decode",
    "select_decode_block_tokens",
    "select_decode_split_count",
    "triton_is_available",
    "triton_paged_decode_attention",
]
