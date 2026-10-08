from __future__ import annotations

import math
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from minillm_l4.engine.kv_cache import (
    PagedKvAllocator, PagedKvCache, paged_attention, triton_is_available,
)
from minillm_l4.engine.kv_cache.qwen3_paged import PagedQwen3Attention
from minillm_l4.engine.kv_cache.triton_paged_kv import triton_write_decode_kv


pytestmark = pytest.mark.skipif(not triton_is_available(), reason="requires CUDA and Triton")


@pytest.fixture
def pages():
    allocator = PagedKvAllocator(num_blocks=4, block_size=2)
    cache = PagedKvCache(
        allocator, num_layers=1, num_kv_heads=1, head_dim=64,
        dtype=torch.bfloat16, device="cuda",
    )
    ids = ("a", "b")
    values = torch.tensor(
        [[1., 2., 100., 200.], [4., 8., 16., 32.]],
        dtype=torch.bfloat16, device="cuda",
    ).reshape(2, 1, 4, 1).expand(2, 1, 4, 64).contiguous()
    for row, sequence_id in enumerate(ids):
        allocator.allocate(sequence_id, token_count=4)
        cache.write_layer_segment(
            sequence_id, 0, torch.zeros_like(values[row:row+1]),
            values[row:row+1], start_token=0,
        )
    return cache, ids


@pytest.mark.parametrize("split_count", [1, 4])
@pytest.mark.parametrize("gqa", [False, True])
@pytest.mark.parametrize("sliced", ["lengths", "tables", "both"])
def test_decode_accepts_strided_metadata(pages, split_count, gqa, sliced):
    cache, ids = pages
    query = torch.ones((2, 2, 1, 64), dtype=torch.bfloat16, device="cuda")
    tables = cache.block_table_tensor(ids).to(torch.int32)
    lengths = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    if sliced in {"tables", "both"}:
        tables = tables.repeat_interleave(2, dim=1)[:, ::2]
    if sliced in {"lengths", "both"}:
        lengths = torch.tensor([4, 1, 4, 1], dtype=torch.int32, device="cuda")[::2]
    actual = paged_attention(
        query, cache, ids, layer_index=0, query_start_positions=(3, 3),
        num_key_value_groups=2, backend="triton", block_tables=tables,
        sequence_lengths=lengths, decode_split_count=split_count,
        decode_max_sequence_length=4, decode_use_gqa_reuse=gqa,
    )
    torch.testing.assert_close(
        actual[:, 0, 0, 0], torch.tensor([76., 15.], dtype=actual.dtype, device="cuda"),
        atol=0, rtol=0,
    )


@pytest.mark.parametrize("position", [-1, 4])
def test_decode_rejects_invalid_device_positions(pages, position):
    cache, ids = pages
    with pytest.raises(ValueError, match="position"):
        paged_attention(
            torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda"),
            cache, ids, layer_index=0,
            query_start_positions=torch.tensor([position, 3], device="cuda"),
            block_tables=cache.block_table_tensor(ids).to(torch.int32),
            sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
            backend="triton",
        )


@pytest.mark.parametrize("positions", [(1.5, 3), torch.tensor([1., 3.])])
def test_decode_rejects_noninteger_host_positions(pages, positions):
    cache, ids = pages
    with pytest.raises(ValueError, match="integer"):
        paged_attention(
            torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda"),
            cache, ids, layer_index=0, query_start_positions=positions,
            block_tables=cache.block_table_tensor(ids).to(torch.int32),
            sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
            backend="triton",
        )


def test_decode_rejects_undersized_split_limit(pages):
    cache, ids = pages
    with pytest.raises(ValueError, match="max_sequence_length"):
        paged_attention(
            torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda"),
            cache, ids, layer_index=0, query_start_positions=(3, 3),
            block_tables=cache.block_table_tensor(ids).to(torch.int32),
            sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
            backend="triton", decode_split_count=2, decode_max_sequence_length=2,
        )


class _IdentityAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.layer_idx = 0
        self.head_dim = 64
        self.num_key_value_groups = 1
        self.scaling = 1.0 / math.sqrt(64)
        self.sliding_window = None
        for name in ("q_norm", "k_norm", "q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Identity())


def test_adapter_writes_at_requested_position_not_kv_tail(pages):
    cache, ids = pages
    hidden = torch.full((2, 1, 64), 7., dtype=torch.bfloat16, device="cuda")
    attention = PagedQwen3Attention(_IdentityAttention())
    with torch.inference_mode():
        actual, _ = attention(
            hidden_states=hidden,
            position_embeddings=(torch.ones_like(hidden), torch.zeros_like(hidden)),
            attention_mask=None, paged_kv_cache=cache, paged_sequence_ids=ids,
            paged_query_start_positions=torch.tensor([[1], [1]], device="cuda"),
            paged_block_tables=cache.block_table_tensor(ids).to(torch.int32),
            paged_sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
            paged_attention_backend="triton",
        )
    _, stored = cache.gather_layer("a", 0)
    torch.testing.assert_close(
        stored[0, 0, :, 0], torch.tensor([1., 7., 100., 200.], dtype=stored.dtype, device="cuda"),
        atol=0, rtol=0,
    )
    torch.testing.assert_close(actual, torch.full_like(actual, 7.), atol=0, rtol=0)


def test_kv_writer_accepts_strided_metadata_and_independent_value_layout(pages):
    cache, ids = pages
    keys = torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda")
    values = torch.arange(128, dtype=torch.bfloat16, device="cuda").reshape(1, 1, 1, 128).expand(2, 1, 1, 128)[..., ::2]
    tables = cache.block_table_tensor(ids).to(torch.int32).repeat_interleave(2, dim=1)[:, ::2]
    lengths = torch.tensor([4, 1, 4, 1], dtype=torch.int32, device="cuda")[::2]
    triton_write_decode_kv(
        keys, values, tables, lengths, cache.key_blocks[0], cache.value_blocks[0],
    )
    for row, sequence_id in enumerate(ids):
        stored_keys, stored_values = cache.gather_layer(sequence_id, 0)
        torch.testing.assert_close(stored_keys[:, :, 3:4], keys[row:row+1], atol=0, rtol=0)
        torch.testing.assert_close(stored_values[:, :, 3:4], values[row:row+1], atol=0, rtol=0)


@pytest.mark.parametrize("invalid", ["zero_length", "too_long", "invalid_block", "floating_lengths"])
def test_decode_rejects_invalid_metadata_before_gpu_access(pages, invalid):
    cache, ids = pages
    tables = cache.block_table_tensor(ids).to(torch.int32)
    lengths = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
    if invalid == "zero_length":
        lengths[0] = 0
    elif invalid == "too_long":
        lengths[0] = 5
    elif invalid == "invalid_block":
        tables[0, 1] = 100
    else:
        lengths = lengths.float()
    with pytest.raises(ValueError):
        paged_attention(
            torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda"),
            cache, ids, layer_index=0, query_start_positions=(3, 3),
            block_tables=tables, sequence_lengths=lengths, backend="triton",
        )
    # A rejected call must leave the CUDA context usable.
    torch.cuda.synchronize()


def test_split_limit_covers_query_prefix_not_future_cached_tokens(pages):
    cache, ids = pages
    actual = paged_attention(
        torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda"),
        cache, ids, layer_index=0, query_start_positions=(1, 1),
        block_tables=cache.block_table_tensor(ids).to(torch.int32),
        sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
        backend="triton", decode_split_count=2, decode_max_sequence_length=2,
    )
    torch.testing.assert_close(
        actual[:, 0, 0, 0], torch.tensor([1.5, 6.], dtype=actual.dtype, device="cuda"),
        atol=0, rtol=0,
    )


def test_adapter_position_write_updates_on_graph_replay(pages):
    cache, ids = pages
    attention = PagedQwen3Attention(_IdentityAttention())
    hidden = torch.full((2, 1, 64), 7., dtype=torch.bfloat16, device="cuda")
    positions = torch.tensor([[1], [1]], device="cuda")
    kwargs = dict(
        hidden_states=hidden,
        position_embeddings=(torch.ones_like(hidden), torch.zeros_like(hidden)),
        attention_mask=None, paged_kv_cache=cache, paged_sequence_ids=ids,
        paged_query_start_positions=positions,
        paged_block_tables=cache.block_table_tensor(ids).to(torch.int32),
        paged_sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
        paged_attention_backend="triton",
    )
    with torch.inference_mode():
        warmup = torch.cuda.Stream()
        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            for _ in range(3):
                attention(**kwargs)
        torch.cuda.current_stream().wait_stream(warmup)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual, _ = attention(**kwargs)
        graph.replay()
        positions.fill_(3)
        hidden.fill_(9.)
        graph.replay()
        _, values = cache.gather_layer("a", 0)
        torch.testing.assert_close(
            values[0, 0, :, 0], torch.tensor([1., 7., 100., 9.], dtype=values.dtype, device="cuda"),
            atol=0, rtol=0,
        )
        torch.testing.assert_close(actual, torch.full_like(actual, 9.), atol=0, rtol=0)


@pytest.mark.parametrize("gqa", [False, True])
def test_sdpa_compatible_decode_keeps_causal_mask_and_metadata_strides(pages, gqa):
    cache, ids = pages
    tables = cache.block_table_tensor(ids).to(torch.int32).repeat_interleave(2, dim=1)[:, ::2]
    lengths = torch.tensor([4, 1, 4, 1], dtype=torch.int32, device="cuda")[::2]
    actual = paged_attention(
        torch.ones((2, 2, 1, 64), dtype=torch.bfloat16, device="cuda"),
        cache, ids, layer_index=0, query_start_positions=(1, 3),
        num_key_value_groups=2, block_tables=tables, sequence_lengths=lengths,
        backend="triton", decode_sdpa_compat=True, decode_use_gqa_reuse=gqa,
    )
    torch.testing.assert_close(
        actual[:, 0, 0, 0], torch.tensor([1.5, 15.], dtype=actual.dtype, device="cuda"),
        atol=0, rtol=0,
    )


def test_sdpa_compatible_decode_rejects_split_reduction(pages):
    cache, ids = pages
    with pytest.raises(ValueError, match="single KV split"):
        paged_attention(
            torch.ones((2, 1, 1, 64), dtype=torch.bfloat16, device="cuda"),
            cache, ids, layer_index=0, query_start_positions=(3, 3),
            block_tables=cache.block_table_tensor(ids).to(torch.int32),
            sequence_lengths=torch.tensor([4, 4], dtype=torch.int32, device="cuda"),
            backend="triton", decode_sdpa_compat=True, decode_split_count=2,
            decode_max_sequence_length=4,
        )


def test_graph_validates_replay_positions_in_an_isolated_cuda_context():
    # A device assertion intentionally invalidates its CUDA context. Isolate
    # the negative replay so the remaining GPU tests stay usable.
    script = r'''
import torch
from minillm_l4.engine.kv_cache import PagedKvAllocator, PagedKvCache, paged_attention
allocator = PagedKvAllocator(num_blocks=2, block_size=2)
allocator.allocate("a", token_count=4)
cache = PagedKvCache(allocator, num_layers=1, num_kv_heads=1, head_dim=64, dtype=torch.bfloat16, device="cuda")
data = torch.ones((1, 1, 4, 64), dtype=torch.bfloat16, device="cuda")
cache.write_layer_segment("a", 0, data, data, start_token=0)
query = data[:, :, :1]
position = torch.tensor([1], device="cuda")
kwargs = dict(layer_index=0, query_start_positions=position, block_tables=cache.block_table_tensor(("a",)).to(torch.int32), sequence_lengths=torch.tensor([4], dtype=torch.int32, device="cuda"), backend="triton")
warmup = torch.cuda.Stream()
warmup.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(warmup):
    for _ in range(3):
        paged_attention(query, cache, ("a",), **kwargs)
torch.cuda.current_stream().wait_stream(warmup)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = paged_attention(query, cache, ("a",), **kwargs)
graph.replay()
torch.cuda.synchronize()
position.fill_(-1)
try:
    graph.replay()
    torch.cuda.synchronize()
except RuntimeError as error:
    if "device-side assert" not in str(error):
        raise
    print("REPLAY_POSITION_REJECTED", flush=True)
else:
    raise AssertionError("invalid replay position was not rejected")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REPLAY_POSITION_REJECTED" in completed.stdout
