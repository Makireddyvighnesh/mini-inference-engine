"""CPU-only checks of FP8 dispatch, tuning coverage, and validation tools."""

from collections import Counter
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from minillm_l4.engine.kernels import sm89_fp8 as fp8
from minillm_l4.engine.kernels.sm89_fp8_validation import (
    EXACT_MS, MODEL_SHAPES, bit_mismatch_details, bitwise_equal, make_activations, make_weights,
)
from minillm_l4.scripts import bench_fp8_gemm, check_fp8_exact


@pytest.mark.parametrize("rows,expected", [(0, 1), (1, 1), (2, 2), (3, 4), (17, 32),
                                          (255, 256), (1000, 1024), (4097, 8192),
                                          (16384, 16384), (20000, 16384)])
def test_m_bucket(rows, expected):
    assert fp8.m_bucket(rows) == expected


def test_dispatch_threshold_is_inclusive_and_explicit_paths_override(monkeypatch):
    monkeypatch.setattr(fp8, "SPLIT_M_THRESHOLD", 100)
    assert fp8.select_fp8_path(99) == "fused"
    assert fp8.select_fp8_path(100) == "split"
    assert fp8.select_fp8_path(20000, "fused") == "fused"
    assert fp8.select_fp8_path(1, "split") == "split"
    with pytest.raises(ValueError, match="Unknown"):
        fp8.select_fp8_path(1, "typo")
    with pytest.raises(ValueError, match="nonnegative"):
        fp8.m_bucket(-1)


def test_split_config_contract():
    configs = fp8.SPLIT_GEMM_CONFIGS
    assert len(set(configs)) == len(configs)
    assert {c.block_m for c in configs} >= {64, 128}
    assert {c.block_n for c in configs} == {64, 128, 256}
    assert {c.num_warps for c in configs} == {4, 8}
    assert {c.num_stages for c in configs} == {3, 4, 5}
    for config in configs:
        assert config.block_k == 128
        assert config.group_m > 1
        assert config.kwargs["BLOCK_K"] == 128
        assert config.launch_kwargs() == {
            **config.kwargs, "num_warps": config.num_warps, "num_stages": config.num_stages,
        }
    with pytest.raises(FrozenInstanceError):
        configs[0].block_k = 256
    # Read decorator metadata without touching the CUDA driver.
    if fp8.triton is not None:
        for kernel in (fp8._fp8_linear_kernel, fp8._fp8_split_gemm_kernel):
            assert kernel.keys == ["M_BUCKET", "N", "K"]
            assert kernel.cache_results
            assert all(c.kwargs["BLOCK_K"] == 128 for c in kernel.configs)
        assert len(fp8._fp8_split_gemm_kernel.configs) == len(configs)


@pytest.mark.parametrize("limit,expected", [(0, (1,)), (1, (1,)), (3, (1, 2, 4)),
                                          (8, (1, 2, 4, 8)), (9, (1, 2, 4, 8, 16))])
def test_pretune_covers_ceiling_bucket(limit, expected):
    assert fp8.pretune_row_buckets(limit) == expected
    assert fp8.pretune_row_buckets(20000)[-1] == fp8.MAX_M_BUCKET


@pytest.fixture
def fp8_model():
    model = torch.nn.Module()
    for name in ("q", "same_shape"):
        projection = torch.nn.Module()
        projection.weight = torch.nn.Parameter(torch.zeros((129, 259)).to(torch.float8_e4m3fn), requires_grad=False)
        projection.weight_scale_inv = torch.ones((2, 3), dtype=torch.float32)
        model.add_module(name, projection)
    model.register_parameter("bf16", torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16)))
    return model


@pytest.mark.parametrize("max_tokens", [1, 3, 17, 1000, 20000])
def test_pretune_warms_both_specializations_and_deduplicates_shapes(monkeypatch, fp8_model, max_tokens):
    calls = []
    monkeypatch.setattr(fp8, "sm89_fp8_linear", lambda x, w, s, **kw: calls.append((x.shape[0], kw["path"])))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    buckets = fp8.pretune_row_buckets(max_tokens)
    assert fp8.pretune_sm89_fp8(fp8_model, max_tokens=max_tokens) == len(buckets)
    for bucket in buckets:
        for path in ("fused", "split"):
            rows = [m for m, p in calls if p == path and fp8.m_bucket(m) == bucket]
            assert rows[0] == bucket  # Tune before compiling the other variant.
            assert len(rows) == (2 if bucket >= 16 else 1)
            assert {m % 16 == 0 for m in rows} == ({True, False} if bucket >= 16 else {False})
    assert calls.count((1, "fused")) == calls.count((1, "split")) == 1


@pytest.mark.skipif(fp8.triton is None, reason="requires Triton metadata, not CUDA")
def test_pretune_specializations_reuse_autotune_search_and_warm_quantizer(monkeypatch, fp8_model):
    # Exercise the real wrappers and Triton autotuners; stub only device checks,
    # benchmarking, and JIT launches so this never initializes CUDA.
    launches = {name: [] for name in ("fused", "split", "quantizer")}
    searches = Counter()

    def instrument_gemm(name, kernel):
        def benchmark(*args, config, **kwargs):
            nargs = dict(zip(kernel.arg_names, args))
            searches[name, nargs["M_BUCKET"], nargs["N"], nargs["K"]] += 1
            return 0.0

        def launch(*args, **kwargs):
            launches[name].append(dict(zip(kernel.arg_names, args)))

        monkeypatch.setattr(kernel, "cache", {})
        monkeypatch.setattr(kernel, "cache_results", False)
        monkeypatch.setattr(kernel, "_bench", benchmark)
        monkeypatch.setattr(kernel.fn, "run", launch)

    instrument_gemm("fused", fp8._fp8_linear_kernel)
    instrument_gemm("split", fp8._fp8_split_gemm_kernel)
    quantizer = fp8._quantize_activation_kernel
    monkeypatch.setattr(quantizer, "run", lambda *args, **kw: launches["quantizer"].append(
        dict(zip(quantizer.arg_names, args))))
    monkeypatch.setattr(fp8, "_supports_call", lambda *args: True)
    monkeypatch.setattr(fp8, "sm89_available", lambda *args: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    real_quantize = fp8.quantize_sm89_fp8_activation

    def quantize(input_2d):
        # Retain CPU allocation and strides, but satisfy the CUDA-only guard.
        return real_quantize(SimpleNamespace(
            ndim=input_2d.ndim, dtype=input_2d.dtype, shape=input_2d.shape,
            device=input_2d.device, is_cuda=True, stride=input_2d.stride,
        ))

    monkeypatch.setattr(fp8, "quantize_sm89_fp8_activation", quantize)
    buckets = fp8.pretune_row_buckets(2048)
    assert fp8.pretune_sm89_fp8(fp8_model, max_tokens=2048) == len(buckets)
    for name, kernel in (("fused", fp8._fp8_linear_kernel), ("split", fp8._fp8_split_gemm_kernel)):
        assert {key[1] for key in searches if key[0] == name} == set(buckets)
        assert all(searches[name, bucket, 129, 259] == len(kernel.configs) for bucket in buckets)
        assert len(kernel.cache) == len(buckets)
        for bucket in buckets:
            calls = [call for call in launches[name] if call["M_BUCKET"] == bucket]
            assert len(calls) == (2 if bucket >= 16 else 1)
            assert calls[0]["M"] == bucket
            assert len({(call["stride_am"], call["stride_om"], call["stride_sn"]) for call in calls}) == 1
            if name == "fused" and bucket >= 16:
                assert calls[0]["activation"].data_ptr() == calls[1]["activation"].data_ptr()
    assert [call["M"] for call in launches["quantizer"]] == [call["M"] for call in launches["split"]]
    assert {call["M"] % 16 == 0 for call in launches["quantizer"]} == {True, False}
    # A second pretune also reuses every already-tuned key.
    before = searches.copy()
    fp8.pretune_sm89_fp8(fp8_model, max_tokens=2048)
    assert searches == before


def test_cpu_fallback_preserves_transformers_arguments(monkeypatch):
    calls = []
    output = torch.ones(1)

    def fallback(*args, **kwargs):
        calls.append((args, kwargs))
        return output

    monkeypatch.setattr(fp8, "_original_fp8_linear", fallback)
    x, w, s = torch.zeros(1, 128), torch.zeros(1, 128), torch.ones(1, 1)
    bias = torch.ones(1)
    result = fp8.sm89_fp8_linear(x, w, s, [128, 128], bias=bias, path="split")
    assert result is output
    assert calls[0][0] == (x, w, s)
    assert calls[0][1]["bias"] is bias
    assert "path" not in calls[0][1]
    assert "gemm_config" not in calls[0][1]


def test_forced_config_requires_explicit_split():
    x, w, s = torch.zeros(1, 128), torch.zeros(1, 128), torch.ones(1, 1)
    with pytest.raises(ValueError, match="gemm_config"):
        fp8.sm89_fp8_linear(x, w, s, gemm_config=fp8.SPLIT_GEMM_CONFIGS[0])


def test_bit_comparison_detects_zero_sign_and_reports_mismatch():
    left = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
    right = torch.tensor([[-0.0, 1.0]], dtype=torch.bfloat16)
    assert torch.equal(left, right)  # Numeric equality misses the sign bit.
    assert not bitwise_equal(left, right)
    assert "1/2 elements" in bit_mismatch_details(left, right)
    assert "first=(0, 0)" in bit_mismatch_details(left, right)
    assert bitwise_equal(left, left.clone())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_validation_data_is_deterministic_finite_and_covers_scale_extremes(dtype):
    x = make_activations(17, 259, 42, dtype, device="cpu")
    assert bitwise_equal(x, make_activations(17, 259, 42, dtype, device="cpu"))
    assert torch.isfinite(x).all()
    assert not x[1].count_nonzero()
    assert x[2].abs().max() > x[0].abs().max()
    assert x[3].abs().max() < 1e-4
    shape = MODEL_SHAPES[1]
    w, s = make_weights(shape, 42, device="cpu")
    assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.float32
    assert s.shape == (8, 20)
    assert sum(shape.count for shape in MODEL_SHAPES) == 7


def test_gpu_script_defaults_and_exhaustive_row_selection():
    args = check_fp8_exact.parse_args([])
    assert tuple(args.ms) == EXACT_MS
    assert args.row_samples == 0 and not args.no_tails
    assert not args.check_pretune_only
    assert check_fp8_exact.parse_args(["--check-pretune-only", "--ms", "1000", "1001", "2047"]).check_pretune_only
    assert list(check_fp8_exact.row_indices(17, 0)) == list(range(17))
    sampled = check_fp8_exact.row_indices(100, 3)
    assert set(range(7)) <= set(sampled) and 99 in sampled
    assert bench_fp8_gemm.parse_args([]).vllm is False
    assert bench_fp8_gemm.tflops(2048, 4096, 2560, 1.0) == pytest.approx(42.94967296)


def test_pretune_checker_counts_autotune_and_jit_entries_without_cuda(monkeypatch):
    def jit(*sizes):
        return SimpleNamespace(device_caches={device: (dict.fromkeys(range(size)),) for device, size in enumerate(sizes)})

    monkeypatch.setattr(fp8, "_fp8_linear_kernel", SimpleNamespace(cache={1: None, 2: None}, fn=jit(1, 3)), raising=False)
    monkeypatch.setattr(fp8, "_fp8_split_gemm_kernel", SimpleNamespace(cache={1: None}, fn=jit(5)), raising=False)
    monkeypatch.setattr(fp8, "_quantize_activation_kernel", jit(6), raising=False)
    assert check_fp8_exact._pretune_cache_sizes() == (2, 1, 4, 5, 6)
