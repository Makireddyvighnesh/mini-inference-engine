"""L4 bitwise smoke gates; skipped on CPU and GPUs other than SM89."""

import pytest
import torch

from minillm_l4.engine.kernels import sm89_fp8 as fp8
from minillm_l4.engine.kernels.sm89_fp8_validation import (
    ModelShape, bit_mismatch_details, bitwise_equal, make_activations, make_weights,
)


pytestmark = pytest.mark.skipif(not fp8.sm89_available(), reason="requires SM89 CUDA and Triton")


@pytest.mark.parametrize("shape", [ModelShape("tail", 193, 259), ModelShape("kv", 1024, 2560),
                                  ModelShape("o", 2560, 4096)])
@pytest.mark.parametrize("rows", [1, 7, 129, 1000])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_every_split_config_bitwise_and_row_independent(shape, rows, dtype):
    x = make_activations(rows, shape.k, 42, dtype)
    weight, scales = make_weights(shape, 42)
    reference = fp8.sm89_fp8_linear(x, weight, scales, [128, 128], path="fused")
    for path in ("auto", "split"):
        actual = fp8.sm89_fp8_linear(x, weight, scales, [128, 128], path=path)
        assert bitwise_equal(reference, actual), bit_mismatch_details(reference, actual)
    selected_rows = sorted(set(range(min(rows, 7))) | {rows - 1})
    for row in selected_rows:
        single = fp8.sm89_fp8_linear(x[row:row + 1], weight, scales, [128, 128], path="fused")
        assert bitwise_equal(single, reference[row:row + 1])
    for config in fp8.SPLIT_GEMM_CONFIGS:
        actual = fp8.sm89_fp8_linear(x, weight, scales, [128, 128], path="split", gemm_config=config)
        assert bitwise_equal(reference, actual), f"{config.label}: {bit_mismatch_details(reference, actual)}"
        for row in selected_rows:
            single = fp8.sm89_fp8_linear(x[row:row + 1], weight, scales, [128, 128],
                                         path="split", gemm_config=config)
            assert bitwise_equal(single, actual[row:row + 1]), config.label


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_bias_output_dtype_and_flattened_input(output_dtype):
    x = make_activations(6, 259, 42).reshape(2, 3, 259)
    weight, scales = make_weights(ModelShape("tail", 193, 259), 42)
    bias = torch.linspace(-1, 1, 193, device=x.device, dtype=output_dtype)
    fused = fp8.sm89_fp8_linear(x, weight, scales, [128, 128], bias=bias,
                                output_dtype=output_dtype, path="fused")
    split = fp8.sm89_fp8_linear(x, weight, scales, [128, 128], bias=bias,
                                output_dtype=output_dtype, path="split")
    assert split.shape == (2, 3, 193) and split.dtype == output_dtype
    assert bitwise_equal(fused, split), bit_mismatch_details(fused, split)


@torch.inference_mode()
def test_quantizer_stores_raw_scale_and_is_row_independent():
    x = make_activations(7, 259, 42)
    q, scales = fp8.quantize_sm89_fp8_activation(x)
    assert scales[1].count_nonzero().item() == 0
    assert (scales[3] < 1e-12).all().item()
    for row in range(7):
        q1, s1 = fp8.quantize_sm89_fp8_activation(x[row:row + 1])
        assert bitwise_equal(q1, q[row:row + 1])
        assert bitwise_equal(s1, scales[row:row + 1])
