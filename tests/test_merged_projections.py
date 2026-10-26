"""Merged q/k/v and gate/up FP8 GEMMs must equal the separate projections bit for bit (L4 only)."""

from __future__ import annotations

import types

import pytest
import torch

from minillm_l4.engine.kernels import sm89_fp8
from minillm_l4.engine.kernels.merged_projections import (
    GATE_UP, QKV, _merge, merged_linears, project_gate_up, project_qkv,
)
from minillm_l4.engine.kernels.sm89_fp8_validation import ModelShape, make_activations, make_weights

pytestmark = pytest.mark.skipif(not sm89_fp8.sm89_available(), reason="needs an SM89 GPU and Triton")


@pytest.fixture(autouse=True)
def sm89_dispatch():
    from transformers.integrations import finegrained_fp8

    original = finegrained_fp8.fp8_linear
    sm89_fp8.install_sm89_fp8_dispatch()
    yield
    finegrained_fp8.fp8_linear = original


def _fp8_linear(n, k, seed):
    from transformers.integrations.finegrained_fp8 import FP8Linear

    linear = FP8Linear(k, n, block_size=(128, 128)).cuda()
    weight, scales = make_weights(ModelShape("p", n, k), seed)
    linear.weight.data, linear.weight_scale_inv.data = weight, scales
    return linear


def _module(names, widths, k=2560):
    return types.SimpleNamespace(**{name: _fp8_linear(n, k, i) for i, (name, n) in enumerate(zip(names, widths))})


@pytest.mark.parametrize("rows", [1, 8, 64, 255, 256, 2048, 4099])
def test_merged_qkv_and_gate_up_bitwise(rows):
    x = make_activations(rows, 2560, rows)
    for names, widths, project, attr in ((QKV, (4096, 1024, 1024), project_qkv, "_minillm_qkv"),
                                         (GATE_UP, (9728, 9728), project_gate_up, "_minillm_gate_up")):
        module = _module(names, widths)
        separate = project(module, x)
        merged = _merge(module, names)
        setattr(module, attr, merged)
        # Originals still compute the same thing and now view the merged storage.
        assert getattr(module, names[0]).weight.data_ptr() == merged.weight.data_ptr()
        for a, b in zip(separate, project(module, x)):
            assert torch.equal(a, b)
        for name, expected in zip(names, separate):
            assert torch.equal(getattr(module, name)(x), expected)


def test_unmergeable_shapes_are_left_alone():
    module = _module(QKV, (4096, 1024, 1024))
    module.k_proj.bias = torch.nn.Parameter(torch.zeros(1024, device="cuda"))
    assert _merge(module, QKV) is None


def test_merged_linears_ignores_unmerged_models():
    assert merged_linears(torch.nn.Linear(4, 4)) == []
