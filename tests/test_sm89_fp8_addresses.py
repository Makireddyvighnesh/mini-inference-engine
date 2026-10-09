"""Evaluate the kernels' address expressions on CPU, without allocating tensors of their target size."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from minillm_l4.engine.kernels import sm89_fp8 as fp8


@pytest.mark.parametrize("kernel_name", [
    "_fp8_linear_kernel", "_quantize_activation_kernel", "_fp8_split_gemm_kernel",
])
def test_address_products_are_64_bit_before_multiplication(kernel_name):
    # Read the actual address expressions even if Triton is not installed.
    # Small CPU index vectors cross the signed-int32 boundary at M*K, and
    # also exercise column/stride products without multi-GB allocations.
    module = ast.parse(Path(fp8.__file__).read_text())
    kernel = next(node for node in ast.walk(module)
                  if isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    boundary = 2**31 // 9728
    indices = torch.tensor([0, boundary, boundary + 1, 262143], dtype=torch.int32)
    common = dict(
        tl=SimpleNamespace(int64=torch.int64, cast=lambda value, dtype: torch.as_tensor(value, dtype=dtype)),
        activation=0, activation_fp8=0, activation_scales=0, weight=0, weight_scales=0, output=0,
        stride_am=9728, stride_ak=1, stride_wn=9728, stride_wk=1, stride_om=9728, stride_on=1,
        stride_asm=76, stride_ask=1, stride_sn=76, stride_sk=1, K=9728, groups_k=76, BLOCK_K=128,
    )

    def addresses(dtype):
        env = dict(common, offsets_m=indices.to(dtype), offsets_n=indices.to(dtype),
                   offsets_k=torch.tensor([0, 127], dtype=dtype), k_block=torch.tensor(75, dtype=dtype))

        def evaluate(expression):
            return eval(compile(ast.Expression(expression), "<kernel address>", "eval"), {"__builtins__": {}}, env)

        for statement in kernel.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and (target.id.startswith("address_") or target.id.endswith("_ptrs")):
                    env[target.id] = evaluate(statement.value)
        result = []
        for node in ast.walk(kernel):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl"
                    and node.func.attr in {"load", "store"}):
                result.append(evaluate(node.args[0]))
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id.endswith("_ptrs"):
                result.append(evaluate(node.value))
        return result

    actual, expected = addresses(torch.int32), addresses(torch.int64)
    assert actual and any(offset.max().item() >= 2**31 for offset in expected)
    for offset, reference in zip(actual, expected, strict=True):
        assert offset.dtype == torch.int64
        assert torch.equal(offset, reference), "address arithmetic overflowed before widening"
