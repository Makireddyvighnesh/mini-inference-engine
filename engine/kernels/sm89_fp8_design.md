# Split SM89 FP8 linear: implementation and L4 validation

The fused kernel body and original nine autotune configs are unchanged.
`sm89_fp8_linear(..., path="fused")` selects it, `path="split"` selects the
quantizer plus GEMM, and `path="auto"` uses `SPLIT_M_THRESHOLD` (provisionally
256, inclusive). This crossover is a starting point, not a measured result.
`gemm_config=SPLIT_GEMM_CONFIGS[i]` with `path="split"` bypasses the GEMM
autotuner. Both autotuners keep `M_BUCKET,N,K`, disk caching, and the 16384
bucket cap. Pretuning warms both paths at each bucket, including the ceiling
bucket for a non-power-of-two limit; its return value remains shape/bucket
pairs. The quantizer has one fixed config, compiled during split pretuning.

The quantizer loads bf16/fp16 into fp32, substitutes zero for a K tail,
reduces max(abs) across exactly 128 columns, and divides by 448.0. It stores
the **raw** fp32 scale (including zero), then divides by max(scale, 1e-12)
and converts to e4m3. Storing an already-fp32 scale or already-fp8 value adds
no rounding step. The GEMM reads these values and applies the original
`accumulator += dot(a_fp8, w) * activation_scale * weight_scale` expression
once per 128-column block in increasing K order, then converts to the output
dtype. Every config fixes BLOCK_K=128. There is no split-K, combined scale
product, multi-block dot, or changed floating-point fusion setting. Bias
addition and output reshaping follow the original wrapper.

The remaining hardware assumption is that FP8 dot results and the compiled
scaling/accumulation arithmetic agree across tile layouts, warps, and pipeline
depths. Source arithmetic alone is not a bitwise proof of compiler behavior.
The L4 checker therefore forces **every** split config, compares raw bits
against the original fused kernel, and checks every row against real 1-row
fused calls by default. It also runs real M=1 split GEMMs under every forced
config, and separately checks 1-row quantization, including scales. These
checks establish both paths' 1-row vs M-row equivalence for every checked row.
The checker covers all requested M, model shapes, and small N/K tails, with
zero, signed-zero, large, tiny, and outlier input rows. `--row-samples` is an
explicitly sampled development mode, not the exhaustive gate.

The 14 candidate configs use 64/128-row tiles, 64/128/256-column tiles,
4/8 warps, 3/4/5 stages, and grouped M ordering. Explicit `tl.range`
staging enables pipelining even when dot has a fresh zero accumulator each
iteration. CPU-only compilation with Triton 3.7.1 to an SM89 target emits
asynchronous copies and uses 33--99 KiB shared memory for aligned model
shapes. The largest candidates sit at the 99 KiB CTA limit; actual launch
resources, register pressure, exactness, and performance still need the L4.
Compilation here did not initialize CUDA or launch a GPU kernel.

From `/home/ubuntu/STT/LLMPerfLab` on the L4:

```bash
.conda-env/bin/python -m pytest -q minillm_l4/tests/test_sm89_fp8_cpu.py minillm_l4/tests/test_sm89_fp8_exact.py
.conda-env/bin/python -m minillm_l4.scripts.check_fp8_exact
.conda-env/bin/python -m minillm_l4.scripts.bench_fp8_gemm
/home/ubuntu/miniconda3/envs/nemo35_vllm/bin/python -m minillm_l4.scripts.bench_fp8_gemm --vllm
```

Optional wider dtype sweep:

```bash
.conda-env/bin/python -m minillm_l4.scripts.check_fp8_exact --dtypes bf16 fp16 --output-dtypes bf16 fp16
```

Return the checker table and any mismatch/launch diagnostics, plus the
benchmark tables including chosen configs, quantizer/GEMM component timings,
seven-projection per-layer totals, and allocation retries. Those measurements
will set the dispatch threshold and guide config pruning. The benchmark uses
median `triton.testing.do_bench` timings with warmup and tuning outside the
timed region. Complete-call times include quantization and allocations;
component times use preallocated buffers. Their sum need not equal a complete
call because launch overhead and cache flushing differ. Optional vLLM timing
uses its own quantizer and is a performance comparison, not an exactness
reference. No model weights are downloaded or loaded.

## Future merged projections (not implemented)

Concatenate q/k/v weights and their scale matrices along N to form
`[6144,2560]`; concatenate gate/up to form `[19456,2560]`. Quantize each shared
activation once, run one GEMM for each concatenation, and slice the output
into the original projections. All current projection widths are multiples
of 128, so concatenation boundaries preserve weight scale block boundaries.
Keep BLOCK_K=128 and the same scaling/accumulation expression; each output
column has the same input values and K order. A new N and tile choice still
require the bitwise gate, just as the split kernel does. Apply each original
bias after slicing with the same dtype and operation, and account for the
slices' noncontiguous row strides when integrating with the model runner.

This would reduce launch count, reuse quantized activations, and expose more
N tiles to the GPU. It requires changes to model projection orchestration
and loading/scale concatenation beyond this kernel-only task. Validate
merged output slices against separate calls for every config, and rerun the
model/policy exactness gates before using it in inference.
