"""Profile where one greedy decode step spends its time on the L4.

For each prompt/batch shape this command times plain decode steps without the
profiler, then records a few steps with ``torch.profiler`` and reports, per
step: wall time, GPU-busy time (the union of kernel and memcpy intervals), the
number of GPU operations, and the weight-bandwidth floor. It also writes the
top CPU operators, the top GPU kernels, and a Chrome/Perfetto trace per shape.

The decode loop mirrors the manual backend: an explicit attention mask, a
``DynamicCache`` carried between steps, and one host read of the next token
per step.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..core.hardware import collect_environment_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/profile_decode"
STEP_MARKER = "decode_step_"
L4_PEAK_BANDWIDTH_BYTES_PER_SECOND = 300e9


def parse_shape(value: str) -> tuple[int, int]:
    """Parse ``PROMPTxBATCH`` such as ``2048x4``."""

    try:
        prompt, batch = (int(part) for part in value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"shape must look like PROMPTxBATCH, received {value!r}"
        ) from error
    if prompt < 1 or batch < 1:
        raise argparse.ArgumentTypeError("prompt and batch must be positive")
    return prompt, batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile MiniLLM-L4 decode steps with torch.profiler."
    )
    parser.add_argument(
        "--shapes",
        type=parse_shape,
        nargs="+",
        default=[(128, 1), (2048, 1), (2048, 4)],
        help="PROMPTxBATCH shapes to profile, e.g. 128x1 2048x4",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--profiled-steps", type=int, default=5)
    parser.add_argument(
        "--fp8-kernel-path",
        choices=("auto", "sm89"),
        default="auto",
        help="auto is the Transformers path used by Phases 1-5",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.warmup_steps < 0 or args.timed_steps < 1 or args.profiled_steps < 2:
        parser.error("need warmup >= 0, timed >= 1, and profiled >= 2 steps")
    return args


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def merged_duration_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Return the length of the union of ``[start, end)`` intervals."""

    total = 0
    current_start: int | None = None
    current_end = 0
    for start, end in sorted(intervals):
        if current_start is None or start > current_end:
            if current_start is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_start is not None:
        total += current_end - current_start
    return total


def attribute_gpu_work(
    steps: Sequence[tuple[int, int]],
    gpu_intervals: Sequence[tuple[int, int]],
) -> list[dict[str, float]]:
    """Assign GPU intervals to the step window that fully contains them.

    Every step ends with a host read of the next token, which waits for that
    step's GPU work, so each step's kernels start and finish inside its CPU
    window.
    """

    per_step: list[dict[str, float]] = []
    for step_start, step_end in steps:
        contained = [
            interval
            for interval in gpu_intervals
            if step_start <= interval[0] and interval[1] <= step_end
        ]
        per_step.append(
            {
                "wall_ms": (step_end - step_start) / 1e6,
                "gpu_busy_ms": merged_duration_ns(contained) / 1e6,
                "gpu_operations": float(len(contained)),
            }
        )
    return per_step


def _kineto_windows(profiler: Any) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    from torch.autograd import DeviceType

    steps: list[tuple[int, int]] = []
    gpu: list[tuple[int, int]] = []
    for event in profiler.profiler.kineto_results.events():
        interval = (event.start_ns(), event.start_ns() + event.duration_ns())
        if event.name().startswith(STEP_MARKER):
            # record_function markers appear on both the CPU and the GPU
            # timeline; only the CPU copy is a step window, and neither copy
            # is GPU work.
            if event.device_type() == DeviceType.CPU:
                steps.append(interval)
        elif event.device_type() == DeviceType.CUDA:
            gpu.append(interval)
    return sorted(steps), gpu


def profile_shape(
    model: Any,
    *,
    prompt_tokens: int,
    batch_size: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    input_ids = torch.randint(
        100, 20_000, (batch_size, prompt_tokens), generator=generator
    ).to(device)
    total_steps = args.warmup_steps + args.timed_steps + args.profiled_steps
    attention_mask = torch.ones(
        (batch_size, prompt_tokens + total_steps + 1), dtype=torch.long, device=device
    )

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask[:, :prompt_tokens],
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        next_token = output.logits[:, -1:].argmax(dim=-1)
        position = prompt_tokens

        def step() -> None:
            nonlocal next_token, position
            result = model(
                input_ids=next_token,
                attention_mask=attention_mask[:, : position + 1],
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            next_token = result.logits[:, -1:].argmax(dim=-1)
            position += 1
            next_token.tolist()  # the runners read each step's tokens on the host

        for _ in range(args.warmup_steps):
            step()
        timed_ms: list[float] = []
        for _ in range(args.timed_steps):
            started = time.perf_counter()
            step()
            timed_ms.append((time.perf_counter() - started) * 1e3)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
        ) as profiler:
            for index in range(args.profiled_steps):
                with record_function(f"{STEP_MARKER}{index}"):
                    step()

    tag = f"p{prompt_tokens}_b{batch_size}"
    profiler.export_chrome_trace(str(output_dir / f"trace_{tag}.json"))
    averages = profiler.key_averages()
    (output_dir / f"cpu_ops_{tag}.txt").write_text(
        averages.table(sort_by="self_cpu_time_total", row_limit=20), encoding="utf-8"
    )
    (output_dir / f"gpu_kernels_{tag}.txt").write_text(
        averages.table(sort_by="self_device_time_total", row_limit=20), encoding="utf-8"
    )

    steps, gpu_intervals = _kineto_windows(profiler)
    per_step = attribute_gpu_work(steps, gpu_intervals)[1:]  # drop profiler warm-up
    step_ms = statistics.median(timed_ms)
    gpu_busy_ms = statistics.median(row["gpu_busy_ms"] for row in per_step)
    return {
        "prompt_tokens": prompt_tokens,
        "batch_size": batch_size,
        "step_ms_p50": step_ms,
        "step_ms_samples": timed_ms,
        "profiled_step_ms_p50": statistics.median(row["wall_ms"] for row in per_step),
        "gpu_busy_ms_p50": gpu_busy_ms,
        "gpu_busy_fraction": gpu_busy_ms / step_ms,
        "gpu_operations_per_step": statistics.median(
            row["gpu_operations"] for row in per_step
        ),
        "profiled_steps": per_step,
    }


def main() -> None:
    args = parse_args()
    import torch

    from ..runners.huggingface_baseline import load_qwen_fp8

    output_dir = _project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_qwen_fp8(device=args.device, fp8_kernel_path=args.fp8_kernel_path)
    weight_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in bundle.model.parameters()
    )
    floor_ms = weight_bytes / L4_PEAK_BANDWIDTH_BYTES_PER_SECOND * 1e3
    print(
        f"weights {weight_bytes / 2**30:.2f} GiB; "
        f"bandwidth floor {floor_ms:.1f} ms/step at 300 GB/s",
        flush=True,
    )

    shapes: list[dict[str, Any]] = []
    for prompt_tokens, batch_size in args.shapes:
        row = profile_shape(
            bundle.model,
            prompt_tokens=prompt_tokens,
            batch_size=batch_size,
            args=args,
            output_dir=output_dir,
        )
        shapes.append(row)
        print(
            f"prompt={prompt_tokens:<5} batch={batch_size:<2} "
            f"step_P50={row['step_ms_p50']:.2f} ms "
            f"gpu_busy_P50={row['gpu_busy_ms_p50']:.2f} ms "
            f"({100 * row['gpu_busy_fraction']:.0f}%) "
            f"gpu_ops/step={row['gpu_operations_per_step']:.0f}",
            flush=True,
        )
        torch.cuda.empty_cache()

    summary = {
        "schema_version": 1,
        "model": bundle.metadata(),
        "environment": collect_environment_metadata(),
        "configuration": {
            "shapes": [list(shape) for shape in args.shapes],
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.timed_steps,
            "profiled_steps": args.profiled_steps,
            "fp8_kernel_path": args.fp8_kernel_path,
            "seed": args.seed,
            "prompt_source": "uniform random token IDs in [100, 20000)",
        },
        "weight_bytes": weight_bytes,
        "bandwidth_floor_ms": floor_ms,
        "shapes": shapes,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
