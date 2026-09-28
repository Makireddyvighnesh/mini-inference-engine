"""Compare contiguous and paged KV allocation under a fixed token budget.

This is a CPU-only allocator experiment; no model runs. It uses the real
``PagedKvAllocator`` for the paged side and measures two effects:

* Static capacity: how many requests from the Phase 6 shape mix fit at once
  when each is allocated at its final length, for each block size, compared
  with contiguous reservation of each request's final length and with
  contiguous reservation of the longest supported length for every request.
* Churn: a FIFO stream of mixed requests that grow one token per decode step
  and release on completion. Paged admission holds only tokens that exist but
  guarantees every admitted request can finish; contiguous admission must
  reserve each request's final length in one unbroken first-fit region, so it
  can be blocked by external fragmentation even when enough total space is
  free.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from minillm_l4.engine.kv_cache.paged import PagedKvAllocator, PagedKvOutOfMemoryError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results/kv_capacity/kv_capacity.json"
# Prompt/output shapes from configs/workloads/qwen3_fp8_paged.yaml.
PHASE6_SHAPES: tuple[tuple[int, int], ...] = ((128, 32), (512, 64), (2048, 128))


@dataclass(frozen=True)
class Request:
    request_id: str
    prompt_tokens: int
    output_tokens: int

    @property
    def final_tokens(self) -> int:
        """KV positions held when the last output token is produced."""

        return self.prompt_tokens + self.output_tokens - 1


def request_mix(count: int, *, seed: int, shapes: Sequence[tuple[int, int]] = PHASE6_SHAPES) -> list[Request]:
    """Return ``count`` requests cycling through ``shapes`` in a seeded order."""

    order = [shapes[index % len(shapes)] for index in range(count)]
    random.Random(seed).shuffle(order)
    return [Request(f"r{index:05d}", prompt, output) for index, (prompt, output) in enumerate(order)]


def static_capacity(
    requests: Sequence[Request],
    *,
    capacity_slots: int,
    block_sizes: Sequence[int],
) -> dict[str, dict[str, float]]:
    """Admit requests in order at their final length until the first does not fit."""

    results: dict[str, dict[str, float]] = {}
    longest = max(request.final_tokens for request in requests)
    for name, reservation in (
        ("contiguous_final_length", None),
        ("contiguous_max_length", longest),
    ):
        reserved = used = admitted = 0
        for request in requests:
            need = request.final_tokens if reservation is None else reservation
            if reserved + need > capacity_slots:
                break
            reserved += need
            used += request.final_tokens
            admitted += 1
        results[name] = _capacity_row(admitted, used, reserved, capacity_slots)
    for block_size in block_sizes:
        allocator = PagedKvAllocator(num_blocks=capacity_slots // block_size, block_size=block_size)
        used = admitted = 0
        for request in requests:
            try:
                allocator.allocate(request.request_id, token_count=request.final_tokens)
            except PagedKvOutOfMemoryError:
                break
            used += request.final_tokens
            admitted += 1
        reserved = allocator.allocated_block_count * block_size
        results[f"paged_b{block_size}"] = _capacity_row(admitted, used, reserved, capacity_slots)
    return results


def _capacity_row(admitted: int, used: int, reserved: int, capacity: int) -> dict[str, float]:
    return {
        "requests_admitted": admitted,
        "used_token_slots": used,
        "reserved_token_slots": reserved,
        "wasted_token_slots": reserved - used,
        "internal_waste_fraction": (reserved - used) / reserved if reserved else 0.0,
        "budget_utilization": used / capacity,
    }


class FirstFitContiguous:
    """A first-fit allocator over one linear range of token slots."""

    def __init__(self, capacity_slots: int) -> None:
        self.capacity = int(capacity_slots)
        self._holes: list[tuple[int, int]] = [(0, self.capacity)]  # (start, length)
        self._regions: dict[str, tuple[int, int]] = {}

    @property
    def free_slots(self) -> int:
        return sum(length for _, length in self._holes)

    @property
    def largest_hole(self) -> int:
        return max((length for _, length in self._holes), default=0)

    def allocate(self, owner: str, slots: int) -> bool:
        for index, (start, length) in enumerate(self._holes):
            if length >= slots:
                self._regions[owner] = (start, slots)
                if length == slots:
                    del self._holes[index]
                else:
                    self._holes[index] = (start + slots, length - slots)
                return True
        return False

    def release(self, owner: str) -> None:
        start, slots = self._regions.pop(owner)
        holes = sorted(self._holes + [(start, slots)])
        merged: list[tuple[int, int]] = []
        for hole_start, hole_length in holes:
            if merged and merged[-1][0] + merged[-1][1] == hole_start:
                merged[-1] = (merged[-1][0], merged[-1][1] + hole_length)
            else:
                merged.append((hole_start, hole_length))
        self._holes = merged


def simulate_churn(
    requests: Sequence[Request],
    *,
    capacity_slots: int,
    policy: str,
    block_size: int = 16,
    contiguous_reservation: int | None = None,
) -> dict[str, float]:
    """Run a FIFO decode loop: admit, grow one token per step, release on completion.

    Every request is available at step 0, so the queue is never empty until the
    tail; the result measures how densely each policy packs a steady backlog.
    ``contiguous_reservation`` makes every contiguous request reserve that many
    slots instead of its own final length (a server sized for its longest
    request).
    """

    if policy not in {"paged", "contiguous"}:
        raise ValueError("policy must be 'paged' or 'contiguous'")
    pending = list(requests)
    active: dict[str, tuple[Request, int]] = {}  # request -> tokens currently held
    paged = (
        PagedKvAllocator(num_blocks=capacity_slots // block_size, block_size=block_size)
        if policy == "paged"
        else None
    )
    contiguous = FirstFitContiguous(capacity_slots) if policy == "contiguous" else None
    steps = 0
    active_sum = used_sum = reserved_sum = 0.0
    fragmentation_blocked_steps = 0

    def blocks(tokens: int) -> int:
        return math.ceil(tokens / block_size)

    while pending or active:
        # Admission (FIFO, head-of-line).
        while pending:
            request = pending[0]
            if paged is not None:
                future_growth = sum(
                    blocks(item.final_tokens) - blocks(held) for item, held in active.values()
                )
                if paged.free_block_count - future_growth < blocks(request.final_tokens):
                    break
                paged.allocate(request.request_id, token_count=request.prompt_tokens)
            else:
                assert contiguous is not None
                need = contiguous_reservation or request.final_tokens
                if not contiguous.allocate(request.request_id, need):
                    if contiguous.free_slots >= need:
                        fragmentation_blocked_steps += 1
                    break
            active[request.request_id] = (request, request.prompt_tokens)
            pending.pop(0)
        if not active:
            raise RuntimeError(f"request {pending[0].request_id} can never fit the budget")

        # One decode step: every active request produces a token.
        steps += 1
        active_sum += len(active)
        used_sum += sum(held for _, held in active.values())
        reserved_sum += (
            paged.allocated_block_count * block_size
            if paged is not None
            else sum(
                contiguous_reservation or item.final_tokens for item, _ in active.values()
            )
        )
        finished: list[str] = []
        for request_id, (request, held) in list(active.items()):
            generated = held - request.prompt_tokens + 1
            if generated >= request.output_tokens:
                finished.append(request_id)
                continue
            if paged is not None:
                paged.append(request_id, 1)
            active[request_id] = (request, held + 1)
        for request_id in finished:
            del active[request_id]
            if paged is not None:
                paged.release(request_id)
            else:
                assert contiguous is not None
                contiguous.release(request_id)

    return {
        "decode_steps": steps,
        "mean_active_requests": active_sum / steps,
        "mean_used_budget_fraction": used_sum / steps / capacity_slots,
        "mean_reserved_budget_fraction": reserved_sum / steps / capacity_slots,
        "fragmentation_blocked_steps": fragmentation_blocked_steps,
    }


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == PROJECT_ROOT.name:
        return PROJECT_ROOT.parent / path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contiguous vs paged KV allocation under a fixed budget.")
    parser.add_argument("--capacity-token-slots", type=int, default=32_768)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--static-requests", type=int, default=300)
    parser.add_argument("--churn-requests", type=int, default=600)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    static = static_capacity(
        request_mix(args.static_requests, seed=args.seed),
        capacity_slots=args.capacity_token_slots,
        block_sizes=args.block_sizes,
    )
    churn_requests = request_mix(args.churn_requests, seed=args.seed)
    churn: dict[str, Any] = {
        "contiguous_final_length": simulate_churn(
            churn_requests, capacity_slots=args.capacity_token_slots, policy="contiguous"
        ),
        "contiguous_max_length": simulate_churn(
            churn_requests,
            capacity_slots=args.capacity_token_slots,
            policy="contiguous",
            contiguous_reservation=max(request.final_tokens for request in churn_requests),
        ),
    }
    for block_size in args.block_sizes:
        churn[f"paged_b{block_size}"] = simulate_churn(
            churn_requests,
            capacity_slots=args.capacity_token_slots,
            policy="paged",
            block_size=block_size,
        )
    payload = {
        "schema_version": 1,
        "capacity_token_slots": args.capacity_token_slots,
        "shapes": [list(shape) for shape in PHASE6_SHAPES],
        "seed": args.seed,
        "static_capacity": static,
        "churn": churn,
    }
    output = _project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("static capacity (requests admitted at final length):")
    for name, row in static.items():
        print(
            f"  {name:26s} admitted={row['requests_admitted']:4d} "
            f"waste={100 * row['internal_waste_fraction']:5.2f}% "
            f"budget_used={100 * row['budget_utilization']:5.1f}%"
        )
    print("churn (FIFO backlog, grow one token per step):")
    for name, row in churn.items():
        print(
            f"  {name:26s} steps={row['decode_steps']:5d} "
            f"mean_active={row['mean_active_requests']:6.2f} "
            f"used={100 * row['mean_used_budget_fraction']:5.1f}% "
            f"reserved={100 * row['mean_reserved_budget_fraction']:5.1f}% "
            f"frag_blocked_steps={row['fragmentation_blocked_steps']}"
        )
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
