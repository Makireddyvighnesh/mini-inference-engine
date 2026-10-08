"""Stream tokens from the MiniLLM-L4 engine and report TTFT / TPOT per request.

Requests run through the continuous paged engine (resource admission: nothing
queues while KV pages are free).  With ``--arrival-interval-ms`` request i
arrives at i * interval, so later prompts are prefilled while earlier requests
are already decoding (mixed prefill + decode).  One silent warm-up run with
identical shapes and arrivals is excluded from the timings.

Timing follows docs/metric_definitions.md; every token timestamp is taken
after the GPU result is copied to the host:
  TTFT    = first token ready - arrival
  TPOT    = (last token ready - first token ready) / (tokens - 1)
  E2E     = last token ready - arrival
  max gap = longest interval between two consecutive tokens of a request

Run from the LLMPerfLab root, e.g. mixed prefill and decode:
  .conda-env/bin/python minillm_l4/scripts/stream_generate.py \
      --prompt-length 128 2048 --max-new-tokens 64 --batch-size 6 --arrival-interval-ms 400
"""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from minillm_l4.benchmarks.core.harness import RequestEventRecorder  # noqa: E402
from minillm_l4.benchmarks.core.schemas import RequestSpec  # noqa: E402
from minillm_l4.benchmarks.core.synthetic import SyntheticSample, exact_token_ids  # noqa: E402
from minillm_l4.benchmarks.runners.chunked_prefill import ChunkedPrefillPagedRunner  # noqa: E402
from minillm_l4.benchmarks.runners.huggingface_baseline import load_qwen_fp8  # noqa: E402

SEED_TEXT = (
    "Explain how prefill, cached decoding, and batching affect language-model "
    "inference. Give concrete comparisons."
)


class Session:
    """Shared live state so the timeline can show how many requests are decoding."""

    def __init__(self, stream: bool):
        self.stream = stream
        self.decoding: set[str] = set()

    def say(self, line: str) -> None:
        if self.stream:
            sys.stdout.write(f"\n  {line}\n")
            sys.stdout.flush()


class StreamingRecorder(RequestEventRecorder):
    """Event recorder that also prints tokens and a prefill/decode timeline."""

    def __init__(self, request_id, *, run_started_ns, request, session, tokenizer, show_text):
        super().__init__(request_id, run_started_ns=run_started_ns)
        self.request, self.session = request, session
        self.tokenizer, self.show_text = tokenizer, show_text
        self.arrival_ns = int(request.scheduled_arrival_ms * 1e6)
        self.token_ids: list[int] = []
        self.ready_ns: list[int] = []
        self._printed = ""
        self._prefill_start_ns: int | None = None

    def record(self, event, *, timestamp_ns=None, token_index=None, metadata=None):
        record = super().record(event, timestamp_ns=timestamp_ns, token_index=token_index, metadata=metadata)
        t = record.timestamp_ns / 1e9
        if event == "prefill_start":
            self._prefill_start_ns = record.timestamp_ns
            packed = (metadata or {}).get("packed_batch_size")
            together = f", flattened with {packed - 1} other prompt(s)" if packed else ""
            self.session.say(f"[{t:7.3f}s] {self.request_id} ({self.request.prompt_tokens}-token prompt) "
                             f"prefill starts; {len(self.session.decoding)} request(s) decoding{together}")
        elif event == "prefill_chunk_end" and not (metadata or {}).get("complete", True):
            meta = metadata or {}
            self.session.say(f"[{t:7.3f}s] {self.request_id} prompt chunk {meta.get('start_token')}->"
                             f"{meta.get('end_token')} of {self.request.prompt_tokens} done")
        elif event == "prefill_end":
            took = (record.timestamp_ns - self._prefill_start_ns) / 1e6
            self.session.say(f"[{t:7.3f}s] {self.request_id} prefill done in {took:.0f} ms -> first token, joins decode batch")
            self.session.decoding.add(self.request_id)
        return record

    def mark_token_ready(self, token_index, *, token_id=None, timestamp_ns=None):
        super().mark_token_ready(token_index, token_id=token_id, timestamp_ns=timestamp_ns)
        self.ready_ns.append(self.last_timestamp_ns)
        self.token_ids.append(int(token_id))
        if self.show_text and self.session.stream:
            text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
            sys.stdout.write(text[len(self._printed):])
            sys.stdout.flush()
            self._printed = text
        if len(self.token_ids) == self.request.max_new_tokens:
            self.session.decoding.discard(self.request_id)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt-length", type=int, nargs="+", required=True,
                        help="exact prompt tokens; several values are cycled across requests (e.g. 128 2048)")
    parser.add_argument("--max-new-tokens", type=int, required=True, help="maximum generated tokens per request")
    parser.add_argument("--batch-size", type=int, required=True, help="number of requests")
    parser.add_argument("--arrival-interval-ms", type=float, default=0.0,
                        help="request i arrives at i * interval (0: all arrive together)")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="chunked prefill size (tokens per iteration); default: whole-prompt prefill")
    parser.add_argument("--mixed", action="store_true",
                        help="vLLM-style mixed batching: one forward per step with all decode tokens "
                             "plus prompt chunks, shortest prompt first, within --token-budget")
    parser.add_argument("--adaptive", action="store_true",
                        help="mixed batching with traffic-aware, time-budgeted chunk sizes")
    parser.add_argument("--token-budget", type=int, default=2048,
                        help="with --mixed: maximum tokens (decode + prompt) per forward")
    parser.add_argument("--sequential-prefill", action="store_true",
                        help="prefill simultaneous prompts one at a time instead of one flattened forward")
    parser.add_argument("--show-request", type=int, default=-1,
                        help="stream this request's text (default -1: timeline only)")
    parser.add_argument("--ignore-eos", action="store_true", help="always generate --max-new-tokens tokens")
    parser.add_argument("--warmup", type=int, default=1, help="silent warm-up runs before the measured run")
    parser.add_argument("--check-hf", action="store_true",
                        help="compare every request's tokens with Hugging Face greedy generate")
    parser.add_argument("--prompt-text", default=SEED_TEXT, help="seed text expanded to each prompt length")
    args = parser.parse_args()
    if any(n < 1 for n in args.prompt_length) or args.max_new_tokens < 1 or args.batch_size < 1:
        parser.error("prompt lengths, --max-new-tokens and --batch-size must be positive")
    if args.token_budget < 1:
        parser.error("--token-budget must be positive")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.arrival_interval_ms < 0:
        parser.error("--arrival-interval-ms must be non-negative")
    return args


def build_runner(model, args, requests, eos):
    block = 16
    blocks = sum(math.ceil((r.prompt_tokens + r.max_new_tokens - 1) / block) for r in requests)
    return ChunkedPrefillPagedRunner(
        model, block_size=block, num_blocks=blocks, max_batch_size=len(requests),
        max_prefill_tokens=args.token_budget if args.mixed else (args.chunk_size or 256),
        prefill_chunk_size=args.chunk_size,
        device="cuda", decode_backend="auto", decode_sdpa_compat=True,
        enable_prefix=False, eos_token_id=eos, batched_prefill=not args.sequential_prefill,
        mixed_batch=args.mixed or args.adaptive, adaptive_chunking=args.adaptive,
    )


class GpuMonitor:
    """Sample nvidia-smi every 100 ms in a separate process (never blocks the engine)."""

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
             "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        out, _ = self.proc.communicate()
        self.samples = []
        for line in out.splitlines():
            try:
                self.samples.append(tuple(float(v) for v in line.split(",")))
            except ValueError:
                pass

    def report(self):
        if not self.samples:
            return "GPU monitor: no samples (is nvidia-smi visible?)"
        util = [s[0] for s in self.samples]
        mem = [s[1] for s in self.samples]
        power = [s[2] for s in self.samples]
        return (f"GPU busy (nvidia-smi, {len(util)} samples @100 ms): mean {statistics.mean(util):.0f}%  "
                f"median {statistics.median(util):.0f}%  peak {max(util):.0f}%\n"
                f"GPU memory used: peak {max(mem) / 1024:.1f} GiB   power: mean {statistics.mean(power):.0f} W\n"
                "(busy % = share of time any kernel is running, not how full the GPU's compute is)")


def run_once(model, tokenizer, args, requests, eos, *, stream):
    runner = build_runner(model, args, requests, eos)
    session = Session(stream)
    try:
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        recorders = []
        for index, request in enumerate(requests):
            recorder = StreamingRecorder(
                request.request_id, run_started_ns=start, request=request, session=session,
                tokenizer=tokenizer, show_text=index == args.show_request,
            )
            recorder.record("arrival", timestamp_ns=recorder.arrival_ns)
            recorders.append(recorder)
        outcomes = runner(requests, recorders)
        wall_s = (time.perf_counter_ns() - start) / 1e9
    finally:
        runner.close()
    for outcome in outcomes:
        if outcome.status != "completed":
            raise RuntimeError(f"request did not complete: {outcome}")
    return recorders, wall_s


def event_ms(recorder, name):
    return next((e.timestamp_ns / 1e6 for e in recorder.events if e.event == name), None)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not visible; run this on the L4 host.")
    print("Loading Qwen3-4B FP8 ...", flush=True)
    bundle = load_qwen_fp8(fp8_kernel_path="sm89", local_files_only=True,
                           pretune_tokens=max(16384, max(args.prompt_length) * args.batch_size))
    model, tokenizer = bundle.model, bundle.tokenizer
    print(f"FP8 kernel tuned for {bundle.kernel_strategy['pretuned_row_buckets']} shape buckets in "
          f"{bundle.kernel_strategy['pretune_ms'] / 1000:.1f} s (cached on disk after the first run).", flush=True)
    prompts = {n: exact_token_ids(tokenizer, SyntheticSample(
        sample_id=f"stream-{n}", category="interactive", target_prompt_tokens=n,
        target_output_tokens=args.max_new_tokens, seed_text=args.prompt_text,
    )) for n in set(args.prompt_length)}
    eos = None if args.ignore_eos else model.generation_config.eos_token_id
    requests = tuple(
        RequestSpec(f"r{i}", tuple(prompts[args.prompt_length[i % len(args.prompt_length)]]), args.max_new_tokens,
                    scheduled_arrival_ms=i * args.arrival_interval_ms)
        for i in range(args.batch_size)
    )
    if args.adaptive:
        policy = "adaptive: chunk size from a step-time limit (150 ms busy / 400 ms idle)"
    elif args.mixed:
        policy = (f"mixed batching: one forward per step, decode + prompt chunks, "
                  f"<= {args.token_budget} tokens per step")
    elif args.chunk_size:
        policy = f"chunked prefill, {args.chunk_size} tokens/iteration"
    elif args.sequential_prefill:
        policy = "whole-prompt prefill, one prompt per forward"
    else:
        policy = "whole-prompt prefill, simultaneous prompts flattened into one forward"
    print(f"{args.batch_size} requests, prompt lengths {args.prompt_length} (cycled), up to "
          f"{args.max_new_tokens} new tokens, arrivals every {args.arrival_interval_ms:g} ms, {policy}.")

    for index in range(args.warmup):
        print(f"Warm-up {index + 1}/{args.warmup} (not timed) ...", flush=True)
        run_once(model, tokenizer, args, requests, eos, stream=False)

    print("\n=== Measured run (live timeline) ===")
    if 0 <= args.show_request < args.batch_size:
        print(f"Streaming text of request r{args.show_request} between timeline lines.")
    with GpuMonitor() as monitor:
        recorders, wall_s = run_once(model, tokenizer, args, requests, eos, stream=True)

    print("\n\n=== Per-request timing (ms, relative to each request's own arrival unless noted) ===")
    header = (f"{'req':>4} {'prompt':>6} {'arrive@':>8} {'tokens':>6} {'wait':>7} {'prefill':>8} "
              f"{'TTFT':>8} {'TPOT':>7} {'max gap':>8} {'E2E':>9}")
    print(header)
    print("-" * len(header))
    ttfts, tpots, gaps, total_tokens = [], [], [], 0
    fmt = lambda v, w, d=1: f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}"
    for r in recorders:
        n = len(r.ready_ns)
        total_tokens += n
        arrival = r.arrival_ns / 1e6
        ttft = r.ready_ns[0] / 1e6 - arrival
        e2e = r.ready_ns[-1] / 1e6 - arrival
        tpot = (r.ready_ns[-1] - r.ready_ns[0]) / 1e6 / (n - 1) if n > 1 else None
        gap = max((b - a for a, b in zip(r.ready_ns, r.ready_ns[1:])), default=0) / 1e6 if n > 1 else None
        prefill_start, prefill_end = event_ms(r, "prefill_start"), event_ms(r, "prefill_end")
        wait = None if prefill_start is None else prefill_start - arrival
        prefill = None if prefill_start is None or prefill_end is None else prefill_end - prefill_start
        ttfts.append(ttft)
        if tpot is not None:
            tpots.append(tpot)
            gaps.append(gap)
        print(f"{r.request_id:>4} {r.request.prompt_tokens:>6} {arrival:>8.0f} {n:>6} {fmt(wait, 7)} {fmt(prefill, 8)} "
              f"{ttft:>8.1f} {fmt(tpot, 7, 2)} {fmt(gap, 8)} {e2e:>9.1f}")

    print("\n=== Summary ===")
    print(f"TTFT     median {statistics.median(ttfts):.1f} ms   min {min(ttfts):.1f}   max {max(ttfts):.1f}")
    if tpots:
        print(f"TPOT     median {statistics.median(tpots):.2f} ms   min {min(tpots):.2f}   max {max(tpots):.2f}")
        print(f"max gap  median {statistics.median(gaps):.1f} ms   worst {max(gaps):.1f}")
    print(f"Wall time {wall_s:.3f} s, {total_tokens} tokens generated, "
          f"aggregate throughput {total_tokens / wall_s:.1f} tokens/s")
    print(monitor.report())
    print("arrive@ = arrival time from the run start; wait = arrival -> this request's prefill starts;\n"
          "prefill = its own prompt compute (all chunks, including gaps between chunks);\n"
          "max gap = longest pause between two of its tokens (a decode stall while another prompt prefills).")

    if args.check_hf:
        from minillm_l4.engine.generation.huggingface import transformers_greedy_generate
        references = {}
        for n, prompt in prompts.items():
            inputs = torch.tensor([prompt], device="cuda")
            references[n] = transformers_greedy_generate(
                model, {"input_ids": inputs, "attention_mask": torch.ones_like(inputs)},
                output_tokens=args.max_new_tokens,
            ).token_ids[0].tolist()
        ok = all(r.token_ids == references[r.request.prompt_tokens][:len(r.token_ids)] for r in recorders)
        print(f"\nHF greedy check: {'PASS - every request matches exactly' if ok else 'FAIL - tokens differ'}")


if __name__ == "__main__":
    main()
