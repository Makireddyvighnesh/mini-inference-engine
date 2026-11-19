"""Benchmark LFM2.5-2.6B with and without the DSpark draft model on one GPU.

What it does, per config (target quantization x speculative setting):
  1. starts llama-server with that config and waits for /health,
  2. sends one warm-up request (not recorded),
  3. sends every prompt in prompts.jsonl `--reps` times, greedy, one at a time,
  4. records llama-server's own timings for each request plus the output text,
  5. stops the server.

Then it runs llama-bench on each target (raw prefill/decode speed, no
speculation) and writes the report with analyze.py.

Results go to results/<YYYYMMDD-HHMM>_<gpu>/ next to this file:
  env.json          GPU, driver, CUDA, llama.cpp commit, model files, arguments
  configs.json      the configs that ran, their server command, and status
  requests.jsonl    one row per request (timings, draft stats, output text)
  llama_bench.jsonl llama-bench rows (pp512 / tg128 per target)
  summary.csv       one row per config (written by analyze.py)
  report.md         readable report (written by analyze.py)
  logs/             server and llama-bench logs (last 3,000 lines each)

Run on Colab after setup_llamacpp.sh:
  python run_bench.py --llama-dir /content/llama.cpp
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze  # noqa: E402

TARGET_REPO = "LiquidAI/LFM2.5-2.6B-GGUF"
DRAFT_REPO = "LiquidAI/LFM2.5-2.6B-DSpark-GGUF"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------- helpers
def sh(cmd: list[str] | str, **kw) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                             shell=isinstance(cmd, str), **kw)
        return out.stdout.strip() if out.returncode == 0 else "<unavailable>"
    except Exception as exc:  # noqa: BLE001
        return f"<error: {exc}>"


def resolve_gguf(repo: str, quant: str, local_dir: Path) -> Path:
    """Download the GGUF in `repo` whose name ends with -<quant>.gguf."""
    if quant.endswith(".gguf") and Path(quant).exists():
        return Path(quant)
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
    want = f"-{quant.lower()}.gguf"
    matches = [f for f in files if f.lower().endswith(want) and "qad" not in f.lower()]
    if not matches:
        raise SystemExit(f"No file ending in {want} in {repo}. Files: {files}")
    name = sorted(matches, key=len)[0]
    print(f"  {repo}/{name}")
    return Path(hf_hub_download(repo, name, local_dir=str(local_dir)))


def gpu_info() -> dict:
    q = "name,driver_version,memory.total,clocks.max.sm,clocks.max.memory,power.limit"
    raw = sh(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    keys = q.split(",")
    vals = [v.strip() for v in raw.split(",")]
    info = dict(zip(keys, vals)) if len(vals) == len(keys) else {"raw": raw}
    info["cuda_version"] = (re.search(r"CUDA Version:\s*([\d.]+)", sh(["nvidia-smi"])) or [None, None])[1]
    info["nvcc"] = sh(["bash", "-lc", "nvcc --version | tail -n 2"])
    return info


def http_json(url: str, payload: dict | None = None, timeout: float = 600) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tail_file(path: Path, n: int = 3000) -> None:
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) > n:
        path.write_text("\n".join([f"[... {len(lines) - n} earlier lines cut ...]"] + lines[-n:]) + "\n")


# ---------------------------------------------------------------- server
class Server:
    def __init__(self, cmd: list[str], log_path: Path, port: int):
        self.cmd, self.log_path, self.port = cmd, log_path, port
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 600) -> tuple[bool, float]:
        t0 = time.time()
        self.log = open(self.log_path, "w")
        self.log.write("$ " + " ".join(self.cmd) + "\n\n")
        self.log.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=self.log, stderr=subprocess.STDOUT)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False, time.time() - t0
            try:
                if http_json(f"http://127.0.0.1:{self.port}/health", timeout=5).get("status") == "ok":
                    return True, time.time() - t0
            except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError, OSError):
                pass
            time.sleep(1)
        return False, time.time() - t0

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if hasattr(self, "log"):
            self.log.close()
        tail_file(self.log_path)


def server_cmd(args, target: Path, draft: Path | None, n_max: int | None, ngld: bool) -> list[str]:
    cmd = [str(args.server_bin), "-m", str(target), "-ngl", "99", "-fa", args.flash_attn,
           "-c", str(args.ctx), "-np", "1", "--port", str(args.port), "--host", "127.0.0.1",
           "--no-warmup"]
    if draft is not None:
        cmd += ["-md", str(draft), "--spec-type", "draft-dspark",
                "--spec-draft-n-max", str(n_max), "--spec-draft-n-min", "0"]
        if ngld:
            cmd += ["-ngld", "99"]
    return cmd + list(args.extra_server_args)


def one_request(port: int, prompt: str, max_tokens: int) -> tuple[dict, float]:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 0,
        "cache_prompt": False,  # every request does its own prefill
    }
    t0 = time.perf_counter()
    resp = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", payload)
    return resp, time.perf_counter() - t0


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llama-dir", type=Path, default=Path("/content/llama.cpp"))
    p.add_argument("--models-dir", type=Path, default=Path("/content/models"))
    p.add_argument("--targets", nargs="+", default=["F16", "Q4_K_M"],
                   help="target quantizations from LFM2.5-2.6B-GGUF (or paths to .gguf)")
    p.add_argument("--draft", default="F16", help="draft quantization from LFM2.5-2.6B-DSpark-GGUF")
    p.add_argument("--n-max", nargs="+", type=int, default=[3, 6, 10],
                   help="--spec-draft-n-max values to try (0 = no speculation is always run)")
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"])
    p.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    p.add_argument("--limit-prompts", type=int, default=0, help="use only the first N prompts (smoke test)")
    p.add_argument("--tag", default="", help="suffix for the results folder name")
    p.add_argument("--out", type=Path, default=None, help="results folder (default: results/<date>_<gpu>)")
    p.add_argument("--skip-llama-bench", action="store_true")
    p.add_argument("--extra-server-args", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    bin_dir = args.llama_dir / "build" / "bin"
    args.server_bin = bin_dir / "llama-server"
    bench_bin = bin_dir / "llama-bench"
    if not args.server_bin.exists():
        raise SystemExit(f"{args.server_bin} not found: run setup_llamacpp.sh first")

    gpu = gpu_info()
    gpu_slug = re.sub(r"[^a-z0-9]+", "-", gpu.get("name", "gpu").lower()).strip("-").replace("nvidia-", "")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M")
    out = args.out or HERE / "results" / f"{stamp}_{gpu_slug}{('_' + args.tag) if args.tag else ''}"
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    (out / "logs").mkdir(parents=True, exist_ok=True)
    print(f"Results -> {out}")

    prompts = [json.loads(line) for line in args.prompts.read_text().splitlines() if line.strip()]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]

    print("Downloading models")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    # label -> file; a quantization name ("F16") is its own label, a path uses the file stem
    targets = {(Path(q).stem if q.endswith(".gguf") else q): resolve_gguf(TARGET_REPO, q, args.models_dir)
               for q in args.targets}
    draft = resolve_gguf(DRAFT_REPO, args.draft, args.models_dir)

    env = {
        "schema_version": SCHEMA_VERSION,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "gpu": gpu,
        "host": {"python": platform.python_version(), "platform": platform.platform(),
                 "cpu_count": os.cpu_count()},
        "llama_cpp": {"commit": sh(["git", "-C", str(args.llama_dir), "rev-parse", "HEAD"]),
                      "subject": sh(["git", "-C", str(args.llama_dir), "log", "-1", "--format=%s"]),
                      "date": sh(["git", "-C", str(args.llama_dir), "log", "-1", "--format=%cI"])},
        "repo_commit": sh(["git", "-C", str(HERE), "rev-parse", "HEAD"]),
        "models": {"targets": {q: {"file": t.name, "bytes": t.stat().st_size} for q, t in targets.items()},
                   "draft": {"file": draft.name, "bytes": draft.stat().st_size,
                             "repo": DRAFT_REPO}, "target_repo": TARGET_REPO},
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_prompts": len(prompts),
    }
    (out / "env.json").write_text(json.dumps(env, indent=2))

    configs = []
    for q in targets:
        configs.append({"name": f"{q}_base", "target": q, "spec": False, "n_max": 0})
        for n in args.n_max:
            configs.append({"name": f"{q}_dspark{n}", "target": q, "spec": True, "n_max": n})

    req_path = out / "requests.jsonl"
    for ci, cfg in enumerate(configs):
        print(f"\n[{ci + 1}/{len(configs)}] {cfg['name']}")
        log_path = out / "logs" / f"server_{cfg['name']}.log"
        t_draft = draft if cfg["spec"] else None
        cmd = server_cmd(args, targets[cfg["target"]], t_draft, cfg["n_max"], ngld=True)
        server = Server(cmd, log_path, args.port)
        ok, load_s = server.start()
        if not ok and cfg["spec"]:
            # Some llama.cpp versions drop -ngld (the draft follows -ngl): retry without it.
            server.stop()
            print("  server did not start; retrying without -ngld")
            cmd = server_cmd(args, targets[cfg["target"]], t_draft, cfg["n_max"], ngld=False)
            server = Server(cmd, log_path.with_name(log_path.stem + "_retry.log"), args.port)
            ok, load_s = server.start()
        cfg.update({"cmd": " ".join(cmd), "load_s": round(load_s, 1), "status": "ok" if ok else "server_failed"})
        if not ok:
            server.stop()
            print(f"  FAILED to start, see {log_path}")
            continue
        try:
            one_request(args.port, "Say hello in one short sentence.", 16)  # warm-up, not recorded
            for rep in range(args.reps):
                for pr in prompts:
                    try:
                        resp, wall = one_request(args.port, pr["prompt"], args.max_tokens)
                        err = None
                    except Exception as exc:  # noqa: BLE001
                        resp, wall, err = {}, None, repr(exc)
                    timings = resp.get("timings", {})
                    choice = (resp.get("choices") or [{}])[0]
                    row = {
                        "config": cfg["name"], "target": cfg["target"], "spec": cfg["spec"],
                        "n_max": cfg["n_max"], "rep": rep, "prompt_id": pr["id"],
                        "category": pr["category"], "wall_s": wall, "error": err,
                        "finish_reason": choice.get("finish_reason"),
                        "usage": resp.get("usage"), "timings": timings,
                        "text": (choice.get("message") or {}).get("content"),
                    }
                    with req_path.open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    tps = timings.get("predicted_per_second")
                    acc = (f" accepted {timings.get('draft_n_accepted')}/{timings.get('draft_n')}"
                           if cfg["spec"] else "")
                    print(f"  rep{rep} {pr['id']:<14} {timings.get('predicted_n', '?'):>4} tok "
                          f"{tps if tps is None else round(tps, 1)} tok/s{acc}" + (f" ERROR {err}" if err else ""))
        finally:
            server.stop()
        (out / "configs.json").write_text(json.dumps(configs, indent=2))
    (out / "configs.json").write_text(json.dumps(configs, indent=2))

    if not args.skip_llama_bench and bench_bin.exists():
        print("\nllama-bench (no speculation): pp512 = prefill speed, tg128 = decode speed")
        for q, t in targets.items():
            cmd = [str(bench_bin), "-m", str(t), "-ngl", "99", "-fa", "1" if args.flash_attn == "on" else "0",
                   "-p", "512", "-n", "128", "-r", "3", "-o", "jsonl"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            (out / "logs" / f"llama_bench_{q}.log").write_text("$ " + " ".join(cmd) + "\n" + res.stderr)
            with (out / "llama_bench.jsonl").open("a") as f:
                for line in res.stdout.splitlines():
                    if line.strip().startswith("{"):
                        row = json.loads(line)
                        row["target"] = q
                        f.write(json.dumps(row) + "\n")
                        print(f"  {q}: n_prompt={row.get('n_prompt')} n_gen={row.get('n_gen')} "
                              f"{row.get('avg_ts', 0):.1f} tok/s")

    env["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    (out / "env.json").write_text(json.dumps(env, indent=2))
    report = analyze.write_report(out)
    print("\n" + report)
    print(f"\nDone. Results in {out}")


if __name__ == "__main__":
    main()
