"""Phase 1 Hugging Face baseline for MiniLLM-L4.

This module intentionally keeps the baseline small and explicit.  It loads one
pinned checkpoint, materializes exact-token workloads with that checkpoint's
tokenizer, and invokes ``model.generate`` for each static batch.  Later phases
can replace the runner while reusing the workload, event, and result
contracts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..core.harness import BenchmarkResult, RequestEventRecorder
from ..core.schemas import RequestOutcome, RequestSpec, WorkloadSpec
from minillm_l4.engine.generation.huggingface import (
    transformers_greedy_generate,
)
from minillm_l4.engine.generation.manual import output_token_digest
from minillm_l4.engine.model_loading import (
    checkpoint_quantization_config,
    configure_kernel_strategy,
    model_load_kwargs,
    precision_execution_metadata,
)
from ..core.synthetic import (
    exact_token_ids,
    read_synthetic_samples,
    samples_for_length,
)


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507-FP8"
# This is the immutable snapshot available in the local Hugging Face cache.
MODEL_REVISION = "8591804019c8b22094c3b5b4454e0edc05dffc98"
BASELINE_PRECISION = "fp8"
BASELINE_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("short", 128, 32),
    ("medium", 512, 64),
    ("long", 2048, 128),
)


@dataclass(frozen=True)
class HfModelBundle:
    """Loaded model state and identity recorded by the Phase 1 CLI."""

    model: Any
    tokenizer: Any
    model_config: Any
    model_id: str
    revision: str
    source: str
    precision_description: str
    precision_execution: dict[str, Any]
    kernel_strategy: dict[str, Any]
    load_time_ms: float
    tokenizer_load_time_ms: float
    model_load_time_ms: float

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "source": self.source,
            "precision": BASELINE_PRECISION,
            "precision_description": self.precision_description,
            "precision_execution": self.precision_execution,
            "kernel_strategy": self.kernel_strategy,
            "tokenizer_load_time_ms": self.tokenizer_load_time_ms,
            "model_load_time_ms": self.model_load_time_ms,
            "total_load_time_ms": self.load_time_ms,
            "model_config": {
                "model_type": getattr(self.model_config, "model_type", None),
                "architectures": list(
                    getattr(self.model_config, "architectures", None) or []
                ),
                "vocab_size": getattr(self.model_config, "vocab_size", None),
                "hidden_size": getattr(self.model_config, "hidden_size", None),
                "num_hidden_layers": getattr(
                    self.model_config, "num_hidden_layers", None
                ),
                "num_attention_heads": getattr(
                    self.model_config, "num_attention_heads", None
                ),
                "num_key_value_heads": getattr(
                    self.model_config, "num_key_value_heads", None
                ),
                "torch_dtype": str(
                    getattr(self.model_config, "torch_dtype", None)
                ),
                "quantization_config": checkpoint_quantization_config(
                    self.model_config
                ),
            },
        }


def _cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for variable in (
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    unique: list[Path] = []
    for root in roots:
        resolved = root.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def _snapshot_is_complete(snapshot: Path) -> bool:
    if not (snapshot / "config.json").is_file():
        return False
    has_tokenizer = any(
        (snapshot / filename).is_file()
        for filename in ("tokenizer.json", "tokenizer.model", "vocab.json")
    )
    if not has_tokenizer:
        return False
    return any(
        path.is_file() and path.suffix in {".safetensors", ".bin"}
        for path in snapshot.iterdir()
    )


def resolve_model_source(
    model_id: str,
    revision: str,
    *,
    model_path: Path | None = None,
    local_files_only: bool = True,
) -> str:
    """Resolve an exact local snapshot, or return the Hub ID when allowed."""

    if not model_id.strip():
        raise ValueError("model_id must not be empty")
    if not revision.strip():
        raise ValueError("revision must be pinned and non-empty")

    explicit = model_path
    if explicit is None and Path(model_id).expanduser().exists():
        explicit = Path(model_id).expanduser()
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Model path is not a directory: {resolved}")
        if not _snapshot_is_complete(resolved):
            raise FileNotFoundError(
                "Model path is missing config, tokenizer, or weight files: "
                f"{resolved}"
            )
        return str(resolved)

    cache_name = f"models--{model_id.replace('/', '--')}"
    for root in _cache_roots():
        snapshot = root / cache_name / "snapshots" / revision
        if snapshot.is_dir() and _snapshot_is_complete(snapshot):
            return str(snapshot)

    if local_files_only:
        searched = ", ".join(
            str(root / cache_name / "snapshots" / revision)
            for root in _cache_roots()
        )
        raise FileNotFoundError(
            f"Pinned local model snapshot was not found for {model_id}@{revision}. "
            f"Searched: {searched}. Pass --allow-download to permit Hub access."
        )
    return model_id


def _cuda_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError(
            "Phase 1 is pinned to the NVIDIA L4 CUDA path; device must be cuda or cuda:N"
        )
    if selected.index is None:
        return torch.device("cuda:0")
    return selected


def load_qwen_fp8(
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    model_path: Path | None = None,
    device: str = "cuda:0",
    local_files_only: bool = True,
    fp8_fallback_dtype: str = "auto",
    fp8_kernel_path: str = "auto",
) -> HfModelBundle:
    """Load the pinned Qwen checkpoint using Transformers' native FP8 path."""

    selected_device = _cuda_device(device)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Phase 1 requires torch.cuda.is_available() == True for the NVIDIA L4. "
            "The checkpoint is cached locally, but this Python environment cannot "
            "initialize CUDA."
        )

    source = resolve_model_source(
        model_id,
        revision,
        model_path=model_path,
        local_files_only=local_files_only,
    )
    kernel_strategy = configure_kernel_strategy(
        BASELINE_PRECISION,
        fp8_kernel_path=fp8_kernel_path,
    )

    # Import Transformers only after the explicit kernel strategy has been set.
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    total_start = time.perf_counter()
    model_config = AutoConfig.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    load_kwargs, precision_description = model_load_kwargs(
        BASELINE_PRECISION,
        model_config=model_config,
        fp8_fallback_dtype=fp8_fallback_dtype,
    )
    precision_execution = precision_execution_metadata(
        BASELINE_PRECISION,
        model_config=model_config,
        fp8_fallback_dtype=fp8_fallback_dtype,
        fp8_kernel_path=fp8_kernel_path,
    )

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        local_files_only=local_files_only,
    )
    tokenizer_load_time_ms = (time.perf_counter() - tokenizer_start) * 1000.0

    # model_load_kwargs defaults to GPU 0 for the project. Keep the explicit
    # device argument authoritative for reproducible single-GPU runs.
    load_kwargs["device_map"] = {"": selected_device.index or 0}
    model_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=revision,
        config=model_config,
        local_files_only=local_files_only,
        **load_kwargs,
    )
    model.eval()
    model_load_time_ms = (time.perf_counter() - model_start) * 1000.0

    return HfModelBundle(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        model_id=model_id,
        revision=revision,
        source=source,
        precision_description=precision_description,
        precision_execution=precision_execution,
        kernel_strategy=kernel_strategy,
        load_time_ms=(time.perf_counter() - total_start) * 1000.0,
        tokenizer_load_time_ms=tokenizer_load_time_ms,
        model_load_time_ms=model_load_time_ms,
    )


class _TokenEventStreamer:
    """Forward Hugging Face streamer callbacks into request event records."""

    def __init__(
        self,
        recorders: Sequence[RequestEventRecorder],
        *,
        prompt_tokens: int,
    ) -> None:
        if not recorders:
            raise ValueError("At least one request recorder is required")
        if prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        self.recorders = tuple(recorders)
        self.prompt_tokens = prompt_tokens
        self._received_prompt = False
        self._generated_index = 0

    @property
    def generated_tokens(self) -> int:
        return self._generated_index

    def put(self, value: Any) -> None:
        tensor = torch.as_tensor(value)
        if not self._received_prompt:
            self._received_prompt = True
            return

        cpu_tokens = tensor.detach().to(device="cpu")
        if cpu_tokens.ndim == 0:
            cpu_tokens = cpu_tokens.reshape(1, 1)
        elif cpu_tokens.ndim == 1:
            cpu_tokens = cpu_tokens.reshape(len(self.recorders), -1)
        elif cpu_tokens.ndim != 2:
            raise ValueError(
                "Hugging Face streamer emitted unsupported token shape: "
                f"{tuple(cpu_tokens.shape)}"
            )
        if cpu_tokens.shape[0] != len(self.recorders):
            raise ValueError(
                "Hugging Face streamer batch size changed: "
                f"expected {len(self.recorders)}, received {cpu_tokens.shape[0]}"
            )
        if cpu_tokens.shape[1] < 1:
            raise ValueError("Hugging Face streamer emitted an empty token row")

        # GenerationMixin emits one token per row per callback for the greedy
        # path. Taking the final column also tolerates a custom streamer test
        # that sends a short sequence in one callback.
        token_ids = cpu_tokens[:, -1].tolist()
        timestamp_ns = max(recorder.now_ns() for recorder in self.recorders)
        for recorder, token_id in zip(self.recorders, token_ids, strict=True):
            if self._generated_index == 0:
                recorder.record("prefill_end", timestamp_ns=timestamp_ns)
            recorder.mark_token_ready(
                self._generated_index,
                token_id=int(token_id),
                timestamp_ns=timestamp_ns,
            )
            recorder.mark_token_sent(
                self._generated_index,
                token_id=int(token_id),
                timestamp_ns=timestamp_ns,
            )
        self._generated_index += 1

    def end(self) -> None:
        return None


class HuggingFaceGreedyBatchRunner:
    """Static-batch ``model.generate`` runner used by the Phase 1 baseline."""

    def __init__(
        self,
        model: Any,
        *,
        device: str | torch.device | None = None,
        logits_mode: str = "last",
    ) -> None:
        self.model = model
        self.device = (
            torch.device(device) if device is not None else _first_model_device(model)
        )
        self.logits_mode = logits_mode

    def __call__(
        self,
        requests: Sequence[RequestSpec],
        recorders: Sequence[RequestEventRecorder],
    ) -> tuple[RequestOutcome, ...]:
        if not requests:
            raise ValueError("A generation batch must contain at least one request")
        if len(requests) != len(recorders):
            raise ValueError("requests and recorders must have equal lengths")
        prompt_lengths = {request.prompt_tokens for request in requests}
        output_lengths = {request.max_new_tokens for request in requests}
        if len(prompt_lengths) != 1:
            raise ValueError(
                "Phase 1 static batching requires equal prompt lengths; "
                f"received {sorted(prompt_lengths)}"
            )
        if len(output_lengths) != 1:
            raise ValueError(
                "Phase 1 static batching requires equal output lengths; "
                f"received {sorted(output_lengths)}"
            )
        prompt_tokens = next(iter(prompt_lengths))
        output_tokens = next(iter(output_lengths))

        for recorder in recorders:
            recorder.record(
                "prefill_start",
                metadata={
                    "runner": "huggingface_model_generate",
                    "batch_size": len(requests),
                },
            )

        input_ids = torch.tensor(
            [request.prompt_token_ids for request in requests],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        streamer = _TokenEventStreamer(
            recorders,
            prompt_tokens=prompt_tokens,
        )
        with torch.inference_mode():
            generation = transformers_greedy_generate(
                self.model,
                {
                    "input_ids": input_ids.to(self.device),
                    "attention_mask": attention_mask.to(self.device),
                },
                output_tokens=output_tokens,
                logits_mode=self.logits_mode,
                streamer=streamer,
            )

        token_rows = generation.token_ids.detach().to(device="cpu").tolist()
        if len(token_rows) != len(requests):
            raise RuntimeError(
                "model.generate returned an unexpected batch size: "
                f"expected {len(requests)}, received {len(token_rows)}"
            )
        if any(len(row) != output_tokens for row in token_rows):
            raise RuntimeError(
                "model.generate returned an unexpected continuation length"
            )

        outcomes: list[RequestOutcome] = []
        for request, recorder, token_row in zip(
            requests,
            recorders,
            token_rows,
            strict=True,
        ):
            recorder.record("completion")
            row_tensor = torch.tensor([token_row], dtype=torch.long)
            model_config = getattr(self.model, "config", None)
            outcomes.append(
                RequestOutcome(
                    status="completed",
                    generated_token_ids=tuple(int(token) for token in token_row),
                    metadata={
                        "runner": "huggingface_model_generate",
                        "model_id": getattr(model_config, "_name_or_path", None),
                        "batch_size": len(requests),
                        "output_token_sha256": output_token_digest(row_tensor),
                    },
                )
            )
        return tuple(outcomes)


def _first_model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as error:
        raise ValueError(
            "device must be supplied when the model has no discoverable parameters"
        ) from error


def build_hf_workload(
    tokenizer: Any,
    dataset_path: Path,
    *,
    bucket_name: str,
    prompt_tokens: int,
    output_tokens: int,
    count: int,
    seed: int,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    device: str = "cuda:0",
) -> WorkloadSpec:
    """Build a deterministic exact-token workload with the pinned tokenizer."""

    if count <= 0:
        raise ValueError("count must be positive")
    samples = samples_for_length(
        read_synthetic_samples(dataset_path),
        prompt_tokens,
    )
    if len(samples) < count:
        raise ValueError(
            f"Dataset has {len(samples)} samples for {prompt_tokens} prompt tokens; "
            f"requested {count}"
        )

    requests: list[RequestSpec] = []
    for index, sample in enumerate(samples[:count]):
        token_ids = tuple(exact_token_ids(tokenizer, sample))
        if len(token_ids) != prompt_tokens:
            raise RuntimeError(
                f"Tokenizer produced {len(token_ids)} tokens for {sample.sample_id}; "
                f"expected {prompt_tokens}"
            )
        requests.append(
            RequestSpec(
                request_id=f"baseline-{bucket_name}-{index:03d}",
                prompt_token_ids=token_ids,
                max_new_tokens=output_tokens,
                category=bucket_name,
                metadata={
                    "source_sample_id": sample.sample_id,
                    "source_category": sample.category,
                    "tokenizer": model_id,
                    "target_prompt_tokens": prompt_tokens,
                },
            )
        )

    return WorkloadSpec(
        name=f"baseline_{bucket_name}",
        seed=seed,
        requests=tuple(requests),
        model_id=model_id,
        model_revision=revision,
        dtype=BASELINE_PRECISION,
        device=device,
        arrival_pattern="closed_loop",
        metadata={
            "phase": 1,
            "tokenization": "Hugging Face tokenizer exact-token materialization",
            "dataset": str(dataset_path),
            "bucket": bucket_name,
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "request_count": count,
        },
    )


def build_reference_corpus(result: BenchmarkResult) -> dict[str, Any]:
    """Create a small token-ID corpus from the first measured repetition."""

    if not result.runs:
        raise ValueError("A measured result is required")
    first_run = result.runs[0]
    request_specs = {request.request_id: request for request in result.workload.requests}
    records: dict[str, Any] = {}
    for request_record in first_run["requests"]:
        request_id = str(request_record["request_id"])
        request = request_specs[request_id]
        outcome = request_record["outcome"]
        records[request_id] = {
            "prompt_sha256": request.prompt_sha256,
            "max_new_tokens": request.max_new_tokens,
            "generated_token_ids": list(outcome["generated_token_ids"]),
            "output_token_sha256": output_token_digest(
                torch.tensor([outcome["generated_token_ids"]], dtype=torch.long)
            ),
        }
    return {
        "schema_version": 1,
        "model_id": result.workload.model_id,
        "model_revision": result.workload.model_revision,
        "workload": result.workload.name,
        "requests": records,
    }


def verify_or_write_reference(
    result: BenchmarkResult,
    path: Path,
) -> dict[str, Any]:
    """Check a saved token corpus, or create it on the first successful run."""

    measured_outputs = build_reference_corpus(result)
    stable = outputs_stable_across_repetitions(result)
    report: dict[str, Any] = {
        "status": "pass" if stable else "fail",
        "outputs_stable_across_repetitions": stable,
        "reference_path": str(path),
        "reference_checked": path.exists(),
        "reference_created": False,
        "reference_match": None,
    }
    if path.exists():
        try:
            expected = json.loads(path.read_text(encoding="utf-8"))
            expected_requests = expected.get("requests", {})
            measured_requests = measured_outputs.get("requests", {})
            identity_matches = all(
                expected.get(field) == measured_outputs.get(field)
                for field in (
                    "schema_version",
                    "model_id",
                    "model_revision",
                    "workload",
                )
            )
            report["reference_match"] = (
                identity_matches
                and all(
                    expected_requests.get(request_id) == record
                    for request_id, record in measured_requests.items()
                )
            )
            report["reference_scope"] = (
                "exact"
                if set(expected_requests) == set(measured_requests)
                else "measured_subset"
            )
            if not report["reference_match"]:
                report["status"] = "fail"
        except (OSError, json.JSONDecodeError) as error:
            report["status"] = "fail"
            report["reference_error"] = f"{type(error).__name__}: {error}"
    elif stable:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(measured_outputs, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report["reference_created"] = True
    return report


def outputs_stable_across_repetitions(result: BenchmarkResult) -> bool:
    if not result.runs:
        return False
    expected: dict[str, tuple[int, ...]] | None = None
    for run in result.runs:
        current: dict[str, tuple[int, ...]] = {}
        for request in run["requests"]:
            outcome = request["outcome"]
            if outcome["status"] != "completed":
                return False
            current[str(request["request_id"])] = tuple(
                int(token) for token in outcome["generated_token_ids"]
            )
        if expected is None:
            expected = current
        elif current != expected:
            return False
    return True


def write_baseline_result(
    result: BenchmarkResult,
    path: Path,
    *,
    model_metadata: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> None:
    """Write the common harness result plus Phase 1 model/correctness fields."""

    payload = result.to_dict()
    payload["baseline"] = {
        "model": dict(model_metadata),
        "correctness": dict(correctness),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
