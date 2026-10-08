from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _nvidia_smi_snapshot(device_index: int | None) -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}

    fields = (
        "driver_version",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    )
    command = [
        executable,
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.extend(["--id", str(device_index)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    payload: dict[str, Any] = {
        "available": result.returncode == 0,
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "fields": list(fields),
    }
    if result.returncode != 0 or not result.stdout.strip():
        return payload
    row = next(csv.reader([result.stdout.strip().splitlines()[0]]))
    values = [value.strip() for value in row]
    if len(values) != len(fields):
        payload["available"] = False
        payload["reason"] = "unexpected nvidia-smi output"
        payload["stdout"] = result.stdout.strip()
        return payload

    def number(value: str) -> float | None:
        if not value or value in {"N/A", "[Not Supported]"}:
            return None
        try:
            return float(value)
        except ValueError:
            return None

    payload.update(
        {
            "driver_version": values[0],
            "gpu_utilization_percent": number(values[1]),
            "memory_utilization_percent": number(values[2]),
            "memory_used_mib": number(values[3]),
            "memory_total_mib": number(values[4]),
            "temperature_c": number(values[5]),
            "power_w": number(values[6]),
        }
    )
    return payload


def collect_environment_metadata() -> dict[str, Any]:
    """Collect software and source identity for reproducible result files."""

    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": os.path.realpath(sys.executable),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit_sha": None,
        "git_commit_status": "unavailable",
        "git_worktree_dirty": None,
        "torch": {
            "installed": False,
            "version": None,
            "cuda_build_version": None,
            "cuda_available": False,
        },
        "packages": {},
    }
    try:
        import torch

        metadata["torch"] = {
            "installed": True,
            "version": torch.__version__,
            "cuda_build_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except ImportError:
        pass

    for package_name in ("transformers", "triton", "numpy", "pytest"):
        try:
            metadata["packages"][package_name] = importlib.metadata.version(
                package_name
            )
        except importlib.metadata.PackageNotFoundError:
            metadata["packages"][package_name] = None

    try:
        project_repo = Path(__file__).resolve().parents[2]
        git_result = subprocess.run(
            ["git", "-C", str(project_repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        git_result = None
    if git_result is not None and git_result.returncode == 0 and git_result.stdout.strip():
        metadata["git_commit_sha"] = git_result.stdout.strip()
        metadata["git_commit_status"] = "available"
        status_result = subprocess.run(
            ["git", "-C", str(project_repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if status_result.returncode == 0:
            metadata["git_worktree_dirty"] = bool(status_result.stdout.strip())
    else:
        metadata["git_commit_status"] = "git metadata unavailable"
    return metadata


_NVML: dict[str, Any] = {}


def _nvml_utilization_percent(properties: Any) -> int | None:
    """GPU busy percent from NVML (the counter nvidia-smi reports), or None.

    Reads libnvidia-ml directly through ctypes: microseconds per call and no
    subprocess, so periodic samples can run during a timed window.
    """

    import ctypes

    try:
        if "lib" not in _NVML:
            _NVML["lib"] = None
            lib = ctypes.CDLL("libnvidia-ml.so.1")
            if lib.nvmlInit_v2() != 0:
                return None
            _NVML["lib"] = lib
        lib = _NVML["lib"]
        if lib is None:
            return None
        bus_id = (f"{properties.pci_domain_id:08X}:{properties.pci_bus_id:02X}:"
                  f"{properties.pci_device_id:02X}.0")
        handle = _NVML.get(bus_id)
        if handle is None:
            handle = ctypes.c_void_p()
            if lib.nvmlDeviceGetHandleByPciBusId_v2(bus_id.encode(), ctypes.byref(handle)) != 0:
                return None
            _NVML[bus_id] = handle

        class Utilization(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        rates = Utilization()
        if lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(rates)) != 0:
            return None
        return int(rates.gpu)
    except (OSError, AttributeError):
        return None


def capture_gpu_snapshot(
    label: str,
    *,
    device: Any = None,
    include_system_telemetry: bool = True,
    started_ns: int | None = None,
) -> dict[str, Any]:
    """Capture best-effort CUDA allocator and coarse device telemetry."""

    captured_ns = time.perf_counter_ns()
    snapshot: dict[str, Any] = {
        "label": label,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "timestamp_ns": captured_ns,
        "elapsed_ms": (
            (captured_ns - started_ns) / 1_000_000.0
            if started_ns is not None
            else None
        ),
        "cuda_available": False,
        "device_index": None,
        "device_name": None,
        "compute_capability": None,
        "total_memory_bytes": None,
        "free_memory_bytes": None,
        "torch_memory_allocated_bytes": None,
        "torch_memory_reserved_bytes": None,
        "torch_peak_memory_allocated_bytes": None,
        "torch_peak_memory_reserved_bytes": None,
        "gpu_utilization_percent": None,
        "nvidia_smi": None,
    }

    try:
        import torch
    except ImportError:
        if include_system_telemetry:
            snapshot["nvidia_smi"] = _nvidia_smi_snapshot(None)
        return snapshot

    if torch.cuda.is_available():
        if device is None:
            device_index = torch.cuda.current_device()
        else:
            selected = torch.device(device)
            if selected.type != "cuda":
                device_index = torch.cuda.current_device()
            else:
                device_index = (
                    torch.cuda.current_device()
                    if selected.index is None
                    else selected.index
                )
        properties = torch.cuda.get_device_properties(device_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        snapshot.update(
            {
                "cuda_available": True,
                "device_index": device_index,
                "device_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "total_memory_bytes": int(total_bytes),
                "free_memory_bytes": int(free_bytes),
                "torch_memory_allocated_bytes": int(
                    torch.cuda.memory_allocated(device_index)
                ),
                "torch_memory_reserved_bytes": int(
                    torch.cuda.memory_reserved(device_index)
                ),
                "torch_peak_memory_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device_index)
                ),
                "torch_peak_memory_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device_index)
                ),
                "gpu_utilization_percent": _nvml_utilization_percent(properties),
            }
        )
    if include_system_telemetry:
        snapshot["nvidia_smi"] = _nvidia_smi_snapshot(snapshot["device_index"])
    return snapshot


class GpuSampler:
    """Collect start/end and optional periodic GPU/VRAM snapshots."""

    def __init__(
        self,
        *,
        device: Any = None,
        interval_seconds: float = 0.25,
        enabled: bool = True,
        include_system_telemetry: bool = True,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.device = device
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.include_system_telemetry = include_system_telemetry
        self.samples: list[dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns: int | None = None

    def reset_peak_memory(self) -> None:
        if not self.enabled:
            return
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            if self.device is not None and torch.device(self.device).type != "cuda":
                return
            torch.cuda.reset_peak_memory_stats(self.device)

    def snapshot(self, label: str, *, include_system_telemetry: bool | None = None) -> dict[str, Any]:
        sample = capture_gpu_snapshot(
            label,
            device=self.device,
            include_system_telemetry=(
                self.include_system_telemetry
                if include_system_telemetry is None
                else include_system_telemetry
            ),
            started_ns=self._started_ns,
        )
        self.samples.append(sample)
        return sample

    def start(self, *, started_ns: int | None = None) -> None:
        if not self.enabled:
            return
        if self._thread is not None:
            raise RuntimeError("GpuSampler has already been started")
        self._started_ns = (
            time.perf_counter_ns() if started_ns is None else int(started_ns)
        )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 5))
        return list(self.samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.snapshot("periodic")
            self._stop_event.wait(self.interval_seconds)
