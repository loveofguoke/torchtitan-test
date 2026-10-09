"""Optional low-rate device telemetry for profiler-off benchmark evidence."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .benchmark_metrics import metric_statistics


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")


def _number(value: str) -> float | None:
    match = _NUMBER.search(value)
    return float(match.group()) if match else None


def parse_nvidia_csv(text: str) -> list[dict[str, Any]]:
    """Parse the fixed ``nvidia-smi --query-gpu`` column contract."""

    records = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) != 7:
            continue
        values = [value.strip() for value in row]
        metrics = {
            "power_w": _number(values[2]),
            "temperature_c": _number(values[3]),
            "compute_clock_mhz": _number(values[4]),
            "utilization_percent": _number(values[5]),
        }
        records.append(
            {
                "device_id": values[1],
                "device_timestamp": values[0],
                "metrics": {
                    key: value for key, value in metrics.items() if value is not None
                },
                "throttle_reasons": values[6],
            }
        )
    return records


def parse_npu_common(text: str, *, device_id: str) -> dict[str, Any]:
    """Parse version-tolerant key/value fields from ``npu-smi`` output."""

    aliases = {
        "power_w": ("real-timepower", "power usage", "power(w)"),
        "temperature_c": ("temperature", "npu temperature"),
        "compute_clock_mhz": ("aicore curfreq", "aicore frequency"),
        "utilization_percent": ("aicore usage", "ai core usage"),
        "hbm_usage_percent": ("hbm usage rate", "memory usage"),
    }
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized = re.sub(r"\s+", " ", key.strip().lower())
        parsed = _number(value)
        if parsed is None:
            continue
        for metric, markers in aliases.items():
            if any(marker in normalized for marker in markers):
                metrics.setdefault(metric, parsed)
    return {"device_id": device_id, "metrics": metrics}


def summarize_telemetry(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "ok"]
    by_device: dict[str, list[dict[str, Any]]] = {}
    for record in successful:
        by_device.setdefault(str(record["device_id"]), []).append(record)
    devices: dict[str, Any] = {}
    for device_id, samples in sorted(by_device.items()):
        names = sorted(
            {
                name
                for sample in samples
                for name in sample.get("metrics", {})
            }
        )
        devices[device_id] = {
            "sample_count": len(samples),
            "metrics": {
                name: metric_statistics(
                    sample["metrics"][name]
                    for sample in samples
                    if name in sample.get("metrics", {})
                )
                for name in names
            },
            "active_throttle_reasons": sorted(
                {
                    str(sample["throttle_reasons"])
                    for sample in samples
                    if sample.get("throttle_reasons")
                    not in {None, "", "0x0000000000000000", "Not Active"}
                }
            ),
        }
    return {
        "status": "observed" if successful else "not_available",
        "sample_count": len(records),
        "successful_sample_count": len(successful),
        "failed_sample_count": len(records) - len(successful),
        "devices": devices,
        "scope": (
            "whole training process including startup, compilation, warmup, and "
            "measured steps; not a steady-window energy measurement"
        ),
    }


def align_steady_energy(
    records: Sequence[dict[str, Any]],
    metric_records: Sequence[dict[str, Any]],
    steady_steps: Sequence[int],
    *,
    scheduled_tokens_per_step: int | None,
    effective_tokens_per_step: int | None = None,
) -> dict[str, Any]:
    """Integrate device power inside the measured optimizer-step window."""

    selected = [
        row
        for row in metric_records
        if int(row.get("step", -1)) in set(steady_steps)
        and row.get("step_start_monotonic_ns") is not None
        and row.get("step_end_monotonic_ns") is not None
    ]
    if not selected:
        return {"status": "not_available", "reason": "step timestamps unavailable"}
    start_ns = min(int(row["step_start_monotonic_ns"]) for row in selected)
    end_ns = max(int(row["step_end_monotonic_ns"]) for row in selected)
    by_device: dict[str, list[tuple[int, float]]] = {}
    for row in records:
        power = row.get("metrics", {}).get("power_w")
        timestamp = row.get("monotonic_ns")
        if row.get("status") != "ok" or power is None or timestamp is None:
            continue
        if start_ns <= int(timestamp) <= end_ns:
            by_device.setdefault(str(row["device_id"]), []).append(
                (int(timestamp), float(power))
            )
    energy_by_device: dict[str, float] = {}
    for device_id, points in by_device.items():
        points.sort()
        if len(points) < 2:
            continue
        joules = sum(
            (right_t - left_t) / 1e9 * (left_w + right_w) / 2
            for (left_t, left_w), (right_t, right_w) in zip(points, points[1:])
        )
        energy_by_device[device_id] = joules
    if not energy_by_device:
        return {
            "status": "not_available",
            "reason": "fewer than two power samples per device in steady window",
        }
    num_steps = len({int(row["step"]) for row in selected})
    total_joules = sum(energy_by_device.values())
    scheduled_tokens = (
        scheduled_tokens_per_step * num_steps
        if scheduled_tokens_per_step is not None
        else None
    )
    effective_tokens = (
        effective_tokens_per_step * num_steps
        if effective_tokens_per_step is not None
        else None
    )
    return {
        "status": "observed",
        "scope": "steady optimizer-step window",
        "start_monotonic_ns": start_ns,
        "end_monotonic_ns": end_ns,
        "num_steps": num_steps,
        "device_energy_joules": energy_by_device,
        "total_device_energy_joules": total_joules,
        "scheduled_tokens_per_joule": (
            scheduled_tokens / total_joules if scheduled_tokens is not None else None
        ),
        "effective_tokens_per_joule": (
            effective_tokens / total_joules if effective_tokens is not None else None
        ),
        "effective_token_status": (
            "observed" if effective_tokens is not None else "not_available"
        ),
        "note": "Management-interface power is sampled, not board-level metrology.",
    }


class DeviceTelemetrySampler:
    """Poll vendor management CLIs without making training depend on them."""

    def __init__(
        self,
        *,
        device_type: str,
        device_ids: Sequence[str],
        interval_seconds: float,
        output_path: Path,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.device_type = device_type
        self.device_ids = tuple(str(device_id) for device_id in device_ids)
        self.interval_seconds = interval_seconds
        self.output_path = output_path
        self.runner = runner
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.records: list[dict[str, Any]] = []

    def start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2))
        summary = summarize_telemetry(self.records)
        summary_path = self.output_path.with_name("telemetry_summary.json")
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return summary

    def _run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return self.runner(
            list(command),
            text=True,
            capture_output=True,
            check=False,
            timeout=max(5.0, self.interval_seconds),
        )

    def _sample_once(self) -> list[dict[str, Any]]:
        if self.device_type == "cuda":
            executable = shutil.which("nvidia-smi")
            if executable is None:
                raise FileNotFoundError("nvidia-smi is not available")
            result = self._run(
                (
                    executable,
                    "--query-gpu=timestamp,index,power.draw,temperature.gpu,"
                    "clocks.current.sm,utilization.gpu,clocks_throttle_reasons.active",
                    "--format=csv,noheader,nounits",
                    "--id=" + ",".join(self.device_ids),
                )
            )
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or result.stdout.strip())
            return parse_nvidia_csv(result.stdout)

        executable = shutil.which("npu-smi")
        if executable is None:
            raise FileNotFoundError("npu-smi is not available")
        def sample_device(device_id: str) -> dict[str, Any]:
            common = self._run(
                (
                    executable,
                    "info",
                    "-t",
                    "common",
                    "-i",
                    device_id,
                    "-c",
                    "0",
                )
            )
            if common.returncode == 0:
                parsed = parse_npu_common(common.stdout, device_id=device_id)
                if parsed["metrics"]:
                    return parsed
            combined = ""
            errors = []
            for query in ("power", "temp", "usages", "memory"):
                result = self._run(
                    (
                        executable,
                        "info",
                        "-t",
                        query,
                        "-i",
                        device_id,
                        "-c",
                        "0",
                    )
                )
                if result.returncode == 0:
                    combined += "\n" + result.stdout
                else:
                    errors.append(result.stderr.strip() or result.stdout.strip())
            parsed = parse_npu_common(combined, device_id=device_id)
            if parsed["metrics"]:
                return parsed
            common_error = common.stderr.strip() or common.stdout.strip()
            raise RuntimeError("; ".join([common_error, *errors]))

        with ThreadPoolExecutor(max_workers=max(1, len(self.device_ids))) as pool:
            return list(pool.map(sample_device, self.device_ids))

    def _sample_loop(self) -> None:
        started = time.monotonic()
        with self.output_path.open("w", encoding="utf-8") as output:
            while not self._stop.is_set():
                timestamp = time.time()
                monotonic_ns = time.monotonic_ns()
                elapsed = time.monotonic() - started
                try:
                    samples = self._sample_once()
                    records = [
                        {
                            "timestamp_unix": timestamp,
                            "monotonic_ns": monotonic_ns,
                            "elapsed_seconds": elapsed,
                            "device_type": self.device_type,
                            "status": "ok",
                            **sample,
                        }
                        for sample in samples
                    ]
                except Exception as error:
                    records = [
                        {
                            "timestamp_unix": timestamp,
                            "monotonic_ns": monotonic_ns,
                            "elapsed_seconds": elapsed,
                            "device_type": self.device_type,
                            "status": "unavailable",
                            "error": repr(error),
                        }
                    ]
                for record in records:
                    self.records.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                if any(record["status"] == "unavailable" for record in records):
                    return
                self._stop.wait(self.interval_seconds)


__all__ = [
    "DeviceTelemetrySampler",
    "parse_npu_common",
    "parse_nvidia_csv",
    "summarize_telemetry",
    "align_steady_energy",
]
