#!/usr/bin/env python
"""Focused benchmark for CUDA fused melspectrogram dispatch.

Measures full ``librosa.feature.melspectrogram(y=...)`` latency for:
  * classic CPU pipeline
  * CUDA fused mono path
  * CUDA fused multichannel batch path
  * CUDA fused multichannel batch-fallback-to-single path

The script also validates output parity against the classic CPU path and records
which fused entry points were exercised.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

import librosa
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
from librosa.feature import _spectral_mfcc_mel as mel_mod


WORKLOADS = [
    {
        "name": "mono_medium",
        "channels": 1,
        "n_samples": 65536,
        "sr": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "seed": 20260414,
    },
    {
        "name": "batch4_medium",
        "channels": 4,
        "n_samples": 65536,
        "sr": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "seed": 20260415,
    },
    {
        "name": "batch4_large",
        "channels": 4,
        "n_samples": 262144,
        "sr": 22050,
        "n_fft": 2048,
        "hop_length": 512,
        "n_mels": 128,
        "seed": 20260416,
    },
]


def _make_audio(spec: dict[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(spec["seed"])
    shape = (spec["channels"], spec["n_samples"])
    y = rng.standard_normal(shape, dtype=np.float32)
    if spec["channels"] == 1:
        return y[0]
    return y


@contextmanager
def _patched_env(**updates: str):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextmanager
def _patched_fused_proxy(*, batch_mode: str | None):
    original = mel_mod._rust_ext
    if batch_mode is None:
        yield {"single_calls": None, "batch_calls": None, "mode": "unpatched"}
        return

    if _rust_ext is None:
        raise RuntimeError("Rust extension is not available")

    calls = {"single_calls": 0, "batch_calls": 0, "mode": batch_mode}

    def _single(*args, **kwargs):
        calls["single_calls"] += 1
        return _rust_ext.melspectrogram_fused_f32(*args, **kwargs)

    def _batch(*args, **kwargs):
        calls["batch_calls"] += 1
        if batch_mode == "raise":
            raise RuntimeError("intentional batch benchmark fallback")
        return _rust_ext.melspectrogram_fused_batch_f32(*args, **kwargs)

    mel_mod._rust_ext = types.SimpleNamespace(
        melspectrogram_fused_f32=_single,
        melspectrogram_fused_batch_f32=_batch,
        rust_backend_info=_rust_ext.rust_backend_info,
        cuda_diagnostics=getattr(_rust_ext, "cuda_diagnostics", None),
    )
    try:
        yield calls
    finally:
        mel_mod._rust_ext = original


def _compute_output(y: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    return librosa.feature.melspectrogram(
        y=y,
        sr=spec["sr"],
        n_fft=spec["n_fft"],
        hop_length=spec["hop_length"],
        n_mels=spec["n_mels"],
        center=True,
        power=2.0,
        dtype=np.float32,
        norm="slaney",
    )


def _time_case(fn, *, warmup: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(vals)),
        "median_ms": float(statistics.median(vals)),
        "min_ms": float(min(vals)),
        "max_ms": float(max(vals)),
        "stdev_ms": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "samples_ms": vals,
    }


def _backend_snapshot() -> dict[str, Any]:
    if not RUST_AVAILABLE or _rust_ext is None:
        return {"rust_available": False, "has_ext": False}
    out: dict[str, Any] = {
        "rust_available": True,
        "has_ext": True,
        "backend_info": _rust_ext.rust_backend_info(),
    }
    if hasattr(_rust_ext, "cuda_diagnostics"):
        out["cuda_diagnostics"] = _rust_ext.cuda_diagnostics()
    return out


def _run_mode(
    *,
    mode_name: str,
    y: np.ndarray,
    spec: dict[str, Any],
    warmup: int,
    repeats: int,
    env: dict[str, str],
    batch_mode: str | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    with _patched_env(**env), _patched_fused_proxy(batch_mode=batch_mode) as calls:
        output = _compute_output(y, spec)
        timings = _time_case(lambda: _compute_output(y, spec), warmup=warmup, repeats=repeats)
    details = {
        "mode": mode_name,
        "timings": timings,
        "dispatch": calls,
        "output_shape": list(output.shape),
    }
    return output, details


def run_benchmark(*, warmup: int, repeats: int) -> dict[str, Any]:
    if not RUST_AVAILABLE or _rust_ext is None:
        raise SystemExit("Rust extension unavailable; benchmark requires the built Python extension.")

    baseline_env = {
        "IRON_LIBROSA_CUDA_MEL_FUSED_EXPERIMENTAL": "0",
        "IRON_LIBROSA_RUST_DEVICE": "cpu",
        "IRON_LIBROSA_CUDA_RUNTIME_FORCE": "0",
    }
    fused_env = {
        "IRON_LIBROSA_CUDA_MEL_FUSED_EXPERIMENTAL": "force-on",
        "IRON_LIBROSA_RUST_DEVICE": "cuda-gpu",
        "IRON_LIBROSA_CUDA_RUNTIME_FORCE": "1",
    }

    results: dict[str, Any] = {
        "backend": _backend_snapshot(),
        "config": {"warmup": warmup, "repeats": repeats},
        "workloads": [],
    }

    for spec in WORKLOADS:
        y = _make_audio(spec)
        workload = {
            "spec": spec,
            "modes": {},
            "comparisons": {},
        }

        cpu_out, cpu_info = _run_mode(
            mode_name="cpu_classic",
            y=y,
            spec=spec,
            warmup=warmup,
            repeats=repeats,
            env=baseline_env,
            batch_mode=None,
        )
        workload["modes"]["cpu_classic"] = cpu_info

        if spec["channels"] == 1:
            gpu_out, gpu_info = _run_mode(
                mode_name="cuda_fused_single",
                y=y,
                spec=spec,
                warmup=warmup,
                repeats=repeats,
                env=fused_env,
                batch_mode="passthrough",
            )
            workload["modes"]["cuda_fused_single"] = gpu_info
            workload["comparisons"]["cuda_fused_single"] = {
                "allclose": bool(np.allclose(cpu_out, gpu_out, rtol=1e-4, atol=1e-4)),
                "max_abs_diff": float(np.max(np.abs(cpu_out - gpu_out))),
                "speedup_vs_cpu_median": cpu_info["timings"]["median_ms"] / gpu_info["timings"]["median_ms"],
                "speedup_vs_cpu_min": cpu_info["timings"]["min_ms"] / gpu_info["timings"]["min_ms"],
            }
        else:
            batch_out, batch_info = _run_mode(
                mode_name="cuda_fused_batch",
                y=y,
                spec=spec,
                warmup=warmup,
                repeats=repeats,
                env=fused_env,
                batch_mode="passthrough",
            )
            fallback_out, fallback_info = _run_mode(
                mode_name="cuda_fused_batch_fallback_single",
                y=y,
                spec=spec,
                warmup=warmup,
                repeats=repeats,
                env=fused_env,
                batch_mode="raise",
            )
            workload["modes"]["cuda_fused_batch"] = batch_info
            workload["modes"]["cuda_fused_batch_fallback_single"] = fallback_info
            workload["comparisons"]["cuda_fused_batch"] = {
                "allclose": bool(np.allclose(cpu_out, batch_out, rtol=1e-4, atol=1e-4)),
                "max_abs_diff": float(np.max(np.abs(cpu_out - batch_out))),
                "speedup_vs_cpu_median": cpu_info["timings"]["median_ms"] / batch_info["timings"]["median_ms"],
                "speedup_vs_cpu_min": cpu_info["timings"]["min_ms"] / batch_info["timings"]["min_ms"],
            }
            workload["comparisons"]["cuda_fused_batch_fallback_single"] = {
                "allclose": bool(np.allclose(cpu_out, fallback_out, rtol=1e-4, atol=1e-4)),
                "max_abs_diff": float(np.max(np.abs(cpu_out - fallback_out))),
                "speedup_vs_cpu_median": cpu_info["timings"]["median_ms"] / fallback_info["timings"]["median_ms"],
                "speedup_vs_cpu_min": cpu_info["timings"]["min_ms"] / fallback_info["timings"]["min_ms"],
                "speedup_batch_vs_fallback_median": batch_info["timings"]["median_ms"] / fallback_info["timings"]["median_ms"],
                "speedup_batch_vs_fallback_min": batch_info["timings"]["min_ms"] / fallback_info["timings"]["min_ms"],
            }

        results["workloads"].append(workload)

    return results


def _to_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# CUDA Fused Melspectrogram Benchmark",
        "",
        f"Warmup runs: {results['config']['warmup']}",
        f"Timed repeats: {results['config']['repeats']}",
        "",
    ]

    backend = results.get("backend", {})
    backend_info = backend.get("backend_info", {})
    if backend_info:
        lines += [
            "## Backend",
            "",
            f"- Requested: `{backend_info.get('requested')}`",
            f"- Resolved: `{backend_info.get('resolved')}`",
            f"- Reason: `{backend_info.get('resolved_reason')}`",
            "",
        ]

    for workload in results["workloads"]:
        spec = workload["spec"]
        lines += [
            f"## {spec['name']}",
            "",
            f"- channels: {spec['channels']}",
            f"- samples: {spec['n_samples']}",
            f"- n_fft: {spec['n_fft']}",
            f"- hop_length: {spec['hop_length']}",
            f"- n_mels: {spec['n_mels']}",
            "",
            "| Mode | Median ms | Min ms | Mean ms | Dispatch |",
            "|---|---:|---:|---:|---|",
        ]
        for mode_name, mode in workload["modes"].items():
            dispatch = mode.get("dispatch", {})
            dispatch_summary = "n/a"
            if dispatch.get("single_calls") is not None:
                dispatch_summary = (
                    f"single={dispatch.get('single_calls', 0)}, "
                    f"batch={dispatch.get('batch_calls', 0)}"
                )
            lines.append(
                f"| {mode_name} | {mode['timings']['median_ms']:.3f} | {mode['timings']['min_ms']:.3f} | "
                f"{mode['timings']['mean_ms']:.3f} | {dispatch_summary} |"
            )
        if workload["comparisons"]:
            lines += ["", "### Comparisons", ""]
            for label, cmp in workload["comparisons"].items():
                lines.append(
                    f"- `{label}`: allclose={cmp['allclose']}, "
                    f"max_abs_diff={cmp['max_abs_diff']:.6g}, "
                    f"speedup_vs_cpu_median={cmp['speedup_vs_cpu_median']:.3f}x, "
                    f"speedup_vs_cpu_min={cmp['speedup_vs_cpu_min']:.3f}x"
                    + (
                        f", batch_vs_fallback_median={cmp['speedup_batch_vs_fallback_median']:.3f}x, "
                        f"batch_vs_fallback_min={cmp['speedup_batch_vs_fallback_min']:.3f}x"
                        if 'speedup_batch_vs_fallback_median' in cmp
                        else ""
                    )
                )
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("Benchmarks/results/cuda_fused_melspectrogram_2026-04-14.json"),
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=Path("Benchmarks/results/cuda_fused_melspectrogram_2026-04-14.md"),
    )
    args = parser.parse_args()

    results = run_benchmark(warmup=args.warmup, repeats=args.repeats)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    args.md_out.write_text(_to_markdown(results), encoding="utf-8")


if __name__ == "__main__":
    main()

