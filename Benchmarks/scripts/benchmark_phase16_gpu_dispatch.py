#!/usr/bin/env python
"""Phase 16 benchmark: compare CPU vs AppleGpu request on GPU-enabled kernels.

This benchmark is intentionally small and focused; it reports warm-call minimum
latency for each kernel plus numerical deltas.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy.fft

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
from benchmark_guard import REVIEW_SPEEDUP_THRESHOLD, evaluate_speedup


@dataclass
class BenchRow:
    kernel: str
    cpu_min_ms: float
    gpu_min_ms: float
    cpu_over_gpu: float
    max_abs_diff: float
    mean_abs_diff: float


def _time_min(fn, warm: int = 2, runs: int = 6):
    for _ in range(warm):
        fn()
    vals = []
    out = None
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn()
        vals.append((time.perf_counter() - t0) * 1e3)
    return out, float(min(vals))


def _with_env(temp: dict[str, str]):
    old = {k: os.environ.get(k) for k in temp}
    os.environ.update(temp)
    return old


def _restore_env(old: dict[str, str | None]):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _bench_pair(kernel: str, fn, env_gpu: dict[str, str]):
    old = _with_env({"IRON_LIBROSA_RUST_DEVICE": "cpu"})
    try:
        out_cpu, cpu_min = _time_min(fn)
    finally:
        _restore_env(old)

    old = _with_env({"IRON_LIBROSA_RUST_DEVICE": "apple-gpu", **env_gpu})
    try:
        out_gpu, gpu_min = _time_min(fn)
    finally:
        _restore_env(old)

    d = np.abs(out_cpu - out_gpu)
    return BenchRow(
        kernel=kernel,
        cpu_min_ms=cpu_min,
        gpu_min_ms=gpu_min,
        cpu_over_gpu=(cpu_min / gpu_min) if gpu_min > 0 else 0.0,
        max_abs_diff=float(d.max()),
        mean_abs_diff=float(d.mean()),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not RUST_AVAILABLE:
        print("Rust extension unavailable; skipping benchmark")
        return 0

    rng = np.random.default_rng(1616)
    rows: list[BenchRow] = []

    # mel_project_f32
    s = np.ascontiguousarray(np.abs(rng.standard_normal((2049, 1200), dtype=np.float32)))
    mb = np.ascontiguousarray(np.abs(rng.standard_normal((256, 2049), dtype=np.float32)))
    rows.append(
        _bench_pair(
            "mel_project_f32",
            lambda: _rust_ext.mel_project_f32(s, mb),
            {"IRON_LIBROSA_GPU_WORK_THRESHOLD": "1"},
        )
    )

    # chroma_project_f32
    cs = np.ascontiguousarray(np.abs(rng.standard_normal((2049, 1200), dtype=np.float32)))
    cb = np.ascontiguousarray(np.abs(rng.standard_normal((256, 2049), dtype=np.float32)))
    rows.append(
        _bench_pair(
            "chroma_project_f32",
            lambda: _rust_ext.chroma_project_f32(cs, cb),
            {"IRON_LIBROSA_GPU_WORK_THRESHOLD": "1"},
        )
    )

    # cqt_project_f32
    d = (
        rng.standard_normal((2, 2049, 800)).astype(np.float32)
        + 1j * rng.standard_normal((2, 2049, 800)).astype(np.float32)
    )
    basis = (
        rng.standard_normal((256, 2049)).astype(np.float32)
        + 1j * rng.standard_normal((256, 2049)).astype(np.float32)
    )
    d = np.ascontiguousarray(d)
    basis = np.ascontiguousarray(basis)
    rows.append(
        _bench_pair(
            "cqt_project_f32",
            lambda: _rust_ext.cqt_project_f32(d, basis),
            {"IRON_LIBROSA_GPU_WORK_THRESHOLD": "1"},
        )
    )

    # piptrack_from_spectrogram_f32
    ps = np.ascontiguousarray(np.abs(rng.standard_normal((2049, 800), dtype=np.float32)))
    pshift = np.ascontiguousarray(rng.standard_normal((2049, 800), dtype=np.float32) * 0.01)
    pdskew = np.ascontiguousarray(rng.standard_normal((2049, 800), dtype=np.float32) * 0.01)
    pref = np.ascontiguousarray(np.percentile(ps, 75.0, axis=0).astype(np.float32))

    def _piptrack_call():
        pitch, mag = _rust_ext.piptrack_from_spectrogram_f32(
            ps, pshift, pdskew, pref, 0, ps.shape[0], 22050.0 / 4096.0
        )
        return pitch + mag

    rows.append(
        _bench_pair(
            "piptrack_from_spectrogram_f32",
            _piptrack_call,
            {"IRON_LIBROSA_PIPTRACK_GPU_WORK_THRESHOLD": "1"},
        )
    )

    # tempogram_ac_f32
    tw = np.ascontiguousarray(rng.random((384, 1200), dtype=np.float32))
    n_pad = int(scipy.fft.next_fast_len(2 * 384 - 1, real=True))
    rows.append(
        _bench_pair(
            "tempogram_ac_f32",
            lambda: _rust_ext.tempogram_ac_f32(tw, n_pad),
            {"IRON_LIBROSA_TEMPOGRAM_GPU_WORK_THRESHOLD": "1"},
        )
    )

    print("kernel,cpu_min_ms,gpu_min_ms,cpu_over_gpu,max_abs_diff,mean_abs_diff")
    for r in rows:
        print(
            f"{r.kernel},{r.cpu_min_ms:.3f},{r.gpu_min_ms:.3f},"
            f"{r.cpu_over_gpu:.3f},{r.max_abs_diff:.8f},{r.mean_abs_diff:.8f}"
        )

    auto_review_cases = []
    for r in rows:
        review = evaluate_speedup(r.cpu_over_gpu, REVIEW_SPEEDUP_THRESHOLD)
        if review["review_required"]:
            auto_review_cases.append(
                {
                    "kernel": r.kernel,
                    "speedup": review["speedup"],
                    "review_threshold": review["review_threshold"],
                    "reason": f"speedup {r.cpu_over_gpu:.3f}x below review threshold",
                }
            )

    payload = {
        "meta": {
            "benchmark": "phase16_gpu_dispatch",
            "backend_info": _rust_ext.rust_backend_info(),
            "review_speedup_threshold": REVIEW_SPEEDUP_THRESHOLD,
        },
        "auto_review_cases": auto_review_cases,
        "rows": [asdict(r) for r in rows],
    }
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json,{args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

