#!/usr/bin/env python
"""Calibrate a recommended value for IRON_LIBROSA_GPU_WORK_THRESHOLD.

This script compares forced CPU vs forced apple-gpu request timings for
`_rust_ext.mel_project_f32` across a sweep of matrix sizes.

It prints per-case timings and recommends a threshold in multiply-add operations
(`n_mels * n_bins * n_frames`) above which GPU dispatch should be enabled.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


@dataclass
class BenchCase:
    n_fft: int
    n_frames: int
    n_mels: int

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    @property
    def work(self) -> int:
        return self.n_bins * self.n_frames * self.n_mels


@dataclass
class BenchResult:
    case: BenchCase
    cpu_min_ms: float
    gpu_min_ms: float
    speedup_cpu_over_gpu: float


def _timed_min_ms(fn, n_warm: int, n_runs: int) -> float:
    for _ in range(n_warm):
        fn()
    vals = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        vals.append((time.perf_counter() - t0) * 1e3)
    return float(min(vals))


def _set_env_temporarily(pairs: dict[str, str]):
    old = {k: os.environ.get(k) for k in pairs}
    os.environ.update(pairs)
    return old


def _restore_env(old: dict[str, Optional[str]]):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _bench_one(_rust_ext, case: BenchCase, n_warm: int, n_runs: int, force_gpu_path: bool) -> BenchResult:
    rng = np.random.default_rng(9000 + case.n_fft + case.n_frames + case.n_mels)
    s = np.ascontiguousarray(np.abs(rng.standard_normal((case.n_bins, case.n_frames), dtype=np.float32)))
    mel = np.ascontiguousarray(np.abs(rng.standard_normal((case.n_mels, case.n_bins), dtype=np.float32)))

    old = _set_env_temporarily({"IRON_LIBROSA_RUST_DEVICE": "cpu"})
    try:
        cpu_min = _timed_min_ms(lambda: _rust_ext.mel_project_f32(s, mel), n_warm=n_warm, n_runs=n_runs)
    finally:
        _restore_env(old)

    gpu_env = {"IRON_LIBROSA_RUST_DEVICE": "apple-gpu"}
    if force_gpu_path:
        gpu_env["IRON_LIBROSA_GPU_WORK_THRESHOLD"] = "1"

    old = _set_env_temporarily(gpu_env)
    try:
        gpu_min = _timed_min_ms(lambda: _rust_ext.mel_project_f32(s, mel), n_warm=n_warm, n_runs=n_runs)
    finally:
        _restore_env(old)

    speed = cpu_min / gpu_min if gpu_min > 0 else 0.0
    return BenchResult(
        case=case,
        cpu_min_ms=cpu_min,
        gpu_min_ms=gpu_min,
        speedup_cpu_over_gpu=speed,
    )


def recommend_threshold(results: Iterable[BenchResult], safety_factor: float = 1.10) -> Optional[int]:
    rows = sorted(results, key=lambda r: r.case.work)
    if not rows:
        return None

    gpu_wins = [r for r in rows if r.gpu_min_ms < r.cpu_min_ms]
    if not gpu_wins:
        return None

    losing_work = [r.case.work for r in rows if r.gpu_min_ms >= r.cpu_min_ms]
    if not losing_work:
        return max(1, int(rows[0].case.work))

    cutoff = max(losing_work)
    return int(cutoff * safety_factor)


def _default_cases() -> list[BenchCase]:
    return [
        BenchCase(1024, 300, 64),
        BenchCase(2048, 800, 128),
        BenchCase(4096, 1200, 256),
        BenchCase(4096, 1600, 256),
        BenchCase(8192, 1200, 256),
    ]


def main() -> int:
    from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext

    parser = argparse.ArgumentParser()
    parser.add_argument("--warm", type=int, default=4)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--safety-factor", type=float, default=1.10)
    parser.add_argument("--allow-threshold-fallback", action="store_true", help="Do not force GPU path during calibration")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not RUST_AVAILABLE or not hasattr(_rust_ext, "mel_project_f32"):
        print("ERROR: Rust mel_project_f32 is not available in this environment")
        return 2

    info = _rust_ext.rust_backend_info()
    print("backend_info:", info)

    cases = _default_cases()
    results: list[BenchResult] = []

    print("case,n_fft,bins,frames,n_mels,work,cpu_min_ms,gpu_min_ms,cpu_over_gpu")
    for idx, case in enumerate(cases, start=1):
        row = _bench_one(
            _rust_ext,
            case,
            n_warm=args.warm,
            n_runs=args.runs,
            force_gpu_path=not args.allow_threshold_fallback,
        )
        results.append(row)
        print(
            f"{idx},{case.n_fft},{case.n_bins},{case.n_frames},{case.n_mels},{case.work},"
            f"{row.cpu_min_ms:.3f},{row.gpu_min_ms:.3f},{row.speedup_cpu_over_gpu:.3f}"
        )

    recommended = recommend_threshold(results, safety_factor=args.safety_factor)
    ratios = [r.speedup_cpu_over_gpu for r in results]
    median_ratio = statistics.median(ratios) if ratios else 0.0

    print()
    if recommended is None:
        print("recommended_threshold: none (GPU did not beat CPU on sampled cases)")
    else:
        print(f"recommended_threshold: {recommended}")
        print("export IRON_LIBROSA_GPU_WORK_THRESHOLD=" + str(recommended))
    print(f"median_cpu_over_gpu: {median_ratio:.3f}")

    if args.json_out is not None:
        payload = {
            "recommended_threshold": recommended,
            "median_cpu_over_gpu": median_ratio,
            "force_gpu_path": not args.allow_threshold_fallback,
            "results": [
                {
                    "case": asdict(r.case),
                    "n_bins": r.case.n_bins,
                    "work": r.case.work,
                    "cpu_min_ms": r.cpu_min_ms,
                    "gpu_min_ms": r.gpu_min_ms,
                    "cpu_over_gpu": r.speedup_cpu_over_gpu,
                }
                for r in results
            ],
        }
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote_json: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


