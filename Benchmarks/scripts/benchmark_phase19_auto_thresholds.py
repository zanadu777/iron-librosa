#!/usr/bin/env python
"""Quick benchmark sweep for FFT Auto threshold tuning.

Measures STFT/iSTFT in five workloads across:
- cpu baseline
- auto mode with configurable work thresholds
- forced gpu mode
"""

from __future__ import annotations

import argparse
import json
import statistics
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

WORKLOADS = [
    ("short_512", 22050 * 1, 512, 128),
    ("short_1024", 22050 * 1, 1024, 256),
    ("medium_512", 22050 * 5, 512, 128),
    ("medium_1024", 22050 * 5, 1024, 256),
    ("long_1024", 22050 * 30, 1024, 256),
]

REPEATS = 4
WARMUP = 1
OUTER_RUNS = 5


@dataclass
class Case:
    name: str
    device: str
    threshold: int | None
    metal: str
    min_frames: int | None = None


def _bench_pair(rust_ext, n_samples: int, n_fft: int, hop: int) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    y = rng.standard_normal(n_samples).astype(np.float32)

    for _ in range(WARMUP):
        rust_ext.stft_complex(y, n_fft, hop, True, None)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        rust_ext.stft_complex(y, n_fft, hop, True, None)
    stft_ms = (time.perf_counter() - t0) * 1000.0 / REPEATS

    stft_m = rust_ext.stft_complex(y, n_fft, hop, True, None)
    for _ in range(WARMUP):
        rust_ext.istft_f32(stft_m, n_fft, hop, None, None)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        rust_ext.istft_f32(stft_m, n_fft, hop, None, None)
    istft_ms = (time.perf_counter() - t0) * 1000.0 / REPEATS

    return stft_ms, istft_ms


def _apply_case_env(case: Case) -> None:
    os.environ["IRON_LIBROSA_RUST_DEVICE"] = case.device
    if case.threshold is None:
        os.environ.pop("IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD", None)
    else:
        os.environ["IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD"] = str(case.threshold)
    if case.metal:
        os.environ["IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL"] = case.metal
    else:
        os.environ.pop("IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL", None)
    if case.min_frames is None:
        os.environ.pop("IRON_LIBROSA_FFT_GPU_MIN_FRAMES", None)
    else:
        os.environ["IRON_LIBROSA_FFT_GPU_MIN_FRAMES"] = str(case.min_frames)


def _run_case_in_subprocess(
    case: Case,
    repeats: int,
    warmup: int,
    outer_runs: int,
) -> dict[str, tuple[float, float]]:
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--device",
        case.device,
        "--metal",
        case.metal,
        "--repeats",
        str(repeats),
        "--warmup",
        str(warmup),
        "--outer-runs",
        str(outer_runs),
    ]
    if case.threshold is not None:
        cmd.extend(["--threshold", str(case.threshold)])
    if case.min_frames is not None:
        cmd.extend(["--min-frames", str(case.min_frames)])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Case {case.name} failed (exit={proc.returncode}):\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    payload = json.loads(proc.stdout)
    return {
        label: (float(v["stft_ms"]), float(v["istft_ms"]))
        for label, v in payload["workloads"].items()
    }


def _worker_main(
    device: str,
    threshold: int | None,
    metal: str,
    repeats: int,
    warmup: int,
    outer_runs: int,
    min_frames: int | None = None,
) -> None:
    from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext

    if not RUST_AVAILABLE or _rust_ext is None:
        raise SystemExit("ERROR: Rust extension unavailable")

    _apply_case_env(Case("worker", device, threshold, metal, min_frames))

    global REPEATS, WARMUP
    REPEATS = repeats
    WARMUP = warmup

    out: dict[str, dict[str, float]] = {}
    for label, n_samples, n_fft, hop in WORKLOADS:
        stft_runs: list[float] = []
        istft_runs: list[float] = []
        for _ in range(outer_runs):
            stft_ms, istft_ms = _bench_pair(_rust_ext, n_samples, n_fft, hop)
            stft_runs.append(stft_ms)
            istft_runs.append(istft_ms)
        out[label] = {
            "stft_ms": statistics.median(stft_runs),
            "istft_ms": statistics.median(istft_runs),
        }

    print(json.dumps({"workloads": out}))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 19 Auto-threshold benchmark")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--metal", default="")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--outer-runs", type=int, default=OUTER_RUNS)
    ap.add_argument("--min-frames", type=int, default=None)
    args = ap.parse_args()

    if args.repeats <= 0 or args.warmup < 0 or args.outer_runs <= 0:
        raise SystemExit("ERROR: repeats>0, warmup>=0, outer-runs>0 are required")

    if args.worker:
        _worker_main(
            args.device,
            args.threshold,
            args.metal,
            args.repeats,
            args.warmup,
            args.outer_runs,
            args.min_frames,
        )
        return

    cases = [
        Case("cpu", "cpu", None, ""),
        Case("auto_50m", "auto", 50_000_000, "force-on"),
        Case("auto_100m", "auto", 100_000_000, "force-on"),
        Case("auto_200m", "auto", 200_000_000, "force-on"),
        Case("gpu", "apple-gpu", None, "force-on"),
    ]

    results: dict[str, dict[str, tuple[float, float]]] = {}

    for case in cases:
        results[case.name] = _run_case_in_subprocess(
            case,
            args.repeats,
            args.warmup,
            args.outer_runs,
        )

    print("case,workload,stft_ms,istft_ms,stft_speedup_vs_cpu,istft_speedup_vs_cpu")
    for case in cases:
        for label, *_ in WORKLOADS:
            stft_ms, istft_ms = results[case.name][label]
            cpu_stft_ms, cpu_istft_ms = results["cpu"][label]
            print(
                f"{case.name},{label},{stft_ms:.3f},{istft_ms:.3f},"
                f"{cpu_stft_ms / stft_ms:.3f},{cpu_istft_ms / istft_ms:.3f}"
            )


if __name__ == "__main__":
    main()

