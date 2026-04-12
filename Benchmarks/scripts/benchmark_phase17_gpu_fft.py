#!/usr/bin/env python
"""
Phase 17 GPU FFT Benchmark
==========================
Measures STFT/iSTFT performance across CPU vs. Apple GPU (Metal) dispatch modes.

Usage::

    # Baseline CPU
    python Benchmarks/scripts/benchmark_phase17_gpu_fft.py \
        --mode cpu --json-out Benchmarks/results/phase17_cpu_baseline.json

    # GPU (experimental Metal FFT)
    IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on \
    python Benchmarks/scripts/benchmark_phase17_gpu_fft.py \
        --mode gpu --json-out Benchmarks/results/phase17_gpu.json

    # Compare
    python Benchmarks/scripts/benchmark_phase17_gpu_fft.py --compare \
        --baseline Benchmarks/results/phase17_cpu_baseline.json \
        --candidate Benchmarks/results/phase17_gpu.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

WORKLOADS = [
    # (label, n_samples, n_fft, hop_length)
    ("short_512",   22050 * 1,  512,  128),
    ("short_1024",  22050 * 1, 1024,  256),
    ("medium_512",  22050 * 5,  512,  128),
    ("medium_1024", 22050 * 5, 1024,  256),
    ("long_1024",   22050 * 30, 1024,  256),
]

REPEATS = 5   # warm-up + timed runs
WARMUP  = 1


def _bench_stft_complex(rust_ext, n_samples: int, n_fft: int, hop_length: int) -> float:
    rng = np.random.default_rng(42)
    y = rng.standard_normal(n_samples).astype(np.float32)
    # Warm-up
    for _ in range(WARMUP):
        rust_ext.stft_complex(y, n_fft, hop_length, True, None)
    # Timed
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        rust_ext.stft_complex(y, n_fft, hop_length, True, None)
    return (time.perf_counter() - t0) / REPEATS


def _bench_istft_f32(rust_ext, n_samples: int, n_fft: int, hop_length: int) -> float:
    rng = np.random.default_rng(42)
    y = rng.standard_normal(n_samples).astype(np.float32)
    stft_m = rust_ext.stft_complex(y, n_fft, hop_length, True, None)
    # Warm-up
    for _ in range(WARMUP):
        rust_ext.istft_f32(stft_m, n_fft, hop_length, None, None)
    # Timed
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        rust_ext.istft_f32(stft_m, n_fft, hop_length, None, None)
    return (time.perf_counter() - t0) / REPEATS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmarks(mode: str, rust_ext) -> dict:
    """Run all workloads and return a results dict."""
    if mode == "gpu":
        os.environ["IRON_LIBROSA_RUST_DEVICE"] = "apple-gpu"
    else:
        os.environ["IRON_LIBROSA_RUST_DEVICE"] = "cpu"

    results = {
        "mode": mode,
        "metal_fft_experimental": os.getenv("IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL", ""),
        "workloads": {},
    }

    for label, n_samples, n_fft, hop in WORKLOADS:
        stft_t  = _bench_stft_complex(rust_ext, n_samples, n_fft, hop)
        istft_t = _bench_istft_f32(rust_ext, n_samples, n_fft, hop)
        results["workloads"][label] = {
            "n_samples":  n_samples,
            "n_fft":      n_fft,
            "hop_length": hop,
            "stft_complex_ms":  round(stft_t  * 1000, 3),
            "istft_f32_ms":     round(istft_t * 1000, 3),
        }
        print(f"  {label:<20}  stft={stft_t*1000:7.2f}ms  istft={istft_t*1000:7.2f}ms")

    return results


def compare(baseline_path: str, candidate_path: str) -> None:
    with open(baseline_path) as f:
        base = json.load(f)
    with open(candidate_path) as f:
        cand = json.load(f)

    auto_review_cases: list[str] = []

    print(f"\n{'Workload':<22} {'STFT speedup':>14} {'iSTFT speedup':>14}")
    print("-" * 52)
    for label, bw in base["workloads"].items():
        cw = cand["workloads"].get(label)
        if cw is None:
            continue
        stft_speedup  = bw["stft_complex_ms"]  / cw["stft_complex_ms"]
        istft_speedup = bw["istft_f32_ms"]     / cw["istft_f32_ms"]
        flag = ""
        if stft_speedup < 1.0 or istft_speedup < 1.0:
            flag = " ⚠"
            auto_review_cases.append(label)
        print(f"  {label:<20}  {stft_speedup:>12.2f}x  {istft_speedup:>12.2f}x{flag}")

    if auto_review_cases:
        print(f"\n⚠  Auto-review required for: {auto_review_cases}")
        print("   Regressions below 1.0x must be documented before promotion.")
    else:
        print("\n✅ All workloads at or above 1.0x — no auto-review required.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 17 GPU FFT benchmark")
    ap.add_argument("--mode", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--candidate", default=None)
    args = ap.parse_args()

    if args.compare:
        if not args.baseline or not args.candidate:
            ap.error("--compare requires --baseline and --candidate")
        compare(args.baseline, args.candidate)
        return

    try:
        from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE
    except ImportError:
        print("ERROR: librosa Rust extension not available.", file=sys.stderr)
        sys.exit(1)

    if not RUST_AVAILABLE or _rust_ext is None:
        print("ERROR: RUST_AVAILABLE=False. Build with maturin first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Phase 17 GPU FFT Benchmark — mode={args.mode} ===\n")
    results = run_benchmarks(args.mode, _rust_ext)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {args.json_out}")
    else:
        print("\n(use --json-out <file> to persist results)")


if __name__ == "__main__":
    main()

