"""Benchmark harness for Phase 5 chroma filter-bank acceleration.

Sections
--------
  1. Direct `filters.chroma` Rust vs forced Python fallback
  2. End-to-end `chroma_stft(S=..., tuning=0.0)` with tuning estimation bypassed
"""

import argparse
import importlib
import json
import platform
import time
from contextlib import contextmanager

import numpy as np
import librosa
from benchmark_guard import (
    REVIEW_SPEEDUP_THRESHOLD,
    assert_benchmark_payload_schema,
    evaluate_speedup,
)

N_RUNS = 10
SR = 22050
CASES = [(2048, 12), (4096, 12), (4096, 24)]


def _timeit(fn):
    fn()
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def _row(label, avg, minv):
    print(f"  {label:<30s} avg={avg*1e3:8.3f} ms  min={minv*1e3:8.3f} ms")


@contextmanager
def _force_python_chroma_filter(enabled: bool):
    filters_mod = importlib.import_module("librosa.filters")
    prev_available = filters_mod.RUST_AVAILABLE
    prev_ext = filters_mod._rust_ext
    try:
        if enabled:
            filters_mod.RUST_AVAILABLE = False
            filters_mod._rust_ext = None
        yield
    finally:
        filters_mod.RUST_AVAILABLE = prev_available
        filters_mod._rust_ext = prev_ext


def bench_filters_chroma(rows, auto_review_cases, review_threshold: float):
    print("=" * 72)
    print("Phase 5 chroma filter benchmark")
    print("=" * 72)

    for n_fft, n_chroma in CASES:
        print(f"\ncase: n_fft={n_fft}, n_chroma={n_chroma}")
        with _force_python_chroma_filter(False):
            rs_avg, rs_min = _timeit(
                lambda: librosa.filters.chroma(sr=SR, n_fft=n_fft, n_chroma=n_chroma, tuning=0.0)
            )
        with _force_python_chroma_filter(True):
            py_avg, py_min = _timeit(
                lambda: librosa.filters.chroma(sr=SR, n_fft=n_fft, n_chroma=n_chroma, tuning=0.0)
            )
        _row("filters.chroma rust", rs_avg, rs_min)
        _row("filters.chroma py", py_avg, py_min)
        speed = py_min / rs_min
        print(f"  speedup (min)                {speed:8.2f}x")
        review = evaluate_speedup(speed, review_threshold)
        rows.append(
            {
                "section": "filters.chroma",
                "case": f"n_fft={n_fft},n_chroma={n_chroma}",
                "py_min_ms": py_min * 1e3,
                "rust_min_ms": rs_min * 1e3,
                "speedup": review["speedup"],
                "review_required": review["review_required"],
            }
        )
        if review["review_required"]:
            auto_review_cases.append(f"filters.chroma n_fft={n_fft} n_chroma={n_chroma} ({speed:.2f}x)")


def bench_chroma_stft_fixed_tuning(rows, auto_review_cases, review_threshold: float):
    print("\n" + "=" * 72)
    print("Phase 5 chroma_stft benchmark (fixed tuning)")
    print("=" * 72)

    for n_fft, n_chroma in CASES:
        n_bins = n_fft // 2 + 1
        frames = 800
        S = np.abs(np.random.randn(n_bins, frames).astype(np.float32)) ** 2
        print(f"\ncase: n_fft={n_fft}, n_chroma={n_chroma}, frames={frames}")
        with _force_python_chroma_filter(False):
            rs_avg, rs_min = _timeit(
                lambda: librosa.feature.chroma_stft(S=S, sr=SR, n_fft=n_fft, n_chroma=n_chroma, tuning=0.0)
            )
        with _force_python_chroma_filter(True):
            py_avg, py_min = _timeit(
                lambda: librosa.feature.chroma_stft(S=S, sr=SR, n_fft=n_fft, n_chroma=n_chroma, tuning=0.0)
            )
        _row("chroma_stft rust", rs_avg, rs_min)
        _row("chroma_stft py", py_avg, py_min)
        speed = py_min / rs_min
        print(f"  speedup (min)                {speed:8.2f}x")
        review = evaluate_speedup(speed, review_threshold)
        rows.append(
            {
                "section": "feature.chroma_stft",
                "case": f"n_fft={n_fft},n_chroma={n_chroma},frames={frames}",
                "py_min_ms": py_min * 1e3,
                "rust_min_ms": rs_min * 1e3,
                "speedup": review["speedup"],
                "review_required": review["review_required"],
            }
        )
        if review["review_required"]:
            auto_review_cases.append(f"chroma_stft n_fft={n_fft} n_chroma={n_chroma} ({speed:.2f}x)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 chroma benchmark")
    parser.add_argument("--review-threshold", type=float, default=REVIEW_SPEEDUP_THRESHOLD)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    np.random.seed(2065)
    rows = []
    auto_review_cases = []
    bench_filters_chroma(rows, auto_review_cases, args.review_threshold)
    bench_chroma_stft_fixed_tuning(rows, auto_review_cases, args.review_threshold)

    if auto_review_cases:
        print("\nauto-review required (< threshold):")
        for item in auto_review_cases:
            print(f"  - {item}")

    if args.json_out:
        payload = {
            "meta": {
                "benchmark": "phase5_chroma",
                "review_threshold": args.review_threshold,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "librosa": getattr(librosa, "__version__", "unknown"),
            },
            "auto_review_cases": auto_review_cases,
            "rows": rows,
        }
        assert_benchmark_payload_schema(payload, "phase5_chroma")
        with open(args.json_out, "w", encoding="utf-8") as fdesc:
            json.dump(payload, fdesc, indent=2)
        print(f"wrote json report: {args.json_out}")

