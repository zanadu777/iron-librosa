"""Benchmark harness for Phase 5 tuning estimation acceleration.

Measures the post-piptrack estimate_tuning acceleration.
"""

import argparse
import importlib
import json
import platform
import time
from contextlib import contextmanager

import numpy as np
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE
from benchmark_guard import (
    REVIEW_SPEEDUP_THRESHOLD,
    assert_benchmark_payload_schema,
    evaluate_speedup,
)

N_RUNS = 10
SR = 22050
SIZES = [(1025, 200), (1025, 800), (2049, 1200)]


def _timeit(fn):
    fn()
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def _row(label, avg, minv):
    print(f"  {label:<28s} avg={avg*1e3:8.3f} ms  min={minv*1e3:8.3f} ms")


@contextmanager
def _force_python_fallback(enabled: bool):
    pitch_mod = importlib.import_module("librosa.core.pitch")
    prev_available = pitch_mod.RUST_AVAILABLE
    prev_ext = pitch_mod._rust_ext
    prev_enable = getattr(pitch_mod, "_ENABLE_RUST_TUNING", False)
    try:
        pitch_mod._ENABLE_RUST_TUNING = not enabled
        if enabled:
            pitch_mod.RUST_AVAILABLE = False
            pitch_mod._rust_ext = None
        yield
    finally:
        pitch_mod._ENABLE_RUST_TUNING = prev_enable
        pitch_mod.RUST_AVAILABLE = prev_available
        pitch_mod._rust_ext = prev_ext


def bench_estimate_tuning(rows, auto_review_cases, review_threshold: float):
    print("=" * 72)
    print("Phase 5 tuning benchmark")
    print("=" * 72)

    for n_bins, n_frames in SIZES:
        S = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        n_fft = (n_bins - 1) * 2

        print(f"\ncase: bins={n_bins}, frames={n_frames}, n_fft={n_fft}")

        with _force_python_fallback(False):
            rs_avg, rs_min = _timeit(
                lambda: librosa.estimate_tuning(S=S, sr=SR, n_fft=n_fft)
            )

        with _force_python_fallback(True):
            py_avg, py_min = _timeit(
                lambda: librosa.estimate_tuning(S=S, sr=SR, n_fft=n_fft)
            )

        _row("estimate_tuning rust", rs_avg, rs_min)
        _row("estimate_tuning py", py_avg, py_min)
        speed = py_min / rs_min
        print(f"  speedup (min)              {speed:8.2f}x")
        review = evaluate_speedup(speed, review_threshold)
        rows.append(
            {
                "section": "estimate_tuning",
                "case": f"bins={n_bins},frames={n_frames}",
                "py_min_ms": py_min * 1e3,
                "rust_min_ms": rs_min * 1e3,
                "speedup": review["speedup"],
                "review_required": review["review_required"],
            }
        )
        if review["review_required"]:
            auto_review_cases.append(f"estimate_tuning bins={n_bins} frames={n_frames} ({speed:.2f}x)")


def bench_piptrack_api(rows, auto_review_cases, review_threshold: float):
    print("\n" + "=" * 72)
    print("Phase 5 tuning benchmark - piptrack API")
    print("=" * 72)

    pitch_mod = importlib.import_module("librosa.core.pitch")
    prev_mode = getattr(pitch_mod, "_PIPTRACK_RUST_MODE", "auto")
    prev_work = getattr(pitch_mod, "_PIPTRACK_RUST_MIN_WORK", 0)

    try:
        pitch_mod._PIPTRACK_RUST_MODE = "rust"
        pitch_mod._PIPTRACK_RUST_MIN_WORK = 0

        for n_bins, n_frames in SIZES:
            S = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
            n_fft = (n_bins - 1) * 2

            print(f"\ncase: bins={n_bins}, frames={n_frames}, n_fft={n_fft}")

            rs_avg, rs_min = _timeit(lambda: librosa.piptrack(S=S, sr=SR, n_fft=n_fft))

            with _force_python_fallback(True):
                py_avg, py_min = _timeit(lambda: librosa.piptrack(S=S, sr=SR, n_fft=n_fft))

            _row("piptrack rust", rs_avg, rs_min)
            _row("piptrack py", py_avg, py_min)
            speed = py_min / rs_min
            print(f"  speedup (min)              {speed:8.2f}x")
            review = evaluate_speedup(speed, review_threshold)
            rows.append(
                {
                    "section": "piptrack",
                    "case": f"bins={n_bins},frames={n_frames}",
                    "py_min_ms": py_min * 1e3,
                    "rust_min_ms": rs_min * 1e3,
                    "speedup": review["speedup"],
                    "review_required": review["review_required"],
                }
            )
            if review["review_required"]:
                auto_review_cases.append(f"piptrack bins={n_bins} frames={n_frames} ({speed:.2f}x)")
    finally:
        pitch_mod._PIPTRACK_RUST_MODE = prev_mode
        pitch_mod._PIPTRACK_RUST_MIN_WORK = prev_work


def _python_postprocess(pitch, mag, resolution=0.01, bins_per_octave=12):
    pitch_mask = pitch > 0
    threshold = np.median(mag[pitch_mask]) if pitch_mask.any() else 0.0
    return librosa.pitch_tuning(
        pitch[(mag >= threshold) & pitch_mask],
        resolution=resolution,
        bins_per_octave=bins_per_octave,
    )


def bench_postprocess_only(rows, auto_review_cases, review_threshold: float):
    print("\n" + "=" * 72)
    print("Phase 5 tuning benchmark - postprocess only (fixed pitch/mag)")
    print("=" * 72)

    if not RUST_AVAILABLE:
        print("Rust extension unavailable; skipping.")
        return

    for n_bins, n_frames in SIZES:
        # Realistic synthetic piptrack-like tensors
        S = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        pitch, mag = librosa.piptrack(S=S, sr=SR, n_fft=(n_bins - 1) * 2)
        pitch = np.ascontiguousarray(pitch)
        mag = np.ascontiguousarray(mag)

        print(f"\ncase: bins={n_bins}, frames={n_frames}")
        nnz = int((pitch > 0).sum())
        total = pitch.size
        print(f"  pitched bins: {nnz}/{total} ({100.0*nnz/total:5.2f}%)")

        py_avg, py_min = _timeit(
            lambda: _python_postprocess(pitch, mag, resolution=0.01, bins_per_octave=12)
        )
        rs_avg, rs_min = _timeit(
            lambda: _rust_ext.estimate_tuning_from_piptrack_f32(
                pitch, mag, 0.01, 12
            )
        )

        _row("postprocess python", py_avg, py_min)
        _row("postprocess rust", rs_avg, rs_min)
        speed = py_min / rs_min
        print(f"  speedup (min)              {speed:8.2f}x")
        review = evaluate_speedup(speed, review_threshold)
        rows.append(
            {
                "section": "estimate_tuning_from_piptrack",
                "case": f"bins={n_bins},frames={n_frames}",
                "py_min_ms": py_min * 1e3,
                "rust_min_ms": rs_min * 1e3,
                "speedup": review["speedup"],
                "review_required": review["review_required"],
            }
        )
        if review["review_required"]:
            auto_review_cases.append(
                f"estimate_tuning_from_piptrack bins={n_bins} frames={n_frames} ({speed:.2f}x)"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 tuning benchmark")
    parser.add_argument("--review-threshold", type=float, default=REVIEW_SPEEDUP_THRESHOLD)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    np.random.seed(2064)
    rows = []
    auto_review_cases = []
    bench_piptrack_api(rows, auto_review_cases, args.review_threshold)
    bench_estimate_tuning(rows, auto_review_cases, args.review_threshold)
    bench_postprocess_only(rows, auto_review_cases, args.review_threshold)

    if auto_review_cases:
        print("\nauto-review required (< threshold):")
        for item in auto_review_cases:
            print(f"  - {item}")

    if args.json_out:
        payload = {
            "meta": {
                "benchmark": "phase5_tuning",
                "review_threshold": args.review_threshold,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "librosa": getattr(librosa, "__version__", "unknown"),
            },
            "auto_review_cases": auto_review_cases,
            "rows": rows,
        }
        assert_benchmark_payload_schema(payload, "phase5_tuning")
        with open(args.json_out, "w", encoding="utf-8") as fdesc:
            json.dump(payload, fdesc, indent=2)
        print(f"wrote json report: {args.json_out}")


