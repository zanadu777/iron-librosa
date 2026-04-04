"""Benchmark harness for Phase 5 chroma filter-bank acceleration.

Sections
--------
  1. Direct `filters.chroma` Rust vs forced Python fallback
  2. End-to-end `chroma_stft(S=..., tuning=0.0)` with tuning estimation bypassed
"""

import importlib
import time
from contextlib import contextmanager

import numpy as np
import librosa

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


def bench_filters_chroma():
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
        print(f"  speedup (min)                {py_min/rs_min:8.2f}x")


def bench_chroma_stft_fixed_tuning():
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
        print(f"  speedup (min)                {py_min/rs_min:8.2f}x")


if __name__ == "__main__":
    np.random.seed(2065)
    bench_filters_chroma()
    bench_chroma_stft_fixed_tuning()

