"""Benchmark nn_filter: Python (librosa) vs Rust-accelerated (iron-librosa).

Runs warmup + multiple timed iterations for several (n_features × n_frames)
sizes and for both mean and weighted-average aggregation modes.
"""
import importlib
import time

import numpy as np
import librosa

SIZES = [
    (12, 500),    # small  – chroma-like
    (128, 1000),  # medium – mel spectrogram
    (128, 5000),  # large  – ~2-min song at 43 fps
]
N_RUNS = 5  # timed repetitions per case (after a warmup run)


def _timeit(fn, n_runs):
    """Return (mean_sec, min_sec) over n_runs calls."""
    fn()  # warmup
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.min(times))


def bench_module(mod, label, X, rec_dense, rec_affinity):
    print(f"\n  [{label}]")
    # --- mean (default aggregate) ---
    mean_avg, mean_min = _timeit(
        lambda: mod.decompose.nn_filter(X, rec=rec_dense), N_RUNS
    )
    print(f"    mean      – avg {mean_avg*1e3:7.2f} ms  min {mean_min*1e3:7.2f} ms")
    # --- weighted average (non-local means) ---
    w_avg, w_min = _timeit(
        lambda: mod.decompose.nn_filter(X, rec=rec_affinity, aggregate=np.average),
        N_RUNS,
    )
    print(f"    weighted  – avg {w_avg*1e3:7.2f} ms  min {w_min*1e3:7.2f} ms")
    return mean_avg, w_avg


if __name__ == "__main__":
    iron_librosa = importlib.import_module("iron_librosa")
    np.random.seed(42)

    print("=" * 60)
    print(f"nn_filter benchmark  ({N_RUNS} runs after warmup)")
    print("=" * 60)

    for n_feat, n_frames in SIZES:
        X = np.random.randn(n_feat, n_frames).astype(np.float64)
        # Binary recurrence matrix (sparse, symmetric)
        rec_dense = librosa.segment.recurrence_matrix(X, sparse=True)
        # Affinity matrix for weighted mode
        rec_affinity = librosa.segment.recurrence_matrix(
            X, mode="affinity", metric="cosine", sparse=True
        )

        print(f"\nSize: {n_feat} × {n_frames}  "
              f"(rec nnz={rec_dense.nnz}, affinity nnz={rec_affinity.nnz})")

        py_mean, py_w = bench_module(librosa, "librosa  (Python)", X, rec_dense, rec_affinity)
        ru_mean, ru_w = bench_module(iron_librosa, "iron-librosa (Rust)", X, rec_dense, rec_affinity)

        print(f"\n  Speedup  mean:     {py_mean/ru_mean:5.1f}×")
        print(f"  Speedup  weighted: {py_w/ru_w:5.1f}×")

    print("\n" + "=" * 60)
