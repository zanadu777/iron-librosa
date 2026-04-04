"""Benchmark onset strength: Python (librosa) vs Rust-accelerated (iron-librosa).

Focuses on the phase-1 fast path where S is already computed and the onset
flux kernel can run in Rust:
- onset_strength_multi(S=..., aggregate=np.mean, max_size=1)
- onset_strength(S=..., aggregate=np.mean, max_size=1)

Also includes one fallback case (aggregate=np.max) for context.
"""

import importlib
import time

import numpy as np
import librosa

N_RUNS = 12
SIZES = [
    (128, 500),
    (128, 2000),
    (256, 4000),
]


def _timeit(fn):
    fn()  # warmup
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def _bench_pair(py_fn, rs_fn, label):
    py_avg, py_min = _timeit(py_fn)
    rs_avg, rs_min = _timeit(rs_fn)

    print(f"  {label}")
    print(f"    librosa      avg={py_avg*1e3:8.3f} ms  min={py_min*1e3:8.3f} ms")
    print(f"    iron-librosa avg={rs_avg*1e3:8.3f} ms  min={rs_min*1e3:8.3f} ms")
    print(f"    speedup(min) {py_min / rs_min:8.2f}x")


if __name__ == "__main__":
    np.random.seed(17)
    iron_librosa = importlib.import_module("iron_librosa")

    print("=" * 72)
    print(f"onset benchmark ({N_RUNS} runs after warmup)")
    print("=" * 72)

    for n_bins, n_frames in SIZES:
        s = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32)

        print(f"\ncase: S.shape={s.shape}, dtype={s.dtype}")

        _bench_pair(
            lambda: librosa.onset.onset_strength_multi(
                S=s, lag=1, max_size=1, aggregate=np.mean, center=False
            ),
            lambda: iron_librosa.onset.onset_strength_multi(
                S=s, lag=1, max_size=1, aggregate=np.mean, center=False
            ),
            "onset_strength_multi fast path (mean)",
        )

        _bench_pair(
            lambda: librosa.onset.onset_strength(
                S=s, lag=1, max_size=1, aggregate=np.mean, center=False
            ),
            lambda: iron_librosa.onset.onset_strength(
                S=s, lag=1, max_size=1, aggregate=np.mean, center=False
            ),
            "onset_strength wrapper fast path (mean)",
        )

        _bench_pair(
            lambda: librosa.onset.onset_strength_multi(
                S=s, lag=1, max_size=1, aggregate=np.max, center=False
            ),
            lambda: iron_librosa.onset.onset_strength_multi(
                S=s, lag=1, max_size=1, aggregate=np.max, center=False
            ),
            "fallback path (max aggregate)",
        )

    print("\n" + "=" * 72)

