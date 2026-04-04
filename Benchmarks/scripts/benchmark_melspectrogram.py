"""Benchmark mel projection in melspectrogram.

Two sections:
  1. Full melspectrogram(S=…) call  – includes filters.mel() overhead (same for both).
  2. Projection-only                – pre-computes mel_basis; measures pure GEMM.
"""

import importlib
import time

import numpy as np
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE

N_RUNS = 8
CASES = [
    # (n_fft, n_frames, n_mels)
    (1024, 300, 64),
    (2048, 800, 128),
    (4096, 1200, 256),
]


def _timeit(fn):
    fn()  # warmup
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def bench_one(mod, label, S, sr, n_fft, n_mels):
    avg, minv = _timeit(
        lambda: mod.feature.melspectrogram(S=S, sr=sr, n_fft=n_fft, n_mels=n_mels)
    )
    print(f"  {label:20s} avg={avg*1e3:8.2f} ms  min={minv*1e3:8.2f} ms")
    return minv


if __name__ == "__main__":
    np.random.seed(7)
    iron_librosa = importlib.import_module("iron_librosa")

    # ── Section 1: full melspectrogram() call (includes filters.mel overhead) ─
    print("=" * 64)
    print("Section 1: full melspectrogram(S=…) — includes filters.mel()")
    print("=" * 64)

    for n_fft, n_frames, n_mels in CASES:
        n_bins = n_fft // 2 + 1
        S = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32) ** 2
        print(f"\ncase: n_fft={n_fft}, bins={n_bins}, frames={n_frames}, n_mels={n_mels}")
        t_py = bench_one(librosa, "librosa (Python)", S, 22050, n_fft, n_mels)
        t_rs = bench_one(iron_librosa, "iron-librosa (Rust)", S, 22050, n_fft, n_mels)
        print(f"  speedup (min)         {t_py / t_rs:8.2f}x")

    # ── Section 2: projection only (pure GEMM, mel_basis pre-computed) ────────
    print("\n" + "=" * 64)
    print("Section 2: projection only — mel_basis pre-computed (pure GEMM)")
    print("=" * 64)

    for n_fft, n_frames, n_mels in CASES:
        n_bins = n_fft // 2 + 1
        S32 = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float32) ** 2
        mb32 = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=n_mels)  # float32
        S_c = np.ascontiguousarray(S32)
        mb_c = np.ascontiguousarray(mb32)

        print(f"\ncase: n_fft={n_fft}, bins={n_bins}, frames={n_frames}, n_mels={n_mels}")

        # NumPy / MKL baseline
        _, t_np = _timeit(lambda: mb_c.dot(S_c))
        print(f"  {'numpy dot (MKL)':20s} avg=n/a        min={t_np*1e3:8.3f} ms")

        # Rust faer kernel
        if RUST_AVAILABLE and hasattr(_rust_ext, "mel_project_f32"):
            _, t_rs = _timeit(lambda: _rust_ext.mel_project_f32(S_c, mb_c))
            print(f"  {'Rust faer f32':20s} avg=n/a        min={t_rs*1e3:8.3f} ms")
            print(f"  speedup vs numpy      {t_np / t_rs:8.2f}x")
        else:
            print("  Rust mel_project_f32 not available")

    print("\n" + "=" * 64)

