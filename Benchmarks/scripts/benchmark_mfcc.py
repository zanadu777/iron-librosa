"""Benchmark MFCC in librosa vs iron-librosa.

Sections:
  1) Full mfcc(y=...) pipeline
  2) DCT stage only (S input), fast-path config (type=2, norm='ortho', float32)
  3) DCT fallback config (norm=None) for context
"""

import importlib
import time

import numpy as np
import librosa
import scipy

N_RUNS = 10
DURATIONS = [1, 5, 15, 30]
SR = 22050
N_FFT = 2048
HOP = 512
N_MELS = 128
N_MFCC = 20


def _timeit(fn):
    fn()  # warmup
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def _bench_mfcc(mod, y, label):
    avg, minv = _timeit(
        lambda: mod.feature.mfcc(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP,
            n_mels=N_MELS,
            n_mfcc=N_MFCC,
            dct_type=2,
            norm="ortho",
            lifter=0,
        )
    )
    print(f"  {label:20s} avg={avg*1e3:8.2f} ms  min={minv*1e3:8.2f} ms")
    return minv


if __name__ == "__main__":
    np.random.seed(7)
    iron_librosa = importlib.import_module("iron_librosa")

    print("=" * 70)
    print("Section 1: full mfcc(y=...) pipeline")
    print("=" * 70)

    for dur in DURATIONS:
        y = np.random.randn(SR * dur).astype(np.float32)
        print(f"\ncase: duration={dur:2d}s, samples={len(y)}")
        t_py = _bench_mfcc(librosa, y, "librosa (Python)")
        t_rs = _bench_mfcc(iron_librosa, y, "iron-librosa (Rust)")
        print(f"  speedup (min)         {t_py / t_rs:8.2f}x")

    print("\n" + "=" * 70)
    print("Section 2: DCT only from S (fast-path config)")
    print("=" * 70)

    for dur in DURATIONS:
        n_frames = 1 + (SR * dur) // HOP
        S = np.random.randn(N_MELS, n_frames).astype(np.float32)

        print(f"\ncase: duration={dur:2d}s, S.shape={S.shape}")

        _, t_sp = _timeit(lambda: scipy.fft.dct(S, axis=-2, type=2, norm="ortho")[:N_MFCC, :])
        print(f"  {'scipy dct baseline':20s} avg=n/a        min={t_sp*1e3:8.3f} ms")

        _, t_py = _timeit(
            lambda: librosa.feature.mfcc(
                S=S,
                n_mfcc=N_MFCC,
                dct_type=2,
                norm="ortho",
                lifter=0,
            )
        )
        _, t_rs = _timeit(
            lambda: iron_librosa.feature.mfcc(
                S=S,
                n_mfcc=N_MFCC,
                dct_type=2,
                norm="ortho",
                lifter=0,
            )
        )

        print(f"  {'librosa.feature.mfcc':20s} avg=n/a        min={t_py*1e3:8.3f} ms")
        print(f"  {'iron_librosa.mfcc':20s} avg=n/a        min={t_rs*1e3:8.3f} ms")
        print(f"  speedup vs librosa     {t_py / t_rs:8.2f}x")

    print("\n" + "=" * 70)
    print("Section 3: DCT fallback (norm=None)")
    print("=" * 70)

    n_frames = 1 + (SR * 15) // HOP
    S = np.random.randn(N_MELS, n_frames).astype(np.float32)
    _, t_py = _timeit(
        lambda: librosa.feature.mfcc(
            S=S,
            n_mfcc=N_MFCC,
            dct_type=2,
            norm=None,
            lifter=0,
        )
    )
    _, t_rs = _timeit(
        lambda: iron_librosa.feature.mfcc(
            S=S,
            n_mfcc=N_MFCC,
            dct_type=2,
            norm=None,
            lifter=0,
        )
    )

    print(f"\ncase: S.shape={S.shape}")
    print(f"  {'librosa.feature.mfcc':20s} avg=n/a        min={t_py*1e3:8.3f} ms")
    print(f"  {'iron_librosa.mfcc':20s} avg=n/a        min={t_rs*1e3:8.3f} ms")
    print(f"  speedup vs librosa     {t_py / t_rs:8.2f}x")

    print("\n" + "=" * 70)

