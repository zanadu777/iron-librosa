"""Benchmark harness for Phase 4C chroma filter projection.

Sections
--------
  1. Raw kernel: chroma_project_f32 / f64 vs NumPy einsum
  2. Public chroma_stft(S=…) — librosa dispatch comparison
  3. Multichannel chroma_stft — 2-channel, 4-channel workloads
  4. Fallback paths — dtype guard verification
"""

import importlib
import time

import numpy as np
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE

# ── configuration ────────────────────────────────────────────────────────────
N_RUNS = 10
RANDOM_SEED = 42

# (n_fft, n_frames) pairs — n_bins = n_fft // 2 + 1
SIZES = [
    (1024, 300),   # small  – short clip
    (2048, 800),   # medium – ~10 s @ 22 kHz, 512 hop
    (4096, 1200),  # large  – dense analysis window
]

SR = 22050
N_CHROMA = 12


# ── helpers ──────────────────────────────────────────────────────────────────

def _timeit(fn):
    fn()  # warmup
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def _row(label, avg, minv):
    print(f"    {label:<28s}  avg={avg*1e3:8.3f} ms  min={minv*1e3:8.3f} ms")


def _speedup(t_py, t_rs):
    print(f"    {'speedup (min)':28s}  {t_py / t_rs:8.2f}x")


def _section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: raw chroma project kernel
# ─────────────────────────────────────────────────────────────────────────────

def bench_raw_chroma():
    _section(
        "Section 1: raw chroma_project kernel  (chroma_project_f32 / _f64 vs NumPy einsum)"
    )

    if not RUST_AVAILABLE:
        print("  [SKIP] Rust extension not available.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32)) ** 2
        S64 = S32.astype(np.float64)
        chromafb32 = np.abs(np.random.randn(N_CHROMA, n_bins).astype(np.float32))
        chromafb64 = chromafb32.astype(np.float64)

        print(f"\n  case: n_fft={n_fft}, bins={n_bins}, frames={n_frames}")

        # f32
        if hasattr(_rust_ext, "chroma_project_f32"):
            def py_f32():
                return np.einsum("cf,ft->ct", chromafb32, S32, optimize=True)

            avg_py, min_py = _timeit(py_f32)
            avg_rs, min_rs = _timeit(
                lambda: _rust_ext.chroma_project_f32(
                    np.ascontiguousarray(S32),
                    np.ascontiguousarray(chromafb32),
                )
            )
            print("  [f32]")
            _row("numpy einsum", avg_py, min_py)
            _row("Rust f32 kernel", avg_rs, min_rs)
            _speedup(min_py, min_rs)
        else:
            print("  chroma_project_f32 not registered – skipping")

        # f64
        if hasattr(_rust_ext, "chroma_project_f64"):
            def py_f64():
                return np.einsum("cf,ft->ct", chromafb64, S64, optimize=True)

            avg_py, min_py = _timeit(py_f64)
            avg_rs, min_rs = _timeit(
                lambda: _rust_ext.chroma_project_f64(
                    np.ascontiguousarray(S64),
                    np.ascontiguousarray(chromafb64),
                )
            )
            print("  [f64]")
            _row("numpy einsum", avg_py, min_py)
            _row("Rust f64 kernel", avg_rs, min_rs)
            _speedup(min_py, min_rs)
        else:
            print("  chroma_project_f64 not registered – skipping")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: public chroma_stft(S=…) dispatch
# ─────────────────────────────────────────────────────────────────────────────

def bench_public_chroma_stft(iron_librosa):
    _section(
        "Section 2: public chroma_stft(S=…) — librosa dispatch comparison"
    )

    if not RUST_AVAILABLE:
        print("  [SKIP] Rust not available.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32)) ** 2
        S64 = S32.astype(np.float64)

        for S, dtype_label in [(S32, "float32"), (S64, "float64")]:
            Sc = np.ascontiguousarray(S)
            print(f"\n  case: n_fft={n_fft}, frames={n_frames}, dtype={dtype_label}")

            avg_py, min_py = _timeit(
                lambda: librosa.feature.chroma_stft(S=Sc, sr=SR, n_fft=n_fft)
            )
            avg_rs, min_rs = _timeit(
                lambda: iron_librosa.feature.chroma_stft(S=Sc, sr=SR, n_fft=n_fft)
            )

            _row("librosa  (Python)", avg_py, min_py)
            _row("iron-librosa (Rust)", avg_rs, min_rs)
            _speedup(min_py, min_rs)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: multichannel chroma_stft(S=…)
# ─────────────────────────────────────────────────────────────────────────────

def bench_multichannel_chroma(iron_librosa):
    _section("Section 3: multichannel chroma_stft(S=…) — per-channel Rust dispatch")

    n_fft, n_frames = 2048, 800
    n_bins = n_fft // 2 + 1

    for n_channels in [2, 4, 8]:
        S = np.abs(
            np.random.randn(n_channels, n_bins, n_frames).astype(np.float32)
        ) ** 2
        Sc = np.ascontiguousarray(S)

        print(f"\n  case: channels={n_channels}, n_fft={n_fft}, frames={n_frames}, dtype=float32")

        avg_py, min_py = _timeit(
            lambda: librosa.feature.chroma_stft(S=Sc, sr=SR, n_fft=n_fft)
        )
        avg_rs, min_rs = _timeit(
            lambda: iron_librosa.feature.chroma_stft(S=Sc, sr=SR, n_fft=n_fft)
        )

        _row("librosa  (Python)", avg_py, min_py)
        _row("iron-librosa (Rust)", avg_rs, min_rs)
        _speedup(min_py, min_rs)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: fallback paths
# ─────────────────────────────────────────────────────────────────────────────

def bench_fallback_paths(iron_librosa):
    _section(
        "Section 4: fallback paths — dtype guard confirms Python path"
    )

    n_fft = 2048
    n_bins = n_fft // 2 + 1
    n_frames = 400

    # chroma_stft with float16 S (unsupported dtype → Python fallback)
    S_f16 = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float16) ** 2

    print(f"\n  chroma_stft fallback: float16 S (S.dtype not in {{float32, float64}} → Python path)")
    avg_py, min_py = _timeit(
        lambda: librosa.feature.chroma_stft(S=S_f16, sr=SR, n_fft=n_fft)
    )
    avg_rs, min_rs = _timeit(
        lambda: iron_librosa.feature.chroma_stft(S=S_f16, sr=SR, n_fft=n_fft)
    )
    _row("librosa  (Python)", avg_py, min_py)
    _row("iron-librosa (Rust)", avg_rs, min_rs)
    ratio = min_py / min_rs if min_rs > 0 else float("nan")
    print(f"    {'ratio (≈1.0 expected)':28s}  {ratio:8.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    iron_librosa = importlib.import_module("iron_librosa")

    print("=" * 72)
    print(f"Phase 4C benchmark  ({N_RUNS} timed runs after warmup)")
    print(f"Rust available: {RUST_AVAILABLE}")
    if RUST_AVAILABLE:
        kernels = [
            "chroma_project_f32",
            "chroma_project_f64",
        ]
        for k in kernels:
            status = "✓" if hasattr(_rust_ext, k) else "✗ MISSING"
            print(f"  {k:35s}  {status}")
    print("=" * 72)

    bench_raw_chroma()
    bench_public_chroma_stft(iron_librosa)
    bench_multichannel_chroma(iron_librosa)
    bench_fallback_paths(iron_librosa)

    print()
    print("=" * 72)
    print("Phase 4C benchmark complete.")
    print("=" * 72)

