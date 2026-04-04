"""Benchmark harness for Phase 4B features: rms(S=…) and spectral_centroid().

Sections
--------
  1. Raw kernel: rms_spectrogram_f32 / f64 vs NumPy reference
  2. Public rms(S=…) — full librosa dispatch comparison
  3. Raw kernel: spectral_centroid_f32 / f64 vs NumPy reference
  4. Public spectral_centroid(S=…) — full librosa dispatch comparison
  5. Multichannel rms(S=…) — 2-channel and 4-channel workloads
  6. Fallback paths — dtype/shape guard, verify Rust is skipped
"""

import importlib
import time

import numpy as np
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE
from librosa.core.convert import fft_frequencies

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


# ── NumPy reference implementations (pure Python / NumPy) ────────────────────

def _rms_numpy(S, frame_length):
    """Exact same formula as the librosa Python path, row-vectorised."""
    x = S.astype(np.float64, copy=True) ** 2
    x[0] *= 0.5
    if frame_length % 2 == 0:
        x[-1] *= 0.5
    power = 2.0 * x.sum(axis=0, keepdims=True) / frame_length ** 2
    return np.sqrt(power)


def _centroid_numpy(S, freq):
    """Vectorised NumPy spectral centroid (single-channel 2-D S)."""
    S64 = S.astype(np.float64)
    freq64 = freq.astype(np.float64)
    denom = S64.sum(axis=0, keepdims=True)
    numer = (S64 * freq64[:, None]).sum(axis=0, keepdims=True)
    mask = denom > np.finfo(np.float64).tiny
    out = np.zeros_like(numer)
    out[mask] = numer[mask] / denom[mask]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: raw RMS kernel
# ─────────────────────────────────────────────────────────────────────────────

def _rms_librosa_python(S, frame_length, dtype=np.float32):
    """Mirror of the librosa Python path (abs2 + two row mults + sum + sqrt).

    Used as a fair baseline so Section 1 measures the Rust kernel against the
    *actual* Python implementation rather than a hand-written naive loop.
    """
    from librosa import util as _util
    x = _util.abs2(S, dtype=dtype)
    x[0] *= 0.5
    if frame_length % 2 == 0:
        x[-1] *= 0.5
    power = 2.0 * np.sum(x, axis=0, keepdims=True) / frame_length ** 2
    return np.sqrt(power)


def bench_raw_rms():
    _section(
        "Section 1: raw RMS kernel  (rms_spectrogram_f32 / _f64 vs NumPy)\n"
        "  Baselines: naive scalar loop  AND  librosa-style vectorised NumPy"
    )

    if not RUST_AVAILABLE:
        print("  [SKIP] Rust extension not available.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        frame_length = n_fft
        S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        S64 = S32.astype(np.float64)
        S32_c = np.ascontiguousarray(S32)
        S64_c = np.ascontiguousarray(S64)

        print(f"\n  case: n_fft={n_fft}, bins={n_bins}, frames={n_frames}")

        # f32
        if hasattr(_rust_ext, "rms_spectrogram_f32"):
            avg_naive, min_naive = _timeit(lambda: _rms_numpy(S32_c, frame_length))
            avg_li, min_li = _timeit(
                lambda: _rms_librosa_python(S32_c, frame_length, np.float32)
            )
            avg_rs, min_rs = _timeit(
                lambda: _rust_ext.rms_spectrogram_f32(S32_c, frame_length)
            )
            print("  [f32]")
            _row("naive scalar NumPy", avg_naive, min_naive)
            _row("librosa-style NumPy", avg_li, min_li)
            _row("Rust f32 kernel", avg_rs, min_rs)
            print(f"    {'vs naive (min)':28s}  {min_naive / min_rs:8.2f}x")
            print(f"    {'vs librosa NumPy (min)':28s}  {min_li / min_rs:8.2f}x")
        else:
            print("  rms_spectrogram_f32 not registered – skipping")

        # f64
        if hasattr(_rust_ext, "rms_spectrogram_f64"):
            avg_naive, min_naive = _timeit(lambda: _rms_numpy(S64_c, frame_length))
            avg_li, min_li = _timeit(
                lambda: _rms_librosa_python(S64_c, frame_length, np.float64)
            )
            avg_rs, min_rs = _timeit(
                lambda: _rust_ext.rms_spectrogram_f64(S64_c, frame_length)
            )
            print("  [f64]")
            _row("naive scalar NumPy", avg_naive, min_naive)
            _row("librosa-style NumPy", avg_li, min_li)
            _row("Rust f64 kernel", avg_rs, min_rs)
            print(f"    {'vs naive (min)':28s}  {min_naive / min_rs:8.2f}x")
            print(f"    {'vs librosa NumPy (min)':28s}  {min_li / min_rs:8.2f}x")
        else:
            print("  rms_spectrogram_f64 not registered – skipping")


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: public rms(S=…) — overhead analysis
# ─────────────────────────────────────────────────────────────────────────────

def bench_public_rms(iron_librosa):
    _section(
        "Section 2: public rms(S=…) — kernel vs API overhead\n"
        "  (librosa in this repo is the Rust-patched package;\n"
        "   this section measures dispatch overhead over raw kernel cost)"
    )

    if not RUST_AVAILABLE:
        print("  [SKIP] Rust not available.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        frame_length = n_fft
        S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        S64 = S32.astype(np.float64)

        for S, dtype_label, kern_name in [
            (S32, "float32", "rms_spectrogram_f32"),
            (S64, "float64", "rms_spectrogram_f64"),
        ]:
            Sc = np.ascontiguousarray(S)
            kern = getattr(_rust_ext, kern_name, None)
            if kern is None:
                print(f"  {kern_name} not registered – skipping")
                continue

            print(f"\n  case: n_fft={n_fft}, frames={n_frames}, dtype={dtype_label}")

            _, min_kern = _timeit(lambda: kern(Sc, frame_length))
            _, min_api = _timeit(
                lambda: iron_librosa.feature.rms(S=Sc, frame_length=frame_length)
            )
            _, min_py = _timeit(lambda: _rms_librosa_python(Sc, frame_length, S.dtype))

            overhead_ms = (min_api - min_kern) * 1e3
            pct = min_kern / min_api * 100

            _row(f"raw Rust kernel ({kern_name[-3:]})", 0.0, min_kern)
            _row("librosa-style Python path", 0.0, min_py)
            _row("public API (Rust active)", 0.0, min_api)
            print(f"    {'dispatch overhead':28s}  {overhead_ms:+8.3f} ms")
            print(f"    {'kernel share of API time':28s}  {pct:8.1f}%")
            print(f"    {'API vs Python path (min)':28s}  {min_py / min_api:8.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: raw spectral centroid kernel
# ─────────────────────────────────────────────────────────────────────────────

def _centroid_librosa_python(S, freq):
    """Mirror of the librosa Python path (util.normalize + weighted sum)."""
    from librosa import util as _util
    f = _util.expand_to(freq, ndim=S.ndim, axes=-2)
    return np.sum(f * _util.normalize(S, norm=1, axis=-2), axis=-2, keepdims=True)


def bench_raw_centroid():
    _section(
        "Section 3: raw spectral centroid kernel  "
        "(spectral_centroid_f32 / _f64 vs NumPy)\n"
        "  Baselines: naive vectorised NumPy  AND  librosa-style normalize+sum"
    )

    if not RUST_AVAILABLE:
        print("  [SKIP] Rust extension not available.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        freq = fft_frequencies(sr=SR, n_fft=n_fft)  # float64, shape (n_bins,)
        S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        S64 = S32.astype(np.float64)
        S32_c = np.ascontiguousarray(S32)
        S64_c = np.ascontiguousarray(S64)
        freq_c = np.ascontiguousarray(freq)

        print(f"\n  case: n_fft={n_fft}, bins={n_bins}, frames={n_frames}")

        # f32
        if hasattr(_rust_ext, "spectral_centroid_f32"):
            avg_naive, min_naive = _timeit(lambda: _centroid_numpy(S32_c, freq_c))
            avg_li, min_li = _timeit(lambda: _centroid_librosa_python(S32_c, freq_c))
            avg_rs, min_rs = _timeit(
                lambda: _rust_ext.spectral_centroid_f32(S32_c, freq_c)
            )
            print("  [f32 S → f64 output]")
            _row("naive vectorised NumPy", avg_naive, min_naive)
            _row("librosa-style NumPy", avg_li, min_li)
            _row("Rust f32 kernel", avg_rs, min_rs)
            print(f"    {'vs naive (min)':28s}  {min_naive / min_rs:8.2f}x")
            print(f"    {'vs librosa NumPy (min)':28s}  {min_li / min_rs:8.2f}x")
        else:
            print("  spectral_centroid_f32 not registered – skipping")

        # f64
        if hasattr(_rust_ext, "spectral_centroid_f64"):
            avg_naive, min_naive = _timeit(lambda: _centroid_numpy(S64_c, freq_c))
            avg_li, min_li = _timeit(lambda: _centroid_librosa_python(S64_c, freq_c))
            avg_rs, min_rs = _timeit(
                lambda: _rust_ext.spectral_centroid_f64(S64_c, freq_c)
            )
            print("  [f64 S → f64 output]")
            _row("naive vectorised NumPy", avg_naive, min_naive)
            _row("librosa-style NumPy", avg_li, min_li)
            _row("Rust f64 kernel", avg_rs, min_rs)
            print(f"    {'vs naive (min)':28s}  {min_naive / min_rs:8.2f}x")
            print(f"    {'vs librosa NumPy (min)':28s}  {min_li / min_rs:8.2f}x")
        else:
            print("  spectral_centroid_f64 not registered – skipping")


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: public spectral_centroid(S=…) — overhead analysis
# ─────────────────────────────────────────────────────────────────────────────

def bench_public_centroid(iron_librosa):
    _section(
        "Section 4: public spectral_centroid(S=…) — kernel vs API overhead"
    )

    if not RUST_AVAILABLE:
        print("  [SKIP] Rust not available.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        freq = fft_frequencies(sr=SR, n_fft=n_fft)
        S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        S64 = S32.astype(np.float64)

        for S, dtype_label, kern_name in [
            (S32, "float32", "spectral_centroid_f32"),
            (S64, "float64", "spectral_centroid_f64"),
        ]:
            Sc = np.ascontiguousarray(S)
            freq_c = np.ascontiguousarray(freq)
            kern = getattr(_rust_ext, kern_name, None)
            if kern is None:
                print(f"  {kern_name} not registered – skipping")
                continue

            print(f"\n  case: n_fft={n_fft}, frames={n_frames}, dtype={dtype_label}")

            _, min_kern = _timeit(lambda: kern(Sc, freq_c))
            _, min_api = _timeit(
                lambda: iron_librosa.feature.spectral_centroid(S=Sc, sr=SR, n_fft=n_fft)
            )
            _, min_py = _timeit(lambda: _centroid_librosa_python(Sc, freq_c))

            overhead_ms = (min_api - min_kern) * 1e3
            pct = min_kern / min_api * 100

            _row(f"raw Rust kernel ({kern_name[-3:]})", 0.0, min_kern)
            _row("librosa-style Python path", 0.0, min_py)
            _row("public API (Rust active)", 0.0, min_api)
            print(f"    {'dispatch overhead':28s}  {overhead_ms:+8.3f} ms")
            print(f"    {'kernel share of API time':28s}  {pct:8.1f}%")
            print(f"    {'API vs Python path (min)':28s}  {min_py / min_api:8.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: multichannel rms(S=…)
# ─────────────────────────────────────────────────────────────────────────────

def bench_multichannel_rms(iron_librosa):
    _section("Section 5: multichannel rms(S=…) — per-channel Rust dispatch overhead")

    n_fft, n_frames = 2048, 800
    n_bins = n_fft // 2 + 1

    for n_channels in [2, 4, 8]:
        S = np.abs(
            np.random.randn(n_channels, n_bins, n_frames).astype(np.float32)
        )
        Sc = np.ascontiguousarray(S)

        print(f"\n  case: channels={n_channels}, n_fft={n_fft}, frames={n_frames}, dtype=float32")

        # Python reference: manually loop over channels (same formula librosa uses)
        def py_ref(Sc=Sc, n_channels=n_channels):
            return [_rms_librosa_python(Sc[i], n_fft, np.float32) for i in range(n_channels)]

        _, min_py = _timeit(py_ref)
        _, min_api = _timeit(
            lambda: iron_librosa.feature.rms(S=Sc, frame_length=n_fft)
        )

        _row(f"Python path × {n_channels} channels", 0.0, min_py)
        _row("public API (Rust active)", 0.0, min_api)
        print(f"    {'API vs Python (min)':28s}  {min_py / min_api:8.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: multichannel spectral_centroid(S=…)
# ─────────────────────────────────────────────────────────────────────────────

def bench_multichannel_centroid(iron_librosa):
    _section(
        "Section 6: multichannel spectral_centroid(S=…) — per-channel Rust dispatch overhead"
    )

    n_fft, n_frames = 2048, 800
    n_bins = n_fft // 2 + 1
    freq = fft_frequencies(sr=SR, n_fft=n_fft)
    freq_c = np.ascontiguousarray(freq)

    for n_channels in [2, 4, 8]:
        S = np.abs(
            np.random.randn(n_channels, n_bins, n_frames).astype(np.float32)
        )
        Sc = np.ascontiguousarray(S)

        print(f"\n  case: channels={n_channels}, n_fft={n_fft}, frames={n_frames}, dtype=float32")

        def py_ref(Sc=Sc, n_channels=n_channels):
            return [_centroid_librosa_python(Sc[i], freq_c) for i in range(n_channels)]

        _, min_py = _timeit(py_ref)
        _, min_api = _timeit(
            lambda: iron_librosa.feature.spectral_centroid(S=Sc, sr=SR, n_fft=n_fft)
        )

        _row(f"Python path × {n_channels} channels", 0.0, min_py)
        _row("public API (Rust active)", 0.0, min_api)
        print(f"    {'API vs Python (min)':28s}  {min_py / min_api:8.2f}x")


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: fallback paths (Rust guard should not fire)
# ─────────────────────────────────────────────────────────────────────────────

def bench_fallback_paths(iron_librosa):
    _section(
        "Section 7: fallback paths — dtype guard (complex / int) confirms Python path"
    )

    n_fft = 2048
    n_bins = n_fft // 2 + 1
    n_frames = 400

    # ── RMS: float16 S trips the guard (S.dtype not in {float32, float64}) ──
    # Pre-convert outside the lambda so the dtype stays float16 inside the call.
    S_f16 = np.abs(np.random.randn(n_bins, n_frames)).astype(np.float16)

    print(f"\n  RMS fallback: float16 S (S.dtype not in {{float32, float64}} → Python path)")
    avg_py, min_py = _timeit(
        lambda: librosa.feature.rms(S=S_f16, frame_length=n_fft)
    )
    avg_rs, min_rs = _timeit(
        lambda: iron_librosa.feature.rms(S=S_f16, frame_length=n_fft)
    )
    _row("librosa  (Python)", avg_py, min_py)
    _row("iron-librosa (Rust)", avg_rs, min_rs)
    ratio = min_py / min_rs if min_rs > 0 else float("nan")
    print(f"    {'ratio (≈1.0 expected)':28s}  {ratio:8.2f}x")

    # ── spectral_centroid with 2-D freq (variable grid → Python fallback) ──
    freq_2d = np.tile(
        fft_frequencies(sr=SR, n_fft=n_fft)[:, None], (1, n_frames)
    )
    S32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
    print(f"\n  spectral_centroid fallback: 2-D freq array (variable grid, no Rust fast-path)")
    avg_py, min_py = _timeit(
        lambda: librosa.feature.spectral_centroid(S=S32, sr=SR, n_fft=n_fft, freq=freq_2d)
    )
    avg_rs, min_rs = _timeit(
        lambda: iron_librosa.feature.spectral_centroid(
            S=S32, sr=SR, n_fft=n_fft, freq=freq_2d
        )
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
    print(f"Phase 4B benchmark  ({N_RUNS} timed runs after warmup)")
    print(f"Rust available: {RUST_AVAILABLE}")
    if RUST_AVAILABLE:
        kernels = [
            "rms_spectrogram_f32",
            "rms_spectrogram_f64",
            "spectral_centroid_f32",
            "spectral_centroid_f64",
        ]
        for k in kernels:
            status = "✓" if hasattr(_rust_ext, k) else "✗ MISSING"
            print(f"  {k:35s}  {status}")
    print("=" * 72)

    bench_raw_rms()
    bench_public_rms(iron_librosa)
    bench_raw_centroid()
    bench_public_centroid(iron_librosa)
    bench_multichannel_rms(iron_librosa)
    bench_multichannel_centroid(iron_librosa)
    bench_fallback_paths(iron_librosa)

    print()
    print("=" * 72)
    print("Phase 4B benchmark complete.")
    print("=" * 72)

