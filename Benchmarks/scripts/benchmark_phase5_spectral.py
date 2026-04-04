"""Benchmark harness for Phase 5 kickoff: spectral_rolloff + spectral_bandwidth.

Sections
--------
  1. Raw spectral_rolloff kernels vs NumPy reference
  2. Raw spectral_bandwidth kernels vs NumPy reference (manual + fused)
  3. Public API rolloff / bandwidth dispatch timing (centroid omitted/provided)
  4. Public API rms(y=...) Rust vs forced Python fallback
  5. Public API spectral_flatness Rust vs forced Python fallback (Phase 6)
  6. Public API spectral_contrast Rust vs forced Python fallback (Phase 7)
"""

import importlib
import time
from contextlib import contextmanager

import numpy as np
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE

N_RUNS = 10
SR = 22050
SIZES = [(1024, 300), (2048, 800), (4096, 1200)]


def _timeit(fn):
    fn()
    vals = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return float(np.mean(vals)), float(np.min(vals))


def _section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _row(label, avg, minv):
    print(f"  {label:<24s} avg={avg*1e3:8.3f} ms  min={minv*1e3:8.3f} ms")


@contextmanager
def _force_python_fallback(enabled):
    """Temporarily disable spectral Rust dispatch to measure pure Python path."""
    spectral_mod = importlib.import_module("librosa.feature.spectral")
    prev_available = spectral_mod.RUST_AVAILABLE
    prev_ext = spectral_mod._rust_ext
    try:
        if enabled:
            spectral_mod.RUST_AVAILABLE = False
            spectral_mod._rust_ext = None
        yield
    finally:
        spectral_mod.RUST_AVAILABLE = prev_available
        spectral_mod._rust_ext = prev_ext


def _rolloff_numpy(S, freq, roll_percent):
    total_energy = np.cumsum(S, axis=-2)
    threshold = np.expand_dims(roll_percent * total_energy[-1, :], axis=-2)
    ind = np.where(total_energy < threshold, np.nan, 1)
    return np.nanmin(ind * freq[:, None], axis=-2, keepdims=True)


def _bandwidth_numpy(S, freq, p=2.0):
    centroid = np.sum(freq[:, None] * librosa.util.normalize(S, norm=1, axis=-2), axis=-2, keepdims=True)
    deviation = np.abs(np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1))
    return np.sum(librosa.util.normalize(S, norm=1, axis=-2) * deviation**p, axis=-2, keepdims=True) ** (1.0 / p)


def bench_raw_rolloff():
    _section("Section 1: raw spectral_rolloff kernel")
    if not RUST_AVAILABLE:
        print("Rust extension unavailable; skipping.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        freq = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
        s32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        s64 = s32.astype(np.float64)

        print(f"\ncase: n_fft={n_fft}, bins={n_bins}, frames={n_frames}")
        _, t_np = _timeit(lambda: _rolloff_numpy(s32, freq, 0.85))
        _, t32 = _timeit(lambda: _rust_ext.spectral_rolloff_f32(s32, freq, 0.85))
        _, t64 = _timeit(lambda: _rust_ext.spectral_rolloff_f64(s64, freq, 0.85))
        _row("numpy", 0.0, t_np)
        _row("rust f32", 0.0, t32)
        _row("rust f64", 0.0, t64)
        print(f"  speedup f32 (min)      {t_np/t32:8.2f}x")


def bench_raw_bandwidth():
    _section("Section 2: raw spectral_bandwidth kernel")
    if not RUST_AVAILABLE:
        print("Rust extension unavailable; skipping.")
        return

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        freq = librosa.fft_frequencies(sr=SR, n_fft=n_fft)
        s32 = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        s64 = s32.astype(np.float64)
        c32 = librosa.feature.spectral_centroid(S=s32, sr=SR, n_fft=n_fft)
        c64 = librosa.feature.spectral_centroid(S=s64, sr=SR, n_fft=n_fft)

        print(f"\ncase: n_fft={n_fft}, bins={n_bins}, frames={n_frames}")
        _, t_np = _timeit(lambda: _bandwidth_numpy(s32, freq, 2.0))
        _, t32 = _timeit(lambda: _rust_ext.spectral_bandwidth_f32(s32, freq, c32, True, 2.0))
        _, t32_auto = _timeit(lambda: _rust_ext.spectral_bandwidth_auto_centroid_f32(s32, freq, True, 2.0))
        _, t64 = _timeit(lambda: _rust_ext.spectral_bandwidth_f64(s64, freq, c64, True, 2.0))
        _row("numpy", 0.0, t_np)
        _row("rust f32 (manual)", 0.0, t32)
        _row("rust f32 (auto)", 0.0, t32_auto)
        _row("rust f64", 0.0, t64)
        print(f"  speedup f32 manual     {t_np/t32:8.2f}x")
        print(f"  speedup f32 auto       {t_np/t32_auto:8.2f}x")


def bench_public_api(iron_librosa):
    _section("Section 3: public API rolloff/bandwidth (auto vs provided centroid)")

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        s = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        print(f"\ncase: n_fft={n_fft}, frames={n_frames}")

        py_avg, py_min = _timeit(lambda: librosa.feature.spectral_rolloff(S=s, sr=SR, n_fft=n_fft))
        rs_avg, rs_min = _timeit(lambda: iron_librosa.feature.spectral_rolloff(S=s, sr=SR, n_fft=n_fft))
        _row("rolloff librosa", py_avg, py_min)
        _row("rolloff iron", rs_avg, rs_min)

        py_avg, py_min = _timeit(lambda: librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft))
        rs_avg, rs_min = _timeit(lambda: iron_librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft))
        _row("bandwidth librosa (auto)", py_avg, py_min)
        _row("bandwidth iron (auto)", rs_avg, rs_min)

        c = librosa.feature.spectral_centroid(S=s, sr=SR, n_fft=n_fft)
        py_avg, py_min = _timeit(lambda: librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft, centroid=c))
        rs_avg, rs_min = _timeit(lambda: iron_librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft, centroid=c))
        _row("bandwidth librosa (provided c)", py_avg, py_min)
        _row("bandwidth iron (provided c)", rs_avg, rs_min)


def bench_public_forced_baseline():
    _section("Section 4: public API Rust vs forced Python fallback")

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        s = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        c = librosa.feature.spectral_centroid(S=s, sr=SR, n_fft=n_fft)
        print(f"\ncase: n_fft={n_fft}, frames={n_frames}")

        with _force_python_fallback(False):
            rs_avg, rs_min = _timeit(lambda: librosa.feature.spectral_rolloff(S=s, sr=SR, n_fft=n_fft))
            bw_auto_rs_avg, bw_auto_rs_min = _timeit(
                lambda: librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft)
            )
            bw_prov_rs_avg, bw_prov_rs_min = _timeit(
                lambda: librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft, centroid=c)
            )

        with _force_python_fallback(True):
            py_avg, py_min = _timeit(lambda: librosa.feature.spectral_rolloff(S=s, sr=SR, n_fft=n_fft))
            bw_auto_py_avg, bw_auto_py_min = _timeit(
                lambda: librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft)
            )
            bw_prov_py_avg, bw_prov_py_min = _timeit(
                lambda: librosa.feature.spectral_bandwidth(S=s, sr=SR, n_fft=n_fft, centroid=c)
            )

        _row("rolloff rust on", rs_avg, rs_min)
        _row("rolloff forced py", py_avg, py_min)
        print(f"  speedup rolloff         {py_min/rs_min:8.2f}x")

        _row("bandwidth auto rust", bw_auto_rs_avg, bw_auto_rs_min)
        _row("bandwidth auto py", bw_auto_py_avg, bw_auto_py_min)
        print(f"  speedup bw auto         {bw_auto_py_min/bw_auto_rs_min:8.2f}x")

        _row("bandwidth prov rust", bw_prov_rs_avg, bw_prov_rs_min)
        _row("bandwidth prov py", bw_prov_py_avg, bw_prov_py_min)
        print(f"  speedup bw provided     {bw_prov_py_min/bw_prov_rs_min:8.2f}x")


def bench_public_rms_time_forced_baseline():
    _section("Section 5: public API rms(y=...) Rust vs forced Python fallback")
    spectral_mod = importlib.import_module("librosa.feature.spectral")

    for n_fft, n_frames in SIZES:
        hop = 512
        n_samples = n_fft + (n_frames - 1) * hop
        y = np.random.randn(n_samples).astype(np.float32)
        print(f"\ncase: frame_length={n_fft}, hop_length={hop}, samples={n_samples}")

        prev_rms_flag = getattr(spectral_mod, "_ENABLE_RUST_RMS_TIME", False)

        with _force_python_fallback(False):
            spectral_mod._ENABLE_RUST_RMS_TIME = True
            rs_avg, rs_min = _timeit(
                lambda: librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, dtype=np.float32)
            )
        with _force_python_fallback(True):
            spectral_mod._ENABLE_RUST_RMS_TIME = False
            py_avg, py_min = _timeit(
                lambda: librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop, dtype=np.float32)
            )
        spectral_mod._ENABLE_RUST_RMS_TIME = prev_rms_flag

        _row("rms(y) rust on", rs_avg, rs_min)
        _row("rms(y) forced py", py_avg, py_min)
        print(f"  speedup rms(y)          {py_min/rs_min:8.2f}x")


def bench_public_flatness_forced_baseline():
    _section("Section 6: public API spectral_flatness Rust vs forced Python fallback")

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        s = np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))
        print(f"\ncase: n_fft={n_fft}, bins={n_bins}, frames={n_frames}")

        with _force_python_fallback(False):
            rs_avg, rs_min = _timeit(
                lambda: librosa.feature.spectral_flatness(S=s)
            )
            rs_p1_avg, rs_p1_min = _timeit(
                lambda: librosa.feature.spectral_flatness(S=s, power=1.0)
            )

        with _force_python_fallback(True):
            py_avg, py_min = _timeit(
                lambda: librosa.feature.spectral_flatness(S=s)
            )
            py_p1_avg, py_p1_min = _timeit(
                lambda: librosa.feature.spectral_flatness(S=s, power=1.0)
            )

        _row("flatness rust (p=2)", rs_avg, rs_min)
        _row("flatness py   (p=2)", py_avg, py_min)
        print(f"  speedup p=2             {py_min/rs_min:8.2f}x")

        _row("flatness rust (p=1)", rs_p1_avg, rs_p1_min)
        _row("flatness py   (p=1)", py_p1_avg, py_p1_min)
        print(f"  speedup p=1             {py_p1_min/rs_p1_min:8.2f}x")


def bench_public_contrast_forced_baseline():
    _section("Section 7: public API spectral_contrast Rust vs forced Python fallback")

    for n_fft, n_frames in SIZES:
        n_bins = n_fft // 2 + 1
        cases = [
            ("mono", np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))),
            ("stereo", np.abs(np.random.randn(2, n_bins, n_frames).astype(np.float32))),
            ("quad", np.abs(np.random.randn(4, n_bins, n_frames).astype(np.float32))),
        ]

        for label, s in cases:
            print(f"\ncase: {label}, n_fft={n_fft}, bins={n_bins}, frames={n_frames}")

            with _force_python_fallback(False):
                rs_avg, rs_min = _timeit(
                    lambda: librosa.feature.spectral_contrast(S=s, sr=SR)
                )
                rs_q01_avg, rs_q01_min = _timeit(
                    lambda: librosa.feature.spectral_contrast(S=s, sr=SR, quantile=0.01)
                )

            with _force_python_fallback(True):
                py_avg, py_min = _timeit(
                    lambda: librosa.feature.spectral_contrast(S=s, sr=SR)
                )
                py_q01_avg, py_q01_min = _timeit(
                    lambda: librosa.feature.spectral_contrast(S=s, sr=SR, quantile=0.01)
                )

            _row("contrast rust (q=0.02)", rs_avg, rs_min)
            _row("contrast py   (q=0.02)", py_avg, py_min)
            print(f"  speedup q=0.02          {py_min/rs_min:8.2f}x")

            _row("contrast rust (q=0.01)", rs_q01_avg, rs_q01_min)
            _row("contrast py   (q=0.01)", py_q01_avg, py_q01_min)
            print(f"  speedup q=0.01          {py_q01_min/rs_q01_min:8.2f}x")


if __name__ == "__main__":
    np.random.seed(2051)
    iron_librosa = importlib.import_module("iron_librosa")

    print("=" * 72)
    print("Phase 5 kickoff benchmark (rolloff + bandwidth)")
    print(f"Rust available: {RUST_AVAILABLE}")
    print("=" * 72)

    bench_raw_rolloff()
    bench_raw_bandwidth()
    bench_public_api(iron_librosa)
    bench_public_forced_baseline()
    bench_public_rms_time_forced_baseline()
    bench_public_flatness_forced_baseline()
    bench_public_contrast_forced_baseline()

