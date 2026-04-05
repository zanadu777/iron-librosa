#!/usr/bin/env python
"""
Phase 15 parity tests: onset_flux_median_ref and tempogram_ac kernels.

Tests verify:
  1. New Rust symbols are present in the extension.
  2. onset_flux_median_ref_f32/f64 matches np.median(max(0, S-ref)) exactly.
  3. tempogram_ac_f32/f64 matches librosa.core.audio.autocorrelate within tolerance.
  4. beat_track end-to-end parity with Rust dispatch enabled.
  5. Fallback works when FORCE_NUMPY_BEAT is set.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.fft

import librosa
from librosa._rust_bridge import _rust_ext, RUST_EXTENSION_AVAILABLE
from librosa.core.audio import autocorrelate

skip_no_rust = pytest.mark.skipif(
    not RUST_EXTENSION_AVAILABLE, reason="Rust extension not installed"
)

SR = 22050
HOP = 512


# ── helpers ──────────────────────────────────────────────────────────────────

def _noisy(sec: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, sec, sec * SR, dtype=np.float32)
    y = 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    return (y + 0.2 * rng.standard_normal(sec * SR).astype(np.float32))


def _run_beat_track(y: np.ndarray, rust: bool):
    """Run beat_track with Rust dispatch on or off."""
    import librosa.beat as bm
    old_np, old_rs, old_av = bm.FORCE_NUMPY_BEAT, bm.FORCE_RUST_BEAT, bm.RUST_AVAILABLE
    try:
        bm.FORCE_NUMPY_BEAT = not rust
        bm.FORCE_RUST_BEAT = rust
        bm.RUST_AVAILABLE = True
        return librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
    finally:
        bm.FORCE_NUMPY_BEAT = old_np
        bm.FORCE_RUST_BEAT = old_rs
        bm.RUST_AVAILABLE = old_av


# ── symbol presence ──────────────────────────────────────────────────────────

@skip_no_rust
def test_phase15_symbols_present():
    assert hasattr(_rust_ext, "onset_flux_median_ref_f32")
    assert hasattr(_rust_ext, "onset_flux_median_ref_f64")
    assert hasattr(_rust_ext, "tempogram_ac_f32")
    assert hasattr(_rust_ext, "tempogram_ac_f64")


# ── onset_flux_median_ref parity ─────────────────────────────────────────────

@skip_no_rust
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_onset_flux_median_ref_parity(dtype):
    rng = np.random.default_rng(99)
    S = rng.random((128, 400)).astype(dtype)
    S_c = np.ascontiguousarray(S)

    lag = 1
    flux = np.maximum(0.0, S[:, lag:] - S[:, :-lag])
    py_out = np.median(flux, axis=0).astype(dtype)

    suffix = "f32" if dtype == np.float32 else "f64"
    fn = getattr(_rust_ext, f"onset_flux_median_ref_{suffix}")
    rs_out = fn(S_c, S_c, lag).ravel().astype(dtype)

    tol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(rs_out, py_out, rtol=tol, atol=tol)


@skip_no_rust
def test_onset_flux_median_ref_empty():
    """Empty or zero-lag inputs should not raise."""
    S = np.zeros((128, 0), dtype=np.float32)
    out = _rust_ext.onset_flux_median_ref_f32(S, S, 1)
    assert out.shape == (1, 0)


@skip_no_rust
def test_onset_flux_median_ref_single_frame():
    """Single-frame input with lag=1 should produce empty output."""
    S = np.ones((128, 1), dtype=np.float32)
    out = _rust_ext.onset_flux_median_ref_f32(S, S, 1)
    assert out.shape == (1, 0)


# ── tempogram_ac parity ───────────────────────────────────────────────────────

@skip_no_rust
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tempogram_ac_parity(dtype):
    rng = np.random.default_rng(77)
    win_len = 80
    n_frames = 300
    W = rng.random((win_len, n_frames)).astype(dtype)
    W_c = np.ascontiguousarray(W)

    py_ac = autocorrelate(W, axis=-2)

    n_pad = int(scipy.fft.next_fast_len(2 * win_len - 1, real=True))
    suffix = "f32" if dtype == np.float32 else "f64"
    fn = getattr(_rust_ext, f"tempogram_ac_{suffix}")
    rs_ac = fn(W_c, n_pad)

    # f32 has limited precision; f64 should be very tight
    atol = 1e-4 if dtype == np.float32 else 1e-9
    np.testing.assert_allclose(rs_ac, py_ac, atol=atol, rtol=atol)


@skip_no_rust
def test_tempogram_ac_zero_input():
    W = np.zeros((50, 100), dtype=np.float32)
    n_pad = int(scipy.fft.next_fast_len(2 * 50 - 1, real=True))
    out = _rust_ext.tempogram_ac_f32(np.ascontiguousarray(W), n_pad)
    assert out.shape == (50, 100)
    assert np.allclose(out, 0.0)


# ── end-to-end beat_track parity ─────────────────────────────────────────────

@skip_no_rust
def test_phase15_beat_track_end_to_end_parity():
    """beat_track with RUST_DISPATCH=1 should produce identical beats to numpy."""
    import os, importlib
    old_disp = os.environ.get("IRON_LIBROSA_RUST_DISPATCH", None)
    try:
        os.environ["IRON_LIBROSA_RUST_DISPATCH"] = "1"
        import librosa._rust_bridge as brg
        importlib.reload(brg)  # pick up env change
        # Re-import modules that read RUST_AVAILABLE at import time
        import librosa.onset as ons
        import librosa.feature.rhythm as rhy
        ons.RUST_AVAILABLE = brg.RUST_AVAILABLE
        rhy.RUST_AVAILABLE = brg.RUST_AVAILABLE

        y = _noisy(sec=10, seed=42)
        bpm_np, beats_np = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)

        # Force Rust DP too
        bm = librosa.beat
        old_np, old_rs, old_av = bm.FORCE_NUMPY_BEAT, bm.FORCE_RUST_BEAT, bm.RUST_AVAILABLE
        bm.FORCE_RUST_BEAT = True
        bm.FORCE_NUMPY_BEAT = False
        bm.RUST_AVAILABLE = True
        try:
            bpm_rs, beats_rs = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
        finally:
            bm.FORCE_NUMPY_BEAT = old_np
            bm.FORCE_RUST_BEAT = old_rs
            bm.RUST_AVAILABLE = old_av

        np.testing.assert_allclose(
            np.asarray(bpm_rs), np.asarray(bpm_np), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_array_equal(beats_rs, beats_np)
    finally:
        if old_disp is None:
            os.environ.pop("IRON_LIBROSA_RUST_DISPATCH", None)
        else:
            os.environ["IRON_LIBROSA_RUST_DISPATCH"] = old_disp


# ── onset_strength dispatch parity ───────────────────────────────────────────

@skip_no_rust
def test_phase15_onset_strength_median_dispatch_parity():
    """onset_strength(aggregate=np.median) should give same result with/without Rust."""
    import librosa.onset as ons
    y = _noisy(sec=10, seed=7)

    old_av = ons.RUST_AVAILABLE
    try:
        ons.RUST_AVAILABLE = False
        env_np = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, aggregate=np.median)

        ons.RUST_AVAILABLE = True
        env_rs = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP, aggregate=np.median)
    finally:
        ons.RUST_AVAILABLE = old_av

    np.testing.assert_allclose(env_rs, env_np, rtol=1e-5, atol=1e-5)

