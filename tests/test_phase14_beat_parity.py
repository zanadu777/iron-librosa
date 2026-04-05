#!/usr/bin/env python

import numpy as np
import pytest

import librosa
from librosa import beat as beat_mod
from librosa._rust_bridge import _rust_ext, RUST_EXTENSION_AVAILABLE


SR = 22050
HOP = 512


def _click_track(seconds: int, bpm: float) -> np.ndarray:
    n = seconds * SR
    y = np.zeros(n, dtype=np.float32)
    interval = int((60.0 / bpm) * SR)
    width = min(256, max(16, interval // 4))
    for idx in range(0, n, interval):
        y[idx : idx + width] += np.hanning(width).astype(np.float32)
    return y


def _run_beat(y: np.ndarray, force_rust: bool, sparse: bool = True):
    old_numpy = beat_mod.FORCE_NUMPY_BEAT
    old_rust = beat_mod.FORCE_RUST_BEAT
    old_available = beat_mod.RUST_AVAILABLE
    try:
        beat_mod.FORCE_NUMPY_BEAT = not force_rust
        beat_mod.FORCE_RUST_BEAT = force_rust
        beat_mod.RUST_AVAILABLE = True
        return librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP, sparse=sparse)
    finally:
        beat_mod.FORCE_NUMPY_BEAT = old_numpy
        beat_mod.FORCE_RUST_BEAT = old_rust
        beat_mod.RUST_AVAILABLE = old_available


@pytest.mark.skipif(not RUST_EXTENSION_AVAILABLE, reason="Rust extension not installed")
def test_phase14_beat_dp_symbols_present_in_extension():
    assert _rust_ext is not None
    assert hasattr(_rust_ext, "beat_track_dp_f32")
    assert hasattr(_rust_ext, "beat_track_dp_f64")


@pytest.mark.skipif(not RUST_EXTENSION_AVAILABLE, reason="Rust extension not installed")
def test_phase14_beat_track_mono_numpy_vs_rust_parity():
    if _rust_ext is None or not hasattr(_rust_ext, "beat_track_dp_f32"):
        pytest.skip("Phase 14 beat DP symbols unavailable")

    y = _click_track(seconds=12, bpm=120.0)

    tempo_np, beats_np = _run_beat(y, force_rust=False, sparse=True)
    tempo_rs, beats_rs = _run_beat(y, force_rust=True, sparse=True)

    np.testing.assert_allclose(np.asarray(tempo_rs), np.asarray(tempo_np), rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(beats_rs, beats_np)


@pytest.mark.skipif(not RUST_EXTENSION_AVAILABLE, reason="Rust extension not installed")
def test_phase14_beat_track_multichannel_guarded_fallback_parity():
    if _rust_ext is None or not hasattr(_rust_ext, "beat_track_dp_f32"):
        pytest.skip("Phase 14 beat DP symbols unavailable")

    y = _click_track(seconds=8, bpm=100.0)
    y_stereo = np.stack([y, y], axis=0)

    tempo_np, beats_np = _run_beat(y_stereo, force_rust=False, sparse=False)
    tempo_rs, beats_rs = _run_beat(y_stereo, force_rust=True, sparse=False)

    # Multichannel path is currently guarded to Python fallback for DP.
    np.testing.assert_allclose(np.asarray(tempo_rs), np.asarray(tempo_np), rtol=1e-7, atol=1e-7)
    np.testing.assert_array_equal(beats_rs, beats_np)

