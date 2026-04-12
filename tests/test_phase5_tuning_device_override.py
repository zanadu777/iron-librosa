#!/usr/bin/env python
"""Rust-device override parity checks for tuning/piptrack kernels."""

import numpy as np
import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_piptrack_from_spectrogram_f32_device_override_matches_cpu(monkeypatch):
    rng = np.random.default_rng(55001)
    s = np.abs(rng.standard_normal((513, 120))).astype(np.float32)
    shift = rng.standard_normal((513, 120)).astype(np.float32) * 0.01
    dskew = rng.standard_normal((513, 120)).astype(np.float32) * 0.01
    ref_values = np.percentile(s, 75.0, axis=0).astype(np.float32)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    pitch_cpu, mag_cpu = _rust_ext.piptrack_from_spectrogram_f32(
        np.ascontiguousarray(s),
        np.ascontiguousarray(shift),
        np.ascontiguousarray(dskew),
        np.ascontiguousarray(ref_values),
        0,
        s.shape[0],
        22050.0 / 1024.0,
    )

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    pitch_gpu, mag_gpu = _rust_ext.piptrack_from_spectrogram_f32(
        np.ascontiguousarray(s),
        np.ascontiguousarray(shift),
        np.ascontiguousarray(dskew),
        np.ascontiguousarray(ref_values),
        0,
        s.shape[0],
        22050.0 / 1024.0,
    )

    np.testing.assert_allclose(pitch_cpu, pitch_gpu, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(mag_cpu, mag_gpu, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_piptrack_from_spectrogram_f32_device_override_preserves_shape_errors(monkeypatch):
    rng = np.random.default_rng(55002)
    s = np.abs(rng.standard_normal((513, 40))).astype(np.float32)
    shift_bad = rng.standard_normal((256, 40)).astype(np.float32)
    dskew = rng.standard_normal((513, 40)).astype(np.float32)
    ref_values = np.percentile(s, 75.0, axis=0).astype(np.float32)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    with pytest.raises(ValueError, match="s, shift, and dskew must have the same shape"):
        _rust_ext.piptrack_from_spectrogram_f32(
            np.ascontiguousarray(s),
            np.ascontiguousarray(shift_bad),
            np.ascontiguousarray(dskew),
            np.ascontiguousarray(ref_values),
            0,
            s.shape[0],
            22050.0 / 1024.0,
        )

