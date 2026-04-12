#!/usr/bin/env python
"""Rust-device override parity checks for STFT kernels."""

import numpy as np
import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_stft_complex_f32_device_override_matches_cpu(monkeypatch):
    rng = np.random.default_rng(42001)
    y = rng.standard_normal(4096).astype(np.float32)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.stft_complex(y, 1024, 256, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.stft_complex(y, 1024, 256, True, None)

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_stft_power_f64_device_override_matches_cpu(monkeypatch):
    rng = np.random.default_rng(42002)
    y = rng.standard_normal(4096).astype(np.float64)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.stft_power_f64(y, 1024, 256, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.stft_power_f64(y, 1024, 256, True, None)

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-12, atol=1e-12)

