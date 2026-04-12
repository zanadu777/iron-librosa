#!/usr/bin/env python
"""Phase 17 Metal FFT dispatch safety checks.

These tests ensure AppleGpu requests preserve numerical parity while the
Metal FFT path is wired through CPU fallback for unsupported cases.
"""

import numpy as np
import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_stft_complex_f32_non_power_two_gpu_request_matches_cpu(monkeypatch):
    rng = np.random.default_rng(91701)
    y = rng.standard_normal(5000).astype(np.float32)

    # Use a non-power-of-two FFT size to validate fallback correctness.
    n_fft = 1000
    hop = 250

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.stft_complex(y, n_fft, hop, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.stft_complex(y, n_fft, hop, True, None)

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_istft_f32_gpu_request_matches_cpu(monkeypatch):
    rng = np.random.default_rng(91702)
    y = rng.standard_normal(4096).astype(np.float32)

    n_fft = 1024
    hop = 256

    # Build a consistent STFT input matrix once.
    stft_m = _rust_ext.stft_complex(y, n_fft, hop, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    y_cpu = _rust_ext.istft_f32(stft_m, n_fft, hop, None, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    y_gpu_req = _rust_ext.istft_f32(stft_m, n_fft, hop, None, None)

    np.testing.assert_allclose(y_cpu, y_gpu_req, rtol=5e-4, atol=5e-4)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_stft_complex_batch_f32_gpu_request_matches_cpu(monkeypatch):
    rng = np.random.default_rng(91703)
    y = rng.standard_normal((2, 4096)).astype(np.float32)

    n_fft = 1024
    hop = 256

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.stft_complex_batch(y, n_fft, hop, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.stft_complex_batch(y, n_fft, hop, True, None)

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_stft_complex_f32_experimental_metal_gate_executes(monkeypatch):
    rng = np.random.default_rng(91704)
    y = rng.standard_normal(4096).astype(np.float32)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.stft_complex(y, 1024, 256, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    monkeypatch.setenv("IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL", "force-on")
    out_exp = _rust_ext.stft_complex(y, 1024, 256, True, None)

    assert out_exp.shape == out_cpu.shape
    np.testing.assert_allclose(out_cpu, out_exp, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_istft_f32_experimental_metal_gate_executes(monkeypatch):
    rng = np.random.default_rng(91705)
    y = rng.standard_normal(4096).astype(np.float32)
    d = _rust_ext.stft_complex(y, 1024, 256, True, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    y_cpu = _rust_ext.istft_f32(d, 1024, 256, None, None)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    monkeypatch.setenv("IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL", "force-on")
    y_exp = _rust_ext.istft_f32(d, 1024, 256, None, None)

    assert y_exp.shape == y_cpu.shape
    np.testing.assert_allclose(y_cpu, y_exp, rtol=1e-2, atol=1e-2)


