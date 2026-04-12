#!/usr/bin/env python
"""Rust-device override parity checks for mel projection kernels."""

import builtins
import numpy as np
import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_mel_project_f32_device_override_matches_cpu(monkeypatch):
    rng = np.random.default_rng(44001)
    s = np.abs(rng.standard_normal((257, 96))).astype(np.float32)
    mel_basis = np.abs(rng.standard_normal((40, 257))).astype(np.float32)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.mel_project_f32(np.ascontiguousarray(s), np.ascontiguousarray(mel_basis))

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.mel_project_f32(np.ascontiguousarray(s), np.ascontiguousarray(mel_basis))

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_mel_project_f32_device_override_preserves_shape_errors(monkeypatch):
    rng = np.random.default_rng(44002)
    s = np.abs(rng.standard_normal((257, 16))).astype(np.float32)
    mel_basis_bad = np.abs(rng.standard_normal((40, 128))).astype(np.float32)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    with pytest.raises(builtins.ValueError, match="Incompatible shapes"):
        _rust_ext.mel_project_f32(
            np.ascontiguousarray(s),
            np.ascontiguousarray(mel_basis_bad),
        )


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_mel_project_f32_repeated_apple_gpu_requests_match_cpu(monkeypatch):
    rng = np.random.default_rng(44003)
    s = np.abs(rng.standard_normal((513, 64))).astype(np.float32)
    mel_basis = np.abs(rng.standard_normal((80, 513))).astype(np.float32)
    s = np.ascontiguousarray(s)
    mel_basis = np.ascontiguousarray(mel_basis)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.mel_project_f32(s, mel_basis)

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_first = _rust_ext.mel_project_f32(s, mel_basis)
    out_gpu_second = _rust_ext.mel_project_f32(s, mel_basis)

    np.testing.assert_allclose(out_cpu, out_gpu_first, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(out_gpu_first, out_gpu_second, rtol=1e-6, atol=1e-6)


