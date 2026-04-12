#!/usr/bin/env python
"""Rust-device override parity checks for Phase 13 CQT/VQT kernels."""

import numpy as np
import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cqt_project_f32_device_override_matches_cpu(monkeypatch):
    rng = np.random.default_rng(13013)
    d = (
        rng.standard_normal((2, 16, 20)).astype(np.float32)
        + 1j * rng.standard_normal((2, 16, 20)).astype(np.float32)
    )
    basis = (
        rng.standard_normal((12, 16)).astype(np.float32)
        + 1j * rng.standard_normal((12, 16)).astype(np.float32)
    )

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.cqt_project_f32(np.ascontiguousarray(d), np.ascontiguousarray(basis))

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.cqt_project_f32(np.ascontiguousarray(d), np.ascontiguousarray(basis))

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cqt_project_f64_device_override_matches_cpu(monkeypatch):
    rng = np.random.default_rng(13014)
    d = (
        rng.standard_normal((1, 10, 12)).astype(np.float64)
        + 1j * rng.standard_normal((1, 10, 12)).astype(np.float64)
    )
    basis = (
        rng.standard_normal((7, 10)).astype(np.float64)
        + 1j * rng.standard_normal((7, 10)).astype(np.float64)
    )

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    out_cpu = _rust_ext.cqt_project_f64(np.ascontiguousarray(d), np.ascontiguousarray(basis))

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_gpu_req = _rust_ext.cqt_project_f64(np.ascontiguousarray(d), np.ascontiguousarray(basis))

    np.testing.assert_allclose(out_cpu, out_gpu_req, rtol=1e-11, atol=1e-12)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cqt_project_f32_device_override_repeated_requests_stable(monkeypatch):
    rng = np.random.default_rng(13015)
    d = (
        rng.standard_normal((1, 64, 96)).astype(np.float32)
        + 1j * rng.standard_normal((1, 64, 96)).astype(np.float32)
    )
    basis = (
        rng.standard_normal((24, 64)).astype(np.float32)
        + 1j * rng.standard_normal((24, 64)).astype(np.float32)
    )

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    out_first = _rust_ext.cqt_project_f32(np.ascontiguousarray(d), np.ascontiguousarray(basis))
    out_second = _rust_ext.cqt_project_f32(np.ascontiguousarray(d), np.ascontiguousarray(basis))

    np.testing.assert_allclose(out_first, out_second, rtol=1e-6, atol=1e-6)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_cqt_project_f32_device_override_preserves_shape_errors(monkeypatch):
    rng = np.random.default_rng(13016)
    d = (
        rng.standard_normal((2, 32, 20)).astype(np.float32)
        + 1j * rng.standard_normal((2, 32, 20)).astype(np.float32)
    )
    basis_bad = (
        rng.standard_normal((12, 16)).astype(np.float32)
        + 1j * rng.standard_normal((12, 16)).astype(np.float32)
    )

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    with pytest.raises(ValueError, match="Incompatible shapes"):
        _rust_ext.cqt_project_f32(np.ascontiguousarray(d), np.ascontiguousarray(basis_bad))


