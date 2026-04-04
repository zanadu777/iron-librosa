#!/usr/bin/env python
"""Phase 13 parity scaffolding for CQT/VQT Rust acceleration.

This file intentionally starts with Python-reference tests so we can lock
baseline behavior before wiring any Rust kernel dispatch.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

import librosa
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext
from librosa.core import constantq as constantq_mod


_PHASE13_RUST_SYMBOLS = (
    "cqt_project_f32",
    "cqt_project_f64",
    "vqt_project_f32",
    "vqt_project_f64",
)


def _phase13_rust_ready() -> bool:
    return bool(RUST_AVAILABLE and _rust_ext is not None and all(hasattr(_rust_ext, name) for name in _PHASE13_RUST_SYMBOLS))


def _phase13_dense_kwargs() -> dict:
    return {
        "sr": 22050,
        "hop_length": 512,
        "n_bins": 84,
        "bins_per_octave": 12,
        "sparsity": 0.0,
        "res_type": "soxr_hq",
    }


def _phase13_pseudo_cqt_kwargs() -> dict:
    kwargs = _phase13_dense_kwargs().copy()
    kwargs.pop("res_type", None)
    return kwargs


def _force_rust_backend(monkeypatch) -> None:
    monkeypatch.setattr(constantq_mod, "FORCE_NUMPY_CQT_VQT", False)
    monkeypatch.setattr(constantq_mod, "FORCE_RUST_CQT_VQT", True)


def _force_numpy_backend(monkeypatch) -> None:
    monkeypatch.setattr(constantq_mod, "FORCE_NUMPY_CQT_VQT", True)
    monkeypatch.setattr(constantq_mod, "FORCE_RUST_CQT_VQT", False)


def _stereo_signal(dtype) -> np.ndarray:
    y = _phase13_signal().astype(dtype)
    return np.stack([y, 0.5 * np.roll(y, 257)], axis=0).astype(dtype, copy=False)


def _backend_tolerances(dtype):
    if dtype == np.float32:
        return 1e-5, 1e-6
    return 1e-11, 1e-12


def _phase13_signal(sr: int = 22050, seconds: float = 3.0, seed: int = 13001) -> np.ndarray:
    """Deterministic test signal with harmonics + noise for parity checks."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.linspace(0.0, seconds, n, endpoint=False, dtype=np.float32)
    y = 0.65 * np.sin(2.0 * np.pi * 220.0 * t)
    y += 0.2 * np.sin(2.0 * np.pi * 440.0 * t)
    y += 0.08 * np.sin(2.0 * np.pi * 660.0 * t)
    y += 0.02 * rng.standard_normal(n).astype(np.float32)
    return y.astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_cqt_reference_deterministic(dtype):
    """Reference CQT path should be deterministic under fixed inputs/settings."""
    y = _phase13_signal().astype(dtype)

    cqt_a = librosa.cqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        sparsity=0.01,
        res_type="soxr_hq",
    )
    cqt_b = librosa.cqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        sparsity=0.01,
        res_type="soxr_hq",
    )

    assert cqt_a.shape == cqt_b.shape
    np.testing.assert_allclose(cqt_a, cqt_b, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_vqt_reference_deterministic(dtype):
    """Reference VQT path should be deterministic under fixed inputs/settings."""
    y = _phase13_signal().astype(dtype)

    vqt_a = librosa.vqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        gamma=0.0,
        sparsity=0.01,
        res_type="soxr_hq",
    )
    vqt_b = librosa.vqt(
        y=y,
        sr=22050,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
        gamma=0.0,
        sparsity=0.01,
        res_type="soxr_hq",
    )

    assert vqt_a.shape == vqt_b.shape
    np.testing.assert_allclose(vqt_a, vqt_b, rtol=1e-7, atol=1e-9)


def test_phase13_rust_kernel_symbols_present_when_extension_available():
    if not RUST_AVAILABLE:
        pytest.skip("Rust extension is not available in this environment")

    missing = [name for name in _PHASE13_RUST_SYMBOLS if not hasattr(_rust_ext, name)]
    assert not missing, f"Missing Phase 13 Rust symbols: {missing}"


def test_phase13_cqt_response_uses_rust_dense_projection(monkeypatch):
    d = np.array(
        [
            [1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j],
            [0.5 - 0.25j, -1.0 + 0.75j, 2.0 - 1.5j],
        ],
        dtype=np.complex64,
    )
    fft_basis = np.array(
        [
            [1.0 + 0.5j, 0.25 - 0.75j],
            [-0.5 + 0.0j, 1.25 + 0.25j],
        ],
        dtype=np.complex64,
    )
    expected = np.tensordot(fft_basis, d.reshape((1, 2, 3)), axes=(1, 1))
    expected = np.moveaxis(expected, 0, 1).reshape((2, 3))

    calls = {"n": 0}

    class _RustStub:
        def cqt_project_f32(self, dr, basis):
            calls["n"] += 1
            np.testing.assert_equal(dr.shape, (1, 2, 3))
            np.testing.assert_equal(basis.shape, (2, 2))
            out = np.tensordot(basis, dr, axes=(1, 1))
            return np.moveaxis(out, 0, 1)

    monkeypatch.setattr(constantq_mod, "stft", lambda *args, **kwargs: d)
    monkeypatch.setattr(constantq_mod, "RUST_AVAILABLE", True)
    monkeypatch.setattr(constantq_mod, "_rust_ext", _RustStub())
    _force_rust_backend(monkeypatch)

    out = constantq_mod.__cqt_response(
        np.zeros(32, dtype=np.float32),
        n_fft=4,
        hop_length=2,
        fft_basis=fft_basis,
        mode="constant",
        dtype=np.complex64,
    )

    assert calls["n"] == 1
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-7)


def test_phase13_cqt_response_dense_enough_sparse_basis_uses_rust(monkeypatch):
    d = np.array(
        [
            [1.0 + 0.0j, 0.5 + 0.5j],
            [0.25 - 0.75j, -1.0 + 0.0j],
        ],
        dtype=np.complex64,
    )
    fft_basis_dense = np.array(
        [
            [1.0 + 0.0j, 0.0 + 0.5j],
            [0.25 + 0.0j, -0.75 + 0.25j],
        ],
        dtype=np.complex64,
    )
    fft_basis = scipy.sparse.csr_matrix(fft_basis_dense)
    calls = {"n": 0}

    class _RustStub:
        def cqt_project_f32(self, dr, basis):
            calls["n"] += 1
            np.testing.assert_allclose(basis, fft_basis_dense)
            out = np.tensordot(basis, dr, axes=(1, 1))
            return np.moveaxis(out, 0, 1)

    monkeypatch.setattr(constantq_mod, "stft", lambda *args, **kwargs: d)
    monkeypatch.setattr(constantq_mod, "RUST_AVAILABLE", True)
    monkeypatch.setattr(constantq_mod, "_rust_ext", _RustStub())
    _force_rust_backend(monkeypatch)

    out = constantq_mod.__cqt_response(
        np.zeros(32, dtype=np.float32),
        n_fft=4,
        hop_length=2,
        fft_basis=fft_basis,
        mode="constant",
        dtype=np.complex64,
    )

    assert calls["n"] == 1
    expected = np.tensordot(fft_basis_dense, d.reshape((1, 2, 2)), axes=(1, 1))
    expected = np.moveaxis(expected, 0, 1).reshape((2, 2))
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-7)


def test_phase13_cqt_response_phase_false_keeps_numpy_dense_path(monkeypatch):
    d = np.array(
        [
            [1.0 + 2.0j, 3.0 + 4.0j],
            [0.5 - 0.25j, -1.0 + 0.75j],
        ],
        dtype=np.complex64,
    )
    fft_basis = np.array(
        [
            [1.0, 0.25],
            [-0.5, 1.25],
        ],
        dtype=np.float32,
    )
    expected = np.tensordot(fft_basis, np.abs(d).reshape((1, 2, 2)), axes=(1, 1))
    expected = np.moveaxis(expected, 0, 1).reshape((2, 2))

    calls = {"n": 0}

    class _RustStub:
        def cqt_project_f32(self, dr, basis):
            calls["n"] += 1
            raise AssertionError("magnitude-only pseudo-CQT should not dispatch to Rust")

    monkeypatch.setattr(constantq_mod, "stft", lambda *args, **kwargs: d)
    monkeypatch.setattr(constantq_mod, "RUST_AVAILABLE", True)
    monkeypatch.setattr(constantq_mod, "_rust_ext", _RustStub())
    _force_rust_backend(monkeypatch)

    out = constantq_mod.__cqt_response(
        np.zeros(32, dtype=np.float32),
        n_fft=4,
        hop_length=2,
        fft_basis=fft_basis,
        mode="constant",
        dtype=np.complex64,
        phase=False,
    )

    assert calls["n"] == 0
    np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.skipif(not _phase13_rust_ready(), reason="Phase 13 Rust kernels are not available")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_cqt_matches_python_backend(dtype, monkeypatch):
    y = _phase13_signal().astype(dtype)
    kwargs = _phase13_dense_kwargs()
    rtol, atol = _backend_tolerances(dtype)

    with monkeypatch.context() as mp:
        _force_numpy_backend(mp)
        reference = librosa.cqt(y=y, **kwargs)

    with monkeypatch.context() as mp:
        _force_rust_backend(mp)
        accelerated = librosa.cqt(y=y, **kwargs)

    np.testing.assert_allclose(accelerated, reference, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _phase13_rust_ready(), reason="Phase 13 Rust kernels are not available")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_vqt_matches_python_backend(dtype, monkeypatch):
    y = _phase13_signal().astype(dtype)
    kwargs = _phase13_dense_kwargs()
    rtol, atol = _backend_tolerances(dtype)

    with monkeypatch.context() as mp:
        _force_numpy_backend(mp)
        reference = librosa.vqt(y=y, gamma=0.0, **kwargs)

    with monkeypatch.context() as mp:
        _force_rust_backend(mp)
        accelerated = librosa.vqt(y=y, gamma=0.0, **kwargs)

    np.testing.assert_allclose(accelerated, reference, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _phase13_rust_ready(), reason="Phase 13 Rust kernels are not available")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_phase13_pseudo_cqt_matches_python_backend(dtype, monkeypatch):
    y = _phase13_signal().astype(dtype)
    kwargs = _phase13_pseudo_cqt_kwargs()
    rtol, atol = _backend_tolerances(dtype)

    with monkeypatch.context() as mp:
        _force_numpy_backend(mp)
        reference = librosa.pseudo_cqt(y=y, **kwargs)

    with monkeypatch.context() as mp:
        _force_rust_backend(mp)
        accelerated = librosa.pseudo_cqt(y=y, **kwargs)

    np.testing.assert_allclose(accelerated, reference, rtol=rtol, atol=atol)


@pytest.mark.skipif(not _phase13_rust_ready(), reason="Phase 13 Rust kernels are not available")
@pytest.mark.parametrize("transform", [librosa.cqt, librosa.vqt, librosa.pseudo_cqt])
def test_phase13_multichannel_matches_python_backend(transform, monkeypatch):
    y = _stereo_signal(np.float32)
    kwargs = (
        _phase13_pseudo_cqt_kwargs()
        if transform is librosa.pseudo_cqt
        else _phase13_dense_kwargs()
    )

    call_kwargs = dict(kwargs)
    if transform is librosa.vqt:
        call_kwargs["gamma"] = 0.0

    with monkeypatch.context() as mp:
        _force_numpy_backend(mp)
        reference = transform(y=y, **call_kwargs)

    with monkeypatch.context() as mp:
        _force_rust_backend(mp)
        accelerated = transform(y=y, **call_kwargs)

    np.testing.assert_allclose(accelerated, reference, rtol=1e-5, atol=1e-6)

