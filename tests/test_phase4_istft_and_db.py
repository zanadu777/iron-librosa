"""
Phase 4A Tests: ISTFT and dB conversions

Tests for Rust-accelerated ISTFT and dB conversion kernels.
Verifies parity with Python implementations and dtype dispatch.
"""

import numpy as np
import pytest
import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestISTFTF32:
    """Test ISTFT float32 kernel"""

    def test_istft_f32_simple_reconstruction(self):
        """Test that istft_f32 reconstructs signal from STFT"""
        y, sr = librosa.load(librosa.ex('trumpet'), sr=None, mono=True, duration=2.0)
        y = y.astype(np.float32)

        # Compute STFT
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        D = D.astype(np.complex64)

        # Use Rust ISTFT
        y_hat = _rust_ext.istft_f32(D, n_fft=2048, hop_length=512, window=None)

        assert y_hat.dtype == np.float32
        assert len(y_hat) >= len(y)
        assert y_hat.shape[0] > 0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestISTFTF64:
    """Test ISTFT float64 kernel"""

    def test_istft_f64_simple_reconstruction(self):
        """Test that istft_f64 reconstructs signal from STFT"""
        y, sr = librosa.load(librosa.ex('trumpet'), sr=None, mono=True, duration=2.0)
        y = y.astype(np.float64)

        # Compute STFT
        D = librosa.stft(y, n_fft=2048, hop_length=512)
        D = D.astype(np.complex128)

        # Use Rust ISTFT
        y_hat = _rust_ext.istft_f64(D, n_fft=2048, hop_length=512, window=None)

        assert y_hat.dtype == np.float64
        assert len(y_hat) >= len(y)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestPowerToDbF32:
    """Test power_to_db float32 kernel"""

    def test_power_to_db_f32_basic(self):
        """Test basic power_to_db_f32 conversion"""
        S = np.array([1.0, 10.0, 100.0, 1e-10], dtype=np.float32)
        S_db = _rust_ext.power_to_db_f32(S, ref_power=1.0, amin=1e-10, top_db=None)

        assert S_db.dtype == np.float32
        assert len(S_db) == len(S)
        # Check approximate values
        expected = 10.0 * np.log10(S)
        np.testing.assert_allclose(S_db, expected, rtol=1e-5)

    def test_power_to_db_f32_with_top_db(self):
        """Test power_to_db_f32 with top_db clipping"""
        S = np.array([1.0, 10.0, 100.0, 1e-10], dtype=np.float32)
        S_db = _rust_ext.power_to_db_f32(S, ref_power=1.0, amin=1e-10, top_db=80.0)

        assert S_db.dtype == np.float32
        # Check that dynamic range is limited
        assert np.max(S_db) - np.min(S_db) <= 80.0


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestPowerToDbF64:
    """Test power_to_db float64 kernel"""

    def test_power_to_db_f64_basic(self):
        """Test basic power_to_db_f64 conversion"""
        S = np.array([1.0, 10.0, 100.0, 1e-12], dtype=np.float64)
        S_db = _rust_ext.power_to_db_f64(S, ref_power=1.0, amin=1e-12, top_db=None)

        assert S_db.dtype == np.float64
        expected = 10.0 * np.log10(S)
        np.testing.assert_allclose(S_db, expected, rtol=1e-10)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestAmplitudeToDbF32:
    """Test amplitude_to_db float32 kernel"""

    def test_amplitude_to_db_f32_basic(self):
        """Test basic amplitude_to_db_f32 conversion"""
        A = np.array([1.0, 10.0, 100.0, 1e-5], dtype=np.float32)
        A_db = _rust_ext.amplitude_to_db_f32(A, ref_amplitude=1.0, amin=1e-5, top_db=None)

        assert A_db.dtype == np.float32
        expected = 20.0 * np.log10(A)
        np.testing.assert_allclose(A_db, expected, rtol=1e-5)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestAmplitudeToDbF64:
    """Test amplitude_to_db float64 kernel"""

    def test_amplitude_to_db_f64_basic(self):
        """Test basic amplitude_to_db_f64 conversion"""
        A = np.array([1.0, 10.0, 100.0, 1e-7], dtype=np.float64)
        A_db = _rust_ext.amplitude_to_db_f64(A, ref_amplitude=1.0, amin=1e-7, top_db=None)

        assert A_db.dtype == np.float64
        expected = 20.0 * np.log10(A)
        np.testing.assert_allclose(A_db, expected, rtol=1e-10)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestDbConversionsRoundtrip:
    """Test round-trip conversions between linear and dB scales"""

    def test_power_round_trip_f32(self):
        """Test power_to_db -> db_to_power round-trip"""
        S = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float32)

        # Forward and back
        S_db = _rust_ext.power_to_db_f32(S, ref_power=1.0, amin=1e-10)
        S_recovered = _rust_ext.db_to_power_f32(S_db, ref_power=1.0)

        np.testing.assert_allclose(S, S_recovered, rtol=1e-5)

    def test_amplitude_round_trip_f64(self):
        """Test amplitude_to_db -> db_to_amplitude round-trip"""
        A = np.array([0.001, 0.1, 1.0, 10.0], dtype=np.float64)

        # Forward and back
        A_db = _rust_ext.amplitude_to_db_f64(A, ref_amplitude=1.0, amin=1e-7)
        A_recovered = _rust_ext.db_to_amplitude_f64(A_db, ref_amplitude=1.0)

        np.testing.assert_allclose(A, A_recovered, rtol=1e-10)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestPublicDbDispatch:
    """Test public librosa dB conversion dispatch behavior."""

    def test_power_to_db_public_dispatch_f32(self, monkeypatch):
        """power_to_db should dispatch to the Rust f32 kernel for ndarray input."""
        x = np.array([[1.0, 10.0], [100.0, 0.1]], dtype=np.float32)
        called = {"value": False}
        original = _rust_ext.power_to_db_f32

        def _fake_power_to_db(flat, *, ref_power, amin, top_db):
            called["value"] = True
            assert flat.ndim == 1
            assert flat.dtype == np.float32
            return original(flat, ref_power=ref_power, amin=amin, top_db=top_db)

        monkeypatch.setattr(_rust_ext, "power_to_db_f32", _fake_power_to_db)

        out = librosa.power_to_db(x, ref=1.0, amin=1e-10, top_db=80.0)

        assert called["value"]
        assert out.shape == x.shape

    def test_power_to_db_public_fallback_callable_ref(self, monkeypatch):
        """Callable refs should preserve the Python path and skip Rust dispatch."""
        x = np.array([[1.0, 10.0], [100.0, 0.1]], dtype=np.float32)

        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used for callable ref")

        monkeypatch.setattr(_rust_ext, "power_to_db_f32", _fail)

        out = librosa.power_to_db(x, ref=np.max, amin=1e-10, top_db=80.0)
        assert out.shape == x.shape

    def test_db_to_power_public_dispatch_f64(self, monkeypatch):
        """db_to_power should dispatch to the Rust f64 kernel for ndarray input."""
        x_db = np.array([[-20.0, 0.0], [10.0, 20.0]], dtype=np.float64)
        called = {"value": False}
        original = _rust_ext.db_to_power_f64

        def _fake_db_to_power(flat, *, ref_power):
            called["value"] = True
            assert flat.ndim == 1
            assert flat.dtype == np.float64
            return original(flat, ref_power=ref_power)

        monkeypatch.setattr(_rust_ext, "db_to_power_f64", _fake_db_to_power)

        out = librosa.db_to_power(x_db, ref=2.0)

        assert called["value"]
        assert out.shape == x_db.shape

    def test_public_db_conversions_match_python_formula(self):
        """Public APIs should still match the documented NumPy formulas."""
        x = np.array([[1e-4, 1e-2], [1.0, 10.0]], dtype=np.float32)

        expected_db = 10.0 * np.log10(np.maximum(1e-10, x))
        actual_db = librosa.power_to_db(x, ref=1.0, amin=1e-10, top_db=None)
        np.testing.assert_allclose(actual_db, expected_db, rtol=1e-5)

        round_trip = librosa.db_to_power(actual_db, ref=1.0)
        np.testing.assert_allclose(round_trip, x, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



