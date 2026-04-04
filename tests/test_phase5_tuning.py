"""Phase 5 tests: estimate_tuning acceleration helpers."""

import numpy as np
import pytest

import librosa
import librosa.core.pitch as pitch_module
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestTuningKernels:
    def test_estimate_tuning_from_piptrack_f32_matches_python_formula(self):
        rng = np.random.default_rng(2060)
        pitch = np.abs(rng.standard_normal((1025, 32), dtype=np.float32)) * 1000.0
        mag = np.abs(rng.standard_normal((1025, 32), dtype=np.float32))

        # Build Python reference with existing formula path
        pitch_mask = pitch > 0
        threshold = np.median(mag[pitch_mask]) if pitch_mask.any() else 0.0
        expected = librosa.pitch_tuning(
            pitch[(mag >= threshold) & pitch_mask],
            resolution=0.01,
            bins_per_octave=12,
        )

        actual = _rust_ext.estimate_tuning_from_piptrack_f32(pitch, mag, 0.01, 12)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)

    def test_estimate_tuning_from_piptrack_f64_matches_python_formula(self):
        rng = np.random.default_rng(2061)
        pitch = np.abs(rng.standard_normal((1025, 24))).astype(np.float64) * 800.0
        mag = np.abs(rng.standard_normal((1025, 24))).astype(np.float64)

        pitch_mask = pitch > 0
        threshold = np.median(mag[pitch_mask]) if pitch_mask.any() else 0.0
        expected = librosa.pitch_tuning(
            pitch[(mag >= threshold) & pitch_mask],
            resolution=0.02,
            bins_per_octave=12,
        )

        actual = _rust_ext.estimate_tuning_from_piptrack_f64(pitch, mag, 0.02, 12)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestEstimateTuningDispatch:
    def test_estimate_tuning_uses_rust_kernel(self, monkeypatch):
        rng = np.random.default_rng(2062)
        S = np.abs(rng.standard_normal((1025, 20), dtype=np.float32))
        calls = {"count": 0}
        original = _rust_ext.estimate_tuning_from_piptrack_f32

        monkeypatch.setattr(pitch_module, "_ENABLE_RUST_TUNING", True)
        monkeypatch.setattr(pitch_module, "_RUST_TUNING_MIN_WORK", 0)

        def _fake_kernel(pitch, mag, resolution, bins_per_octave):
            calls["count"] += 1
            assert pitch.dtype == np.float32
            assert mag.dtype == np.float32
            return original(pitch, mag, resolution, bins_per_octave)

        monkeypatch.setattr(_rust_ext, "estimate_tuning_from_piptrack_f32", _fake_kernel)

        tuning = librosa.estimate_tuning(S=S, sr=22050, n_fft=2048)
        assert isinstance(tuning, float)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestPiptrackDispatch:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("ref", [None, 1.0, np.max])
    def test_piptrack_rust_matches_python(self, monkeypatch, dtype, ref):
        rng = np.random.default_rng(2065)
        S = np.abs(rng.standard_normal((513, 80))).astype(dtype)

        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MODE", "rust")
        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MIN_WORK", 0)

        pitches_rust, mags_rust = librosa.piptrack(S=S, sr=22050, n_fft=1024, ref=ref)

        monkeypatch.setattr(pitch_module, "RUST_AVAILABLE", False)
        monkeypatch.setattr(pitch_module, "_rust_ext", None)
        pitches_py, mags_py = librosa.piptrack(S=S, sr=22050, n_fft=1024, ref=ref)

        np.testing.assert_allclose(pitches_rust, pitches_py, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(mags_rust, mags_py, rtol=1e-6, atol=1e-8)

    def test_piptrack_multichannel_rust_matches_python(self, monkeypatch):
        rng = np.random.default_rng(2066)
        S = np.abs(rng.standard_normal((2, 513, 60), dtype=np.float32))

        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MODE", "rust")
        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MIN_WORK", 0)
        pitches_rust, mags_rust = librosa.piptrack(S=S, sr=22050, n_fft=1024, ref=np.max)

        monkeypatch.setattr(pitch_module, "RUST_AVAILABLE", False)
        monkeypatch.setattr(pitch_module, "_rust_ext", None)
        pitches_py, mags_py = librosa.piptrack(S=S, sr=22050, n_fft=1024, ref=np.max)

        np.testing.assert_allclose(pitches_rust, pitches_py, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(mags_rust, mags_py, rtol=1e-6, atol=1e-8)

    def test_piptrack_respects_min_work_threshold(self, monkeypatch):
        rng = np.random.default_rng(2067)
        S = np.abs(rng.standard_normal((513, 20), dtype=np.float32))

        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MODE", "auto")
        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MIN_WORK", S.size + 1)

        def _fail(*args, **kwargs):
            raise AssertionError("Rust piptrack dispatch should not be used below threshold")

        monkeypatch.setattr(_rust_ext, "piptrack_from_spectrogram_f32", _fail)

        pitches, mags = librosa.piptrack(S=S, sr=22050, n_fft=1024)
        assert pitches.shape == mags.shape == S.shape

    def test_piptrack_uses_rust_kernel(self, monkeypatch):
        rng = np.random.default_rng(2068)
        S = np.abs(rng.standard_normal((513, 80), dtype=np.float32))
        calls = {"count": 0}
        original = _rust_ext.piptrack_from_spectrogram_f32

        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MODE", "rust")
        monkeypatch.setattr(pitch_module, "_PIPTRACK_RUST_MIN_WORK", 0)

        def _fake_kernel(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(_rust_ext, "piptrack_from_spectrogram_f32", _fake_kernel)

        pitches, mags = librosa.piptrack(S=S, sr=22050, n_fft=1024)
        assert pitches.shape == mags.shape == S.shape
        assert calls["count"] == 1
        assert calls["count"] == 1

    def test_estimate_tuning_respects_work_threshold(self, monkeypatch):
        rng = np.random.default_rng(2064)
        S = np.abs(rng.standard_normal((1025, 20), dtype=np.float32))

        monkeypatch.setattr(pitch_module, "_ENABLE_RUST_TUNING", True)
        monkeypatch.setattr(pitch_module, "_RUST_TUNING_MIN_WORK", S.size + 1)

        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used below threshold")

        monkeypatch.setattr(_rust_ext, "estimate_tuning_from_piptrack_f32", _fail)

        tuning = librosa.estimate_tuning(S=S, sr=22050, n_fft=2048)
        assert isinstance(tuning, float)

    def test_estimate_tuning_fallback_without_kernel(self, monkeypatch):
        rng = np.random.default_rng(2063)
        S = np.abs(rng.standard_normal((1025, 18), dtype=np.float64))

        monkeypatch.delattr(_rust_ext, "estimate_tuning_from_piptrack_f64", raising=False)

        tuning = librosa.estimate_tuning(S=S, sr=22050, n_fft=2048)
        assert isinstance(tuning, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


