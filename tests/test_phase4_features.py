"""Phase 4B tests: RMS and spectral centroid acceleration."""

import numpy as np
import pytest

import librosa
import librosa.feature.spectral as spectral_module
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestRmsKernels:
    def test_rms_spectrogram_f32_matches_formula(self):
        rng = np.random.default_rng(2036)
        S = np.abs(rng.standard_normal((1025, 24), dtype=np.float32))
        frame_length = 2048

        expected = S**2
        expected[0] *= 0.5
        expected[-1] *= 0.5
        expected = np.sqrt(2 * np.sum(expected, axis=-2, keepdims=True) / frame_length**2)

        actual = _rust_ext.rms_spectrogram_f32(S, frame_length)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rms_spectrogram_f64_matches_formula(self):
        rng = np.random.default_rng(2037)
        S = np.abs(rng.standard_normal((1025, 18))).astype(np.float64)
        frame_length = 2048

        expected = S**2
        expected[0] *= 0.5
        expected[-1] *= 0.5
        expected = np.sqrt(2 * np.sum(expected, axis=-2, keepdims=True) / frame_length**2)

        actual = _rust_ext.rms_spectrogram_f64(S, frame_length)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_rms_time_f32_matches_formula(self):
        rng = np.random.default_rng(2053)
        x = rng.standard_normal((2048, 17), dtype=np.float32)
        expected = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
        actual = _rust_ext.rms_time_f32(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_rms_time_f64_matches_formula(self):
        rng = np.random.default_rng(2054)
        x = rng.standard_normal((2048, 13)).astype(np.float64)
        expected = np.sqrt(np.mean(x * x, axis=0, keepdims=True))
        actual = _rust_ext.rms_time_f64(x)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSpectralCentroidKernels:
    def test_spectral_centroid_f32_matches_formula(self):
        rng = np.random.default_rng(2038)
        S = np.abs(rng.standard_normal((1025, 20), dtype=np.float32))
        freq = librosa.fft_frequencies(sr=22050, n_fft=2048)

        expected = np.sum(
            freq[:, np.newaxis] * librosa.util.normalize(S, norm=1, axis=-2),
            axis=-2,
            keepdims=True,
        )

        actual = _rust_ext.spectral_centroid_f32(S, freq)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_spectral_centroid_f64_matches_formula(self):
        rng = np.random.default_rng(2039)
        S = np.abs(rng.standard_normal((1025, 12))).astype(np.float64)
        freq = librosa.fft_frequencies(sr=44100, n_fft=2048)

        expected = np.sum(
            freq[:, np.newaxis] * librosa.util.normalize(S, norm=1, axis=-2),
            axis=-2,
            keepdims=True,
        )

        actual = _rust_ext.spectral_centroid_f64(S, freq)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestSpectralRolloffBandwidthKernels:
    def test_spectral_rolloff_f32_matches_formula(self):
        rng = np.random.default_rng(2045)
        S = np.abs(rng.standard_normal((1025, 15), dtype=np.float32))
        freq = librosa.fft_frequencies(sr=22050, n_fft=2048)
        roll_percent = 0.85

        total_energy = np.cumsum(S, axis=-2)
        threshold = np.expand_dims(roll_percent * total_energy[-1, :], axis=-2)
        ind = np.where(total_energy < threshold, np.nan, 1)
        expected = np.nanmin(ind * freq[:, np.newaxis], axis=-2, keepdims=True)

        actual = _rust_ext.spectral_rolloff_f32(S, freq, roll_percent)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    def test_spectral_bandwidth_f32_matches_formula(self):
        rng = np.random.default_rng(2046)
        S = np.abs(rng.standard_normal((1025, 13), dtype=np.float32))
        freq = librosa.fft_frequencies(sr=22050, n_fft=2048)
        centroid = librosa.feature.spectral_centroid(S=S, sr=22050, n_fft=2048)

        deviation = np.abs(np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1))
        expected = np.sum(
            librosa.util.normalize(S, norm=1, axis=-2) * deviation**2,
            axis=-2,
            keepdims=True,
        ) ** 0.5

        actual = _rust_ext.spectral_bandwidth_f32(S, freq, centroid, True, 2.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_spectral_bandwidth_auto_centroid_f32_matches_formula(self):
        rng = np.random.default_rng(2051)
        S = np.abs(rng.standard_normal((1025, 13), dtype=np.float32))
        freq = librosa.fft_frequencies(sr=22050, n_fft=2048)

        centroid = np.sum(
            freq[:, np.newaxis] * librosa.util.normalize(S, norm=1, axis=-2),
            axis=-2,
            keepdims=True,
        )
        deviation = np.abs(np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1))
        expected = np.sum(
            librosa.util.normalize(S, norm=1, axis=-2) * deviation**2,
            axis=-2,
            keepdims=True,
        ) ** 0.5

        actual = _rust_ext.spectral_bandwidth_auto_centroid_f32(S, freq, True, 2.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestPublicFeatureDispatch:
    def test_rms_public_dispatch_on_spectrogram_f32(self, monkeypatch):
        rng = np.random.default_rng(2040)
        S = np.abs(rng.standard_normal((2, 1025, 10), dtype=np.float32))
        calls = {"count": 0}
        original = _rust_ext.rms_spectrogram_f32

        def _fake_kernel(channel, frame_length):
            calls["count"] += 1
            assert channel.dtype == np.float32
            return original(channel, frame_length)

        monkeypatch.setattr(_rust_ext, "rms_spectrogram_f32", _fake_kernel)

        out = librosa.feature.rms(S=S, frame_length=2048, dtype=np.float32)

        assert out.shape == (2, 1, 10)
        assert calls["count"] == 2

    def test_rms_public_dispatch_on_time_signal_f32(self, monkeypatch):
        rng = np.random.default_rng(2055)
        y = rng.standard_normal((2, 4096), dtype=np.float32)
        calls = {"count": 0}
        original = _rust_ext.rms_time_f32

        monkeypatch.setattr(spectral_module, "_ENABLE_RUST_RMS_TIME", True)

        def _fake_kernel(channel):
            calls["count"] += 1
            assert channel.dtype == np.float32
            return original(channel)

        monkeypatch.setattr(_rust_ext, "rms_time_f32", _fake_kernel)

        out = librosa.feature.rms(y=y, frame_length=2048, hop_length=512, dtype=np.float32)

        assert out.shape[0] == 2
        assert out.shape[1] == 1
        assert calls["count"] == 2

    def test_rms_public_time_fallback_for_unsupported_dtype(self, monkeypatch):
        rng = np.random.default_rng(2056)
        y = rng.standard_normal(4096).astype(np.float16)

        def _fail(*args, **kwargs):
            raise AssertionError("Rust time-domain dispatch should not be used for float16")

        monkeypatch.setattr(_rust_ext, "rms_time_f32", _fail)

        out = librosa.feature.rms(y=y, frame_length=2048, hop_length=512, dtype=np.float16)
        assert out.shape[0] == 1

    def test_rms_public_fallback_when_dtype_mismatch(self, monkeypatch):
        rng = np.random.default_rng(2041)
        S = np.abs(rng.standard_normal((1025, 8), dtype=np.float32))

        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used when dtype mismatches")

        monkeypatch.setattr(_rust_ext, "rms_spectrogram_f32", _fail)

        out = librosa.feature.rms(S=S, frame_length=2048, dtype=np.float64)
        assert out.shape == (1, 8)
        assert out.dtype == np.float64

    def test_spectral_centroid_public_dispatch_default_freq(self, monkeypatch):
        rng = np.random.default_rng(2042)
        S = np.abs(rng.standard_normal((2, 1025, 9), dtype=np.float32))
        calls = {"count": 0}
        original = _rust_ext.spectral_centroid_f32

        def _fake_kernel(channel, freq):
            calls["count"] += 1
            assert channel.dtype == np.float32
            assert freq.dtype == np.float64
            return original(channel, freq)

        monkeypatch.setattr(_rust_ext, "spectral_centroid_f32", _fake_kernel)

        out = librosa.feature.spectral_centroid(S=S, sr=22050)

        assert out.shape == (2, 1, 9)
        assert calls["count"] == 2

    def test_spectral_centroid_public_fallback_for_variable_freq(self, monkeypatch):
        rng = np.random.default_rng(2043)
        S = np.abs(rng.standard_normal((1025, 7), dtype=np.float32))
        freq = np.abs(rng.standard_normal((1025, 7)))

        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used for variable frequency grids")

        monkeypatch.setattr(_rust_ext, "spectral_centroid_f32", _fail)

        out = librosa.feature.spectral_centroid(S=S, freq=freq)
        assert out.shape == (1, 7)

    def test_public_feature_outputs_match_existing_formulas(self):
        rng = np.random.default_rng(2044)
        S = np.abs(rng.standard_normal((1025, 11), dtype=np.float32))
        freq = librosa.fft_frequencies(sr=22050, n_fft=2048)

        rms_expected = np.sqrt(
            2
            * np.sum(
                np.concatenate(
                    [
                        0.5 * (S[:1] ** 2),
                        S[1:-1] ** 2,
                        0.5 * (S[-1:] ** 2),
                    ],
                    axis=0,
                ),
                axis=-2,
                keepdims=True,
            )
            / 2048**2
        )
        rms_actual = librosa.feature.rms(S=S, frame_length=2048, dtype=np.float32)
        np.testing.assert_allclose(rms_actual, rms_expected, rtol=1e-5, atol=1e-6)

        centroid_expected = np.sum(
            freq[:, np.newaxis] * librosa.util.normalize(S, norm=1, axis=-2),
            axis=-2,
            keepdims=True,
        )
        centroid_actual = librosa.feature.spectral_centroid(S=S, sr=22050)
        np.testing.assert_allclose(centroid_actual, centroid_expected, rtol=1e-6, atol=1e-6)

    def test_spectral_rolloff_public_dispatch_default_freq(self, monkeypatch):
        rng = np.random.default_rng(2047)
        S = np.abs(rng.standard_normal((2, 1025, 9), dtype=np.float32))
        calls = {"count": 0}
        original = _rust_ext.spectral_rolloff_f32

        def _fake_kernel(channel, freq, roll_percent):
            calls["count"] += 1
            assert channel.dtype == np.float32
            assert freq.dtype == np.float64
            return original(channel, freq, roll_percent)

        monkeypatch.setattr(_rust_ext, "spectral_rolloff_f32", _fake_kernel)

        out = librosa.feature.spectral_rolloff(S=S, sr=22050)
        assert out.shape == (2, 1, 9)
        assert calls["count"] == 2

    def test_spectral_bandwidth_public_dispatch_default_freq(self, monkeypatch):
        rng = np.random.default_rng(2048)
        S = np.abs(rng.standard_normal((2, 1025, 9), dtype=np.float32))
        centroid = librosa.feature.spectral_centroid(S=S, sr=22050)
        calls = {"count": 0}
        original = _rust_ext.spectral_bandwidth_f32

        def _fake_kernel(channel, freq, centroid, norm, p):
            calls["count"] += 1
            assert channel.dtype == np.float32
            assert freq.dtype == np.float64
            assert centroid.shape[0] == 1
            return original(channel, freq, centroid, norm, p)

        monkeypatch.setattr(_rust_ext, "spectral_bandwidth_f32", _fake_kernel)

        out = librosa.feature.spectral_bandwidth(S=S, sr=22050, centroid=centroid)
        assert out.shape == (2, 1, 9)
        assert calls["count"] == 2

    def test_spectral_bandwidth_public_uses_auto_centroid_kernel_when_centroid_none(self, monkeypatch):
        rng = np.random.default_rng(2052)
        S = np.abs(rng.standard_normal((2, 1025, 9), dtype=np.float32))
        calls = {"auto": 0, "manual": 0}
        auto_original = _rust_ext.spectral_bandwidth_auto_centroid_f32
        manual_original = _rust_ext.spectral_bandwidth_f32

        def _auto(channel, freq, norm, p):
            calls["auto"] += 1
            return auto_original(channel, freq, norm, p)

        def _manual(*args, **kwargs):
            calls["manual"] += 1
            return manual_original(*args, **kwargs)

        monkeypatch.setattr(_rust_ext, "spectral_bandwidth_auto_centroid_f32", _auto)
        monkeypatch.setattr(_rust_ext, "spectral_bandwidth_f32", _manual)

        out = librosa.feature.spectral_bandwidth(S=S, sr=22050)
        assert out.shape == (2, 1, 9)
        assert calls["auto"] == 2
        assert calls["manual"] == 0

    def test_spectral_rolloff_public_fallback_for_variable_freq(self, monkeypatch):
        rng = np.random.default_rng(2049)
        S = np.abs(rng.standard_normal((1025, 7), dtype=np.float32))
        freq = np.abs(rng.standard_normal((1025, 7)))

        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used for variable frequency grids")

        monkeypatch.setattr(_rust_ext, "spectral_rolloff_f32", _fail)

        out = librosa.feature.spectral_rolloff(S=S, freq=freq)
        assert out.shape == (1, 7)

    def test_spectral_bandwidth_public_fallback_for_variable_freq(self, monkeypatch):
        rng = np.random.default_rng(2050)
        S = np.abs(rng.standard_normal((1025, 7), dtype=np.float32))
        freq = np.abs(rng.standard_normal((1025, 7)))

        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used for variable frequency grids")

        monkeypatch.setattr(_rust_ext, "spectral_bandwidth_f32", _fail)

        out = librosa.feature.spectral_bandwidth(S=S, freq=freq)
        assert out.shape == (1, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

