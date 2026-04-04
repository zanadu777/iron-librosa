"""Phase 5 tests: chroma filter-bank acceleration."""

import numpy as np
import pytest

import librosa
from librosa._rust_bridge import _rust_ext, RUST_AVAILABLE


def _python_chroma_reference(
    *,
    sr,
    n_fft,
    n_chroma=12,
    tuning=0.0,
    ctroct=5.0,
    octwidth=2.0,
    base_c=True,
    dtype=np.float32,
):
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]
    frqbins = n_chroma * librosa.hz_to_octs(
        frequencies, tuning=tuning, bins_per_octave=n_chroma
    )
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))
    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))
    D = np.subtract.outer(frqbins, np.arange(0, n_chroma, dtype="d")).T
    n_chroma2 = np.round(float(n_chroma) / 2)
    D = np.remainder(D + n_chroma2 + 10 * n_chroma, n_chroma) - n_chroma2
    wts = np.exp(-0.5 * (2 * D / np.tile(binwidthbins, (n_chroma, 1))) ** 2)
    wts = librosa.util.normalize(wts, norm=2, axis=0)
    if octwidth is not None:
        wts *= np.tile(
            np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2)),
            (n_chroma, 1),
        )
    if base_c:
        wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)
    return np.ascontiguousarray(wts[:, : int(1 + n_fft / 2)], dtype=dtype)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestChromaFilterKernels:
    def test_chroma_filter_f32_matches_reference(self):
        actual = _rust_ext.chroma_filter_f32(22050.0, 2048, 12, 0.0, 5.0, 2.0, True)
        expected = _python_chroma_reference(sr=22050, n_fft=2048, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_chroma_filter_f64_matches_reference(self):
        actual = _rust_ext.chroma_filter_f64(22050.0, 4096, 24, 0.1, 5.5, None, False)
        expected = _python_chroma_reference(
            sr=22050,
            n_fft=4096,
            n_chroma=24,
            tuning=0.1,
            ctroct=5.5,
            octwidth=None,
            base_c=False,
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestChromaFilterDispatch:
    def test_filters_chroma_uses_rust_kernel_default_case(self, monkeypatch):
        calls = {"count": 0}
        original = _rust_ext.chroma_filter_f32

        def _fake(*args):
            calls["count"] += 1
            return original(*args)

        monkeypatch.setattr(_rust_ext, "chroma_filter_f32", _fake)
        out = librosa.filters.chroma(sr=22050, n_fft=2048)
        assert out.shape == (12, 1025)
        assert out.dtype == np.float32
        assert calls["count"] == 1

    def test_filters_chroma_uses_rust_kernel_for_norm_one(self, monkeypatch):
        calls = {"count": 0}
        original = _rust_ext.chroma_filter_f32

        def _fake(*args):
            calls["count"] += 1
            return original(*args)

        monkeypatch.setattr(_rust_ext, "chroma_filter_f32", _fake)
        out = librosa.filters.chroma(sr=22050, n_fft=2048, norm=1)
        assert out.shape == (12, 1025)
        assert out.dtype == np.float32
        assert calls["count"] == 1

    def test_filters_chroma_fallback_for_unsupported_norm(self, monkeypatch):
        def _fail(*args, **kwargs):
            raise AssertionError("Rust dispatch should not be used for unsupported norm")

        monkeypatch.setattr(_rust_ext, "chroma_filter_f32", _fail)
        out = librosa.filters.chroma(sr=22050, n_fft=2048, norm=3)
        assert out.shape == (12, 1025)
        assert out.dtype == np.float32

    def test_filters_chroma_matches_reference_default(self):
        actual = librosa.filters.chroma(sr=22050, n_fft=2048)
        expected = _python_chroma_reference(sr=22050, n_fft=2048, dtype=np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
