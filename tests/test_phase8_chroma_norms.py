"""Phase 8 tests: chroma filter norm expansion (None/1/2/inf)."""

import importlib
from contextlib import contextmanager

import numpy as np
import pytest

import librosa
from librosa._rust_bridge import RUST_AVAILABLE


@contextmanager
def _force_python_chroma_fallback(enabled: bool):
    """Temporarily disable Rust dispatch in librosa.filters."""
    filters_mod = importlib.import_module("librosa.filters")
    prev_available = filters_mod.RUST_AVAILABLE
    prev_ext = filters_mod._rust_ext
    try:
        if enabled:
            filters_mod.RUST_AVAILABLE = False
            filters_mod._rust_ext = None
        yield
    finally:
        filters_mod.RUST_AVAILABLE = prev_available
        filters_mod._rust_ext = prev_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestChromaNormDispatchParity:
    @pytest.mark.parametrize("norm", [None, 1, 2, np.inf])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_chroma_norms_match_forced_python_fallback(self, norm, dtype):
        kwargs = dict(sr=22050, n_fft=2048, n_chroma=12, norm=norm, octwidth=2.0, dtype=dtype)

        with _force_python_chroma_fallback(True):
            expected = librosa.filters.chroma(**kwargs)

        with _force_python_chroma_fallback(False):
            actual = librosa.filters.chroma(**kwargs)

        assert actual.shape == expected.shape == (12, 1025)
        assert actual.dtype == expected.dtype == np.dtype(dtype)
        rtol = 1e-5 if dtype == np.float32 else 1e-10
        atol = 1e-6 if dtype == np.float32 else 1e-12
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    @pytest.mark.parametrize("norm", [None, 1, 2, np.inf])
    def test_supported_norms_dispatch_to_rust(self, norm, monkeypatch):
        from librosa._rust_bridge import _rust_ext

        calls = {"f32": 0}
        original = _rust_ext.chroma_filter_f32

        def _wrapped(*args):
            calls["f32"] += 1
            return original(*args)

        monkeypatch.setattr(_rust_ext, "chroma_filter_f32", _wrapped)
        out = librosa.filters.chroma(sr=22050, n_fft=2048, norm=norm, dtype=np.float32)

        assert out.shape == (12, 1025)
        assert calls["f32"] == 1


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
class TestChromaNormInvariants:
    def test_l1_columns_sum_to_one_octwidth_none(self):
        w = librosa.filters.chroma(sr=22050, n_fft=2048, norm=1, octwidth=None, dtype=np.float64)
        col_l1 = np.sum(np.abs(w), axis=0)
        np.testing.assert_allclose(col_l1, np.ones_like(col_l1), rtol=1e-10, atol=1e-12)

    def test_l2_columns_have_unit_norm_octwidth_none(self):
        w = librosa.filters.chroma(sr=22050, n_fft=2048, norm=2, octwidth=None, dtype=np.float64)
        col_l2 = np.sqrt(np.sum(w * w, axis=0))
        np.testing.assert_allclose(col_l2, np.ones_like(col_l2), rtol=1e-10, atol=1e-12)

    def test_linf_columns_have_unit_max_octwidth_none(self):
        w = librosa.filters.chroma(sr=22050, n_fft=2048, norm=np.inf, octwidth=None, dtype=np.float64)
        col_linf = np.max(np.abs(w), axis=0)
        np.testing.assert_allclose(col_linf, np.ones_like(col_linf), rtol=1e-10, atol=1e-12)

    def test_none_norm_is_not_unit_normalized_octwidth_none(self):
        w = librosa.filters.chroma(sr=22050, n_fft=2048, norm=None, octwidth=None, dtype=np.float64)
        col_l2 = np.sqrt(np.sum(w * w, axis=0))
        assert np.max(np.abs(col_l2 - 1.0)) > 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

