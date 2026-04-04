"""Phase 10A baseline tests for HPSS before Rust dispatch integration."""

import numpy as np
import pytest
from scipy.ndimage import median_filter

import librosa
from librosa.util.exceptions import ParameterError


def _hpss_reference(S, *, kernel_size=31, power=2.0, mask=False, margin=1.0):
    """Reference implementation matching the current Python HPSS path."""
    if np.iscomplexobj(S):
        mag, phase = librosa.magphase(S)
    else:
        mag = S
        phase = 1

    if isinstance(kernel_size, (tuple, list)):
        win_harm, win_perc = kernel_size
    else:
        win_harm = win_perc = kernel_size

    if isinstance(margin, (tuple, list)):
        margin_harm, margin_perc = margin
    else:
        margin_harm = margin_perc = margin

    if margin_harm < 1 or margin_perc < 1:
        raise ParameterError("Margins must be >= 1.0. A typical range is between 1 and 10.")

    harm_shape = [1] * mag.ndim
    harm_shape[-1] = win_harm

    perc_shape = [1] * mag.ndim
    perc_shape[-2] = win_perc

    harm = median_filter(mag, size=harm_shape, mode="reflect")
    perc = median_filter(mag, size=perc_shape, mode="reflect")

    split_zeros = margin_harm == 1 and margin_perc == 1
    mask_harm = librosa.util.softmask(harm, perc * margin_harm, power=power, split_zeros=split_zeros)
    mask_perc = librosa.util.softmask(perc, harm * margin_perc, power=power, split_zeros=split_zeros)

    if mask:
        return mask_harm, mask_perc

    return (mag * mask_harm) * phase, (mag * mask_perc) * phase


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hpss_parity_real(dtype):
    rng = np.random.default_rng(1010)
    S = np.abs(rng.standard_normal((257, 48))).astype(dtype)

    expected_h, expected_p = _hpss_reference(S, kernel_size=(17, 31), power=2.0, mask=False, margin=1.0)
    actual_h, actual_p = librosa.decompose.hpss(S, kernel_size=(17, 31), power=2.0, mask=False, margin=1.0)

    rtol = 1e-5 if dtype == np.float32 else 1e-10
    atol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual_h, expected_h, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_p, expected_p, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_hpss_parity_complex(dtype):
    rng = np.random.default_rng(2020)
    real = rng.standard_normal((257, 48))
    imag = rng.standard_normal((257, 48))
    S = (real + 1j * imag).astype(dtype)

    expected_h, expected_p = _hpss_reference(S, kernel_size=31, power=2.0, mask=False, margin=(1.0, 2.0))
    actual_h, actual_p = librosa.decompose.hpss(S, kernel_size=31, power=2.0, mask=False, margin=(1.0, 2.0))

    rtol = 1e-4 if dtype == np.complex64 else 1e-10
    atol = 1e-5 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(actual_h, expected_h, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_p, expected_p, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hpss_parity_multichannel_real(dtype):
    rng = np.random.default_rng(5050)
    S = np.abs(rng.standard_normal((2, 257, 48))).astype(dtype)

    expected_h, expected_p = _hpss_reference(
        S, kernel_size=(17, 31), power=2.0, mask=False, margin=1.0
    )
    actual_h, actual_p = librosa.decompose.hpss(
        S, kernel_size=(17, 31), power=2.0, mask=False, margin=1.0
    )

    rtol = 1e-5 if dtype == np.float32 else 1e-10
    atol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(actual_h, expected_h, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_p, expected_p, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_hpss_parity_multichannel_complex(dtype):
    rng = np.random.default_rng(6060)
    real = rng.standard_normal((2, 257, 48))
    imag = rng.standard_normal((2, 257, 48))
    S = (real + 1j * imag).astype(dtype)

    expected_h, expected_p = _hpss_reference(
        S, kernel_size=31, power=2.0, mask=False, margin=(1.0, 2.0)
    )
    actual_h, actual_p = librosa.decompose.hpss(
        S, kernel_size=31, power=2.0, mask=False, margin=(1.0, 2.0)
    )

    rtol = 1e-4 if dtype == np.complex64 else 1e-10
    atol = 1e-5 if dtype == np.complex64 else 1e-12
    np.testing.assert_allclose(actual_h, expected_h, rtol=rtol, atol=atol)
    np.testing.assert_allclose(actual_p, expected_p, rtol=rtol, atol=atol)


def test_hpss_masks_basic_invariants():
    rng = np.random.default_rng(3030)
    S = np.abs(rng.standard_normal((2, 257, 40))).astype(np.float32)

    mask_h, mask_p = librosa.decompose.hpss(S, kernel_size=(13, 21), mask=True, margin=1.0)

    assert mask_h.shape == S.shape
    assert mask_p.shape == S.shape
    assert np.all(mask_h >= 0)
    assert np.all(mask_p >= 0)
    np.testing.assert_allclose(mask_h + mask_p, 1.0, rtol=1e-5, atol=1e-6)


def test_hpss_reconstruction_margin_one():
    rng = np.random.default_rng(4040)
    S = np.abs(rng.standard_normal((513, 30))).astype(np.float64)

    H, P = librosa.decompose.hpss(S, margin=1.0)
    np.testing.assert_allclose(H + P, S, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("margin", [0.0, 0.5, (1.0, 0.9), (0.5, 1.0)])
def test_hpss_margin_validation(margin):
    S = np.abs(np.random.randn(129, 20)).astype(np.float32)
    with pytest.raises(ParameterError, match="Margins must be >= 1.0"):
        librosa.decompose.hpss(S, margin=margin)

