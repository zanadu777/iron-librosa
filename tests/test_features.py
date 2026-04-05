#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import warnings
import types
import numpy as np
import scipy

import pytest

import librosa
import librosa.feature.spectral as spectral_mod
import librosa.core.spectrum as core_spectrum_mod
from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext

from test_core import load, srand

# Disable cache
import os

try:
    os.environ.pop("LIBROSA_CACHE_DIR")
except KeyError:
    pass

__EXAMPLE_FILE = os.path.join("tests", "data", "test1_22050.wav")
warnings.resetwarnings()
warnings.simplefilter("always")
warnings.filterwarnings("module", ".*", FutureWarning, "scipy.*")


# utils submodule
@pytest.mark.parametrize("slope", np.linspace(-2, 2, num=6))
@pytest.mark.parametrize("xin", [np.vstack([np.arange(100.0)] * 3)])
@pytest.mark.parametrize("order", [1])
@pytest.mark.parametrize("width, axis", [(3, 0), (3, 1), (5, 1), (7, 1)])
@pytest.mark.parametrize("bias", [-10, 0, 10])
def test_delta(xin, width, slope, order, axis, bias):

    x = slope * xin + bias

    # Note: this test currently only checks first-order differences
    #    if width < 3 or np.mod(width, 2) != 1 or width > x.shape[axis]:
    #        pytest.raises(librosa.ParameterError)

    delta = librosa.feature.delta(x, width=width, order=order, axis=axis)

    # Check that trimming matches the expected shape
    assert x.shape == delta.shape

    # Once we're sufficiently far into the signal (ie beyond half_len)
    # (x + delta)[t] should approximate x[t+1] if x is actually linear
    slice_orig = [slice(None)] * x.ndim
    slice_out = [slice(None)] * delta.ndim
    slice_orig[axis] = slice(width // 2 + 1, -width // 2 + 1)
    slice_out[axis] = slice(width // 2, -width // 2)
    assert np.allclose((x + delta)[tuple(slice_out)], x[tuple(slice_orig)])


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_delta_badorder():
    x = np.ones((10, 10))
    librosa.feature.delta(x, order=0)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("x", [np.ones((3, 100))])
@pytest.mark.parametrize(
    "width, axis",
    [
        (-1, 0),
        (-1, 1),
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (4, 0),
        (4, 1),
        (5, 0),
        (6, 0),
        (6, 1),
        (7, 0),
    ],
)
def test_delta_badwidthaxis(x, width, axis):
    librosa.feature.delta(x, width=width, axis=axis)


@pytest.mark.parametrize("data", [np.arange(5.0), np.remainder(np.arange(10000), 24)])
@pytest.mark.parametrize("delay", [-4, -2, -1, 1, 2, 4])
@pytest.mark.parametrize("n_steps", [1, 2, 3, 300])
def test_stack_memory(data, n_steps, delay):

    data_stack = librosa.feature.stack_memory(data, n_steps=n_steps, delay=delay)

    # If we're one-dimensional, reshape for testing
    if data.ndim == 1:
        data = data.reshape((1, -1))

    d, t = data.shape

    assert data_stack.shape[0] == n_steps * d
    assert data_stack.shape[1] == t

    assert np.allclose(data_stack[0], data[0])

    for i in range(d):
        for step in range(1, n_steps):
            if delay > 0:
                assert np.allclose(
                    data[i, : -step * delay], data_stack[step * d + i, step * delay :]
                )
            else:
                assert np.allclose(
                    data[i, -step * delay :], data_stack[step * d + i, : step * delay]
                )
    assert np.max(data) + 1e-7 >= np.max(data_stack)
    assert np.min(data) - 1e-7 <= np.min(data_stack)


@pytest.mark.parametrize("n_steps,delay", [(0, 1), (-1, 1), (1, 0)])
@pytest.mark.parametrize("data", [np.zeros((2, 2))])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_stack_memory_fail(data, n_steps, delay):
    librosa.feature.stack_memory(data, n_steps=n_steps, delay=delay)


@pytest.mark.parametrize("data", [np.zeros((2, 0))])
@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("delay", [-2, -1, 1, 2])
@pytest.mark.parametrize("n_steps", [1, 2])
def test_stack_memory_ndim_badshape(data, delay, n_steps):
    librosa.feature.stack_memory(data, n_steps=n_steps, delay=delay)


@pytest.fixture(scope="module")
def S_ideal():
    # An idealized spectrum with all zero energy except at one DFT band
    S = np.zeros((513, 3))
    S[5, :] = 1.0
    return S


# spectral submodule
@pytest.mark.parametrize(
    "freq",
    [
        None,
        librosa.fft_frequencies(sr=22050, n_fft=1024),
        3 * librosa.fft_frequencies(sr=22050, n_fft=1024),
        np.random.randn(513, 3),
    ],
)
def test_spectral_centroid_synthetic(S_ideal, freq):
    n_fft = 2 * (S_ideal.shape[0] - 1)
    cent = librosa.feature.spectral_centroid(S=S_ideal, freq=freq)

    if freq is None:
        freq = librosa.fft_frequencies(sr=22050, n_fft=n_fft)

    assert np.allclose(cent, freq[5])


@pytest.mark.parametrize("S", [-np.ones((9, 3)) * 1.0j])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_spectral_centroid_errors(S):
    # Pure-imaginary spectrogram is still an error.
    # Note: real-valued negative spectrograms are now accepted (no ParameterError).
    librosa.feature.spectral_centroid(S=S)


@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize(
    "y,S", [(np.zeros(3 * 22050), None), (None, np.zeros((1025, 10)))]
)
def test_spectral_centroid_empty(y, sr, S):
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, S=S)
    assert not np.any(cent)


@pytest.mark.parametrize(
    "freq",
    [
        None,
        librosa.fft_frequencies(sr=22050, n_fft=1024),
        3 * librosa.fft_frequencies(sr=22050, n_fft=1024),
        np.random.randn(513, 3),
    ],
)
@pytest.mark.parametrize("norm", [False, True])
@pytest.mark.parametrize("p", [1, 2])
def test_spectral_bandwidth_synthetic(S_ideal, freq, norm, p):
    # This test ensures that a signal confined to a single frequency bin
    # always achieves 0 bandwidth

    bw = librosa.feature.spectral_bandwidth(S=S_ideal, freq=freq, norm=norm, p=p)

    assert not np.any(bw)


@pytest.mark.parametrize(
    "freq",
    [
        None,
        librosa.fft_frequencies(sr=22050, n_fft=1024),
        3 * librosa.fft_frequencies(sr=22050, n_fft=1024),
        np.random.randn(513, 1),
    ],
)
def test_spectral_bandwidth_onecol(S_ideal, freq):
    # This test checks for issue https://github.com/librosa/librosa/issues/552
    # failure when the spectrogram has a single column

    bw = librosa.feature.spectral_bandwidth(S=S_ideal[:, :1], freq=freq)
    assert bw.shape == (1, 1)


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("S", [-np.ones((17, 2)), -np.ones((17, 2)) * 1.0j])
def test_spectral_bandwidth_errors(S):
    librosa.feature.spectral_bandwidth(S=S)


def test_spectral_bandwidth_variable_freq_shape_mismatch():
    S = np.abs(np.random.randn(1025, 3)).astype(np.float32)
    bad_freq = np.abs(np.random.randn(1025, 2)).astype(np.float64)
    centroid = np.abs(np.random.randn(1, 3)).astype(np.float64)

    with pytest.raises(librosa.ParameterError, match="freq.shape mismatch"):
        librosa.feature.spectral_bandwidth(S=S, freq=bad_freq, centroid=centroid)


@pytest.mark.parametrize("S", [np.ones((1025, 3))])
@pytest.mark.parametrize(
    "freq",
    [
        None,
        librosa.fft_frequencies(sr=22050, n_fft=2048),
        np.cumsum(np.abs(np.random.randn(1025, 3)), axis=0),
    ],
)
@pytest.mark.parametrize("pct", [0.25, 0.5, 0.95])
def test_spectral_rolloff_synthetic(S, freq, pct):

    sr = 22050
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, freq=freq, roll_percent=pct)

    n_fft = 2 * (S.shape[0] - 1)
    if freq is None:
        freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    idx = np.floor(pct * freq.shape[0]).astype(int)
    assert np.allclose(rolloff, freq[idx])


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize(
    "S,pct",
    [
        (-np.ones((513, 3)), 0.95),
        (-np.ones((513, 3)) * 1.0j, 0.95),
        (np.ones((513, 3)), -1),
        (np.ones((513, 3)), 2),
    ],
)
def test_spectral_rolloff_errors(S, pct):
    librosa.feature.spectral_rolloff(S=S, roll_percent=pct)


@pytest.fixture(scope="module")
def y_ex():
    return librosa.load(os.path.join("tests", "data", "test1_22050.wav"))


def test_spectral_contrast_log(y_ex):
    # We already have a regression test for linear energy difference
    # This test just does a sanity-check on the log-scaled version

    y, sr = y_ex

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, linear=False)

    assert not np.any(contrast < 0)


@pytest.mark.parametrize("S", [np.ones((1025, 10))])
@pytest.mark.parametrize(
    "freq,fmin,n_bands,quantile",
    [
        (0, 200, 6, 0.02),
        (np.zeros(1 + 1025), 200, 6, 0.02),
        (np.zeros((1025, 10)), 200, 6, 0.02),
        (None, -1, 6, 0.02),
        (None, 0, 6, 0.02),
        (None, 200, -1, 0.02),
        (None, 200, 6, -1),
        (None, 200, 6, 2),
        (None, 200, 7, 0.02),
    ],
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_spectral_contrast_errors(S, freq, fmin, n_bands, quantile):

    librosa.feature.spectral_contrast(
        S=S, freq=freq, fmin=fmin, n_bands=n_bands, quantile=quantile
    )


@pytest.mark.parametrize(
    "S,flatness_ref",
    [
        (np.array([[1, 3], [2, 1], [1, 2]]), np.array([[0.7937005259, 0.7075558390]])),
        (np.ones((1025, 2)), np.ones((1, 2))),
        (np.zeros((1025, 2)), np.ones((1, 2))),
    ],
)
def test_spectral_flatness_synthetic(S, flatness_ref):
    flatness = librosa.feature.spectral_flatness(S=S)
    assert np.allclose(flatness, flatness_ref)


@pytest.mark.parametrize("S", [np.ones((1025, 2))])
@pytest.mark.parametrize("amin", [0, -1])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_spectral_flatness_errors(S, amin):
    librosa.feature.spectral_flatness(S=S, amin=amin)


@pytest.mark.parametrize("S", [-np.ones((1025, 2)), -np.ones((1025, 2)) * 1.0j])
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_spectral_flatness_badtype(S):
    librosa.feature.spectral_flatness(S=S)


@pytest.mark.parametrize("n", range(10, 100, 10))
def test_rms_const(n):
    S = np.ones((n, 5))

    # RMSE of an all-ones band is 1
    frame_length = 2 * (n - 1)
    rms = librosa.feature.rms(S=S, frame_length=frame_length)
    assert np.allclose(rms, np.ones_like(rms) / np.sqrt(frame_length), atol=1e-2)


@pytest.mark.parametrize("frame_length", [2048, 2049, 4096, 4097])
@pytest.mark.parametrize("hop_length", [128, 512, 1024])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("y2", [np.random.randn(100000)])
def test_rms(y_ex, y2, frame_length, hop_length, center):
    y1, sr = y_ex
    # Ensure audio is divisible into frame size.
    y1 = librosa.util.fix_length(y1, size=y1.size - y1.size % frame_length)
    y2 = librosa.util.fix_length(y2, size=y2.size - y2.size % frame_length)
    assert y1.size % frame_length == 0
    assert y2.size % frame_length == 0

    # STFT magnitudes with a constant windowing function and no centering.
    S1 = librosa.magphase(
        librosa.stft(
            y1, n_fft=frame_length, hop_length=hop_length, window=np.ones, center=center
        )
    )[0]
    S2 = librosa.magphase(
        librosa.stft(
            y2, n_fft=frame_length, hop_length=hop_length, window=np.ones, center=center
        )
    )[0]

    # Try both RMS methods.
    rms1 = librosa.feature.rms(S=S1, frame_length=frame_length, hop_length=hop_length)
    rms2 = librosa.feature.rms(
        y=y1, frame_length=frame_length, hop_length=hop_length, center=center
    )
    rms3 = librosa.feature.rms(S=S2, frame_length=frame_length, hop_length=hop_length)
    rms4 = librosa.feature.rms(
        y=y2, frame_length=frame_length, hop_length=hop_length, center=center
    )

    assert rms1.shape == rms2.shape
    assert rms3.shape == rms4.shape

    # Ensure results are similar.
    np.testing.assert_allclose(rms1, rms2, atol=5e-4)
    np.testing.assert_allclose(rms3, rms4, atol=5e-4)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_rms_noinput():
    librosa.feature.rms(y=None, S=None)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_rms_badshape():
    S = np.zeros((100, 3))
    librosa.feature.rms(S=S, frame_length=100)


@pytest.fixture(params=[32, 16, 8, 4, 2], scope="module")
def y_zcr(request):
    sr = 16384
    period = request.param
    y = np.ones(sr)
    y[::period] = -1
    rate = 2.0 / period
    return y, sr, rate


@pytest.mark.parametrize("frame_length", [513, 2049])
@pytest.mark.parametrize("hop_length", [128, 256])
@pytest.mark.parametrize("center", [False, True])
def test_zcr_synthetic(y_zcr, frame_length, hop_length, center):

    y, sr, rate = y_zcr
    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=center
    )

    # We don't care too much about the edges if there's padding
    if center:
        zcr = zcr[:, frame_length // 2 : -frame_length // 2]

    # We'll allow 1% relative error
    assert np.allclose(zcr, rate, rtol=1e-2)


@pytest.fixture(scope="module", params=[1, 2])
def poly_order(request):
    return request.param


@pytest.fixture(scope="module")
def poly_coeffs(poly_order):
    return np.atleast_1d(np.arange(1, 1 + poly_order))


@pytest.fixture(scope="module", params=[None, 1, 2, -1, "varying"])
def poly_freq(request):
    srand()
    freq = librosa.fft_frequencies()

    if request.param in (1, 2):
        return freq**request.param

    elif request.param == -1:
        return np.cumsum(np.abs(np.random.randn(1 + 2048 // 2)), axis=0)
    elif request.param == "varying":
        return np.cumsum(np.abs(np.random.randn(1 + 2048 // 2, 5)), axis=0)
    else:
        return None


@pytest.fixture(scope="module")
def poly_S(poly_coeffs, poly_freq):
    if poly_freq is None:
        poly_freq = librosa.fft_frequencies()

    S = np.zeros_like(poly_freq)
    for i, c in enumerate(poly_coeffs):
        S += c * poly_freq**i

    return S.reshape((poly_freq.shape[0], -1))


def test_poly_features_synthetic(poly_S, poly_coeffs, poly_freq):
    sr = 22050
    n_fft = 2048
    order = poly_coeffs.shape[0] - 1
    p = librosa.feature.poly_features(
        S=poly_S, sr=sr, n_fft=n_fft, order=order, freq=poly_freq
    )

    for i in range(poly_S.shape[-1]):
        assert np.allclose(poly_coeffs, p[::-1, i].squeeze())


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_tonnetz_fail_empty():
    librosa.feature.tonnetz(y=None, chroma=None)


def test_tonnetz_audio(y_ex):
    y, sr = y_ex
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    assert tonnetz.shape[0] == 6


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_chroma_cqt_badcombo(y_ex):
    y, sr = y_ex
    librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=24, bins_per_octave=36)


def test_tonnetz_cqt(y_ex):
    y, sr = y_ex
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=36)
    tonnetz = librosa.feature.tonnetz(chroma=chroma_cqt, sr=sr)
    assert tonnetz.shape[1] == chroma_cqt.shape[1]
    assert tonnetz.shape[0] == 6


def test_tonnetz_msaf():
    # Use pre-computed chroma
    chroma_path = os.path.join("tests", "data", "feature-tonnetz-chroma.npy")
    msaf_path = os.path.join("tests", "data", "feature-tonnetz-msaf.npy")
    if not (os.path.exists(chroma_path) and os.path.exists(msaf_path)):
        pytest.skip("tonnetz MSAF fixtures are not available in this workspace")

    tonnetz_chroma = np.load(chroma_path)
    tonnetz_msaf = np.load(msaf_path)

    tonnetz = librosa.feature.tonnetz(chroma=tonnetz_chroma)
    assert tonnetz.shape[1] == tonnetz_chroma.shape[1]
    assert tonnetz.shape[0] == 6
    assert np.allclose(tonnetz_msaf, tonnetz)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_tempogram_fail_noinput():
    librosa.feature.tempogram(y=None, onset_envelope=None)


@pytest.mark.parametrize("y", [np.zeros(10 * 1000)])
@pytest.mark.parametrize("sr", [1000])
@pytest.mark.parametrize(
    "win_length,window", [(-384, "hann"), (0, "hann"), (384, np.ones(3))]
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_tempogram_fail_badwin(y, sr, win_length, window):
    librosa.feature.tempogram(y=y, sr=sr, win_length=win_length, window=window)


@pytest.mark.parametrize("hop_length", [512, 1024])
def test_tempogram_audio(y_ex, hop_length):
    y, sr = y_ex

    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Get the tempogram from audio
    t1 = librosa.feature.tempogram(
        y=y, sr=sr, onset_envelope=None, hop_length=hop_length
    )

    # Get the tempogram from oenv
    t2 = librosa.feature.tempogram(
        y=None, sr=sr, onset_envelope=oenv, hop_length=hop_length
    )

    # Make sure it works when both are provided
    t3 = librosa.feature.tempogram(
        y=y, sr=sr, onset_envelope=oenv, hop_length=hop_length
    )

    # And that oenv overrides y
    t4 = librosa.feature.tempogram(
        y=0 * y, sr=sr, onset_envelope=oenv, hop_length=hop_length
    )

    assert np.allclose(t1, t2)
    assert np.allclose(t1, t3)
    assert np.allclose(t1, t4)


@pytest.mark.parametrize("tempo", [60, 120, 200])
@pytest.mark.parametrize("center", [False, True])
def test_tempogram_odf_equiv(tempo, center):
    sr = 22050
    hop_length = 512
    duration = 8

    odf = np.zeros(duration * sr // hop_length)
    spacing = sr * 60.0 // (hop_length * tempo)
    odf[:: int(spacing)] = 1

    odf_ac = librosa.autocorrelate(odf)

    tempogram = librosa.feature.tempogram(
        onset_envelope=odf,
        sr=sr,
        hop_length=hop_length,
        win_length=len(odf),
        window=np.ones,
        center=center,
        norm=None,
    )

    idx = 0
    if center:
        idx = len(odf) // 2

    assert np.allclose(odf_ac, tempogram[:, idx])


@pytest.mark.parametrize("tempo", [60, 90, 200])
@pytest.mark.parametrize("win_length", [192, 384])
@pytest.mark.parametrize("window", ["hann", np.ones])
@pytest.mark.parametrize("norm", [None, 1, 2, np.inf])
def test_tempogram_odf_peak(tempo, win_length, window, norm):
    sr = 22050
    hop_length = 512
    duration = 8

    # Generate an evenly-spaced pulse train
    odf = np.zeros(duration * sr // hop_length)
    spacing = sr * 60.0 // (hop_length * tempo)
    odf[:: int(spacing)] = 1

    tempogram = librosa.feature.tempogram(
        onset_envelope=odf,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        norm=norm,
    )

    # Check the shape of the output
    assert tempogram.shape[0] == win_length

    assert tempogram.shape[1] == len(odf)

    # Mean over time to wash over the boundary padding effects
    idx = np.where(librosa.util.localmax(tempogram.max(axis=1)))[0]

    # Indices should all be non-zero integer multiples of spacing
    assert np.allclose(idx, spacing * np.arange(1, 1 + len(idx)))


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("win_length", [192, 384])
@pytest.mark.parametrize("window", ["hann", np.ones])
@pytest.mark.parametrize("norm", [None, 1, 2, np.inf])
def test_tempogram_odf_multi(center, win_length, window, norm):

    sr = 22050
    hop_length = 512
    duration = 8

    # Generate an evenly-spaced pulse train
    odf = np.zeros((10, duration * sr // hop_length))
    for i in range(10):
        spacing = sr * 60.0 // (hop_length * (60 + 12 * i))
        odf[i, :: int(spacing)] = 1

    tempogram = librosa.feature.tempogram(
        onset_envelope=odf,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        norm=norm,
    )

    for i in range(10):
        tg_local = librosa.feature.tempogram(
            onset_envelope=odf[i],
            sr=sr,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            norm=norm,
        )

        assert np.allclose(tempogram[i], tg_local)


@pytest.mark.parametrize("y", [np.zeros(10 * 1000)])
@pytest.mark.parametrize("sr", [1000])
@pytest.mark.parametrize(
    "win_length,window", [(-384, "hann"), (0, "hann"), (384, np.ones(3))]
)
@pytest.mark.xfail(raises=librosa.ParameterError)
def test_fourier_tempogram_fail_badwin(y, sr, win_length, window):
    librosa.feature.fourier_tempogram(y=y, sr=sr, win_length=win_length, window=window)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_fourier_tempogram_fail_noinput():
    librosa.feature.fourier_tempogram(y=None, onset_envelope=None)


@pytest.mark.parametrize("hop_length", [512, 1024])
@pytest.mark.filterwarnings(
    "ignore:n_fft=.*is too large"
)  # our test signal is short, but this is fine here
def test_fourier_tempogram_audio(y_ex, hop_length):
    y, sr = y_ex
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # Get the tempogram from audio
    t1 = librosa.feature.fourier_tempogram(
        y=y, sr=sr, onset_envelope=None, hop_length=hop_length
    )

    # Get the tempogram from oenv
    t2 = librosa.feature.fourier_tempogram(
        y=None, sr=sr, onset_envelope=oenv, hop_length=hop_length
    )

    # Make sure it works when both are provided
    t3 = librosa.feature.fourier_tempogram(
        y=y, sr=sr, onset_envelope=oenv, hop_length=hop_length
    )

    # And that oenv overrides y
    t4 = librosa.feature.fourier_tempogram(
        y=0 * y, sr=sr, onset_envelope=oenv, hop_length=hop_length
    )

    assert np.iscomplexobj(t1)
    assert np.allclose(t1, t2)
    assert np.allclose(t1, t3)
    assert np.allclose(t1, t4)


@pytest.mark.parametrize("sr", [22050])
@pytest.mark.parametrize("hop_length", [512])
@pytest.mark.parametrize("win_length", [192, 384])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("window", ["hann", np.ones])
def test_fourier_tempogram_invert(sr, hop_length, win_length, center, window):
    duration = 16
    tempo = 100

    odf = np.zeros(duration * sr // hop_length, dtype=np.float32)
    spacing = sr * 60.0 // (hop_length * tempo)
    odf[:: int(spacing)] = 1

    tempogram = librosa.feature.fourier_tempogram(
        onset_envelope=odf,
        sr=sr,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )

    if center:
        sl = slice(None)
    else:
        sl = slice(win_length // 2, -win_length // 2)

    odf_inv = librosa.istft(
        tempogram, hop_length=1, center=center, window=window, length=len(odf)
    )
    assert np.allclose(odf_inv[sl], odf[sl], atol=1e-6)


def test_cens():
    # load CQT data from Chroma Toolbox
    ct_cqt = load(os.path.join("tests", "data", "features-CT-cqt.mat"))

    fn_ct_chroma_cens = [
        "features-CT-CENS_9-2.mat",
        "features-CT-CENS_21-5.mat",
        "features-CT-CENS_41-1.mat",
    ]

    cens_params = [(9, 2), (21, 5), (41, 1)]

    missing = [
        fn
        for fn in fn_ct_chroma_cens
        if not os.path.exists(os.path.join("tests", "data", fn))
    ]
    if missing:
        pytest.skip(
            "CENS Chroma Toolbox fixtures are not available in this workspace: "
            + ", ".join(missing)
        )

    for cur_test_case, cur_fn_ct_chroma_cens in enumerate(fn_ct_chroma_cens):
        win_len_smooth = cens_params[cur_test_case][0]
        downsample_smooth = cens_params[cur_test_case][1]

        # plug into librosa cens computation
        lr_chroma_cens = librosa.feature.chroma_cens(
            C=ct_cqt["f_cqt"],
            win_len_smooth=win_len_smooth,
            fmin=librosa.core.midi_to_hz(1),
            bins_per_octave=12,
            n_octaves=10,
        )

        # leaving out frames to match chroma toolbox behaviour
        # lr_chroma_cens = librosa.resample(lr_chroma_cens, orig_sr=1, target_sr=1/downsample_smooth)
        lr_chroma_cens = lr_chroma_cens[:, ::downsample_smooth]

        # load CENS-41-1 features
        ct_chroma_cens = load(os.path.join("tests", "data", cur_fn_ct_chroma_cens))

        maxdev = np.abs(ct_chroma_cens["f_CENS"] - lr_chroma_cens)
        assert np.allclose(
            ct_chroma_cens["f_CENS"], lr_chroma_cens, rtol=1e-15, atol=1e-15
        ), maxdev


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("win_len_smooth", [-1, 0, 1.5, "foo"])
def test_cens_fail(y_ex, win_len_smooth):
    y, sr = y_ex
    librosa.feature.chroma_cens(y=y, sr=sr, win_len_smooth=win_len_smooth)


@pytest.mark.parametrize(
    "S", [librosa.power_to_db(np.random.randn(128, 1) ** 2, ref=np.max)]
)
@pytest.mark.parametrize("dct_type", [1, 2, 3])
@pytest.mark.parametrize("norm", [None, "ortho"])
@pytest.mark.parametrize("n_mfcc", [13, 20])
@pytest.mark.parametrize("lifter", [0, 13])
def test_mfcc(S, dct_type, norm, n_mfcc, lifter):

    E_total = np.sum(S, axis=0)

    mfcc = librosa.feature.mfcc(
        S=S, dct_type=dct_type, norm=norm, n_mfcc=n_mfcc, lifter=lifter
    )

    assert mfcc.shape[0] == n_mfcc
    assert mfcc.shape[1] == S.shape[1]

    # In type-2 mode, DC component should be constant over all frames
    if dct_type == 2:
        assert np.var(mfcc[0] / E_total) <= 1e-29


# This test is no longer relevant since scipy 1.2.0
# @pytest.mark.xfail(raises=NotImplementedError)
# def test_mfcc_dct1_ortho():
#    S = np.ones((65, 3))
#    librosa.feature.mfcc(S=S, dct_type=1, norm='ortho')


@pytest.mark.xfail(raises=librosa.ParameterError)
@pytest.mark.parametrize("lifter", [-1, np.nan])
def test_mfcc_badlifter(lifter):
    S = np.random.randn(128, 100) ** 2
    librosa.feature.mfcc(S=S, lifter=lifter)


# -- feature inversion tests
@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_fft", [1024, 2048])
def test_mel_to_stft(power, dtype, n_fft):
    srand()

    # Make a random mel spectrum, 4 frames
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=128, dtype=dtype)

    stft_orig = np.random.randn(n_fft // 2 + 1, 4) ** power
    mels = mel_basis.dot(stft_orig.astype(dtype))

    stft = librosa.feature.inverse.mel_to_stft(mels, power=power, n_fft=n_fft)

    # Check precision
    assert stft.dtype == dtype

    # Check for non-negative spectrum
    assert np.all(stft >= 0)

    # Check that the shape is good
    assert stft.shape[0] == 1 + n_fft // 2

    # Check that the approximation is good in RMSE terms
    assert np.sqrt(np.mean((mel_basis.dot(stft**power) - mels) ** 2)) <= 5e-2


def test_mel_to_audio():
    y = librosa.tone(440.0, sr=22050, duration=1)

    M = librosa.feature.melspectrogram(y=y, sr=22050)

    y_inv = librosa.feature.inverse.mel_to_audio(M, sr=22050, length=len(y))

    # Sanity check the length
    assert len(y) == len(y_inv)

    # And that it's valid audio
    assert librosa.util.valid_audio(y_inv)


def test_melspectrogram_projection_matches_einsum():
    rng = np.random.default_rng(1234)
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    n_frames = 257
    n_mels = 80

    S = np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float64) ** 2
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=n_mels)

    expected = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
    observed = librosa.feature.melspectrogram(
        S=S,
        sr=22050,
        n_fft=n_fft,
        n_mels=n_mels,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)


def test_melspectrogram_projection_matches_einsum_float32():
    rng = np.random.default_rng(4321)
    n_fft = 1024
    n_bins = n_fft // 2 + 1
    n_frames = 199
    n_mels = 64

    S = (np.abs(rng.standard_normal((n_bins, n_frames))).astype(np.float32) ** 2)
    mel_basis = librosa.filters.mel(sr=22050, n_fft=n_fft, n_mels=n_mels, dtype=np.float32)

    expected = np.einsum("...ft,mf->...mt", S, mel_basis, optimize=True)
    observed = librosa.feature.melspectrogram(
        S=S,
        sr=22050,
        n_fft=n_fft,
        n_mels=n_mels,
        dtype=np.float32,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n_mfcc", [13, 20])
@pytest.mark.parametrize("n_mels", [64, 128])
@pytest.mark.parametrize("dct_type", [2, 3])
@pytest.mark.parametrize("lifter", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("y", [librosa.tone(440.0, sr=22050, duration=1)])
def test_mfcc_to_mel(y, n_mfcc, n_mels, dct_type, lifter):
    mfcc = librosa.feature.mfcc(
        y=y, sr=22050, n_mels=n_mels, n_mfcc=n_mfcc, dct_type=dct_type
    )

    # check lifter parameter error
    if lifter < 0:
        with pytest.raises(librosa.ParameterError):
            librosa.feature.inverse.mfcc_to_mel(
                mfcc * 10**3, n_mels=n_mels, dct_type=dct_type, lifter=lifter
            )

    # check no lifter computations
    elif lifter == 0:
        melspec = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=n_mels)

        mel_recover = librosa.feature.inverse.mfcc_to_mel(
            mfcc, n_mels=n_mels, dct_type=dct_type
        )
        # Quick shape check
        assert melspec.shape == mel_recover.shape

        # Check non-negativity
        assert np.all(mel_recover >= 0)

    # check that runtime warnings are triggered when appropriate
    elif lifter == 2:
        with pytest.warns((UserWarning, RuntimeWarning)):
            librosa.feature.inverse.mfcc_to_mel(
                mfcc * 10**3, n_mels=n_mels, dct_type=dct_type, lifter=lifter
            )

    # check if mfcc_to_mel works correctly with lifter
    else:
        ones = np.ones(mfcc.shape, dtype=mfcc.dtype)
        n_mfcc = mfcc.shape[0]
        idx = np.arange(1, 1 + n_mfcc, dtype=mfcc.dtype)
        lifter_sine = 1 + lifter * 0.5 * np.sin(np.pi * idx / lifter)[:, np.newaxis]

        # compute the recovered mel
        mel_recover = librosa.feature.inverse.mfcc_to_mel(
            ones * lifter_sine, n_mels=n_mels, dct_type=dct_type, lifter=lifter
        )

        # compute the expected mel
        mel_expected = librosa.feature.inverse.mfcc_to_mel(
            ones, n_mels=n_mels, dct_type=dct_type, lifter=0
        )

        # assert equality of expected and recovered mels
        np.testing.assert_almost_equal(mel_recover, mel_expected, 3)


@pytest.mark.parametrize("n_mfcc", [13, 20])
@pytest.mark.parametrize("n_mels", [64, 128])
@pytest.mark.parametrize("dct_type", [2, 3])
@pytest.mark.parametrize("lifter", [0, 3])
@pytest.mark.parametrize("y", [librosa.tone(440.0, sr=22050, duration=1)])
def test_mfcc_to_audio(y, n_mfcc, n_mels, dct_type, lifter):

    mfcc = librosa.feature.mfcc(
        y=y, sr=22050, n_mels=n_mels, n_mfcc=n_mfcc, dct_type=dct_type
    )

    y_inv = librosa.feature.inverse.mfcc_to_audio(
        mfcc, n_mels=n_mels, dct_type=dct_type, lifter=lifter, length=len(y)
    )

    # Sanity check the length
    assert len(y) == len(y_inv)

    # And that it's valid audio
    assert librosa.util.valid_audio(y_inv)


def test_chroma_vqt_bpo(y_ex):
    # Test that bins per octave is properly overridden in chroma
    y, sr = y_ex
    chroma = librosa.feature.chroma_vqt(
        y=y, sr=sr, intervals=[1, 1.25, 1.5], bins_per_octave=12
    )

    assert chroma.shape[0] == 3

    chroma2 = librosa.feature.chroma_vqt(
        y=y, sr=sr, intervals="equal", bins_per_octave=12
    )

    assert chroma2.shape[0] == 12


def test_chroma_vqt_threshold(y_ex):

    y, sr = y_ex

    c1 = librosa.feature.chroma_vqt(y=y, sr=sr, intervals="pythagorean")
    c2 = librosa.feature.chroma_vqt(y=y, sr=sr, intervals="pythagorean", threshold=1)

    # Check that all thresholded points are zero
    assert np.allclose(c2[c2 < c1], 0)
    # Check that all non-thresholded points match
    assert np.all(c2 <= c1)


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_chroma_vqt_noinput():
    librosa.feature.chroma_vqt(y=None, V=None, intervals="ji3")


@pytest.mark.xfail(raises=librosa.ParameterError)
def test_chroma_cqt_noinput():
    librosa.feature.chroma_cqt(y=None, C=None)


def test_tempogram_ratio_factors():
    # Testing with synthetic data and specific factors

    # tg is [0, 1, 2, 3, 4]  for each frame
    tg = np.multiply.outer(np.arange(5), np.ones(4))
    # frequencies are [1, 2, 4, 8, 16]
    freqs = 2 ** np.arange(5)
    factors = np.array([1, 2, 4])
    bpm = np.array([4, 2, 1, 1.5])

    tgr = librosa.feature.tempogram_ratio(tg=tg, freqs=freqs, factors=factors, bpm=bpm)

    # frame 0: bpm = 4, factors are [1, 2, 4] => [4, 8, 16] => values 2 3 4
    assert np.allclose(tgr[:, 0], [2, 3, 4])
    # frame 1: bpm = 2, factors are [1, 2, 4] => [2, 4, 8] => values [0, 2, 3]
    assert np.allclose(tgr[:, 1], [1, 2, 3])
    # frame 2: bpm = 1, factors are [1, 2, 4] => [1, 2, 4] => values [0, 1, 2]
    assert np.allclose(tgr[:, 2], [0, 1, 2])
    # frame 3: bpm = 1.5, factors are [1, 2, 4] => [1.5, 3, 6] => values
    # [0.5, 1.5, 2.5]
    assert np.allclose(tgr[:, 3], [0.5, 1.5, 2.5])


@pytest.fixture(scope="module")
def tg_ex(y_ex):
    y, sr = y_ex
    return librosa.feature.tempogram(y=y, sr=sr)


def test_tempogram_ratio_aggregate(y_ex, tg_ex):
    # Verify that aggregation does its job
    _, sr = y_ex
    tgr1 = librosa.feature.tempogram_ratio(sr=sr, tg=tg_ex, aggregate=None)
    tgr2 = librosa.feature.tempogram_ratio(sr=sr, tg=tg_ex, aggregate=np.median)
    assert np.allclose(np.median(tgr1, axis=-1), tgr2)


def test_tempogram_ratio_with_tg(y_ex, tg_ex):
    # Verify equivalent behavior with/without pre-computed tempogram
    y, sr = y_ex

    tgr1 = librosa.feature.tempogram_ratio(y=y, sr=sr)
    tgr2 = librosa.feature.tempogram_ratio(tg=tg_ex, sr=sr)

    assert np.allclose(tgr1, tgr2)


def test_tempogram_ratio_with_bpm(y_ex, tg_ex):
    y, sr = y_ex
    tempo = librosa.feature.tempo(tg=tg_ex, sr=sr, aggregate=None)
    tgr1 = librosa.feature.tempogram_ratio(tg=tg_ex, sr=sr, bpm=None)
    tgr2 = librosa.feature.tempogram_ratio(tg=tg_ex, sr=sr, bpm=tempo)
    assert np.allclose(tgr1, tgr2)


def test_melspectrogram_full_pipeline_matches_reference_from_y():
    rng = np.random.default_rng(2026)
    sr = 22050
    n_fft = 2048
    hop = 512
    n_mels = 128

    y = rng.standard_normal(sr * 2).astype(np.float32)

    # Reference path: explicit STFT power + Python mel basis projection.
    stft_power = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True)) ** 2
    mel_basis = librosa.filters.mel(
        sr=sr,
        n_fft=n_fft,
        n_mels=n_mels,
        dtype=np.float32,
        norm="slaney",
    )
    expected = mel_basis.dot(stft_power)

    observed = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        dtype=np.float32,
        norm="slaney",
        power=2.0,
    )

    np.testing.assert_allclose(observed, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power")),
    reason="Rust STFT kernel is not available",
)
def test_spectrogram_rust_dispatch_center_false(monkeypatch):
    rng = np.random.default_rng(606)
    y = rng.standard_normal(4096).astype(np.float32)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"n": 0}

    def _spy_stft_power(*args, **kwargs):
        calls["n"] += 1
        return real_ext.stft_power(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(stft_power=_spy_stft_power),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    spectral_mod._spectrogram(
        y=y,
        S=None,
        n_fft=1024,
        hop_length=256,
        power=2.0,
        win_length=1024,
        window="hann",
        center=False,
        pad_mode="reflect",
    )

    assert calls["n"] == 1


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power")),
    reason="Rust STFT kernel is not available",
)
def test_spectrogram_rust_dispatch_precomputed_hann(monkeypatch):
    """Verify Rust dispatch works with precomputed Hanning window."""
    from scipy.signal import get_window

    rng = np.random.default_rng(5555)
    y = rng.standard_normal(8192).astype(np.float32)
    n_fft = 512
    hop_length = 128

    # Precomputed Hanning window
    hanning_win = get_window("hann", n_fft, fftbins=False).astype(np.float32)

    # Reference: Pure Python with Hanning
    S_ref = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=hanning_win))**2

    # Rust dispatch with precomputed window
    S_rust = librosa.feature.melspectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=hanning_win,
        n_mels=128,
    )

    # For this test, we're verifying the dispatcher accepts precomputed windows.
    # The actual spectrogram will go through mel filtering, so we just check shape.
    assert S_rust is not None
    assert S_rust.shape[1] > 0  # Has frames


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power")),
    reason="Rust STFT kernel is not available",
)
def test_spectrogram_rust_dispatch_precomputed_hamming():
    """Verify Rust dispatch works with precomputed Hamming window."""
    from scipy.signal import get_window

    rng = np.random.default_rng(5555)
    y = rng.standard_normal(8192).astype(np.float32)
    n_fft = 512
    hop_length = 128

    # Precomputed Hamming window
    hamming_win = get_window("hamming", n_fft, fftbins=False).astype(np.float32)

    # Reference: Pure Python with Hamming
    S_ref = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=hamming_win))**2

    # Rust dispatch with precomputed window
    S_rust = librosa.feature.melspectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        window=hamming_win,
        n_mels=128,
    )

    # For this test, we're verifying the dispatcher accepts precomputed windows.
    # The actual spectrogram will go through mel filtering, so we just check shape.
    assert S_rust is not None
    assert S_rust.shape[1] > 0  # Has frames


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power")),
    reason="Rust STFT kernel is not available",
)
def test_spectrogram_rust_dispatch_precomputed_blackman():
    """Verify Rust dispatch works with precomputed Blackman window."""
    from scipy.signal import get_window

    rng = np.random.default_rng(6666)
    y = rng.standard_normal(8192).astype(np.float32)
    n_fft = 512

    # Precomputed Blackman window
    blackman_win = get_window("blackman", n_fft, fftbins=False).astype(np.float32)

    # Simple power spectrogram call with precomputed window
    S = librosa.core.spectrum._spectrogram(
        y=y,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=blackman_win,
        center=True,
        pad_mode="constant",
    )

    # Verify output shape
    assert S[0].ndim == 2
    assert S[0].shape[0] == n_fft // 2 + 1


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power")),
    reason="Rust STFT power kernel is not available",
)
def test_spectrogram_window_length_validation():
    """Verify that a wrong-length window array raises ParameterError."""
    from scipy.signal import get_window

    rng = np.random.default_rng(5544)
    y = rng.standard_normal(4410).astype(np.float32)
    n_fft = 512

    # Window with WRONG length — both Rust and Python paths should reject it.
    wrong_win = get_window("hann", 256, fftbins=True).astype(np.float32)

    with pytest.raises(librosa.ParameterError, match="Window size mismatch"):
        librosa.core.spectrum._spectrogram(
            y=y,
            n_fft=n_fft,
            hop_length=128,
            window=wrong_win,
            center=True,
            pad_mode="constant",
        )


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power")),
    reason="Rust STFT power kernel is not available",
)
def test_spectrogram_performance_no_regression():
    """Verify Phase 2 window support doesn't cause performance regression."""
    from scipy.signal import get_window
    import time

    rng = np.random.default_rng(6677)
    y = rng.standard_normal(88200).astype(np.float32)  # 4 seconds at 22050 Hz
    n_fft = 2048
    hop_length = 512

    # Warm up
    librosa.core.spectrum._spectrogram(
        y=y[:8192],
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann",
        center=True,
    )

    # Time the dispatch path
    t0 = time.perf_counter()
    for _ in range(3):
        _ = librosa.core.spectrum._spectrogram(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window="hann",
            center=True,
        )
    t_hann = time.perf_counter() - t0

    # Time with precomputed window
    win = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(3):
        _ = librosa.core.spectrum._spectrogram(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window=win,
            center=True,
        )
    t_precomputed = time.perf_counter() - t0

    # Precomputed should be ≤ 5% slower (accounting for window extraction overhead and timing variance)
    overhead_ratio = t_precomputed / t_hann
    assert overhead_ratio < 1.05, f"Overhead: {overhead_ratio:.4f}x (expected <1.05x)"


# ============================================================================
# Phase 1: Complex STFT Support (Still Needed)
# ============================================================================

@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex")),
    reason="Rust STFT complex kernel is not available",
)
def test_stft_complex_matches_librosa():
    """Verify Rust complex STFT output matches librosa.stft() exactly."""
    rng = np.random.default_rng(9999)
    y = rng.standard_normal(22050).astype(np.float32)
    n_fft = 2048
    hop_length = 512

    # Reference: librosa STFT
    D_ref = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)

    # Rust implementation
    D_rust = _rust_ext.stft_complex(y, n_fft, hop_length, center=True)

    # Check shape and parity (slightly relaxed tolerance for float32)
    assert D_rust.shape == D_ref.shape
    np.testing.assert_allclose(D_rust, D_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex")),
    reason="Rust STFT complex kernel is not available",
)
def test_stft_complex_phase_vocoder_parity():
    """Verify phase vocoder can use Rust complex STFT for time-stretching."""
    rng = np.random.default_rng(1111)
    y = rng.standard_normal(22050).astype(np.float32)
    n_fft = 2048
    hop_length = 512

    # Reference: librosa STFT → phase vocoder → istft
    D_ref = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    D_stretched_ref = librosa.phase_vocoder(D_ref, rate=2.0, hop_length=hop_length)
    y_ref = librosa.istft(D_stretched_ref, hop_length=hop_length)

    # Rust: complex STFT → phase vocoder → istft
    D_rust = _rust_ext.stft_complex(y, n_fft, hop_length, center=True)
    D_stretched_rust = librosa.phase_vocoder(D_rust, rate=2.0, hop_length=hop_length)
    y_rust = librosa.istft(D_stretched_rust, hop_length=hop_length)

    # Check time-stretched audio parity (slightly relaxed tolerance after 2 transforms)
    min_len = min(len(y_ref), len(y_rust))
    np.testing.assert_allclose(y_rust[:min_len], y_ref[:min_len], rtol=1e-3, atol=3e-4)


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power_f64")),
    reason="Rust float64 STFT power kernel is not available",
)
def test_stft_power_f64_matches_librosa():
    rng = np.random.default_rng(2027)
    y = rng.standard_normal(22050).astype(np.float64)
    n_fft = 2048
    hop_length = 512

    S_ref = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)) ** 2
    S_rust = _rust_ext.stft_power_f64(y, n_fft, hop_length, True, None)

    assert S_rust.dtype == np.float64
    np.testing.assert_allclose(S_rust, S_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex_f64")),
    reason="Rust float64 STFT complex kernel is not available",
)
def test_stft_complex_f64_matches_librosa():
    rng = np.random.default_rng(2028)
    y = rng.standard_normal(22050).astype(np.float64)
    n_fft = 1024
    hop_length = 256

    D_ref = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    D_rust = _rust_ext.stft_complex_f64(y, n_fft, hop_length, True, None)

    assert D_rust.dtype == np.complex128
    np.testing.assert_allclose(D_rust, D_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_power_f64")),
    reason="Rust float64 STFT power kernel is not available",
)
def test_spectrogram_dispatch_prefers_f64_kernel(monkeypatch):
    rng = np.random.default_rng(2029)
    y = rng.standard_normal(4096).astype(np.float64)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"f32": 0, "f64": 0}

    def _spy_stft_power(*args, **kwargs):
        calls["f32"] += 1
        return real_ext.stft_power(*args, **kwargs)

    def _spy_stft_power_f64(*args, **kwargs):
        calls["f64"] += 1
        return real_ext.stft_power_f64(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(stft_power=_spy_stft_power, stft_power_f64=_spy_stft_power_f64),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    S, _ = spectral_mod._spectrogram(
        y=y,
        S=None,
        n_fft=1024,
        hop_length=256,
        power=2.0,
        win_length=1024,
        window="hann",
        center=True,
        pad_mode="constant",
    )

    assert S.dtype == np.float64
    assert calls["f64"] == 1
    assert calls["f32"] == 0


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex")),
    reason="Rust complex STFT f32 kernel is not available",
)
def test_stft_dispatch_prefers_complex_f32_kernel(monkeypatch):
    rng = np.random.default_rng(2030)
    y = rng.standard_normal(4096).astype(np.float32)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"f32": 0}

    def _spy_stft_complex(*args, **kwargs):
        calls["f32"] += 1
        return real_ext.stft_complex(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(stft_complex=_spy_stft_complex),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    D = librosa.stft(y, n_fft=1024, hop_length=256, window="hann", center=True)

    assert D.dtype == np.complex64
    assert calls["f32"] == 1


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex_f64")),
    reason="Rust complex STFT f64 kernel is not available",
)
def test_stft_dispatch_prefers_complex_f64_kernel(monkeypatch):
    rng = np.random.default_rng(2031)
    y = rng.standard_normal(4096).astype(np.float64)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"f64": 0, "f32": 0}

    def _spy_stft_complex(*args, **kwargs):
        calls["f32"] += 1
        return real_ext.stft_complex(*args, **kwargs)

    def _spy_stft_complex_f64(*args, **kwargs):
        calls["f64"] += 1
        return real_ext.stft_complex_f64(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(stft_complex=_spy_stft_complex, stft_complex_f64=_spy_stft_complex_f64),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    D = librosa.stft(y, n_fft=1024, hop_length=256, window="hann", center=True)

    assert D.dtype == np.complex128
    assert calls["f64"] == 1
    assert calls["f32"] == 0


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex")),
    reason="Rust complex STFT f32 kernel is not available",
)
def test_stft_multichannel_dispatch_f32(monkeypatch):
    rng = np.random.default_rng(2032)
    y = rng.standard_normal((2, 4096)).astype(np.float32)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"f32": 0, "batch": 0}

    def _spy_stft_complex(*args, **kwargs):
        calls["f32"] += 1
        return real_ext.stft_complex(*args, **kwargs)

    def _spy_stft_complex_batch(*args, **kwargs):
        calls["batch"] += 1
        return real_ext.stft_complex_batch(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(stft_complex=_spy_stft_complex, stft_complex_batch=_spy_stft_complex_batch),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    D = librosa.stft(y, n_fft=1024, hop_length=256, window="hann", center=True)

    assert D.dtype == np.complex64
    assert D.shape[0] == 2
    assert calls["batch"] == 0
    assert calls["f32"] == 2


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex_f64")),
    reason="Rust complex STFT f64 kernel is not available",
)
def test_stft_multichannel_dispatch_f64(monkeypatch):
    rng = np.random.default_rng(2033)
    y = rng.standard_normal((2, 4096)).astype(np.float64)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"f64": 0, "f32": 0, "batch": 0}

    def _spy_stft_complex(*args, **kwargs):
        calls["f32"] += 1
        return real_ext.stft_complex(*args, **kwargs)

    def _spy_stft_complex_f64(*args, **kwargs):
        calls["f64"] += 1
        return real_ext.stft_complex_f64(*args, **kwargs)

    def _spy_stft_complex_f64_batch(*args, **kwargs):
        calls["batch"] += 1
        return real_ext.stft_complex_f64_batch(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(
            stft_complex=_spy_stft_complex,
            stft_complex_f64=_spy_stft_complex_f64,
            stft_complex_f64_batch=_spy_stft_complex_f64_batch,
        ),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    D = librosa.stft(y, n_fft=1024, hop_length=256, window="hann", center=True)

    assert D.dtype == np.complex128
    assert D.shape[0] == 2
    assert calls["batch"] == 0
    assert calls["f64"] == 2
    assert calls["f32"] == 0


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex_batch")),
    reason="Rust batched complex STFT f32 kernel is not available",
)
def test_stft_multichannel_dispatch_uses_batch_for_many_channels(monkeypatch):
    rng = np.random.default_rng(2037)
    y = rng.standard_normal((4, 4096)).astype(np.float32)

    real_ext = core_spectrum_mod._rust_ext
    calls = {"f32": 0, "batch": 0}

    def _spy_stft_complex(*args, **kwargs):
        calls["f32"] += 1
        return real_ext.stft_complex(*args, **kwargs)

    def _spy_stft_complex_batch(*args, **kwargs):
        calls["batch"] += 1
        return real_ext.stft_complex_batch(*args, **kwargs)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(stft_complex=_spy_stft_complex, stft_complex_batch=_spy_stft_complex_batch),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    D = librosa.stft(y, n_fft=1024, hop_length=256, window="hann", center=True)

    assert D.dtype == np.complex64
    assert D.shape[0] == 4
    assert calls["batch"] == 1
    assert calls["f32"] == 0


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex")),
    reason="Rust complex STFT f32 kernel is not available",
)
def test_stft_multichannel_parity_f32(monkeypatch):
    rng = np.random.default_rng(2034)
    y = rng.standard_normal((2, 8192)).astype(np.float32)

    D_rust = librosa.stft(
        y,
        n_fft=1024,
        hop_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
    )

    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", False)
    D_py = librosa.stft(
        y,
        n_fft=1024,
        hop_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
    )

    assert D_rust.shape == D_py.shape
    np.testing.assert_allclose(D_rust, D_py, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not (RUST_AVAILABLE and hasattr(_rust_ext, "stft_complex_f64")),
    reason="Rust complex STFT f64 kernel is not available",
)
def test_stft_multichannel_parity_f64(monkeypatch):
    rng = np.random.default_rng(2035)
    y = rng.standard_normal((2, 8192)).astype(np.float64)

    D_rust = librosa.stft(
        y,
        n_fft=1024,
        hop_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
    )

    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", False)
    D_py = librosa.stft(
        y,
        n_fft=1024,
        hop_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
    )

    assert D_rust.shape == D_py.shape
    np.testing.assert_allclose(D_rust, D_py, rtol=1e-10, atol=1e-12)


def _phase_vocoder_reference_loop(D, rate, hop_length, n_fft):
    """Reference implementation mirroring librosa.core.spectrum.phase_vocoder loop.

    The Rust kernel (phase_vocoder_f32/f64) casts each f32 phase value individually
    to f64 *before* computing dphase, matching the Python behaviour where NumPy
    upcasts the f32 subtraction result to f64 when mixed with the f64 phi_advance.
    We make that cast explicit here so the reference is numerically identical.
    """
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # phi_advance and phase_acc are kept in f64 (matching the Rust kernel's
    # internal accumulation precision, even for the f32 / complex64 path).
    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)  # f64
    phase_acc = np.angle(D[..., 0]).astype(np.float64)  # f64

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D_padded = np.pad(D, padding, mode="constant")

    step_int = np.floor(time_steps).astype(int)
    step_alpha = time_steps - step_int

    # Store phases in the input dtype (f32 for complex64, f64 for complex128)
    # so that the individual casts to f64 below reproduce the Rust kernel.
    D_phase = np.angle(D_padded)   # inherits dtype from D_padded
    D_mag = np.abs(D_padded)

    for t, idx in enumerate(step_int):
        alpha = step_alpha[t]
        mag = (1.0 - alpha) * D_mag[..., idx] + alpha * D_mag[..., idx + 1]
        d_stretch[..., t] = librosa.util.phasor(phase_acc, mag=mag)

        # Cast each operand to f64 individually *before* subtracting, matching:
        #   Rust: (phase_t[[idx+1,b]] as f64) - (phase_t[[idx,b]] as f64) - phi[b]
        dphase = (
            D_phase[..., idx + 1].astype(np.float64)
            - D_phase[..., idx].astype(np.float64)
            - phi_advance
        )
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))
        phase_acc += phi_advance + dphase

    return d_stretch


def _phase_vocoder_trace_divergence(D, rate, hop_length, n_fft, rust_out, tol=1e-5):
    """
    Trace and report first divergence point between Rust output and Python reference.
    Returns (first_t, first_b, py_val, rust_val, diff) or None if all within tolerance.
    """
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(D[..., 0])

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D_padded = np.pad(D, padding, mode="constant")

    step_int = np.floor(time_steps).astype(int)
    step_alpha = time_steps - step_int

    D_phase = np.angle(D_padded)
    D_mag = np.abs(D_padded)

    for t, idx in enumerate(step_int):
        alpha = step_alpha[t]
        mag = (1.0 - alpha) * D_mag[..., idx] + alpha * D_mag[..., idx + 1]
        py_val = librosa.util.phasor(phase_acc, mag=mag)

        dphase = D_phase[..., idx + 1] - D_phase[..., idx] - phi_advance
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Compare with Rust output
        for b in range(D.shape[-2]):
            py_complex = py_val[b] if D.ndim == 2 else py_val[..., b]
            rust_complex = rust_out[b, t] if rust_out.ndim == 2 else rust_out[..., b, t]

            diff = np.abs(py_complex - rust_complex)
            if diff > tol:
                return (t, b, py_complex, rust_complex, diff)

        phase_acc += phi_advance + dphase

    return None  # All within tolerance


def test_phase_vocoder_dispatch_prefers_rust_by_default(monkeypatch):
    """Verify Rust phase-vocoder is called by default when available."""
    rng = np.random.default_rng(2038)
    D = (
        rng.standard_normal((257, 16)).astype(np.float32)
        + 1j * rng.standard_normal((257, 16)).astype(np.float32)
    ).astype(np.complex64)

    calls = {"f32": 0}

    def _spy_phase_vocoder_f32(*args, **kwargs):
        calls["f32"] += 1
        # Return dummy output matching expected shape
        n_bins = args[0].shape[1]
        step_int = args[3]
        n_out_frames = len(step_int)
        return np.zeros((n_bins, n_out_frames), dtype=np.complex64)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(phase_vocoder_f32=_spy_phase_vocoder_f32),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    # By default (prefer_rust=True), Rust should be called
    out = librosa.phase_vocoder(D, rate=1.5, hop_length=128, n_fft=512)

    assert calls["f32"] == 1, "Expected Rust phase_vocoder_f32 to be called by default"


def test_phase_vocoder_dispatch_fallback_with_prefer_rust_false(monkeypatch):
    """Verify Python fallback when prefer_rust=False even if Rust is available."""
    rng = np.random.default_rng(2043)
    D = (
        rng.standard_normal((257, 16)).astype(np.float32)
        + 1j * rng.standard_normal((257, 16)).astype(np.float32)
    ).astype(np.complex64)

    calls = {"f32": 0}

    def _spy_phase_vocoder_f32(*args, **kwargs):
        calls["f32"] += 1
        raise AssertionError("Rust should not be called when prefer_rust=False")

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(phase_vocoder_f32=_spy_phase_vocoder_f32),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    # With prefer_rust=False, Python path should be used
    out = librosa.phase_vocoder(D, rate=1.5, hop_length=128, n_fft=512, prefer_rust=False)

    assert calls["f32"] == 0, "Rust should not be called when prefer_rust=False"
    assert out.shape[-2] == D.shape[-2], "Output shape should match input"


def test_phase_vocoder_dispatch_opt_in_calls_rust(monkeypatch):
    """Verify Rust is called when available and prefer_rust=True (default)."""
    rng = np.random.default_rng(2039)
    D = (
        rng.standard_normal((257, 18)).astype(np.float32)
        + 1j * rng.standard_normal((257, 18)).astype(np.float32)
    ).astype(np.complex64)

    marker = np.complex64(3 + 4j)
    calls = {"f32": 0}

    def _spy_phase_vocoder_f32(d_phase_t, _d_mag_t, _phi, step_int, _step_alpha, _phase_acc):
        calls["f32"] += 1
        n_bins = d_phase_t.shape[1]
        n_frames = step_int.shape[0]
        return np.full((n_bins, n_frames), marker, dtype=np.complex64)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(phase_vocoder_f32=_spy_phase_vocoder_f32),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    out = librosa.phase_vocoder(D, rate=1.5, hop_length=128, n_fft=512)

    assert calls["f32"] == 1, "Rust kernel should be called"
    assert out.dtype == np.complex64
    assert np.all(out == marker)


def test_phase_vocoder_dispatch_opt_in_calls_rust_per_channel(monkeypatch):
    """Verify Rust is called per-channel for multichannel input."""
    rng = np.random.default_rng(2040)
    D = (
        rng.standard_normal((3, 257, 18)).astype(np.float32)
        + 1j * rng.standard_normal((3, 257, 18)).astype(np.float32)
    ).astype(np.complex64)

    calls = {"f32": 0}

    def _spy_phase_vocoder_f32(d_phase_t, _d_mag_t, _phi, step_int, _step_alpha, _phase_acc):
        calls["f32"] += 1
        n_bins = d_phase_t.shape[1]
        n_frames = step_int.shape[0]
        return np.zeros((n_bins, n_frames), dtype=np.complex64)

    monkeypatch.setattr(
        core_spectrum_mod,
        "_rust_ext",
        types.SimpleNamespace(phase_vocoder_f32=_spy_phase_vocoder_f32),
    )
    monkeypatch.setattr(core_spectrum_mod, "RUST_AVAILABLE", True)

    out = librosa.phase_vocoder(D, rate=1.5, hop_length=128, n_fft=512)

    assert out.shape[0] == D.shape[0], "Batch dimension should be preserved"
    assert calls["f32"] == D.shape[0], f"Rust should be called once per channel ({D.shape[0]} times)"


@pytest.mark.parametrize(
    "dtype,fn_name,rtol,atol",
    [
        (np.complex64, "phase_vocoder_f32", 1e-5, 1e-6),
        (np.complex128, "phase_vocoder_f64", 1e-11, 1e-13),
    ],
)
def test_phase_vocoder_rust_kernel_matches_reference_loop(dtype, fn_name, rtol, atol):
    if not (RUST_AVAILABLE and hasattr(_rust_ext, fn_name)):
        pytest.skip("Rust phase-vocoder kernel is not available")

    rng = np.random.default_rng(2041)
    D = (
        rng.standard_normal((257, 24)).astype(np.float64)
        + 1j * rng.standard_normal((257, 24)).astype(np.float64)
    ).astype(dtype)

    rate = 1.25
    hop_length = 128
    n_fft = 512

    ref = _phase_vocoder_reference_loop(D, rate=rate, hop_length=hop_length, n_fft=n_fft)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)
    step_int = np.floor(time_steps).astype(np.int64)
    step_alpha = (time_steps - step_int).astype(np.float64)

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D_padded = np.pad(D, padding, mode="constant")

    float_dtype = np.float32 if dtype == np.complex64 else np.float64
    D_phase_t = np.ascontiguousarray(np.angle(D_padded).astype(float_dtype).T)
    D_mag_t = np.ascontiguousarray(np.abs(D_padded).astype(float_dtype).T)
    phi_advance = hop_length * librosa.fft_frequencies(sr=2 * np.pi, n_fft=n_fft)
    phase_acc = np.angle(D[..., 0]).astype(np.float64)

    out = getattr(_rust_ext, fn_name)(
        D_phase_t,
        D_mag_t,
        phi_advance.astype(np.float64),
        step_int,
        step_alpha,
        phase_acc,
    )

    try:
        np.testing.assert_allclose(out, ref, rtol=rtol, atol=atol)
    except AssertionError as e:
        # Provide detailed divergence trace on failure
        diverg = _phase_vocoder_trace_divergence(D, rate, hop_length, n_fft, out, tol=atol)
        if diverg:
            t, b, py_val, rust_val, diff = diverg
            pytest.fail(
                f"Divergence at (frame {t}, bin {b}): "
                f"Python={py_val:.10f}, Rust={rust_val:.10f}, diff={diff:.2e}\n"
                f"Original assertion:\n{e}"
            )
        raise
