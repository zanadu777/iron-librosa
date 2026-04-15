import re

import numpy as np
import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


def _cuda_fused_ready() -> bool:
    if not RUST_AVAILABLE or _rust_ext is None:
        return False
    if not hasattr(_rust_ext, "cuda_diagnostics"):
        return False
    diag = _rust_ext.cuda_diagnostics()
    return bool(
        diag.get("cuda_feature_enabled")
        and diag.get("cuda_runtime_available")
        and diag.get("cuda_fused_mel_helper_built")
    )


@pytest.mark.skipif(not _cuda_fused_ready(), reason="CUDA fused mel path unavailable")
def test_fused_mel_reuses_resident_window_and_mel_basis(monkeypatch, capfd):
    rng = np.random.default_rng(20260415)
    n_fft = 1024
    hop = 256
    n_mels = 64

    y = rng.standard_normal(32768).astype(np.float32)
    window = np.hanning(n_fft).astype(np.float32)

    # librosa provides a C-contiguous mel basis; this is the typical fused path input.
    import librosa

    mel_basis = librosa.filters.mel(
        sr=22050,
        n_fft=n_fft,
        n_mels=n_mels,
        dtype=np.float32,
        norm="slaney",
    )

    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cuda-gpu")
    monkeypatch.setenv("IRON_LIBROSA_CUDA_MEL_FUSED_EXPERIMENTAL", "force-on")
    monkeypatch.setenv("IRON_LIBROSA_CUDA_PROFILE", "1")
    monkeypatch.setenv("IRON_LIBROSA_CUDA_DEBUG", "0")

    _rust_ext.melspectrogram_fused_f32(y, n_fft, hop, True, window, mel_basis)
    _rust_ext.melspectrogram_fused_f32(y, n_fft, hop, True, window, mel_basis)

    err = capfd.readouterr().err
    h2d_vals = [
        float(v)
        for v in re.findall(
            r"\[CUDA_PROFILE\].*op=stft_mel_fused(?:_stft_cached)?.*h2d_kb=([\d.]+)",
            err,
        )
    ]

    assert len(h2d_vals) >= 2, f"Expected >=2 fused profile rows, got {len(h2d_vals)}\n{err}"
    first_h2d = h2d_vals[-2]
    second_h2d = h2d_vals[-1]

    # Warm call should avoid re-uploading constant buffers.
    assert second_h2d < first_h2d

    n_frames = 1 + ((len(y) + n_fft - n_fft) // hop)
    expected_signal_kb = ((n_fft + hop * (n_frames - 1)) * 4) / 1024.0
    n_bins = n_fft // 2 + 1
    expected_const_kb = (n_fft * 4 + n_mels * n_bins * 4) / 1024.0
    expected_drop_kb = expected_signal_kb + expected_const_kb
    # Profile rounds to one decimal place; allow slack around expected transfer reduction.
    assert (first_h2d - second_h2d) >= expected_drop_kb * 0.8


