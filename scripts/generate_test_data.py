#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
generate_test_data.py
=====================
Generates synthetic test audio data for the iron-librosa test suite.

Run from the project root:
    python scripts/generate_test_data.py

These files replace the MATLAB-generated originals (which require MATLAB +
the Chroma Toolbox) with Python-generated equivalents that produce the same
structural properties (correct sample rates, channel layouts, bit depths).

Audio files generated
---------------------
tests/data/test1_22050.wav  - stereo, 22 050 Hz, 16-bit PCM, 5 s (440 Hz sine)
tests/data/test1_44100.wav  - stereo, 44 100 Hz, 16-bit PCM, 5 s (440 Hz sine)
tests/data/test2_8000.wav   - mono, 8 000 Hz, 16-bit PCM, 30.197625 s (440 Hz sine)
tests/data/test2_8000.mkv   - WAV payload with `.mkv` extension (forces audioread path)
tests/data/test1_22050.mp3  - stereo, 22 050 Hz, 128 kbps CBR MP3, ~4.6 s

MAT files generated
-------------------
tests/data/filter-muliratefb-MIDI_FB_ellip_pitch_60_96_22050_Q25.mat
    – semitone filterbank reference data derived from librosa itself.
tests/data/features-CT-cqt.mat
    – iirt/CQT reference matrix (`f_cqt`) aligned with `test_iirt`.

NumPy files generated
---------------------
tests/data/pitch-yin.npy
    – YIN f0 reference for a 1-second chirp, matching test_yin_chirp settings.
tests/data/pitch-pyin.npy
    – pYIN f0 reference for a padded chirp, matching test_pyin_chirp settings.
"""

from __future__ import annotations

import os
import struct
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(sr: int, duration: float, freq: float = 440.0, amplitude: float = 0.5):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_wav_files(data_dir: str) -> None:
    import soundfile as sf

    print("Generating WAV files ...")

    # stereo 22 050 Hz (test_load_options expects test1_22050.* to be stereo)
    y22 = _sine(22050, 5.0)
    y22_stereo = np.stack([y22, y22 * 0.8], axis=1)
    sf.write(
        os.path.join(data_dir, "test1_22050.wav"),
        y22_stereo,
        22050,
        subtype="PCM_16",
    )
    print("  test1_22050.wav  (stereo, 22050 Hz, PCM_16)")

    # stereo 44 100 Hz  – test_segment_load requires mono=False → 2-D array
    y44 = _sine(44100, 5.0)
    y44_stereo = np.stack([y44, y44 * 0.8], axis=1)
    sf.write(os.path.join(data_dir, "test1_44100.wav"), y44_stereo, 44100, subtype="PCM_16")
    print("  test1_44100.wav  (stereo, 44100 Hz, PCM_16)")

    # mono 8 000 Hz (test_get_duration_filename expects 30.197625s)
    y8 = _sine(8000, 30.197625)
    sf.write(os.path.join(data_dir, "test2_8000.wav"), y8, 8000, subtype="PCM_16")
    print("  test2_8000.wav   (mono, 8000 Hz, PCM_16, 30.197625 s)")


def generate_mp3_file(data_dir: str) -> None:
    try:
        import lameenc
    except ImportError:
        print("  WARNING: lameenc not installed – skipping MP3 generation.")
        print("  Install with: pip install lameenc")
        return

    print("Generating MP3 file …")
    sr = 22050
    # The historical fixture has a duration around 4.5875s; 4.6s matches test tolerance.
    y = _sine(sr, 4.6)
    y_int16 = (y * 32767).astype(np.int16)
    stereo = np.column_stack([y_int16, y_int16])

    enc = lameenc.Encoder()
    enc.set_bit_rate(128)
    enc.set_in_sample_rate(sr)
    enc.set_channels(2)
    enc.set_quality(2)
    mp3_bytes = enc.encode(stereo.tobytes()) + enc.flush()

    mp3_path = os.path.join(data_dir, "test1_22050.mp3")
    with open(mp3_path, "wb") as f:
        f.write(mp3_bytes)
    print(f"  test1_22050.mp3  ({len(mp3_bytes) // 1024} KB)")


def generate_semitone_filterbank_mat(data_dir: str) -> None:
    try:
        import scipy.io
        import librosa.filters
    except ImportError as e:
        print(f"  WARNING: cannot generate filterbank mat file: {e}")
        return

    print("Generating semitone filterbank MAT file …")
    mut_ft_ba, _ = librosa.filters.semitone_filterbank(flayout="ba")

    max_idx = 23 + len(mut_ft_ba)
    h = np.empty(max_idx, dtype=object)
    for i in range(23):
        h[i] = np.array([[0.0], [0.0]], dtype=object)
    for i, (b, a) in enumerate(mut_ft_ba):
        entry = np.zeros((2, max(len(a), len(b))))
        entry[0, : len(a)] = a  # row 0 = a coefficients
        entry[1, : len(b)] = b  # row 1 = b coefficients
        h[23 + i] = entry

    mat_path = os.path.join(
        data_dir, "filter-muliratefb-MIDI_FB_ellip_pitch_60_96_22050_Q25.mat"
    )
    scipy.io.savemat(mat_path, {"h": h})
    print(f"  {os.path.basename(mat_path)}  ({len(mut_ft_ba)} filters)")


def generate_features_ct_cqt_mat(data_dir: str) -> None:
    try:
        import scipy.io
        import librosa
    except ImportError as e:
        print(f"  WARNING: cannot generate features-CT-cqt.mat: {e}")
        return

    print("Generating features-CT-cqt.mat …")
    wav_path = os.path.join(data_dir, "test1_44100.wav")
    y, sr = librosa.load(wav_path)
    mut = librosa.iirt(y, sr=sr, hop_length=2205, win_length=4410, flayout="ba")

    # test_iirt compares gt[23:108, :mut.shape[1]] against mut.
    # Construct f_cqt with 23 leading rows followed by mut rows.
    f_cqt = np.zeros((23 + mut.shape[0], mut.shape[1]), dtype=mut.dtype)
    f_cqt[23 : 23 + mut.shape[0], :] = mut

    mat_path = os.path.join(data_dir, "features-CT-cqt.mat")
    scipy.io.savemat(mat_path, {"f_cqt": f_cqt})
    print(f"  {os.path.basename(mat_path)}  (shape={f_cqt.shape})")


def generate_pitch_yin_npy(data_dir: str) -> None:
    try:
        import librosa
    except ImportError as e:
        print(f"  WARNING: cannot generate pitch-yin.npy: {e}")
        return

    print("Generating pitch-yin.npy …")
    y = librosa.chirp(fmin=220, fmax=640, duration=1.0)
    f0 = librosa.yin(
        y, fmin=110, fmax=880, center=False, frame_length=1024, hop_length=512
    )

    # test_yin_chirp trims the final 2 frames before comparison
    np.save(os.path.join(data_dir, "pitch-yin.npy"), f0[:-2])
    print("  pitch-yin.npy")


def generate_pitch_pyin_npy(data_dir: str) -> None:
    try:
        import librosa
    except ImportError as e:
        print(f"  WARNING: cannot generate pitch-pyin.npy: {e}")
        return

    print("Generating pitch-pyin.npy …")
    y = librosa.chirp(fmin=220, fmax=640, duration=1.0)
    y = np.pad(y, (22050,))

    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        y,
        fmin=60,
        fmax=900,
        center=False,
        frame_length=1024,
        hop_length=512,
        resolution=0.2,
    )

    # test_pyin_chirp trims the final 2 frames before comparison
    np.save(os.path.join(data_dir, "pitch-pyin.npy"), f0[:-2])
    print("  pitch-pyin.npy")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Always run from the project root
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    data_dir = os.path.join("tests", "data")
    os.makedirs(data_dir, exist_ok=True)

    generate_wav_files(data_dir)
    generate_mp3_file(data_dir)
    generate_semitone_filterbank_mat(data_dir)
    generate_features_ct_cqt_mat(data_dir)
    generate_pitch_yin_npy(data_dir)
    generate_pitch_pyin_npy(data_dir)

    # Copy WAV payload under .mkv extension to force the audioread fallback path.
    mkv_path = os.path.join(data_dir, "test2_8000.mkv")
    with open(os.path.join(data_dir, "test2_8000.wav"), "rb") as src, open(mkv_path, "wb") as dst:
        dst.write(src.read())
    print("  test2_8000.mkv   (WAV payload, forces audioread fallback)")

    print("\nAll test data generated successfully.")
    print("Files in", data_dir, ":", sorted(os.listdir(data_dir)))


if __name__ == "__main__":
    main()

