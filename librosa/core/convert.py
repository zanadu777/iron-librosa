#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit conversion utilities.

All functions are re-exported from focused submodules for maintainability.
Public import paths are unchanged.
"""
from __future__ import annotations

from ._convert_time import (
    frames_to_samples,
    samples_to_frames,
    frames_to_time,
    time_to_frames,
    time_to_samples,
    samples_to_time,
    blocks_to_frames,
    blocks_to_samples,
    blocks_to_time,
)
from ._convert_pitch import (
    note_to_hz,
    note_to_midi,
    midi_to_note,
    midi_to_hz,
    hz_to_midi,
    hz_to_note,
)
from ._convert_freq import (
    hz_to_mel,
    mel_to_hz,
    hz_to_octs,
    octs_to_hz,
    A4_to_tuning,
    tuning_to_A4,
    fft_frequencies,
    cqt_frequencies,
    mel_frequencies,
    tempo_frequencies,
    fourier_tempo_frequencies,
)
from ._convert_weighting import (
    A_weighting,
    B_weighting,
    C_weighting,
    D_weighting,
    Z_weighting,
    WEIGHTING_FUNCTIONS,
    frequency_weighting,
    multi_frequency_weighting,
    times_like,
    samples_like,
)
from ._convert_svara import (
    midi_to_svara_h,
    hz_to_svara_h,
    note_to_svara_h,
    midi_to_svara_c,
    hz_to_svara_c,
    note_to_svara_c,
    hz_to_fjs,
)

__all__ = [
    "frames_to_samples",
    "frames_to_time",
    "samples_to_frames",
    "samples_to_time",
    "time_to_samples",
    "time_to_frames",
    "blocks_to_samples",
    "blocks_to_frames",
    "blocks_to_time",
    "note_to_hz",
    "note_to_midi",
    "midi_to_hz",
    "midi_to_note",
    "hz_to_note",
    "hz_to_midi",
    "hz_to_mel",
    "hz_to_octs",
    "hz_to_fjs",
    "mel_to_hz",
    "octs_to_hz",
    "A4_to_tuning",
    "tuning_to_A4",
    "fft_frequencies",
    "cqt_frequencies",
    "mel_frequencies",
    "tempo_frequencies",
    "fourier_tempo_frequencies",
    "A_weighting",
    "B_weighting",
    "C_weighting",
    "D_weighting",
    "Z_weighting",
    "frequency_weighting",
    "multi_frequency_weighting",
    "samples_like",
    "times_like",
    "midi_to_svara_h",
    "midi_to_svara_c",
    "note_to_svara_h",
    "note_to_svara_c",
    "hz_to_svara_h",
    "hz_to_svara_c",
]
