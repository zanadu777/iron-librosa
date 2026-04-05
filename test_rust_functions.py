#!/usr/bin/env python
"""Test Rust functions directly"""
import numpy as np
from librosa import _rust as rust_ext

# Test onset_flux_median_ref_f32
print("Testing onset_flux_median_ref_f32...")

# Create test input
S = np.random.random((513, 100)).astype(np.float32)  # (freq_bins, time_frames)
S_ref = S.copy()
lag = 2

try:
    result = rust_ext.onset_flux_median_ref_f32(S, S_ref, lag)
    print(f"  Input shape: {S.shape}")
    print(f"  Result shape: {result.shape}")
    print(f"  Result dtype: {result.dtype}")
    print(f"  Result mean: {result.mean():.6f}")
    print(f"  SUCCESS!")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test tempogram_ac_f32
print("Testing tempogram_ac_f32...")

# Create test input (onset envelope)
windowed = np.random.random((1, 256)).astype(np.float32)  # (channels, time_frames)
n_pad = 512

try:
    result = rust_ext.tempogram_ac_f32(windowed, n_pad)
    print(f"  Input shape: {windowed.shape}")
    print(f"  Result shape: {result.shape}")
    print(f"  Result dtype: {result.dtype}")
    print(f"  Result mean: {result.mean():.6f}")
    print(f"  SUCCESS!")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# Test beat_track_dp_f32
print("Testing beat_track_dp_f32...")

# Create test inputs
localscore = np.random.random(100).astype(np.float32)
frames_per_beat = np.full(100, 22050/512*0.5).astype(np.float32)  # ~0.5 beat intervals
tightness = 100.0

try:
    backlink, cumscore = rust_ext.beat_track_dp_f32(localscore, frames_per_beat, tightness)
    print(f"  localscore shape: {localscore.shape}")
    print(f"  backlink shape: {backlink.shape}, dtype: {backlink.dtype}")
    print(f"  cumscore shape: {cumscore.shape}, dtype: {cumscore.dtype}")
    print(f"  cumscore mean: {cumscore.mean():.6f}")
    print(f"  SUCCESS!")
except Exception as e:
    print(f"  ERROR: {e}")

