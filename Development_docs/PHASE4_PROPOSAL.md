# Phase 4: Advanced DSP Operations & Feature Extraction Acceleration

## Status: PROPOSAL (Awaiting Approval)

This document outlines Phase 4, which extends the Rust acceleration to high-impact DSP operations and feature extraction kernels beyond Phase 3's STFT/Mel/DCT foundation.

---

## Executive Summary

Phase 3 established native float32/float64 kernels for:
- ✅ STFT (power & complex)
- ✅ Mel-spectrogram projection
- ✅ DCT (orthogonal)
- ✅ Onset detection / spectral flux
- ✅ NN filtering

Phase 4 proposes adding **5 strategic operations** that are either:
1. **High-call-frequency** (used in every feature extraction pipeline)
2. **Computationally intensive** (matrix reductions, overlap-add, FFT-based reconstruction)
3. **Current bottlenecks** (Numba-jitted or pure-Python implementations)
4. **Multi-channel-friendly** (scale well with stereo/multichannel audio)

---

## Top 5 Recommended Operations (Priority Order)

### 1. **ISTFT** (Inverse Short-Time Fourier Transform)
**Priority: CRITICAL**

**Why:**
- Direct inverse of STFT, essential for phase reconstruction and audio synthesis
- Currently pure Python with complex overlap-add logic
- Bottleneck for phase vocoder, effects processing, and source separation
- Required for end-to-end neural network training loops

**Current Implementation:**
- `librosa/core/spectrum.py::istft()` (~300 lines of Python)
- Performs overlap-add reconstruction with window sum normalization

**Complexity:** **HIGH**
- Requires reverse FFT with proper windowing
- Manages per-sample overlap regions
- Must handle edge cases (frame count, padding modes)

**Expected Speedup:** **2.5–3.5x**
- Phase 3 STFT achieved 2.5–3.3x on f32, 1.5–2.5x on f64
- ISTFT will be similar but constrained by overlap-add serialization

**Implementation Notes:**
- Add `istft_f32()` and `istft_f64()` Rust kernels
- Dtype-aware Python dispatch in `istft()` function
- Conservative fast-path: support `out is None`, single/multi-channel, basic edge cases

---

### 2. **Power-to-dB & Amplitude-to-dB Conversion**
**Priority: HIGH**

**Why:**
- Used in *every* feature extraction pipeline (MFCC, spectrogram logging, etc.)
- Called millions of times in typical workflows (one per spectrogram frame per bin)
- Currently NumPy element-wise operations with conditional branches
- Minimal implementation overhead, high volume

**Current Implementation:**
- `librosa/core/spectrum.py::power_to_db()` and `amplitude_to_db()`
- Pure NumPy with optional reference scaling and NaN/clipping

**Complexity:** **LOW**
- Simple element-wise logarithm with scaling and threshold handling
- Vectorizable operation

**Expected Speedup:** **1.5–2.5x**
- Element-wise Rust operations are typically 1.5–2x faster due to tight loops
- High impact due to call volume

**Implementation Notes:**
- Add `power_to_db_f32()`, `power_to_db_f64()` kernels
- Add `amplitude_to_db_f32()`, `amplitude_to_db_f64()` kernels
- Support `ref_power` scaling and `amin`/`top_db` thresholding

---

### 3. **Spectral Centroid, Bandwidth, and Rolloff**
**Priority: MEDIUM-HIGH**

**Why:**
- Core feature extraction operations used in music analysis and classification
- Frame-wise weighted reductions (sum products over frequency bins)
- Called frequently in feature engineering pipelines
- Currently pure Python loops per frame

**Current Implementation:**
- `librosa/feature/spectral.py::spectral_centroid()`, `spectral_bandwidth()`, `spectral_rolloff()`
- Per-frame weighted average/variance calculations

**Complexity:** **MEDIUM**
- Straightforward weighted reductions
- Vectorizable across frames

**Expected Speedup:** **1.8–2.5x**
- Typical reduction operations see 2–2.5x speedup in Rust
- Impact scales with frame count (audio length)

**Implementation Notes:**
- Add `spectral_centroid_f32()`, `spectral_centroid_f64()` kernels
- Add `spectral_rolloff_f32()`, `spectral_rolloff_f64()` kernels
- (Optional: bandwidth can be derived from centroid variance)

---

### 4. **RMS Energy (Root Mean Square)**
**Priority: MEDIUM**

**Why:**
- Fundamental feature extraction operation
- Used for activity detection, loudness normalization, and many ML pipelines
- High call frequency (one per frame)
- Simple vectorizable operation

**Current Implementation:**
- `librosa/feature/spectral.py::rms()`
- Frame-wise mean-square root calculation

**Complexity:** **LOW**
- Straightforward frame-wise reduction

**Expected Speedup:** **1.5–2x**
- Simple operation, but high volume justifies Rust implementation

**Implementation Notes:**
- Add `rms_f32()`, `rms_f64()` kernels
- Support frame-by-frame or full reduction

---

### 5. **Chroma Filter Bank Application**
**Priority: MEDIUM**

**Why:**
- Music analysis (chord recognition, key estimation)
- High-dimensional reduction (stft bins → 12 chroma bins)
- Currently pure NumPy matrix operations
- Scales well with multi-channel input

**Current Implementation:**
- `librosa/feature/spectral.py::chroma_stft()` builds chroma filters via `librosa.filters.chroma()`
- Matrix multiplication of complex/power spectrogram with chroma filter bank

**Complexity:** **MEDIUM**
- Matrix-vector products per frame
- Similar to Mel-spectrogram projection (already accelerated in Phase 2–3)

**Expected Speedup:** **1.5–2.5x**
- Depends on filter bank size (typically 12 bins) and input spectrogram size

**Implementation Notes:**
- Reuse lessons from Mel-spectrogram projection
- Optionally pre-compute and cache chroma filters
- Support both linear and log-compressed input spectrograms

---

## Summary Table

| Operation | Complexity | Speedup | Call Volume | Priority |
|---|:---:|---:|:---:|:---:|
| **ISTFT** | HIGH | 2.5–3.5x | Medium | **CRITICAL** |
| **power_to_db / amplitude_to_db** | LOW | 1.5–2.5x | **Very High** | **HIGH** |
| **spectral_centroid / rolloff** | MEDIUM | 1.8–2.5x | High | MEDIUM-HIGH |
| **RMS energy** | LOW | 1.5–2x | **Very High** | MEDIUM |
| **Chroma filters** | MEDIUM | 1.5–2.5x | Medium | MEDIUM |

---

## Implementation Strategy

### Phase 4A: Foundation (Weeks 1–2)
1. **ISTFT + dB conversions** (high-impact, relatively isolated)
   - Implement `istft_f32()`, `istft_f64()` kernels
   - Implement `power_to_db_f32/f64()`, `amplitude_to_db_f32/f64()` kernels
   - Wire dtype-aware dispatch in Python
   - Add comprehensive tests and benchmarks

### Phase 4B: Feature Extraction (Weeks 2–3)
2. **Spectral features** (RMS, centroid, rolloff)
   - Implement `rms_f32()`, `rms_f64()` kernels
   - Implement `spectral_centroid_f32/f64()`, `spectral_rolloff_f32/f64()` kernels
   - Tests and benchmarks

### Phase 4C: Advanced (Week 3–4)
3. **Chroma filters** (nice-to-have, leverages existing patterns)
   - Implement chroma filter matrix application
   - Optimize for typical 12-bin chroma output

### Phase 4D: Hardening & Validation
4. **Broad regression testing** (across all STFT/Mel/feature workflows)
5. **Performance benchmarking** against numpy/numba baselines
6. **Documentation & Phase 4 completion summary**

---

## Files to Create/Modify

### New Rust modules
- `src/istft.rs` — ISTFT kernels
- `src/spectrum_utils.rs` — dB conversions, spectral reductions

### Modified Rust files
- `src/lib.rs` — Register new kernels

### Modified Python files
- `librosa/core/spectrum.py` — Dispatch for ISTFT and dB conversions
- `librosa/feature/spectral.py` — Dispatch for spectral features

### New test files
- `tests/test_phase4_istft.py` — ISTFT parity and dispatch tests
- `tests/test_phase4_spectrum.py` — dB conversion and spectral feature tests

### New benchmark files
- `scripts/benchmark_phase4_istft.py` — ISTFT performance
- `scripts/benchmark_phase4_features.py` — Spectral feature performance

---

## Success Criteria

✅ **Phase 4 is complete when:**

1. ISTFT achieves 2.5–3x speedup on both f32 and f64
2. dB conversions achieve 1.5–2x speedup with high-volume call patterns
3. Spectral features (RMS, centroid, rolloff) achieve measurable speedup
4. All operations support both single-channel and multi-channel inputs
5. Broad regression tests pass (test_core.py, test_multichannel.py, test_features.py)
6. Benchmarks document baseline vs. Rust performance for each operation

---

## Deferred / Future Work

- **Time-stretching (phase_vocoder):** Complex operation requiring heuristic phase reconstruction; defer to Phase 5
- **Constant-Q Transform (CQT):** Requires sparse filterbank; defer to Phase 5
- **Decomposition (NMF):** Currently relies on scikit-learn; consider Phase 5+
- **Advanced perceptual weighting:** Depends on ISTFT; consider Phase 5

---

## Performance Appendix (Expectations)

Based on Phase 3 benchmarks and typical audio workloads:

| Operation | Scenario | Rust Time | NumPy Time | Speedup |
|---|---|---:|---:|---:|
| ISTFT | 10s audio, mono, f32 | ~2 ms | ~5–6 ms | 2.5–3x |
| power_to_db | 1000 frames, f32 | ~0.5 ms | ~1 ms | 2x |
| RMS | 1000 frames, f32 | ~0.3 ms | ~0.5 ms | 1.5–2x |
| spectral_centroid | 1000 frames, 1025 bins, f32 | ~1 ms | ~2 ms | 2x |
| chroma | 1000 frames, 12 bins, f32 | ~0.3 ms | ~0.5 ms | 1.5–2x |

Speedups are estimated based on Phase 3 results and typical operation profiles.

---

**Recommendation:** Approve Phase 4 proposal. Start implementation with ISTFT + dB conversions (4A) in next sprint.

**Completion Target Date:** Mid-April 2026 (4 weeks)

