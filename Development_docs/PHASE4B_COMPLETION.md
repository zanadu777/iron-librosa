# Phase 4B: RMS + Spectral Centroid Acceleration

## Status: ✅ COMPLETE (Current Scope)

Phase 4B extends the Rust acceleration layer with two frequently used spectral feature operations:

- spectrogram-based `rms()`
- `spectral_centroid()` for the common static-frequency-bin case

The implementation is conservative by design: fast-paths are enabled only where behavior matches the existing Python implementation closely, and all other cases fall back to the original code paths.

---

## What Was Implemented

### 1) Native RMS kernels for spectrogram input

File: `src/spectrum_utils.rs`

Added:
- `rms_spectrogram_f32(...) -> PyArray2<f32>`
- `rms_spectrogram_f64(...) -> PyArray2<f64>`

Behavior matches the `librosa.feature.rms(S=...)` spectrogram path:
- real-valued magnitude spectrogram input
- DC half-weighting
- Nyquist half-weighting when `frame_length` is even
- output shape `(1, t)`

### 2) Native spectral centroid kernels

File: `src/spectrum_utils.rs`

Added:
- `spectral_centroid_f32(...) -> PyArray2<f64>`
- `spectral_centroid_f64(...) -> PyArray2<f64>`

Behavior matches the common `librosa.feature.spectral_centroid()` case when:
- the spectrogram is real and non-negative
- frequency bins are static 1D values
- default FFT-bin frequencies are used, or an explicit `float64` 1D frequency vector is provided

### 3) Module exports

File: `src/lib.rs`

Registered the new kernels:
- `spectrum_utils::rms_spectrogram_f32`
- `spectrum_utils::rms_spectrogram_f64`
- `spectrum_utils::spectral_centroid_f32`
- `spectrum_utils::spectral_centroid_f64`

### 4) Public Python dispatch

File: `librosa/feature/spectral.py`

Added guarded Rust dispatch for:
- `librosa.feature.rms(S=...)`
- `librosa.feature.spectral_centroid(...)`

Dispatch guardrails:

#### `rms(S=...)`
Rust path is used only when:
- `S` is real-valued
- `S.dtype` is `float32` or `float64`
- output `dtype` matches `S.dtype`

Otherwise, it falls back to Python.

#### `spectral_centroid(...)`
Rust path is used only when:
- `S` is real-valued and non-negative
- `S.dtype` is `float32` or `float64`
- `freq` is `None` or a static 1D `float64` vector

Variable frequency grids and other edge cases still fall back to Python.

### 5) Phase 4B regression tests

File: `tests/test_phase4_features.py`

Added tests for:
- raw RMS kernel parity (`f32`, `f64`)
- raw spectral centroid kernel parity (`f32`, `f64`)
- public dispatch for multichannel RMS
- public dispatch for spectral centroid with default FFT frequencies
- fallback behavior for unsupported cases
- formula parity for public outputs

---

## Verified Test Runs

### Focused Phase 4B tests
Command target:
- `tests/test_phase4_features.py`

Result:
- `9 passed`

### Existing feature regression sweep
Command target:
- `tests/test_features.py -k "rms or spectral_centroid"`

Result:
- `39 passed, 4 xfailed`

### Multichannel regression sweep
Command target:
- `tests/test_multichannel.py -k "rms or spectral_centroid"`

Result:
- `3 passed`

### Additional metric-feature check
Command target:
- `tests/test_met_features.py -k "spectral_centroid"`

Result:
- `1 skipped`

---

## Files Changed in Phase 4B

- `src/spectrum_utils.rs`
- `src/lib.rs`
- `librosa/feature/spectral.py`
- `tests/test_phase4_features.py`

---

## Scope Achieved vs Planned

### Achieved now
- ✅ Native Rust RMS kernels for the spectrogram path
- ✅ Native Rust spectral centroid kernels for static-bin frequencies
- ✅ Conservative public dispatch in `librosa.feature`
- ✅ Multichannel dispatch via Python-side channel flattening + per-channel kernels
- ✅ Focused parity and regression coverage

### Deferred
- time-domain `rms(y=...)` acceleration
- variable-frequency-grid centroid acceleration
- spectral rolloff / bandwidth / chroma acceleration
- dedicated performance benchmark harness for Phase 4B

---

## Notes

This phase prioritized **behavior safety over breadth**:
- common cases are accelerated
- complex or tricky broadcasting cases retain the trusted Python implementation
- multichannel support is preserved by flattening leading dimensions into per-channel Rust calls

This makes the new acceleration usable immediately while keeping the fallback path intact for harder edge cases.

---

## Conclusion

Phase 4B is complete for the current safe scope:
- `rms(S=...)` now has Rust acceleration for common real-valued spectrogram inputs
- `spectral_centroid()` now uses Rust for the common fixed-bin-frequency case
- existing feature and multichannel regression tests remain green

**Completion Date:** April 1, 2026

