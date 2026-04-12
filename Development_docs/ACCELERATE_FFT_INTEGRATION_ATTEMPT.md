# Accelerate FFT Integration Attempt Summary

**Date:** April 9, 2026  
**Status:** Paused (reverted to rustfft baseline for stability)

## Objective
Integrate macOS Accelerate framework (`vDSP_fft*` routines) into STFT/iSTFT kernels to replace `rustfft` with hand-tuned native code, expecting significant performance gains on macOS.

## What Was Done
1. **Created `src/accelerate_fft.rs`** — A wrapper module with:
   - FFI bindings to `vDSP_create_fftsetup`, `vDSP_fft_zip`, `vDSP_destroy_fftsetup`
   - `fft_forward_inplace()` and `fft_inverse_inplace()` implementations
   - Fallback to `rustfft` on non-macOS platforms
   - High-level `fft_forward()` / `fft_inverse()` convenience functions

2. **Modified `src/stft.rs`** — Updated to call Accelerate wrapper instead of inline rustfft

3. **Build succeeded** — `cargo check` and `maturin develop --release` completed without errors

## Issue Encountered
**Segmentation fault** when calling `vDSP_fft_zip` at runtime. Root causes suspected:
- vDSP `Complex` struct layout mismatch with Rust's `num_complex::Complex<f32>`
- Incorrect `vDSP_Complex` FFI repr(C) declaration
- Incorrect stride/setup parameters to the FFT routine
- Memory alignment requirements not met

## Resolution Path
The vDSP FFI layer requires more careful testing/validation. Reverted STFT to use `rustfft` (which was already working well) to restore stability.

## Better Approach for Phase 17+
Rather than hand-rolling vDSP bindings, **use an existing Rust ecosystem crate**:

### Option A: `accelerate-sys` crate (if it exists and is maintained)
- Check crates.io for a well-maintained `accelerate-sys` wrapper
- Simpler to use, vetted by community

### Option B: Use `metal-rs` compute shaders for FFT
- Write a custom Metal FFT kernel (similar to the approach for `mel_project_f32`)
- Gives full GPU acceleration like current GPU paths
- Can leverage existing Metal batching infrastructure
- Metal FFTs can be even faster than Accelerate for large batches

### Option C: Integrate a GPU FFT library
- Libraries like `cufft` (CUDA) have Metal/Apple GPU equivalents via `Metal Performance Shaders`
- May require additional SDK dependencies

## Current State
- **Code status**: `accelerate_fft.rs` created but not integrated into build path
  - STFT/iSTFT reverted to pure `rustfft` (stable)
  - `accelerate_fft.rs` module is declared but unused (no regressions)
- **Test status**: Full test suite passes; STFT/iSTFT numerically validated
- **Performance**: rustfft is already quite fast (~3–5ms for typical STFT); marginal gains expected from Accelerate alone

## Recommendation
**Option B (Metal FFT kernel)** is the best long-term path because:
1. **Consistency**: Aligns with Phase16 GPU strategy (Metal for Apple GPU acceleration)
2. **Scalability**: Batch FFT across frames on GPU = large speedup for long audio
3. **Proven pattern**: Similar to metal-based `mel_project_f32` already implemented
4. **Future-proof**: Handles both Accelerate and GPU paths transparently

### Estimated Effort
- Custom Metal FFT kernel: ~200–300 lines MSL code
- Rust wrapper: ~100–150 lines (similar to mel.rs pattern)
- Testing: use Phase16 regression suite + benchmarking harness
- **Timeline**: 1–2 engineering cycles

### Expected Outcome
- STFT/iSTFT GPU path: **5–20x speedup** for large audio (depending on batch size)
- Complements CQT's 10.75x win with another major GPU accelerator
- Unlocks downstream benefits for all analysis functions that depend on STFT

## Files Affected (Current)
- `/Users/kenjohnson/Dev/Rust/iron-librosa/src/accelerate_fft.rs` — created but unused (safe to delete or keep for future use)
- `/Users/kenjohnson/Dev/Rust/iron-librosa/src/lib.rs` — added module declaration (harmless)
- `/Users/kenjohnson/Dev/Rust/iron-librosa/src/stft.rs` — reverted to rustfft (no changes from baseline)

## Clean-up for Handoff
To prepare for Phase 17 GPU FFT work:
```bash
# Keep accelerate_fft.rs as a reference for later experimentation
# No immediate action needed; code is safe and non-intrusive
```

---

**Note**: This attempt was exploratory and low-risk. The failure to integrate Accelerate FFI is not a blocker; rustfft remains performant. Phase 17 should prioritize the Metal FFT kernel path for GPU acceleration instead.

