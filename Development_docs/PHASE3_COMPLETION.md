# Phase 3: Float64 Native + STFT Dispatch Hardening

## Status: ✅ COMPLETE (Current Scope)

This phase delivered native float64 STFT kernels, wired dtype-aware dispatch in both `stft()` and `_spectrogram()`, added multi-channel `stft()` acceleration, and completed a hardening pass with broader regression tests.

---

## What Was Implemented

### 1) Native float64 STFT kernels (Rust)

File: `src/stft.rs`

Added:
- `stft_power_f64(...) -> PyArray2<f64>`
- `stft_complex_f64(...) -> PyArray2<Complex<f64>>`
- float64 Hann generator: `hann_window_f64(...)`
- dedicated thread-local plan/buffer/scratch for f64

Also fixed a critical buffer-size bug found during broad test sweeps:
- thread-local FFT buffers are now resized when `buf.len() != n_fft` (not only when smaller)
- this prevents RustFFT panics when calls alternate between FFT sizes (e.g., 2049 then 2048)

### 2) Module exports for new kernels

File: `src/lib.rs`

Registered:
- `stft::stft_power_f64`
- `stft::stft_complex_f64`

### 3) `_spectrogram()` dtype-aware dispatch

File: `librosa/core/spectrum.py`

Updated behavior:
- float32 input dispatches to `stft_power`
- float64 input dispatches to `stft_power_f64`
- window extraction is dtype-agnostic and converted to match kernel dtype
- contiguous conversion remains automatic

### 4) `stft()` dtype-aware + multi-channel Rust fast-path

File: `librosa/core/spectrum.py`

Added conservative fast-path in `stft()`:
- float32 uses `stft_complex`
- float64 uses `stft_complex_f64`
- supports multi-channel via native batched kernels (`stft_complex_batch`, `stft_complex_f64_batch`) when available
- uses a tuned dispatch heuristic: small channel counts (e.g., stereo) prefer per-channel kernels; larger batches use native batched kernels

Guardrails kept strict for behavior parity:
- `win_length == n_fft`
- `out is None`
- `(center is False or pad_mode == "constant")`
- `n_fft <= y.shape[-1]` (to preserve Python warning/error behavior for too-short inputs)

If any guard fails, code falls back to existing Python implementation.

---

## Tests Added (Phase 3)

File: `tests/test_features.py`

Added:
- `test_stft_power_f64_matches_librosa`
- `test_stft_complex_f64_matches_librosa`
- `test_spectrogram_dispatch_prefers_f64_kernel`
- `test_stft_dispatch_prefers_complex_f32_kernel`
- `test_stft_dispatch_prefers_complex_f64_kernel`
- `test_stft_multichannel_dispatch_f32`
- `test_stft_multichannel_dispatch_f64`
- `test_stft_multichannel_parity_f32`
- `test_stft_multichannel_parity_f64`

---

## Verified Test Runs

### Focused Phase 3 dispatch/parity
- `6 passed` (float64 kernel parity + dtype dispatch + existing parity regression checks)
- `5 passed` (single-channel dispatch + multichannel dispatch + float64 parity)
- `4 passed` (multichannel parity + multichannel dispatch)

### Broader core sweep
Command target:
- `tests/test_core.py -k "stft or _spectrogram"`

Result:
- `152 passed, 1 skipped, 15 xfailed`
- `0 unexpected failures`

### Multichannel sweep
Command target:
- `tests/test_multichannel.py -k "stft or istft"`

Result:
- `7 passed`

---

## Regression Found and Fixed During Hardening

During broad core tests, RustFFT panics occurred:
- panic message: expected FFT buffer multiple of 2048, got len 2049
- root cause: thread-local buffers only grew, never shrank across mixed `n_fft` calls
- fix: resize buffers to exact `n_fft` in all STFT kernels

After fix + rebuild, broad core and multichannel sweeps passed.

---

## Scope Achieved vs Planned

### Achieved now
- ✅ Native float64 STFT power + complex kernels
- ✅ `_spectrogram()` dtype-aware dispatch (`f32`/`f64`)
- ✅ `stft()` dtype-aware dispatch (`f32`/`f64`)
- ✅ Multi-channel `stft()` acceleration path
- ✅ Hardening with broad core/multichannel test sweeps

### Deferred
- Additional performance benchmark document for Phase 3 (can be added next)

---

## Files Changed in Phase 3

- `src/stft.rs`
- `src/lib.rs`
- `librosa/core/spectrum.py`
- `tests/test_features.py`

---

## Conclusion

Phase 3 (current scope) is complete and stable:
- float64 is now first-class for STFT acceleration
- `stft()` and `_spectrogram()` select Rust kernels by dtype
- multi-channel `stft()` path is active and parity-tested
- broad STFT regression sweeps are green after hardening

**Completion Date:** April 1, 2026

---

## Performance Appendix (Verified)

Benchmark harness:
- `scripts/benchmark_phase3_stft.py`

Run configuration:
- `seed=2036`
- `seconds=10.0`
- `sr=22050`
- `n_fft=2048`
- `hop_length=512`
- `warmup=2`, `runs=8`, `batches=3`

Run command:

```powershell
python scripts/benchmark_phase3_stft.py --seconds 10 --runs 8 --batches 3 --warmup 2
```

Results (lower is better; speedup is `python_fallback / rust_fastpath`):

| center | channels | dtype | rust_ms | py_ms | speedup_x |
|---|---:|---|---:|---:|---:|
| `True` | 1 | `float32` | 1.240 | 4.136 | 3.33 |
| `True` | 1 | `float64` | 2.185 | 5.092 | 2.33 |
| `True` | 2 | `float32` | 3.745 | 9.312 | 2.49 |
| `True` | 2 | `float64` | 7.212 | 11.394 | 1.58 |
| `False` | 1 | `float32` | 1.331 | 4.225 | 3.17 |
| `False` | 1 | `float64` | 2.148 | 5.252 | 2.45 |
| `False` | 2 | `float32` | 3.829 | 9.386 | 2.45 |
| `False` | 2 | `float64` | 7.074 | 11.362 | 1.61 |

Quick read:
- Rust is faster in all tested cases.
- Largest gains appear on mono (`float32`/`float64`) cases.
- Stereo gains improved versus the prior batch-only strategy after adding the small-channel dispatch heuristic.

Follow-up optimization target:
- Tune native batched kernels (layout and parallel decomposition) for stronger stereo throughput.

