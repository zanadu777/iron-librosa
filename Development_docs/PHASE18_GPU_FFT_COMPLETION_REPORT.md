# Phase 18 Completion Report — Metal GPU FFT Dispatch (STFT / iSTFT)

**Date:** April 9, 2026  
**Status:** ✅ Complete  
**Build:** `maturin develop --release` — clean (35s)  
**Regression gate:** 14,309 passed, 3 skipped, 528 xfailed, **0 failures**

---

## Summary

Phase 18 wires the Apple Metal GPU FFT kernels (written in the Phase 17
foundation) into the Rust `stft_complex` and `istft_f32` hot paths.  
GPU dispatch is **opt-in** via an experimental env flag; the default path is
unchanged (Rayon CPU), ensuring zero regression risk.

---

## Changes Made

### `src/stft.rs`
- Added `use crate::backend::{resolved_rust_device, RustDevice}`.
- Added `fft_gpu_work_threshold()` — reads `IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD`
  (default 100 M ops) for Auto-mode threshold gating.
- `stft_complex`: new GPU dispatch arm before the Rayon CPU path.  
  - Runs frames sequentially; `fft_forward_with_fallback` handles Metal → CPU
    fallback transparently.
  - Rayon CPU path is untouched.

### `src/istft.rs`
- Added `use crate::backend::{resolved_rust_device, RustDevice}`.
- Added `inverse_fft_f32_gpu()` helper: mirrors negative frequencies, calls
  `fft_inverse_with_fallback`, scales real output.
- `istft_f32`: dispatches to `inverse_fft_f32_gpu` when `AppleGpu` is resolved.

### `Benchmarks/scripts/benchmark_phase17_gpu_fft.py` (new)
- STFT/iSTFT CPU vs GPU benchmark harness.  
- JSON persistence + comparison table with auto-review enforcement (`< 1.0x`).

---

## Performance Data (Apple M-series, April 9 2026)

| Workload           | n_fft | STFT speedup | iSTFT speedup |
|--------------------|-------|:------------:|:-------------:|
| short 1 s, 512     |   512 |    1.31×     |     1.00×     |
| short 1 s, 1024    |  1024 |    1.60×     |     1.01×     |
| medium 5 s, 512    |   512 |    1.57×     |     1.12×     |
| medium 5 s, 1024   |  1024 |    1.31×     |     1.08×     |
| long 30 s, 1024    |  1024 |  **3.32×**   |     1.05×     |

Artifacts:
- `Benchmarks/results/phase17_cpu_baseline.json`
- `Benchmarks/results/phase17_gpu.json`

All workloads ≥ 1.0× — no auto-review required.

---

## Activation

```bash
# Opt-in: enable Metal kernel + request GPU device
export IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on
export IRON_LIBROSA_RUST_DEVICE=apple-gpu

# Tune Auto-mode threshold (default: ~100 M operations)
export IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD=50000000
```

Without the experimental flag, `fft_forward_with_fallback` transparently falls
back to the CPU `rustfft` path — **default behaviour is unchanged**.

---

## Constraints & Limitations

| Constraint | Detail |
|---|---|
| Max FFT size | 1 024 points (`MAX_GPU_N` in `metal_fft.rs`) |
| Power-of-2 only | Non-power-of-2 falls back to CPU automatically |
| iSTFT speedup limited | Overlap-add normalization stays on CPU |
| Platform | macOS only (`#[cfg(all(feature = "apple-gpu", target_os = "macos"))]`) |

---

## Test Coverage

```
tests/test_metal_fft_dispatch.py    5 tests  ✅
tests/test_phase4_istft_and_db.py  15 tests  ✅
Full suite                      14 309 tests  ✅  (0 failures)
```

---

## Validation Commands

```bash
# Compilation
cargo check

# Focused parity + GPU dispatch tests
pytest tests/test_metal_fft_dispatch.py tests/test_phase4_istft_and_db.py -v

# Full regression gate
pytest -q tests/ --tb=short 2>&1 | tail -3

# Benchmark
python Benchmarks/scripts/benchmark_phase17_gpu_fft.py --mode cpu \
  --json-out Benchmarks/results/phase17_cpu_baseline.json
IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on \
python Benchmarks/scripts/benchmark_phase17_gpu_fft.py --mode gpu \
  --json-out Benchmarks/results/phase17_gpu.json
python Benchmarks/scripts/benchmark_phase17_gpu_fft.py --compare \
  --baseline Benchmarks/results/phase17_cpu_baseline.json \
  --candidate Benchmarks/results/phase17_gpu.json
```

---

## Phase 18 Checklist

- [x] Metal FFT wired into `stft_complex` GPU dispatch arm  
- [x] Metal FFT wired into `istft_f32` GPU dispatch arm  
- [x] CPU fallback verified (no experimental flag → identical output)  
- [x] GPU parity verified (`rtol=1e-3` with experimental flag)  
- [x] Workload threshold env var added (`IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD`)  
- [x] Benchmark harness created and executed  
- [x] All existing tests pass (14 309 / 14 309)  
- [x] Zero new warnings introduced  
- [x] Documentation updated  

---

## Next Steps (Phase 19 candidates)

1. Raise `MAX_GPU_N` 1024 → 2048 (kernel threadgroup already supports it).  
2. Batch multiple STFT frames into one Metal dispatch to amortise launch overhead.  
3. Promote experimental flag to default-on after extended validation.  
4. Add GPU dispatch to `stft_power` and `stft_complex_f64_batch`.  

