# Phase 5 Performance Snapshot
## Date: April 2, 2026

This snapshot captures the latest results from `benchmark_phase5_spectral.py` after Phase 5 tuning updates.

---

## Environment Note

In this workspace, direct `librosa` vs `iron_librosa` comparisons can be misleading because both may resolve to the same patched code path.

For reliable end-to-end impact, use **Section 4** numbers below:
- Rust enabled path vs
- Forced Python fallback path in the same runtime

---

## Raw Kernel Speedups (vs NumPy reference)

### `spectral_rolloff_f32`

| Case | NumPy min (ms) | Rust min (ms) | Speedup |
|---|---:|---:|---:|
| `n_fft=1024`, `frames=300` | 0.872 | 0.195 | **4.47x** |
| `n_fft=2048`, `frames=800` | 4.991 | 0.122 | **40.91x** |
| `n_fft=4096`, `frames=1200` | 19.564 | 0.314 | **62.31x** |

### `spectral_bandwidth_f32` (manual centroid)

| Case | NumPy min (ms) | Rust min (ms) | Speedup |
|---|---:|---:|---:|
| `n_fft=1024`, `frames=300` | 3.502 | 0.260 | **13.47x** |
| `n_fft=2048`, `frames=800` | 19.868 | 0.170 | **116.60x** |
| `n_fft=4096`, `frames=1200` | 67.025 | 0.434 | **154.33x** |

### `spectral_bandwidth_auto_centroid_f32` (fused path)

| Case | NumPy min (ms) | Rust min (ms) | Speedup |
|---|---:|---:|---:|
| `n_fft=1024`, `frames=300` | 3.502 | 0.247 | **14.17x** |
| `n_fft=2048`, `frames=800` | 19.868 | 0.171 | **115.92x** |
| `n_fft=4096`, `frames=1200` | 67.025 | 0.505 | **132.64x** |

---

## Public API A/B (Rust ON vs Forced Python Fallback)

### `spectral_rolloff`

| Case | Rust min (ms) | Forced Python min (ms) | Speedup |
|---|---:|---:|---:|
| `n_fft=1024`, `frames=300` | 0.250 | 1.241 | **4.97x** |
| `n_fft=2048`, `frames=800` | 0.449 | 6.126 | **13.65x** |
| `n_fft=4096`, `frames=1200` | 1.180 | 21.134 | **17.91x** |

### `spectral_bandwidth` (`centroid=None`, auto-centroid path)

| Case | Rust min (ms) | Forced Python min (ms) | Speedup |
|---|---:|---:|---:|
| `n_fft=1024`, `frames=300` | 0.364 | 4.848 | **13.32x** |
| `n_fft=2048`, `frames=800` | 0.460 | 21.000 | **45.67x** |
| `n_fft=4096`, `frames=1200` | 1.234 | 68.166 | **55.23x** |

### `spectral_bandwidth` (provided centroid)

| Case | Rust min (ms) | Forced Python min (ms) | Speedup |
|---|---:|---:|---:|
| `n_fft=1024`, `frames=300` | 0.331 | 3.211 | **9.69x** |
| `n_fft=2048`, `frames=800` | 0.475 | 11.911 | **25.09x** |
| `n_fft=4096`, `frames=1200` | 1.278 | 41.998 | **32.87x** |

---

## Key Takeaways

- Raw kernels are now very fast; the strongest gains appear on larger FFT/frame workloads.
- Public API speedups are substantial when compared against forced fallback:
  - `spectral_rolloff`: up to **17.9x**
  - `spectral_bandwidth` (auto centroid): up to **55.2x**
- The fused auto-centroid path removes extra orchestration in the default bandwidth flow and gives the best practical path when `centroid` is omitted.

---

## Source

- Benchmark script: `benchmark_phase5_spectral.py`
- Latest run output: April 2, 2026

