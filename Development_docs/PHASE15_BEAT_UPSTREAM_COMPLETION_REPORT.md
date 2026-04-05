# Phase 15 Beat-Track Upstream Acceleration — Completion Report

Date: 2026-04-05

---

## Motivation

Phase 14 built the beat-track DP seam but found **zero end-to-end gain** because
the DP step is < 1 % of `beat_track` runtime.  Stage profiling exposed the real
bottlenecks:

| Stage | Share | Root cause |
|---|---|---|
| `onset_strength` | ~49 % | mel-spectrogram + **median aggregation** |
| `tempo` | ~49 % | **autocorrelation** over 5 000+ frames |
| beat DP | ~0.8 % | (Phase 14 seam — negligible) |

Phase 15 attacked both 49 % bottlenecks.

---

## New Rust Seams

### 1. `onset_flux_median_ref_f32 / f64`  (`src/onset.rs`)

Replaces the `util.sync(np.median)` call inside `onset_strength_multi`
when `aggregate=np.median` and `max_size=1` (the `beat_track` default).

- Rayon parallel iteration over time frames.
- Per-frame: collect flux values → `sort_unstable_by` → take middle element.
- Returns `(1, n_frames − lag)` to match existing mean-kernel shape contract.

### 2. `tempogram_ac_f32 / f64`  (`src/rhythm.rs`)

Replaces `autocorrelate(windowed, axis=-2)` inside `librosa.feature.tempogram`
for 2-D (mono) input.

- Rayon `map_init`: one complex FFT buffer + scratch per worker thread.
- rustfft forward → in-place power spectrum → rustfft inverse.
- Normalizes by `n_pad` (matching scipy's `irfft` convention).
- `n_pad` computed by Python (`scipy.fft.next_fast_len`) for exact parity.

### Python dispatch changes

| File | Change |
|---|---|
| `librosa/onset.py` | `_rust_onset_median_eligible` flag; new dispatch branch for `onset_flux_median_ref_*` |
| `librosa/feature/rhythm.py` | Added `_rust_bridge` import; `tempogram_ac_*` dispatch in `tempogram()` |

---

## Parity Test Results

```
tests/test_phase15_beat_upstream.py  10/10 passed
  - symbols present
  - onset_flux_median_ref f32/f64 exact match vs np.median
  - tempogram_ac f32/f64 within 1e-4 (f32) / 1e-9 (f64)
  - beat_track end-to-end frame parity
  - onset_strength median dispatch round-trip parity (rtol=1e-5)

Full beat+onset suite: 1681 passed, 0 failures
```

---

## Performance Results

Benchmark: `Benchmarks/scripts/benchmark_phase15_beat_upstream.py`

| Workload | NumPy baseline | Phase 15 Rust | Speedup |
|---|---|---|---|
| noisy 30 s | 30.2 ms | 11.7 ms | **2.58×** |
| noisy 120 s | 137.0 ms | 44.6 ms | **3.07×** |

Stage improvement (120 s):

| Stage | Before (no Rust) | After (Rust) | Improvement |
|---|---|---|---|
| `onset_strength` | ~86 ms | ~36 ms | 2.4× |
| `tempo` (ac_size=4 s) | ~35 ms | ~15 ms | 2.3× |
| Beat DP | ~1.2 ms | ~1.2 ms | 1.0× |
| **Total** | **~137 ms** | **~45 ms** | **3.1×** |

---

## Promotion Decision: **Promote** (opt-in via env var)

> Minimum threshold: ≥ 1.5× on medium single-channel.
> Achieved: **2.6–3.1×** — threshold exceeded on all workloads.

The Rust upstream paths are enabled via:

```powershell
$env:IRON_LIBROSA_RUST_DISPATCH = "1"
python your_script.py
```

The default remains conservative (NumPy) pending broader CI validation.
Recommendation: promote `IRON_LIBROSA_RUST_DISPATCH=1` to **default** in the
next release after smoke testing on Linux/macOS.

---

## Artifacts

| File | Description |
|---|---|
| `src/onset.rs` | `onset_flux_median_ref_f32/f64` kernels added |
| `src/rhythm.rs` | New file: `tempogram_ac_f32/f64` |
| `src/lib.rs` | Module registration updated |
| `librosa/onset.py` | Median dispatch injected |
| `librosa/feature/rhythm.py` | `tempogram_ac` dispatch injected |
| `tests/test_phase15_beat_upstream.py` | Parity test suite |
| `Benchmarks/scripts/benchmark_phase15_beat_upstream.py` | Benchmark script |
| `Benchmarks/results/phase15_bench_numpy_baseline.json` | Baseline |
| `Benchmarks/results/phase15_bench_phase15_rust.json` | Phase 15 Rust results |

---

## Checklist

- [x] Rust kernels implemented, building without warnings
- [x] Python dispatch seams wired with guards and fallback
- [x] No `unwrap`/`expect` on untrusted inputs
- [x] Parity tests green (10/10)
- [x] Full beat/onset suite green (1681/0)
- [x] Benchmark captured — **2.6–3.1× speedup vs NumPy**
- [x] Promotion decision: **Promote** (opt-in, threshold exceeded)

