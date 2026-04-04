# Phase 11 Status (Kickoff)

Date: 2026-04-03

## Completed in kickoff

- Enabled Rust `spectral_contrast` band-kernel dispatch for multi-dimensional inputs (`S.ndim >= 2`) in `librosa/feature/spectral.py`.
- Added channel-flattened dispatch path for each sub-band and reshaped outputs back to `(..., n_bands+1, n_frames)`.
- Aligned Rust path quantile indexing with Python fallback semantics per band to preserve parity.
- Added multichannel parity tests (3D and 4D) in `tests/test_phase7_contrast.py`.
- Extended contrast benchmark coverage with mono/stereo/quad cases in `benchmark_phase5_spectral.py`.
- Added quick benchmark harness `benchmark_phase11_contrast_multichannel.py`.

## Completed in tuning pass

- Added adaptive dispatch policy in `librosa/feature/spectral.py` for `spectral_contrast`:
  - `_CONTRAST_RUST_MODE` (`auto` / `rust` / `python`)
  - `_CONTRAST_RUST_WORK_THRESHOLD = 1_500_000`
  - `_CONTRAST_RUST_MIN_FRAMES = 1200`
-  - `_CONTRAST_RUST_MULTICHANNEL_MIN_FRAMES = 800`
- Tightened auto policy into channel-aware tiers via `_contrast_rust_auto_ok(...)`:
  - mono: `>= 1200` frames and `>= 1.5M` work
  - 2-3 channels: `>= 800` frames and `>= 1.5M` work
  - 4+ channels: `>= 300` frames and `>= 1.2M` work
- Rust path now auto-enables only on larger workloads in `auto` mode.
- Added threshold/override parity tests in `tests/test_phase7_contrast.py`.

## Completed in fused-kernel pass

- Added fused Rust kernels in `src/spectrum_utils.rs`:
  - `spectral_contrast_fused_f32`
  - `spectral_contrast_fused_f64`
- Registered fused kernels in `src/lib.rs`.
- Updated `librosa/feature/spectral.py` to precompute contiguous band metadata and call the fused Rust kernel once per flattened channel batch.
- Removed the worst Python-side per-band FFI overhead from the Rust path.
- Parallelized fused Rust execution internally over flattened `(channel, band)` tasks for multichannel workloads.
- Added a mono-specialized fused path to preserve the better frame-parallel behavior for single-channel inputs.

## Validation

- `python -m pytest tests/test_phase7_contrast.py -q`
- Result: `28 passed`

## Updated performance snapshot (quick benchmark)

Workload `frames=800`:
- mono: 1.07x vs forced Python fallback
- stereo: 1.53x vs forced Python fallback
- quad: 2.26x vs forced Python fallback

Workload `frames=1600` (auto mode enables Rust path):
- mono: 1.78x vs forced Python fallback
- stereo: 1.60x vs forced Python fallback
- quad: 3.27x vs forced Python fallback

Interpretation: internal Rust parallelization made the `frames=800` multichannel case clearly profitable. Mono-800 is now effectively parity, while stereo and quad are strong wins.

## Post fused-kernel scan (`tmp_phase11_scan.py`)

- `frames=800`:
- `speedup_auto` is now:
  - mono: `1.00x`
  - stereo: `1.39x`
  - quad: `2.57x`
  - 8ch: `4.06x`
- `speedup_rust` is now favorable for multichannel:
  - stereo: `1.48x`
  - quad: `2.43x`
  - 8ch: `4.00x`
- `frames=1600`:
- `speedup_auto` and `speedup_rust` are strongly favorable (`~1.50x` to `~4.00x`)

- `frames=300`:
  - mono stays near Python parity / off-path
  - stereo stays near parity
  - 4+ channels now correctly unlock Rust and show strong wins

Conclusion: the fused kernel plus internal Rust parallelization is the first combination that decisively fixes `frames=800` for multichannel workloads. The tuned auto gate now has a sensible shape:
- keep mono 800-frame workloads near parity
- strongly benefit stereo+ 800-frame workloads
- strongly favor Rust on larger multichannel workloads

## Next planned actions

1. Reduce per-band allocation overhead (reuse work buffers, avoid repeated stack/reshape pressure).
2. Re-benchmark broader shapes (varying bins/bands/quantiles) and refine thresholds if needed.
3. Add benchmark guardrails to catch regressions in CI-sized workloads.


