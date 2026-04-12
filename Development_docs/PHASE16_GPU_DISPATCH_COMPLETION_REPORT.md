# Phase 16 GPU Dispatch Completion Report

Date: 2026-04-09

## Scope

This closeout covers the Apple GPU rollout and optimization pass for the currently implemented f32 kernels:

- `mel_project_f32` (`src/mel.rs`)
- `chroma_project_f32` (`src/chroma.rs`)
- `cqt_project_f32` (`src/cqt_vqt.rs`)
- `piptrack_from_spectrogram_f32` (`src/tuning.rs`)
- `tempogram_ac_f32` (`src/rhythm.rs`)

All kernels keep strict CPU fallback semantics for unavailable/failed GPU paths.

## Regression Evidence

Full gate:

- Log: `full_run_phase16_20260409_141726.log`
- Exit: `full_run_phase16_20260409_141726.exit`
- Result: `14287 passed, 3 skipped, 512 xfailed, 15 warnings`
- Hotfix validation: `Passed: 6/6`

Focused GPU parity/regression gates:

- `tests/test_phase4c_chroma.py`
- `tests/test_phase13_rust_device_override.py`
- `tests/test_phase13_cqt_vqt_parity.py`
- `tests/test_phase5_tuning_device_override.py`
- `tests/test_phase5_tuning.py`
- `tests/test_phase15_beat_upstream.py`

Latest combined focused run result: `61 passed`.

## Optimization Decisions

### 1) Dispatch Thresholds (default policy)

Added conservative defaults with env overrides:

- Mel: `IRON_LIBROSA_GPU_WORK_THRESHOLD` (existing)
- Chroma: `IRON_LIBROSA_CHROMA_GPU_WORK_THRESHOLD` (new)
- Piptrack: `IRON_LIBROSA_PIPTRACK_GPU_WORK_THRESHOLD` (new)
- Tempogram: `IRON_LIBROSA_TEMPOGRAM_GPU_WORK_THRESHOLD` (new)

Purpose: avoid default-on regressions where GPU setup/launch overhead dominates.

### 2) Benchmark Harness + Schema Contract

Added benchmark script:

- `Benchmarks/scripts/benchmark_phase16_gpu_dispatch.py`

Validated payload with:

- `Benchmarks/scripts/validate_benchmark_payloads.py`

Persisted artifact:

- `Benchmarks/results/phase16_gpu_dispatch_20260409_141924.json`

## Latest Phase16 Benchmark Snapshot

From `Benchmarks/results/phase16_gpu_dispatch_20260409_141924.json`:

- `mel_project_f32`: CPU/GPU = `1.418x` (GPU-request faster)
- `chroma_project_f32`: CPU/GPU = `0.995x` (near parity)
- `cqt_project_f32`: CPU/GPU = `10.745x` (GPU-request much faster)
- `piptrack_from_spectrogram_f32`: CPU/GPU = `0.889x` (CPU faster)
- `tempogram_ac_f32`: CPU/GPU = `0.686x` (CPU faster)

Numerical parity stayed within expected f32 tolerance envelopes.

## Promotion Outcome

Phase 16 dispatch rollout is complete for the implemented kernels listed above, with:

- full regression gate green,
- focused parity gates green,
- benchmark artifact schema-compliant,
- conservative default thresholds to avoid regressions.

## Remaining Work (outside Phase 16 closeout)

- Extend Apple GPU implementations to remaining CPU-fallback kernels not yet ported.
- Continue per-kernel optimization where CPU still outperforms GPU on representative shapes.
- Optionally add explicit boundary tests for threshold decisions per kernel.

