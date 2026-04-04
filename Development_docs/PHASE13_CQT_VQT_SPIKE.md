# Phase 13 CQT/VQT Spike Plan

Date: 2026-04-04

## Goal
Implement Rust acceleration for the highest-value Constant-Q paths while preserving
strict parity with Python reference outputs.

## Day-1 Output (this commit)
- Baseline benchmark script upgraded: `Benchmarks/scripts/benchmark_phase12_cqt_vqt.py`
  - Workload matrix from CLI (`--durations`, `--repeats`)
  - Optional machine-readable JSON report (`--json-out`)
  - Stage profiler output retained for hotspot triage
- Parity scaffolding added: `tests/test_phase13_cqt_vqt_parity.py`
  - Deterministic reference tests for CQT and VQT
  - Placeholder symbol-probe test for future Rust exports

## Scope
### In-scope
1. Identify the first Rust seam (recommended: `__cqt_response` projection path)
2. Implement f32/f64 kernels for CQT response projection
3. Wire optional Rust dispatch behind existing `RUST_AVAILABLE` guard
4. Add parity tests for mono and multichannel
5. Add benchmark deltas against day-1 baseline

### Out-of-scope (Phase 13)
- Full rewrite of recursive downsampling path
- NMF or segment/effects acceleration

## Milestones
1. **M1: Baseline lock (DONE)**
   - Benchmark script emits reproducible machine-readable output
   - Reference parity tests deterministic
2. **M2: First Rust kernel path (DONE)**
   - Add `src/cqt_vqt.rs` with f32/f64 projection kernels
   - Register symbols in `src/lib.rs`
3. **M3: Python dispatch integration (DONE)**
   - Wire dispatch in `librosa/core/constantq.py` for compatible dtypes/shapes
4. **M4: Parity hardening (DONE)**
   - Add strict allclose thresholds and multichannel coverage
5. **M5: Promotion decision (DECIDED: keep opt-in)**
   - Benchmark speedup >= 1.2x on medium workloads
   - No parity regressions in `tests/test_constantq.py`

## Final Status
- Rust dense projection seam implemented and validated.
- Parity and regression suites pass.
- Backend comparison benchmark captured in `Benchmarks/results/phase13_cqt_vqt_baseline.json`.
- Default promotion deferred: benchmark gains were near parity / mixed on the validation machine.
- Runtime policy: use `IRON_LIBROSA_CQT_VQT_BACKEND=rust` to opt in while further optimization work is pending.

## Validation commands
```powershell
python -m pytest tests/test_phase13_cqt_vqt_parity.py -q
python Benchmarks/scripts/benchmark_phase12_cqt_vqt.py --durations 10 30 --repeats 3 --json-out Benchmarks/results/phase13_cqt_vqt_baseline.json
python -m pytest tests/test_constantq.py -q
```

## Risks
- CQT/VQT recursive resampling can hide projection speedups if dispatch boundary is too late.
- Basis cache interactions may skew benchmark variance if warmup strategy is inconsistent.
- f32 parity may require explicit casting points to mirror NumPy behavior.

## Exit criteria
- Reproducible benchmark baseline captured in `Benchmarks/results/`
- First Rust kernel integrated with parity tests passing
- Clear speedup evidence on mono and stereo workloads

