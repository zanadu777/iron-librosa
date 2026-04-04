# Phase 13 CQT/VQT Completion Report

Date: 2026-04-04

## Summary

Phase 13 is functionally complete:
- Rust dense projection kernels for CQT/VQT are implemented in `src/cqt_vqt.rs`.
- Python dispatch is integrated in `librosa/core/constantq.py`.
- Parity tests and Constant-Q regressions pass.
- Benchmark artifacts are captured in `Benchmarks/results/phase13_cqt_vqt_baseline.json`.

## Promotion Decision

**Decision:** do not promote the CQT/VQT Rust seam to default dispatch yet.

The implementation is correct, but the benchmark target from the spike plan was not met consistently on the validation machine. The backend remains **opt-in** through:

```powershell
$env:IRON_LIBROSA_CQT_VQT_BACKEND = "rust"
```

Default behavior (`auto`) continues to use the NumPy/SciPy path.

## Final Implementation State

### Delivered
1. Rust dense projection kernels
   - `cqt_project_f32`
   - `cqt_project_f64`
   - `vqt_project_f32`
   - `vqt_project_f64`
2. Python dispatch seam in `__cqt_response`
3. Parity coverage for:
   - deterministic reference behavior
   - dense Rust projection dispatch
   - dense-enough sparse-basis dispatch
   - mono and multichannel backend parity
4. Benchmark harness upgraded to compare Python vs Rust timings
5. Backend policy flag added in `librosa/_rust_bridge.py`
   - `IRON_LIBROSA_CQT_VQT_BACKEND=rust` to opt in
   - `IRON_LIBROSA_CQT_VQT_BACKEND=numpy` to force fallback
   - default `auto` keeps fallback for now

### Hardening completed during sign-off
- `pseudo_cqt` no longer routes through the complex Rust projector.
  - This avoids the extra real-to-complex bridge overhead that regressed the magnitude-only path.
- The Rust projector now avoids Rayon overhead for very small batch counts.
  - Mono/stereo workloads take the direct sequential path.

## Validation

### Tests
```powershell
python -m pytest -q -o addopts="" tests/test_phase13_cqt_vqt_parity.py
python -m pytest -q -o addopts="" tests/test_constantq.py
```

Observed results:
- `tests/test_phase13_cqt_vqt_parity.py` -> `17 passed`
- `tests/test_constantq.py` -> `632 passed, 10 xfailed`

### Type checking
```powershell
python -m mypy librosa/_rust_bridge.py librosa/core/constantq.py tests/test_phase13_cqt_vqt_parity.py Benchmarks/scripts/benchmark_phase12_cqt_vqt.py --no-color-output --no-incremental
```

Observed result:
- `Success: no issues found in 4 source files`

### Benchmark artifact
```powershell
python Benchmarks/scripts/benchmark_phase12_cqt_vqt.py --durations 10 30 --repeats 3 --json-out Benchmarks/results/phase13_cqt_vqt_baseline.json
```

Observed comparison summary:

| Case | CQT avg speedup | VQT avg speedup |
|------|-----------------:|----------------:|
| mono-10s | 1.043x | 0.979x |
| stereo-10s | 1.012x | 0.963x |
| mono-30s | 0.971x | 0.943x |
| stereo-30s | 1.042x | 0.968x |

Notes:
- Results were near parity, with some workloads slightly faster and others slightly slower.
- The Phase 13 promotion target (`>= 1.2x` on medium workloads) was not met consistently.

## Exit Criteria Mapping

- [x] Reproducible benchmark artifact captured in `Benchmarks/results/`
- [x] First Rust kernel integrated
- [x] Parity tests passing
- [x] Multichannel coverage added
- [x] Promotion decision recorded
- [ ] Default promotion to Rust backend

## Recommended Follow-up

If this seam is revisited later, the next optimization pass should focus on:
1. replacing the current `ndarray::dot` projector with a faster GEMM-backed implementation,
2. reducing basis-build and wavelet overhead, which still dominates the stage profile,
3. adding a dedicated real-valued pseudo-CQT kernel before reconsidering that path for acceleration.

