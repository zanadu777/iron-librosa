# Phase 20 Promotion Decision (2026-04-12)

## Scope

- Phase: 20
- Feature: Adaptive GPU FFT dispatch optimization
- Code areas: `src/stft.rs`, `src/istft.rs`, `src/metal_fft.rs`
- Decision date: 2026-04-12

## Required Inputs

- Completion report: `Development_docs/PHASE20_ADAPTIVE_GPU_DISPATCH_REPORT.md`
- Process gate: `Development_docs/PHASE_COMPLETION_PROCESS.md`
- Test logs:
  - `artifacts/run_logs/phase20_pytest_full_2026-04-12.log`
  - `artifacts/run_logs/phase20_stft_istft_2026-04-12.log`
  - `artifacts/run_logs/phase20_metal_dispatch_2026-04-12.log`
- Benchmark artifacts:
  - `Benchmarks/results/phase20_chunk_ab_2026-04-12.json`
  - `Benchmarks/results/phase20_chunk_ab_2026-04-12.md`

## Gate Results

### 1) Code Complete

- [x] Implemented and built
- [x] Dispatch seam and fallback verified
- [x] No unsafe error-path behavior introduced

Notes:

Code paths are present in `src/stft.rs`, `src/istft.rs`, and `src/metal_fft.rs` with additive gating and explicit CPU fallback behavior.

### 2) Parity / Regression Tests

- [x] Full test suite: pass with zero new failures
- [x] STFT/iSTFT focused tests: pass
- [x] Metal dispatch tests: pass

Notes:

- Full suite (`artifacts/run_logs/phase20_pytest_full_2026-04-12.log`): `14309 passed, 3 skipped, 528 xfailed, 38 xpassed`.
- STFT/iSTFT focused (`artifacts/run_logs/phase20_stft_istft_2026-04-12.log`): `444 passed, 14413 deselected, 21 xfailed`.
- Metal dispatch (`artifacts/run_logs/phase20_metal_dispatch_2026-04-12.log`): `5 passed`.

### 3) Performance Review

- [x] Benchmark artifacts captured and stored under `Benchmarks/results/`
- [x] Composite score measured
- [x] Regressions <1.0x documented with workload details

Measured results:

| Metric | Value | Target | Pass? |
|---|---:|---:|:---:|
| Composite score | 1.276 | >= 0.887 | Yes |
| STFT regressions (<1.0x) | 0 | As few as possible | Yes |
| iSTFT regressions (<1.0x) | 0 | As few as possible | Yes |

## Decision Rubric

Choose exactly one:

- **Promote**
  - Tests green
  - Composite score >= 0.887
  - No severe regressions (<0.95x)
- **Opt-in**
  - Tests green
  - Composite score in [0.82, 0.886] or isolated regressions remain
- **Defer/Rework**
  - Any gate failure, or score < 0.82

## Final Decision

- Decision: `Promote`
- Rationale:

- All closeout test gates passed with no new failures.
- Composite score exceeds promotion bar (`1.276` vs `0.887` target).
- No STFT/iSTFT benchmark regressions under this run (`<1.0x` count is zero).
- Dispatch gating + fallback behavior remain additive and low-risk to existing CPU paths.

## Release Configuration

If Promote or Opt-in, record final defaults:

- `IRON_LIBROSA_FFT_GPU_MIN_FRAMES=200`
- `IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD=100000000`
- `IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE=unset (adaptive)`
- `IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL=force-on`

## Follow-up Actions

- [x] Document residual regressions and owner
- [ ] Link benchmark artifacts in phase index
- [x] Open Phase 21 execution issue/doc

Residual regressions observed in current benchmark artifact: none (`stft_regressions=0`, `istft_regressions=0`).

## Sign-off

- Engineer: Ken Johnson
- Reviewer: TODO
- Date: 2026-04-12

