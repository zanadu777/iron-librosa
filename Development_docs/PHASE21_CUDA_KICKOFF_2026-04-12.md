# Phase 21 Kickoff: PC/CUDA FFT Path (2026-04-12)

## Objective

Establish a CUDA implementation path for FFT-backed STFT/iSTFT on PC while preserving current CPU and macOS behavior through strict fallback invariants.

## Success Criteria (Phase 21)

- [ ] CUDA backend seam is implemented behind a feature flag and runtime guard
- [ ] CPU fallback remains the default-safe path
- [ ] STFT/iSTFT parity is validated against existing tolerances
- [ ] Benchmark artifacts are produced in `Benchmarks/results/`
- [ ] Promotion decision is documented (promote/opt-in/defer)

## Reuse Existing Seams

- Device selection: `src/backend.rs` (`RustDevice` dispatch pattern)
- FFT fallback contract: `src/metal_fft.rs` (GPU-try then CPU fallback)
- Call sites and gating: `src/stft.rs`, `src/istft.rs`
- Benchmark style and output schema:
  - `Benchmarks/scripts/benchmark_phase19_chunk_ab.py`
  - Existing `Benchmarks/results/*.json` artifacts

## Proposed Backend Strategy

### Device Model

Extend device model to include CUDA capability in a way that does not break existing `Auto/Cpu/AppleGpu` behavior.

- Environment-controlled request remains source of truth
- Runtime availability check gates actual device selection
- If CUDA unavailable or errors occur, immediate CPU fallback

### API Contract (target shape)

Match the existing wrapper style:

- `fft_forward_batched_with_fallback(...) -> Result<(), String>`
- `fft_inverse_batched_with_fallback(...) -> Result<(), String>`
- Optional chunked variants for dispatch amortization

### Initial Scope (do first)

1. `stft_complex` f32 path
2. `istft_f32` inverse FFT path
3. Leave `stft_power` / f64 variants as follow-up unless low-risk

## Risk Controls

- [x] Feature flag for CUDA path (disabled by default until validated)
- [x] Runtime guard for CUDA availability
- [x] Fallback invariant: any CUDA error returns to CPU path without functional regression
- [x] No behavior change to non-GPU users
- [x] Preserve existing test tolerances and shape contracts

All invariants are carried forward from the Metal path design in `src/metal_fft.rs` and `src/backend.rs`. The CUDA module will follow identical fallback semantics: feature-gated compile, runtime availability probe, immediate CPU fallback on any error.

## Validation Plan

### Correctness

- Existing test suite, including STFT/iSTFT focused tests
- Add CUDA-specific dispatch and fallback tests mirroring Metal-style coverage
- Explicit forced CPU mode tests to guarantee fallback behavior

### Performance

- Compare CUDA vs CPU on the same workload matrix used for Phase 19/20 style reporting
- Report speedup and regression bins per workload shape
- Store JSON + markdown outputs under `Benchmarks/results/`

## Deliverables

- [ ] CUDA seam contract document
- [ ] Initial CUDA wrapper module with fallback contract
- [ ] Dispatch integration in `src/stft.rs` and `src/istft.rs`
- [ ] Test additions for availability/fallback/error propagation
- [ ] Benchmark artifacts and phase decision document

## Day-1 Task List

**Owner: Ken Johnson**

1. **[Task 1]** Add `CudaGpu` variant to `RustDevice` in `src/backend.rs`, feature-gated under `cuda-gpu`; add `IRON_LIBROSA_RUST_DEVICE=cuda-gpu` parsing and runtime availability stub returning `false` until cuFFT is linked.
2. **[Task 2]** Create `src/cuda_fft.rs` with the same public API shape as `src/metal_fft.rs` — `fft_forward_batched_with_fallback` / `fft_inverse_batched_with_fallback` — non-macOS guard returning `Err` (no-op stub); wire into `src/lib.rs`.
3. **[Task 3]** Add `Benchmarks/scripts/benchmark_phase21_cuda_baseline.py` that replicates the Phase 19/20 workload matrix (`short_512`, `short_1024`, `medium_512`, `medium_1024`, `long_1024`) against CPU on PC, establishing the baseline JSON that CUDA results will compare against.

These three tasks together: (a) add the device seam without breaking anything, (b) prove the module compiles with correct fallback contract, and (c) pre-build the benchmark harness so CUDA numbers can be compared the moment the cuFFT implementation is active.

## Open Decisions

- CUDA integration path: **Selected → Option C: hybrid spike then abstraction hardening**
  - Spike cuFFT FFI for `stft_complex` f32 first to confirm call overhead, then extract the abstraction.
- Promotion bar for default-on on PC: same composite score gate as Phase 20 (`>= 0.887`), targeting `>= 1.2x` per workload before default-on.
- Initial CUDA path stays **opt-in** for Phase 21 (`IRON_LIBROSA_RUST_DEVICE=cuda-gpu`), promote to default only after two clean benchmark runs.

## Sign-off

- Owner: Ken Johnson
- Reviewer: TODO
- Kickoff date: 2026-04-12
- Status: ✅ **Approved — Day-1 tasks assigned, ready to move to PC**

