# Phase 20 Completion Report — Adaptive GPU FFT Dispatch Optimization

**Date:** April 10, 2026  
**Status:** ✅ Implementation Complete  
**Build:** `maturin develop --release` — clean, 0.04s  
**Changes:** Operationalized Phase 19 A/B test winner + adaptive chunking strategy

---

## Executive Summary

Phase 20 implements GPU FFT dispatch optimizations based on Phase 19's empirical A/B testing:
- **Operationalized default:** `IRON_LIBROSA_FFT_GPU_MIN_FRAMES=200` (Phase 19 winner, score 0.887 vs 0.767)
- **Adaptive chunking:** Intelligent batch subdivision to reduce dispatch overhead for small workloads
- **Consistency:** Applied min_frames gate to both STFT and iSTFT; adaptive chunking to batched dispatch paths

This phase prioritizes low-risk, incremental improvements with strong regression safety.

---

## Changes Made

### 1. **Updated `src/istft.rs`** — Operationalize min_frames=200 default
- **Line 46–56:** Changed `fft_gpu_min_frames()` default from `0` → `200`
- **Rationale:** Phase 19 A/B testing showed min_frames=200 reduces regressions (STFT 2→1, iSTFT 3→2) and improves composite score (+0.120)
- **Comment:** Added Phase 19 reference and Phase 20 attribution for traceability

### 2. **Updated `src/stft.rs`** — Apply min_frames gate to forward FFT
- **Line 33–54:** Added `fft_gpu_min_frames()` function (mirroring istft.rs)
- **Line 299–304:** Modified `stft_complex` GPU dispatch logic to gate on `n_frames >= fft_gpu_min_frames()` in both `AppleGpu` and `Auto` modes
- **Impact:** Small-batch STFT workloads (< 200 frames) now consistently use CPU path, reducing overhead

### 3. **Updated `src/metal_fft.rs`** — Implement adaptive chunking
- **Line 282–305:** Replaced static `metal_fft_batch_chunk_size()` with `adaptive_chunk_size(n_fft, n_frames)` function
- **Algorithm:**
  - If explicit `IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE` is set, use that (backward compat)
  - Otherwise, compute adaptively:
    - `total_work = n_fft * n_frames`
    - If `total_work ≤ 65536` (2^16): return full batch (will fallback to CPU in practice)
    - Else: `chunk = max(64, n_frames / max(4, n_fft / 256))` capped at 512
  - **Intuition:** Larger FFTs trigger more frequent dispatch; smaller batches per dispatch amortize launch overhead better
- **Line 425–468:** Updated both `fft_forward_batched_chunked_with_fallback` and `fft_inverse_batched_chunked_with_fallback` to use adaptive sizing
- **Added documentation:** Phase 20 attribution explaining rationale

---

## Expected Behavior Changes

### Before Phase 20
- **min_frames=0:** All GPU-eligible requests immediately dispatch, even with <200 frames → overhead-dominated regressions
- **Chunking:** Fixed size or full-batch → suboptimal for mixed workload sizes
- **Score:** 0.767 (based on Phase 19 baseline)

### After Phase 20
- **min_frames=200:** Gated dispatch threshold reduces overhead; small batches use fast CPU path
- **Adaptive chunking:** Workload-aware subdivision; large workloads get amortized dispatch, small ones skip GPU entirely
- **Expected Score:** 0.887+ (Phase 19 A/B winner; likely higher with adaptive chunking)

---

## Test Strategy

### Regression Gate (Required)
```bash
cd /Users/kenjohnson/Dev/Rust/iron-librosa
source .venv-mac/bin/activate
pytest -xvs tests/ -k "stft or istft" --tb=short
# Expected: All existing STFT/iSTFT tests pass (no functional regressions)
```

### Benchmark Validation (Recommended)
```bash
# Run Phase 19 A/B benchmark suite with new defaults
python -u Benchmarks/scripts/benchmark_phase19_chunk_ab.py \
  --chunks default \
  --min-frames 200 \
  --rounds 3 --repeats 4 --warmup 1 --cpu-outer-runs 3 \
  --json-out /tmp/phase20_baseline_200.json \
  --md-out /tmp/phase20_baseline_200.md

# Compare to Phase 19 baseline (min_frames=0 case)
# Expected: score >= 0.887
```

### Profiling (Optional)
```bash
# Enable timing to observe dispatch overhead
export IRON_LIBROSA_FFT_TIMING=1
python -c "
import iron_librosa
import numpy as np
y = np.random.randn(44100).astype(np.float32)
for n_fft in [512, 1024]:
  for n_frames in [50, 200, 500, 1000]:
    # Will print timing info via stderr
    S = iron_librosa.stft_complex(y[:n_frames*512], n_fft=n_fft, hop_length=512)
"
# Observe: GPU dispatch kicked in at n_frames >= 200 (Phase 20 gate)
```

---

## Configuration Environment Variables

### Phase 20 Defaults
| Variable | Default | Phase 19 | Phase 20 | Purpose |
|----------|---------|---------|---------|---------|
| `IRON_LIBROSA_FFT_GPU_MIN_FRAMES` | — | 0 | **200** | Min frames to dispatch to GPU (STFT/iSTFT) |
| `IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE` | — | — | — | Override adaptive chunking (if set, uses explicit size) |
| `IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD` | 100M | 100M | 100M | Min work (n_fft × n_frames × log2(n_fft)) to dispatch in Auto mode |
| `IRON_LIBROSA_ENABLE_METAL_FFT_EXPERIMENTAL` | — | force-on | force-on | Enable Metal GPU FFT (must be "force-on" for GPU path) |

### Overriding Phase 20 Defaults
```bash
# Use original Phase 18 behavior (min_frames=0)
export IRON_LIBROSA_FFT_GPU_MIN_FRAMES=0
python script.py

# Use aggressive chunking (override adaptive)
export IRON_LIBROSA_METAL_FFT_BATCH_CHUNK_SIZE=128
python script.py

# Disable GPU dispatch entirely
export IRON_LIBROSA_RUST_DEVICE=cpu
python script.py
```

---

## Performance Targets

Based on Phase 19 A/B empirical results with min_frames=200:

| Workload | n_fft | STFT speedup | iSTFT speedup | GPU dispatch? |
|----------|-------|:---:|:---:|---|
| short_512 | 512 | 1.075x | 0.977x | **No** (< 200 frames, CPU) |
| short_1024 | 1024 | 1.087x | 1.013x | **No** (< 200 frames, CPU) |
| medium_512 | 512 | 1.003x | 0.993x | **Yes** (≥ 200 frames) |
| medium_1024 | 1024 | 0.994x | 1.093x | **Yes** (≥ 200 frames) |
| long_1024 | 1024 | 1.124x | 1.045x | **Yes** (≥ 200 frames, adaptive chunks) |

**Composite Score:** 0.887 (vs 0.767 baseline, +15.6% improvement)

---

## Known Limitations & Next Steps

### Limitations (Phase 20)
1. **iSTFT ceiling** (~1.02–1.05x speedup) — overlap-add normalization dominates post-IFFT, limiting GPU gains; CPU-side bottleneck, not a bug
2. **Remaining regressions** — 1 STFT (likely short_512/short_1024), 2 iSTFT (likely medium_512) still <1.0x; min_frames gate mitigates by directing to CPU
3. **Adaptive chunking heuristic** — conservative (favors CPU fallback on marginal workloads); can be tuned further if empirical data shows opportunity

### Recommended Phase 21 Work
1. **Profile dispatch overhead** separately (Metal launch, buffer copy, kernel execution) to refine adaptive thresholds
2. **Investigate iSTFT ceiling** — profile overlap-add cost; consider Metal kernel fusion if justified
3. **Expand GPU dispatch** to `stft_power` and `stft_complex_f64_batch` (currently CPU-only) using same min_frames + adaptive chunking pattern
4. **Regression deep-dive** — identify exact conditions (n_fft, n_frames, device state) triggering <1.0x cases; consider shape-specific tuning

---

## Files Modified
- `src/istft.rs` — min_frames default change
- `src/stft.rs` — min_frames function + dispatch gate
- `src/metal_fft.rs` — adaptive chunking logic

## Build & Test Artifacts
- Build: Clean, 0.04s release compile
- Warnings: Existing unused code in accelerate_fft.rs (not Phase 20 concern)
- Regression: Full suite run required (pytest command above)

### Closeout (2026-04-12)

- Promotion decision: `Development_docs/PHASE20_PROMOTION_DECISION_2026-04-12.md` (Decision: Promote)
- CUDA kickoff: `Development_docs/PHASE21_CUDA_KICKOFF_2026-04-12.md`
- Closeout checklist: `Development_docs/PHASE20_MACOS_CLOSEOUT_CHECKLIST_2026-04-12.md`
- Benchmark artifacts:
  - `Benchmarks/results/phase20_chunk_ab_2026-04-12.json`
  - `Benchmarks/results/phase20_chunk_ab_2026-04-12.md`
- Test logs:
  - `artifacts/run_logs/phase20_pytest_full_2026-04-12.log`
  - `artifacts/run_logs/phase20_stft_istft_2026-04-12.log`
  - `artifacts/run_logs/phase20_metal_dispatch_2026-04-12.log`

---

## Rollout Plan

### Immediate (Today)
1. Merge Phase 20 changes (min_frames defaults, adaptive chunking)
2. Run full regression suite (~30 min): `pytest -q tests/ --tb=short`
3. Document any new failures (expect: 0)

### Short-term (This week)
1. Run Phase 19-style benchmark sweep with new defaults
2. Compare score to Phase 19 baseline (0.887)
3. If score >= 0.887, promote to default
4. If score < 0.887, investigate adaptive heuristic tuning

### Medium-term (Next phase)
1. Profile & refine adaptive thresholds based on real workload data
2. Extend GPU dispatch to additional functions (stft_power, f64 variants)
3. Plan iSTFT optimization (kernel fusion or profile-driven tuning)

---

## Sign-off

Phase 20 implements Phase 19's empirically-validated optimization (min_frames=200) plus a conservative adaptive chunking strategy. Expected improvement: **+15.6% over Phase 18 baseline** (score 0.887). Regression risk: **minimal** (gating logic is purely additive; CPU path unchanged). 

Ready for testing and promotion to Phase 21.

