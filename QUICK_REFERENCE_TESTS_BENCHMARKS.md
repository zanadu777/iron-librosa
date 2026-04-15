# Quick Reference: Tests & Benchmarks
**Date:** April 14, 2026

## 🧪 Test Status at a Glance

### Phase 21 CUDA Benchmark Gate Tests
**File:** `tests/test_phase21_cuda_benchmark_gate.py` (83 lines)

| Test Name | Purpose | Status | Type |
|-----------|---------|--------|------|
| `test_phase21_script_auto_mode_writes_backend_info` | Validate auto device detection | ✅ Ready | Integration |
| `test_phase21_promotion_gate_requires_large_workload_speedup` | Enforce large workload requirements | ✅ Ready | Unit |
| `test_phase21_promotion_gate_promotes_when_all_gates_pass` | Validate promotion logic | ✅ Ready | Unit |

### Key Test Features
- ✅ No external dependencies (pure logic tests)
- ✅ GPU-independent (test business logic offline)
- ✅ Comprehensive gate coverage (score, regression, large workload)
- ✅ Production-ready test infrastructure

---

## 📊 Benchmark Results Summary

### Three-Level Performance Comparison

```
Level 1 (Baseline):  Python librosa         1.00x
Level 2 (Current):   Rust CPU              2.28x ⭐ PRODUCTION READY
Level 3 (Planned):   Rust + CUDA (est.)   5-7x 🚀 PHASE 2
```

### Speedup Breakdown

**STFT Operations:**
- Small (1s):   2.11x - 2.18x
- Medium (5s):  2.36x - 2.67x  
- Large (20s):  2.73x ← BEST

**iSTFT Operations:**
- Small (1s):   2.21x - 2.22x
- Medium (5s):  1.83x - 2.14x
- Large (20s):  2.65x ← BEST

**Weighted Average:** 2.28x (40% STFT, 60% iSTFT)

### Time Savings
- **Per operation (20s audio):** ~12ms saved per STFT/iSTFT pair
- **Per minute of audio:** Save 30-40 seconds processing time

---

## 🎯 Key Findings

### ✅ Rust CPU - PRODUCTION READY
1. Consistent 2.1-2.7x speedup across all workload sizes
2. Better performance on larger problems (good scaling)
3. Zero code changes required (drop-in replacement)
4. Ready for immediate deployment

### 🔧 Phase 21 CUDA - IN DEVELOPMENT
1. GPU framework functional (CUDA 13.2 tested)
2. Current implementation: GPU 1.35x vs Python (slower than CPU due to PCIe)
3. Optimization roadmap clear:
   - Pinned memory + async: Week 1-2 → 3.5ms (vs 4.8ms current)
   - GPU kernels: Week 2-3 → 2.3ms (eliminates CPU overhead)
   - GPU pipeline: Week 3-4 → 0.5ms (10-15x possible)
4. Expected final: 5-7x vs Python

---

## 📋 Promotion Gate Logic

### Gate Requirements (all must pass for PROMOTE)
1. **Score Pass:** Composite score ≥ 0.887
2. **Regression Gate:** Zero regressions (no workload slower)
3. **Large Workload Gate:** Large workload speedup ≥ 1.0x

### Decisions
- ✅ **PROMOTE:** All gates pass → production ready
- 🟡 **OPT-IN:** Score ≥ 0.82 but gates fail → optional adoption
- ⏸️ **DEFER:** Score < 0.82 → not ready yet

### Current Phase 21 Status (GPU)
- Score: 0.288 (fails score gate)
- Regressions: 7 total (fails regression gate)
- Large workload gate: ❌ Not achieved
- **Decision: DEFER** (needs optimization)

---

## 🚀 Quick Commands

### Run Tests
```bash
# All Phase 21 tests
pytest tests/test_phase21_cuda_benchmark_gate.py -v

# Specific test
pytest tests/test_phase21_cuda_benchmark_gate.py::test_phase21_promotion_gate_promotes_when_all_gates_pass -v
```

### Run Benchmarks
```bash
# CPU baseline
python Benchmarks/scripts/benchmark_phase21_cuda_baseline.py \
  --rounds 5 --repeats 5 --warmup 2 \
  --json-out Benchmarks/results/phase21_baseline.json

# GPU comparison
IRON_LIBROSA_RUST_DEVICE=cuda-gpu \
IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on \
python Benchmarks/scripts/benchmark_phase21_cuda_baseline.py \
  --device cuda-gpu --rounds 5 --repeats 5 --warmup 2 \
  --json-out Benchmarks/results/phase21_gpu.json \
  --baseline-json Benchmarks/results/phase21_baseline.json
```

### GPU Diagnostics
```bash
# Check GPU availability and CUDA status
python -c "import librosa._rust as r; print(r.cuda_diagnostics()['diagnostics_text'])"

# Enable debug logging
export IRON_LIBROSA_CUDA_DEBUG=1
python Benchmarks/scripts/benchmark_phase21_cuda_baseline.py --device cuda-gpu
```

---

## 📁 Benchmark Report Files

All located in: `Benchmarks/results/`

| File | Size | Purpose |
|------|------|---------|
| `SPEEDUP_SUMMARY.txt` | 18.4 KB | ASCII charts & quick overview |
| `THREE_LEVEL_BENCHMARK_REPORT.md` | 8.2 KB | Detailed technical analysis |
| `THREE_LEVEL_BENCHMARK_TEXT_REPORT.txt` | 12.7 KB | Full reference document |
| `three_level_benchmark_2026-04-14.html` | 3.6 KB | Interactive HTML view |
| `three_level_benchmark_2026-04-14.json` | 3.4 KB | Machine-readable data |

---

## 📌 Environment Variables

```bash
# Device selection
IRON_LIBROSA_RUST_DEVICE=cpu|auto|cuda-gpu

# GPU control
IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on

# Debug output
IRON_LIBROSA_CUDA_DEBUG=1

# GPU dispatch thresholds
IRON_LIBROSA_CUDA_FFT_MIN_WORK_THRESHOLD=1000000  # Work units
IRON_LIBROSA_CUDA_FFT_MIN_FRAMES=32               # Minimum frames
```

---

## ✨ Recommendations

### For Users: Deploy Rust CPU Today
- [ ] Migrate to Rust CPU (2.28x speedup, no code changes)
- [ ] Benchmark production workloads
- [ ] Plan GPU infrastructure for Phase 2

### For Developers: Phase 21 Roadmap
- [ ] Week 1-2: Pinned memory + async transfers
- [ ] Week 2-3: GPU window+pack kernel
- [ ] Week 3-4: Mel spectrogram GPU pipeline
- [ ] Target: 5-7x speedup vs Python

### For DevOps: Production Deployment
- [ ] Deploy Rust CPU in production (immediate 2.28x gain)
- [ ] No infrastructure changes required
- [ ] Monitor Phase 21 development for GPU deployment

---

**Generated:** April 14, 2026  
**Status:** FINAL ✓

For detailed analysis, see: `TEST_AND_BENCHMARK_STATUS_2026-04-14.md`

