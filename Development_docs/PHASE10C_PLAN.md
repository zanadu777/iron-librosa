# Phase 10C: HPSS Optimization & Validation Plan

## Objectives

1. **Masking Optimization** - Parallelize softmask computation
2. **2D Decomposition** - Add frame-wise parallelism for masking layer
3. **Integration Testing** - Full `librosa.decompose.hpss()` validation
4. **Performance Benchmarking** - Measure speedup vs scipy baseline
5. **Quality Assurance** - Parity validation, edge case testing

## Current Implementation Analysis

### Existing (Phase 10A-10B)
- ✅ Median filter kernels (harmonic & percussive)
- ✅ Batch-level parallelism with adaptive dispatch
- ✅ SciPy fallback (stable baseline)

### Bottleneck Analysis
Current masking computation (lines 2022-2037 in spectrum_utils.rs):
```rust
// Sequential nested loops: n_bins × n_frames
for f in 0..n_bins {
    for t in 0..n_frames {
        let harm_val = harm[[f, t]];
        let perc_val = perc[[f, t]];
        let mh = softmask_pair_f32(...);
        let mp = softmask_pair_f32(...);
        // Apply mask or return
    }
}
```

**Problem:** Double nested loop is purely sequential
**Solution:** Parallelize over frames (inner loop) using rayon

## Phase 10C Tasks

### Task 1: Masking Parallelization (f32 & f64)
**Objective:** Replace sequential masking loop with parallel frame-wise computation

**Changes needed:**
- `hpss_fused_core_2d_f32`: Parallelize frame iteration
- `hpss_fused_core_2d_f64`: Parallelize frame iteration
- Use rayon `.into_par_iter()` over frames
- **Expected:** 2-4x speedup on masking (depending on hardware)

**Files:** `src/spectrum_utils.rs` (lines ~2022-2037)

### Task 2: Soft-mask Kernel Optimization
**Objective:** Optimize softmask_pair computation

**Options:**
1. Vectorization hints
2. Branch prediction optimization
3. Power computation caching (if margins are constant)
4. Conditional compilation for release mode

**Files:** `src/spectrum_utils.rs` (lines 1960-1989)

### Task 3: 2D Decomposition Support
**Objective:** Add parallel processing for multi-dimensional spectrograms

**Current:** Reshapes N-D → 3D, processes batch
**Proposed:** Optimize for typical stereo/multichannel cases

**Files:** `librosa/decompose.py` (lines 401-432)

### Task 4: Comprehensive Testing
**Objective:** Full validation suite

**Test cases:**
- ✅ Parity: Rust vs SciPy output
- ✅ Batch correctness: Batch processing matches sequential
- ✅ Edge cases: Small inputs, large inputs, margins
- ✅ Dtype validation: f32 and f64
- ✅ Shape validation: 2D, 3D, 4D inputs
- ✅ Mask mode: Both `mask=True` and `mask=False`

**Files:** `tests/test_phase10c_hpss_optimization.py` (new)

### Task 5: Benchmarking & Validation
**Objective:** Measure speedup and validate performance

**Benchmark types:**
- SciPy baseline comparison
- Scaling analysis (input size impact)
- Batch processing impact
- Masking vs full decomposition
- f32 vs f64 comparison

**Files:** `benchmark_phase10c_hpss_optimization.py` (new)

## Implementation Strategy

### Phase 1: Masking Parallelization (Day 1)
```
├─ Parallelize hpss_fused_core_2d_f32 (frame loop)
├─ Parallelize hpss_fused_core_2d_f64 (frame loop)
├─ Run quick validation tests
└─ Measure performance improvement
```

### Phase 2: Soft-mask Optimization (Day 1-2)
```
├─ Profile softmask_pair computation
├─ Optimize for common case (split_zeros=true)
├─ Add inline hints if beneficial
└─ Validate numerical precision
```

### Phase 3: Testing & Validation (Day 2)
```
├─ Create comprehensive test suite
├─ Parity validation (vs SciPy)
├─ Batch correctness checks
├─ Edge case coverage
└─ All tests pass
```

### Phase 4: Benchmarking (Day 3)
```
├─ Create benchmark suite
├─ Measure baseline improvements
├─ Scaling analysis
├─ Generate final report
└─ Performance validation
```

## Expected Improvements

### Masking Parallelization
- **Single frame:** 2-4× (parallelism factor)
- **Typical case:** 1.5-3× (hardware contention)
- **Batch case:** Already optimized in Phase 10B

### Overall HPSS
- **Combined with Phase 10B:** 2-3× total improvement
- **vs SciPy baseline:** 7.8× × 2-3 = 15-20×

## Success Criteria

✅ All tests passing (parity, batch, edge cases)
✅ Measurable speedup (2-3× on masking layer)
✅ Zero regression (SciPy fallback still works)
✅ Full backward compatibility
✅ Comprehensive documentation

## Risk Assessment

🟢 **Low Risk** - Isolated changes to masking computation
- Doesn't affect median filter kernels
- SciPy fallback always available
- Can disable Rust HPSS via guards
- Conservative default (scipy active)

## Timeline

**Day 1:** Masking parallelization + soft-mask optimization
**Day 2:** Comprehensive testing
**Day 3:** Benchmarking & final validation

**Total:** 2-3 days to production-ready

## Files to Modify/Create

**Modified:**
- `src/spectrum_utils.rs` - Masking parallelization

**Created:**
- `tests/test_phase10c_hpss_optimization.py` - Comprehensive tests
- `benchmark_phase10c_hpss_optimization.py` - Benchmarking suite
- `PHASE10C_STATUS.md` - Progress tracking
- `PHASE10C_COMPLETION_REPORT.md` - Final report

