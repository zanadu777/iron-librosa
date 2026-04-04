# Phase 7+ Planning: Next High-ROI Kernel Targets

Based on profiling, complexity analysis, and user-visible API coverage, here's the recommended prioritization:

## Tier 1 (Immediate, high impact)

### `spectral_contrast` — Medium complexity, high impact
- **Why high ROI**: Widely-used feature, currently pure-Python reduction
- **Complexity**: Medium (octave-band filtering + envelope extraction)
- **Risk**: Low (self-contained feature, no upstream dependencies)
- **Speedup estimate**: 5–15x (similar to flatness scale profile)
- **Effort**: ~200–300 LOC Rust, ~50 LOC Python dispatch
- **Next steps**:
  1. Implement fused Rust kernels for octave-band static-frequency case
  2. Fast paths for default band count (7 bands)
  3. Guarded dispatch like flatness (float32/float64 only)

### `filters.chroma` norm expansion — Low complexity, easy win
- **Why high ROI**: Unblock 2–3 use cases with minimal effort
- **Complexity**: Low (extend existing norm=2 path to norm=1, None, inf)
- **Risk**: Very low (isolated to chroma filter logic)
- **Speedup estimate**: 2–3x for newly-enabled norms
- **Effort**: ~100 LOC Rust, ~20 LOC Python
- **Next steps**:
  1. Add `spectral_chroma_f32_norm1`, `spectral_chroma_f32_norms`, etc.
  2. Extend dispatch guard to check `norm` parameter
  3. Maintain fallback for complex/unsupported norms

## Tier 2 (Medium-term, high upside)

### Variable-frequency fast paths (centroid, rolloff, bandwidth)
- **Why high ROI**: Unlock reassigned/time-frequency adaptive workflows
- **Complexity**: Medium-high (variable indexing, reduced SIMD potential)
- **Risk**: Medium (more pointer chasing, less cache-friendly)
- **Speedup estimate**: 2–4x (lower than static-freq due to indexing overhead)
- **Effort**: ~300–500 LOC Rust, ~100 LOC Python per function
- **Next steps**:
  1. Benchmark current pure-Python variable-freq bottleneck
  2. Implement variable-freq kernel for centroid first (simplest)
  3. Extend to rolloff/bandwidth once centroid validated

## Tier 3 (Major initiative, transformative)

### `piptrack` hot-loop optimization — High complexity, highest upside
- **Why high ROI**: Unlock default-on tuning → `chroma_stft` speedups by 5–20x
- **Complexity**: High (masking, interpolation, peak tracking internals)
- **Risk**: Medium-high (affects core pitch detection algorithm)
- **Speedup estimate**: 10–30x end-to-end for `estimate_tuning()` → 2–5x for `chroma_stft()`
- **Effort**: ~1000+ LOC Rust, multi-day integration
- **Next steps**:
  1. Profile `piptrack.piptrack()` with realistic audio (1–10 min clips)
  2. Identify hot spots (likely: the energy tracking + peak masking loops)
  3. Implement fused Rust kernel for the innermost 2–3 loops
  4. **Expect regressions initially** — tuning is sensitive to numerical precision
  5. Use new `benchmark_phase5_tuning.py` to A/B vs pure-Python

## Tier 4 (Nice-to-have, coverage)

### `spectral_centroid` variable-frequency fast path
- Needed for reassigned workflows; lower priority than static-freq optimization

### `onset.onset_detect` acceleration
- Currently dispatches to `onset_flux`, which is already Rust-accelerated in Phase 4
- Limited upside unless profiling shows CPU hotspot

### `mfcc` DCT-3 inverse (orthogonal DCT)
- Already Rust-accelerated; only benefit if users request non-orthogonal variants

## Execution Strategy

### For each kernel target:
1. **Baseline benchmark** — Use existing `benchmark_phase5_*.py` pattern
2. **Raw kernel tests** — Add `TestXxxKernels` class with parity vs NumPy
3. **API dispatch tests** — Add `TestXxxAPI` class with forced-fallback A/B
4. **Guard design** — Prefer explicit dtype/ndim checks over try-catch
5. **Multichannel support** — Test 3D/4D inputs before shipping
6. **Fallback validation** — Ensure pure-Python fallback always available

### Risk mitigation for `piptrack`:
- **Do NOT make default-on until extensive validation** — keep experimental flag
- **Run full `estimate_tuning()` test suite** before merging
- **A/B real-world audio** (different genres, sample rates, tempos)
- **Validate against librosa 0.11 exact output** where possible

## ROI Scoring (Quick Reference)

| Kernel | Complexity | Impact | Risk | Speedup | Effort | ROI | Priority |
|--------|-----------|--------|------|---------|--------|-----|----------|
| spectral_contrast | M | H | L | 5–15x | M | **H** | **1** |
| filters.chroma norm | L | M | VL | 2–3x | L | **H** | **2** |
| Variable-freq paths | M–H | M | M | 2–4x | M–H | M | **3** |
| piptrack | H | VH | M–H | 10–30x | H | **VH** | **4** |

---

**Recommendation**: Complete Phase 6 (flatness) ✅, then **start Phase 7 with `spectral_contrast`** (medium complexity, solid ROI, low risk). Use results to scope `piptrack` work in Phase 8.

