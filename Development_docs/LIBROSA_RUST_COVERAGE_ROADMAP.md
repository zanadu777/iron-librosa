# Librosa Rust Acceleration: Remaining Work Analysis

**Current Date:** April 4, 2026  
**Status:** Phase 14 kickoff active  
**Phase 13 (CQT/VQT):** ✅ COMPLETE (opt-in backend, not promoted by default)

---

## 📊 CURRENT RUST COVERAGE

### ✅ Fully Accelerated (Ready)
- **STFT** (complex & power, f32 & f64, batched)
- **ISTFT** (f32 & f64)
- **Phase Vocoder** (f32 & f64, just promoted to default)
- **Mel Spectrogram** (project, filter, f32 & f64)
- **DCT** (2D ortho, f32 & f64)
- **Spectral Features** (30+ functions):
  - Centroid, Bandwidth, Rolloff, Flatness, Contrast
  - RMS, Power/DB conversions, Amplitude/DB conversions
  - HPSS (Harmonic-Percussive Source Separation)
  - Median filtering
- **Onset Detection** (flux mean variants, f32 & f64)
- **Chroma** (filter & project, f32 & f64)
- **Tuning** (piptrack, pitch estimation, f32 & f64)
- **Hz/Mel Conversion** (hz_to_mel, mel_to_hz)
- **NN Filter**

### ⏳ Partially Accelerated (In Progress)
- **CQT/VQT** (Rust seam implemented, opt-in via `IRON_LIBROSA_CQT_VQT_BACKEND`)
- **Beat Tracking / Tempo** (Phase 14 kickoff)
- **Mel Multi-Host Threshold** (cross-CPU strategy in development)

### ❌ Not Accelerated Yet (Major Gaps)
1. **Decompose** (NMF - deferred to scikit-learn)
2. **Segment & Beat Tracking** (beat_track, segment)
3. **Temporary Resonance Modification** (TRM)
4. **Chromagram postprocessing** (some advanced variants)
5. **Constant-Q variants** (CQT, VQT - in Phase 12)
6. **Tempo estimation** (full tempo detection)
7. **Harmonic-Percussive source separation advanced** (some variants)
8. **Effects** (time-stretching beyond phase-vocoder, pitch shifting)

---

## 🎯 PHASE 12 TARGETS (Current)

### 1. **Mel Cross-CPU Threshold Strategy** 🔴 IN PROGRESS
**What:** Decide when to use Rust mel projection vs Python based on CPU
**Why:** Different CPUs have different performance characteristics
**Status:** Strategy document needed
**Effort:** 1-2 sessions

### 2. **Phase Vocoder** ✅ COMPLETE
**What:** Rust kernel with parity validation
**Status:** Just promoted to production default
**Effort:** DONE

### 3. **CQT/VQT Acceleration** 🟡 PLANNING
**What:** Constant-Q Transform & Variable-Q Transform kernels
**Why:** Expensive operations for music analysis
**Status:** Planning phase
**Effort:** 3-4 weeks (high complexity)
**Challenge:** Non-uniform frequency bins, complex FFT structure

### 4. **Tonnetz Parity Policy** 🟡 PLANNING
**What:** Ensure tonal centroid network preserves behavior
**Status:** Decision document needed
**Effort:** 1 session

### 5. **Beat Tracking Optimization** ✅ PARTIAL
**What:** Optimize librosa.beat.beat_track() performance
**Status:** Python-level optimization applied (autocorrelation window)
**Result:** Measurable improvement on noisy audio
**Effort:** DONE (Python layer)
**Next:** Rust kernel for lower-level components?

---

## 📋 MAJOR REMAINING FEATURES

### HIGH PRIORITY (High impact, reasonable effort)

#### 1. **CQT/VQT** 
- Lines of code: ~500-800 Rust
- Estimated speedup: 2.0-3.0×
- Difficulty: HIGH (non-uniform FFT required)
- Frequency: Used in pitch/tuning analysis
- Plan: Phase 12 target

#### 2. **Advanced Tempo Estimation**
- Lines of code: ~300-500 Rust
- Estimated speedup: 1.5-2.0×
- Difficulty: MEDIUM (autocorrelation + onset detection)
- Frequency: Popular feature
- Dependencies: Onset detection (done), autocorrelation (partially done)

#### 3. **Segment & Clustering**
- Lines of code: ~400-600 Rust
- Estimated speedup: 1.5-2.5×
- Difficulty: MEDIUM-HIGH (linear algebra heavy)
- Frequency: Music analysis/structure
- Dependencies: librosa.segment module

### MEDIUM PRIORITY (Moderate impact or effort)

#### 4. **Advanced HPSS Variants**
- Lines of code: ~200-300 Rust
- Estimated speedup: 1.3-1.8×
- Difficulty: MEDIUM
- Frequency: Source separation
- Status: Basic HPSS done, variants remain

#### 5. **Effects Module**
- Lines of code: ~600-1000 Rust per effect
- Estimated speedup: 2.0-4.0×
- Difficulty: MEDIUM (but many functions)
- Frequency: Audio processing
- Examples: Time-stretching (beyond phase-vocoder), pitch-shifting, reverb

#### 6. **Decomposition (NMF)**
- Lines of code: ~1000+ Rust
- Estimated speedup: 2.0-3.0× (if replacing sklearn)
- Difficulty: VERY HIGH
- Frequency: Music analysis
- Status: Deferred to scikit-learn (external dependency)
- Decision: Keep external for now (specialized library)

### LOW PRIORITY (Lower frequency or complex)

#### 7. **Fourier Coefficients Analysis**
- Various specialized transforms
- Difficulty: MEDIUM-HIGH
- Frequency: Niche use cases

#### 8. **Psychoacoustic Models**
- Lines of code: ~500-800 Rust
- Difficulty: HIGH
- Frequency: Niche research
- Impact: Moderate

---

## 💪 What's Done vs Remaining

```
FULLY RUST-ACCELERATED (Production Ready):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ STFT / ISTFT (5 functions, 2 dtypes each)
✅ Phase Vocoder (2 functions)
✅ Mel Spectrogram (3 functions)
✅ Spectral Features (30+ functions)
✅ Onset Detection (6 functions)
✅ Chroma (4 functions)
✅ Tuning (4 functions)
✅ Conversions (2 functions)
✅ DCT (2 functions)
✅ NN Filter (1 function)

PARTIALLY ACCELERATED (In Progress):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 Mel Threshold (strategy phase)
🔄 CQT/VQT (planning phase)

NOT ACCELERATED YET:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ Segment/Beat Tracking (high-value targets)
❌ Advanced Effects
❌ CQT/VQT (phase 12 target)
❌ NMF Decomposition (external dependency)
❌ Tempo estimation (advanced)
❌ Some chroma variants
```

---

## 🚀 Recommended Roadmap for Full Rust Coverage

### **Phase 13: CQT/VQT** ✅ COMPLETE
**Duration:** 3-4 weeks  
**Priority:** HIGH (popular, high-value)  
**Effort:** HIGH  
**Expected Gain:** 2.0-3.0× speedup

**Outcome:**
1. Rust dense projection kernels added and wired
2. Parity and regression tests passed
3. Benchmark evidence captured
4. Promotion decision: keep opt-in for now
5. Reference: `Development_docs/PHASE13_CQT_VQT_COMPLETION_REPORT.md`

### **Phase 14: Beat Tracking Acceleration** 🔄 ACTIVE
**Duration:** 2-3 weeks  
**Priority:** HIGH (very popular feature)  
**Effort:** MEDIUM  
**Expected Gain:** 1.5-2.0× speedup

**Kickoff Tasks:**
1. Baseline profile: `librosa.beat.beat_track()` and tempo path
2. Isolate DP hotspot (`librosa.beat.__beat_track_dp`) for Rust candidate seam
3. Add parity tests for mono/stereo beat tracking outputs
4. Add benchmark deltas for Python vs Rust candidate path
5. Reference kickoff doc: `Development_docs/PHASE14_BEAT_TRACK_KICKOFF.md`

### **Phase 15: Effects & Advanced Transforms**
**Duration:** 4+ weeks  
**Priority:** MEDIUM-HIGH  
**Effort:** HIGH (many functions)  
**Expected Gain:** 1.5-3.0× per effect

**Tasks:**
1. Time-stretching variants (beyond phase-vocoder)
2. Pitch-shifting
3. Reverb/convolution effects
4. Each requires careful parity validation

### **Phase 16+: Remaining Features**
- Segment/clustering
- Advanced HPSS
- Psychoacoustic models
- Specialized transforms

---

## 📈 Overall Coverage Target

**Current:** ~70% of hot-path functions accelerated  
**Target:** 95%+ of frequently-used functions  
**Remaining:** ~20-25% of lower-impact functions

---

## ✅ Completion Criteria

For "all features set to use Rust":
1. ✅ All STFT/ISTFT variants accelerated
2. ✅ All spectral features accelerated (done)
3. ✅ All onset/beat-related accelerated (partial)
4. 🔄 CQT/VQT accelerated (Phase 13)
5. 🔄 Segment/clustering accelerated (Phase 14-15)
6. ⏳ Effects accelerated (Phase 15+)
7. ⏳ Advanced transforms accelerated (Phase 16+)

---

## 🎯 Summary: What's Next?

### **Immediate (Phase 14)**
- 🔄 Beat tracking acceleration kickoff
- 🔄 Tempo-path hotspot profiling and seam definition
- 🔄 Mel cross-CPU threshold strategy

### **Near-Term (Phase 14-15)**
- Beat tracking/tempo acceleration promotion path
- Effects module acceleration
- Advanced feature extraction

### **Long-Term (Phase 16+)**
- Remaining specialized transforms
- Research-level algorithms
- Library completeness

---

## 📊 Rough Effort Estimate

| Feature | Effort | Impact | Difficulty |
|---------|--------|--------|-----------|
| CQT/VQT | 3-4w | HIGH | ⭐⭐⭐⭐ |
| Beat Tracking | 2-3w | HIGH | ⭐⭐⭐ |
| Effects | 4-6w | MEDIUM | ⭐⭐⭐ |
| Segment/Clustering | 2-3w | MEDIUM | ⭐⭐⭐ |
| Remaining | 4-6w | LOW-MED | ⭐⭐ |

**Total to 95% coverage:** ~15-20 weeks (3-5 months) of focused work

---

## 🎓 Key Decisions Made

✅ **NMF:** Defer to scikit-learn (external, specialized library)  
✅ **Phase Vocoder:** Enabled by default (ties-to-even rounding fixed)  
⏳ **CQT/VQT:** Targeted for Phase 13 (high impact, manageable)  
⏳ **Beat Tracking:** Hybrid approach (Python optimization + Rust path planning)

---

## Next Steps

1. **Run Phase 14 kickoff checklist** (`Development_docs/PHASE14_BEAT_TRACK_KICKOFF.md`)
2. **Keep Phase 13 seam benchmarked** and revisit promotion only with clear gains
3. **Document beat/tempo parity gates** before enabling any Rust dispatch by default
4. **Continuous:** Benchmark improvements, validate parity

