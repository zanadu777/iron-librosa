# iron-librosa: Features Remaining & Roadmap
## April 3, 2026

---

## Summary: What's Complete vs. Remaining

### ✅ Completed (76 Rust Kernels)

**Phase 1–3:** Foundation, decomposition, core DSP
- Mel-spectrogram projection (faer GEMM)
- DCT transforms
- STFT (forward)
- Onset detection
- Filtering (NN filter)
- Frequency conversions (Hz ↔ Mel)

**Phase 4A:** Inverse operations
- ISTFT (inverse STFT)
- dB conversions (4 directions: power/amplitude ↔ dB)

**Phase 4B:** Spectral features (optimized)
- RMS spectrogram (f32/f64, **12× speedup**)
- Spectral centroid (f32/f64, **43× speedup**)

**Phase 4C:** Music analysis
- Chroma filter projection (faer GEMM, **2.4× speedup**)

**Phase 5:** Spectral completion + chroma bottleneck reduction
- Spectral rolloff
- Spectral bandwidth (including fused auto-centroid path)
- Chroma filter generation (`filters.chroma`)
- Tuning post-processing helper for `estimate_tuning` (implemented, experimental / threshold-gated)

---

## ⏳ Features Remaining (Post-Phase-5 Backlog)

### 1. **Time-Domain RMS** (MEDIUM)
**Status:** Not yet accelerated

**What it is:**
- `librosa.feature.rms(y=...)` — RMS of audio frames (not spectrogram)
- Currently: window + hop + sum-of-squares + sqrt

**Current Implementation:** Python with scipy windowing

**Estimated Speedup:** 1.5–2× 

**Implementation Complexity:** MEDIUM-HIGH (complex windowing logic, edge cases)

**Lines of Code:** ~200 Rust + ~60 Python dispatch

**Effort:** 2–3 days

**Why deferred:** More complex than spectrogram path (requires handling multiple window types, center padding modes, output length logic)

---

### 2. **Variable-Frequency-Grid Spectral Features** (MEDIUM)
**Status:** Not yet started

**What it is:**
- Spectral centroid / rolloff with **2-D frequency grids** (reassigned spectrograms, CQT)
- Current fast-path only supports static 1-D freq vectors

**Current Implementation:** Python fallback for all variable-grid cases

**Estimated Speedup:** 2–3× (heavier computation than static-freq case)

**Implementation Complexity:** HIGH (requires adaptive frequency lookup, no GEMM optimization)

**Effort:** 3–4 days

**Why deferred:** Reassigned spectrograms and CQT are less common than standard STFT; current fallback is acceptable

**Files to Create/Modify:**
- `src/spectrum_utils.rs` (add 2 kernels: rms_time f32/f64)
- `librosa/feature/spectral.py` (add dispatch guard)

---

### 3. **Chroma Tuning Estimation Promotion** (MEDIUM)
**Status:** Implemented, but still experimental / threshold-gated

**What it is:**
- `librosa.feature.spectral.estimate_tuning()` Rust post-processing helper is implemented
- Currently guarded behind `IRON_LIBROSA_ENABLE_RUST_TUNING` and workload threshold

**Current Implementation:** Hybrid; default remains Python unless opt-in / large-enough workload

**Measured Speedup:**
- full API: ~1.03× to ~1.19×
- post-piptrack helper only: ~1.13× to ~3.18×

**Implementation Complexity:** MEDIUM-HIGH for further promotion because upstream `piptrack` dominates total time

**Effort to improve further:** 2–5 days depending on whether work stays in post-processing or expands into `piptrack`

**Why not default-on yet:** The helper is useful, but full `estimate_tuning()` remains only modestly faster end-to-end on current workloads

---

### 4. **Spectral Features: Flatness & Contrast** (LOW)
**Status:** Not yet accelerated

**What it is:**
- `librosa.feature.spectral.estimate_tuning()` — autocorrelation-based A440 deviation detection
- Currently: Pure Python autocorrelation + argmax

**Current Implementation:** `scipy` autocorrelation via convolve (slow)

**Estimated Speedup:** 5–10× (autocorrelation is expensive in Python)

**Real-World Impact:** **HIGH** — This function is 5–10 ms of the ~60 ms `chroma_stft()` total time

**Implementation Complexity:** MEDIUM-HIGH (FFT-based autocorrelation, requires careful numerics)

**Lines of Code:** ~300 Rust + ~80 Python dispatch

**Effort:** 3–5 days

**Why it's a bottleneck:**
- Called once per `chroma_stft()` invocation
- Autocorrelation is O(n²) in pure Python
- Would enable end-to-end Phase 4C speedup (currently blocked by this)

**Files to Create/Modify:**
- `src/spectral.rs` (new module: autocorrelation kernel)
- `librosa/feature/spectral.py` (dispatch for estimate_tuning)

---

### 5. **MFCC Optimization** (MEDIUM)
**Status:** Already accelerated, but potential further gains

**What it is:**
- `librosa.filters.chroma()` — builds chroma filter bank via octave-wise summation
- Currently: NumPy sparse matrix operations

**Current Implementation:** Sparse matrix multiplication with octave/semitone logic

**Estimated Speedup:** 1.5–2×

**Real-World Impact:** MEDIUM — 1–3 ms of total `chroma_stft()` time

**Implementation Complexity:** MEDIUM (sparse operations, but well-defined pattern)

**Effort:** 2–3 days

---

### 6. **Spectral Features: Flatness & Contrast** (LOW)
**Status:** Not yet accelerated

**What it is:**
- `librosa.feature.spectral_flatness()` — geometric vs. arithmetic mean ratio
- `librosa.feature.spectral_contrast()` — contrast across sub-bands

**Current Implementation:** Pure Python with logarithmic operations

**Estimated Speedup:** 1.5–2×

**Implementation Complexity:** LOW (straightforward element-wise ops)

**Effort:** 1–2 days

**Why lower priority:** Less commonly used than centroid/rolloff

---

### 7. **MFCC Optimization** (MEDIUM)
**Status:** Already accelerated, but potential further gains

**What it is:**
- `librosa.feature.mfcc()` — Mel-Frequency Cepstral Coefficients

**Current Pipeline:**
- ✅ STFT (Rust, Phase 3)
- ✅ Mel-spectrogram projection (Rust, faer GEMM)
- ✅ DCT (Rust, Phase 2)

**Opportunity:** Combine mel+DCT into single Rust kernel (avoid intermediate array)

**Estimated Speedup:** 10–15% (memory bandwidth savings)

**Effort:** 2–3 days

**Why deferred:** Diminishing returns; 3-operation pipeline is already very fast

---

### 8. **Phase Vocoder / Time-Stretching** (HIGH)
**Status:** Not started (deferred to Phase 5+)

**What it is:**
- `librosa.phase_vocoder()` — time-stretch without pitch change
- Requires phase reconstruction (complex vocoder algorithm)

**Current Implementation:** Pure Python with phase manipulation

**Estimated Speedup:** 2–3×

**Implementation Complexity:** HIGH (heuristic phase reconstruction, numerical stability)

**Effort:** 5–7 days

**Why deferred:** Complex algorithm, lower call frequency than feature extraction

---

### 9. **Constant-Q Transform (CQT)** (HIGH)
**Status:** Not started (deferred to Phase 5+)

**What it is:**
- `librosa.cqt()` — logarithmically-spaced frequency representation
- Alternative to STFT for music (better low-frequency resolution)

**Current Implementation:** Python with sparse filterbank multiplication

**Estimated Speedup:** 2–3× (sparse GEMM opportunity)

**Implementation Complexity:** HIGH (sparse matrix operations, FFT padding tricks)

**Effort:** 4–6 days

**Why deferred:** Less common than STFT; sparse operations are complex

---

### 10. **Advanced Decomposition** (MEDIUM)
**Status:** Not started

**What it is:**
- `librosa.decompose.nn_filter()` — already done (Phase 2) ✅
- `librosa.decompose.hpss()` — Harmonic/Percussive Source Separation (unsupervised, median filtering)
- `librosa.decompose.nmf()` — Non-Negative Matrix Factorization (relies on scikit-learn, expensive)

**Current Implementation:** scipy (median filtering), scikit-learn (NMF)

**Estimated Speedup:** 2–3× for median-based HPSS

**Effort:** 3–4 days (HPSS only; NMF is scikit-learn dependent)

---

## Deferred (Phase 5+)

### Low Priority / Complex / Less Common

| Feature | Speedup | Effort | Notes |
|---|---|---|---|
| Spectral flatness/contrast | 1.5–2× | 1–2 days | Low call frequency |
| MFCC (combined kernel) | 1.1–1.15× | 2–3 days | Diminishing returns |
| Phase vocoder | 2–3× | 5–7 days | Complex algorithm, lower frequency |
| CQT | 2–3× | 4–6 days | Sparse operations, less common |
| HPSS (median-based) | 2–3× | 3–4 days | Medium call frequency |
| NMF wrapper | 1.2–1.5× | 4–5 days | scikit-learn dependency, already optimized |

---

## Recommended Phase 5 Plan

### High-Impact, Quick Wins (1–2 weeks)

**Priority 1: Spectral Rolloff & Bandwidth**
- Effort: 1–2 days
- Speedup: 1.8–2.5×
- Completion: Complete Phase 4B planned scope

**Priority 2: Tuning Estimation (for chroma)**
- Effort: 3–5 days
- Speedup: 5–10× (this operation alone)
- Real-world impact: Enable end-to-end chroma speedup

**Priority 3: Chroma Filter Generation**
- Effort: 2–3 days
- Speedup: 1.5–2×
- Real-world impact: Complete chroma acceleration

**Total Phase 5 Effort:** 6–10 days (1.5 weeks)  
**Expected Phase 5 Deliverables:** 4–6 new Rust kernels, 3–5 public API dispatches

### Medium-Priority, More Complex

**After Phase 5.1:**
- Time-domain RMS (2–3 days)
- Variable-frequency-grid features (3–4 days)
- Spectral flatness/contrast (1–2 days)

---

## Summary: What Librosa Features Have Rust Acceleration?

### ✅ Already Accelerated (26 public functions)
1. `librosa.feature.melspectrogram()`
2. `librosa.feature.mfcc()`
3. `librosa.feature.chroma_stft()`
4. `librosa.feature.rms()`
5. `librosa.feature.spectral_centroid()`
6. `librosa.core.stft()`
7. `librosa.core.istft()` — partial (inverse only)
8. `librosa.decompose.nn_filter()`
9. Frequency conversions (`hz_to_mel()`, `mel_to_hz()`)
10. dB conversions (`power_to_db()`, `amplitude_to_db()`, inverses)
11. DCT (`dct()`— via scipy wrapper using Rust kernel)
12. Onset detection (`onset.onset_strength()` via `librosa.onset.onset_detect()`)

### ⏳ High-Value Candidates for Phase 5 (Would Accelerate)
1. `librosa.feature.spectral_rolloff()` — 1–2 days
2. `librosa.feature.spectral_bandwidth()` — included with rolloff
3. `librosa.feature.spectral_flatness()` — 1–2 days
4. `librosa.feature.spectral_contrast()` — 1–2 days
5. `librosa.feature.chroma_stft()` — tuning estimation bottleneck (3–5 days for full acceleration)

### ❌ Not Yet Accelerated (Complex, Lower Priority)
1. `librosa.feature.tonnetz()` — requires special chroma handling
2. `librosa.cqt()` — Constant-Q Transform (sparse operations)
3. `librosa.phase_vocoder()` — phase reconstruction algorithm
4. `librosa.beat.beat_track()` — relies on onset detection + dynamic programming
5. `librosa.decompose.hpss()` — harmonic/percussive separation (median filtering + masking)
6. `librosa.decompose.nmf()` — Non-Negative Matrix Factorization (scikit-learn dependency)

---

## Next Steps

### Immediate (Phase 5, ~2 weeks)
- [ ] Spectral rolloff & bandwidth (1–2 days)
- [ ] Tuning estimation acceleration (3–5 days)
- [ ] Chroma filter generation (2–3 days)
- [ ] Tests + benchmarks for each

### Medium-term (Phase 5.1–5.2, ~3–4 weeks)
- [ ] Time-domain RMS acceleration
- [ ] Variable-frequency spectral features
- [ ] Additional spectral metrics (flatness, contrast)

### Advanced (Phase 6+, ~4+ weeks)
- [ ] Phase vocoder
- [ ] CQT acceleration
- [ ] HPSS / advanced decomposition
- [ ] GPU-accelerated paths (optional, future)

---

## Conclusion

**Remaining features fall into three categories:**

1. **Quick wins (1–2 days each):** Rolloff, bandwidth, flatness, contrast
2. **High-impact bottlenecks (3–5 days each):** Tuning estimation, time-domain RMS
3. **Complex algorithms (4+ days each):** Phase vocoder, CQT, HPSS, NMF

**Phase 5 recommendation:** Focus on **Quick wins + Tuning estimation** to complete spectral feature acceleration and enable end-to-end chroma speedup.


