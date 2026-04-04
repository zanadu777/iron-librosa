# Quick Reference: Librosa Rust Acceleration Status

**Date:** April 4, 2026  
**Overall Coverage:** 70% → Target: 95%+

---

## ✅ DONE (Production Ready)

- STFT / ISTFT
- Phase Vocoder ✅ (just promoted)
- Mel Spectrogram
- Spectral Features (30+ functions)
- Onset Detection
- Chroma
- Tuning
- DCT
- Utils (Hz/Mel, NN Filter, etc.)

**= ~40 functions, 2 dtypes each = ~80 kernels**

---

## 🔄 IN PROGRESS (Phase 12)

1. Mel cross-CPU threshold strategy (1-2 weeks)
2. CQT/VQT planning (ongoing)
3. Tonnetz parity policy (1 week)

---

## 🎯 HIGH-PRIORITY NEXT (15-20 weeks to 95% coverage)

| Phase | Feature | Duration | Speedup |
|-------|---------|----------|---------|
| **13** | CQT/VQT | 3-4 weeks | 2.0-3.0× |
| **14** | Beat Tracking | 2-3 weeks | 1.5-2.0× |
| **15** | Segment/Effects | 4-6 weeks | 1.5-4.0× |
| **16+** | Remaining | 4-6 weeks | 1.3-2.0× |

---

## ❌ NOT PLANNED (Strategic Decisions)

- **NMF Decomposition** ← Defer to scikit-learn (external)
- **Niche features** ← Low ROI
- **Some psychoacoustic** ← Research level

---

## 📊 By The Numbers

```
Current Status:
━━━━━━━━━━━━━━━━━
✅ 70% of hot-path functions accelerated
✅ ~80 Rust kernels implemented
⏳ 20-30% remaining (high-value targets)

After Phase 13 (CQT/VQT):
━━━━━━━━━━━━━━━━━━━━━━━━━
📈 ~80% coverage
📈 +2 major feature modules

Full Coverage (Phase 16+):
━━━━━━━━━━━━━━━━━━━━━━━━
✅ ~95% of frequently-used functions
✅ Comprehensive library acceleration
```

---

## 🚀 Next Move

**Start Phase 13:** CQT/VQT acceleration (3-4 weeks)
- Expected: 2.0-3.0× speedup
- Very popular feature
- Reasonable complexity

**See:** `Development_docs/LIBROSA_RUST_COVERAGE_ROADMAP.md`

