# Next Actions: From Phase 12 to Full Rust Coverage

**Current Date:** April 4, 2026  
**Phase:** 12 (wrapping up)  
**Next Phase:** 13 (CQT/VQT)

---

## 🎯 IMMEDIATE ACTIONS (This Week)

### 1. **Complete Phase 12 Documentation**
- [ ] Finalize Mel cross-CPU threshold strategy document
- [ ] Document CQT/VQT planning decisions
- [ ] Write Tonnetz parity policy
- [ ] Update PHASE12_CPU_REMAINING_PLAN.md with final status

**Owner:** Architecture team  
**Time:** 1-2 days

### 2. **Phase 13 Planning Meeting**
- [ ] Decide: CQT/VQT as Phase 13 target ✅ (recommended)
- [ ] Allocate resources (1 dev, 3-4 weeks)
- [ ] Setup benchmarking for CQT baseline
- [ ] Assign spike tasks (research phase)

**Owner:** Project manager + tech lead  
**Time:** 2-4 hours

### 3. **Spike: CQT Algorithm Research**
- [ ] Analyze non-uniform FFT structure
- [ ] Review current librosa.cqt implementation
- [ ] Identify bottlenecks vs librosa.stft
- [ ] Design Rust kernel architecture
- [ ] Prototype f32 & f64 variants

**Owner:** Lead Rust developer  
**Time:** 3-5 days

**Deliverable:** `Development_docs/PHASE13_CQT_VQT_SPIKE.md`

---

## 📅 PHASE 13 PLAN (Starting Next Week)

### **Week 1-2: Implementation**
- [ ] Implement Rust CQT kernel (f32 & f64)
- [ ] Add variable-Q support
- [ ] Wire dispatch in Python
- [ ] Create parity test harness

**Parallel:** Benchmark setup, baseline collection

### **Week 3: Testing & Validation**
- [ ] Run parity tests (aim for < 1e-5 f32, < 1e-11 f64)
- [ ] Benchmark: Python vs Rust
- [ ] Performance regression testing
- [ ] Multichannel validation

**Goal:** All tests green, > 2.0× speedup confirmed

### **Week 4: Polish & Promotion**
- [ ] Remove any xfail markers
- [ ] Update documentation
- [ ] Create release notes
- [ ] Promote to default (prefer_rust=True)
- [ ] Merge to main

**Deliverable:** CQT/VQT Rust acceleration in production

---

## 🚀 PHASE 14 PLAN (After Phase 13)

### **Beat Tracking Acceleration (2-3 weeks)**

**Key Functions to Accelerate:**
- `librosa.beat.beat_track()` (onset-based detection)
- `librosa.beat.estimate_tempo()` (autocorrelation)
- Dynamic programming for beat grid alignment

**Speedup Target:** 1.5-2.0×

**Dependencies:**
- ✅ Onset detection (already Rust)
- ⏳ Autocorrelation optimization

---

## 📊 Coverage Progression

```
TODAY (Phase 12 complete):
  70% Coverage ..................... 80+ Rust kernels
  
AFTER PHASE 13 (CQT/VQT):
  80% Coverage ..................... +CQT/VQT for pitch
  Expected: 2.0-3.0× speedup
  
AFTER PHASE 14 (Beat Tracking):
  85% Coverage ..................... +Beat/Tempo analysis
  Expected: 1.5-2.0× speedup
  
AFTER PHASE 15 (Segment/Effects):
  90% Coverage ..................... +Effects module
  Expected: 1.5-4.0× speedup
  
AFTER PHASE 16+ (Remaining):
  95%+ Coverage .................... Comprehensive
  Expected: 1.3-2.0× speedup
```

---

## 📋 TODO Before Phase 13 Starts

### Code Quality
- [ ] All Phase 12 tests passing
- [ ] Phase vocoder thoroughly benchmarked
- [ ] Mel threshold strategy documented
- [ ] Tonnetz parity policy finalized

### Documentation
- [ ] Update PHASE12_CPU_REMAINING_PLAN.md
- [ ] Create PHASE13_CQT_VQT_SPIKE.md
- [ ] Update LIBROSA_RUST_COVERAGE_ROADMAP.md
- [ ] Brief team on Phase 13 scope

### Infrastructure
- [ ] CQT benchmark baseline collected
- [ ] Rust development environment validated
- [ ] Test harness prepared
- [ ] CI/CD ready for Phase 13

### Planning
- [ ] Resource allocation confirmed
- [ ] Timeline agreed upon
- [ ] Success criteria defined
- [ ] Stakeholder alignment

---

## 🎓 Key Decisions Made

| Decision | Status | Rationale |
|----------|--------|-----------|
| Phase Vocoder → Production | ✅ DONE | Parity validated, 1.57× speedup |
| CQT/VQT → Phase 13 | ✅ RECOMMENDED | Very popular, 2.0-3.0× speedup |
| Beat Tracking → Phase 14 | ✅ RECOMMENDED | Most-used, 1.5-2.0× speedup |
| NMF → External (scikit-learn) | ✅ DECIDED | Specialized library, defer |
| Full Coverage Goal → 95% | ✅ TARGET | Not 100% (niche features low ROI) |

---

## 📞 Escalation Points

**If CQT/VQT complexity higher than expected:**
- Option A: Extend timeline 1-2 weeks
- Option B: Reduce scope (only CQT, defer VQT)
- Option C: Push to Phase 14, start Beat Tracking first

**If Mel threshold strategy unresolved:**
- Decision deadline: End of Phase 12
- Fallback: Use current heuristics, optimize later
- Don't block CQT/VQT work

---

## ✅ Success Criteria for Phase 13

- [ ] CQT kernel implements full algorithm (non-uniform FFT)
- [ ] Parity tests pass (< 1e-5 for f32, < 1e-11 for f64)
- [ ] Speedup > 2.0× on medium workloads
- [ ] Multichannel support working
- [ ] All tests green (no xfails)
- [ ] Documentation complete
- [ ] Promotion to default ready
- [ ] Performance benchmarks published

---

## 📅 Milestone Calendar

| Date | Milestone | Status |
|------|-----------|--------|
| Apr 4 | Phase 12 wrap-up begins | 🔄 NOW |
| Apr 7-8 | Phase 13 planning meeting | 📅 NEXT |
| Apr 8-12 | CQT research spike | 📅 NEXT |
| Apr 15 | Phase 13 development starts | 📅 PLANNED |
| May 1 | Phase 13 completion target | 📅 PLANNED |
| May 8 | Phase 14 starts (Beat Tracking) | 📅 PLANNED |

---

## 🎯 Resources Needed

### Personnel
- 1 senior Rust developer (3-4 weeks for Phase 13)
- 1 test/QA engineer (parity validation, benchmarking)
- 1 project manager (timeline, coordination)

### Infrastructure
- Benchmark machine (consistent environment)
- CI/CD pipeline (automated testing)
- Documentation wiki/system

### Time
- Phase 13: 3-4 weeks (full-time, one dev)
- Phase 14: 2-3 weeks
- Phases 15-16: 6-10 weeks total

---

## 🎉 Vision: Full Rust Coverage

By end of 2026 (Q4):
- ✅ CQT/VQT in production (Phase 13)
- ✅ Beat tracking in production (Phase 14)
- ✅ Effects & segment in production (Phases 15)
- ✅ 95%+ of librosa functions using Rust
- ✅ ~2-3× average speedup across library
- ✅ Comprehensive documentation
- ✅ Full backward compatibility

**Status:** Ready to move forward! 🚀

---

**Prepared by:** iron-librosa development team  
**Date:** April 4, 2026  
**Next Review:** Phase 13 kickoff meeting

