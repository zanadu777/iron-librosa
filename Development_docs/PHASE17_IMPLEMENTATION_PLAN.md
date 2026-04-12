# Phase 17 Implementation Plan - Detailed Scope

**Session Date:** April 9, 2026  
**Current Status:** Foundation ready, GPU dispatch scaffolding in place  
**Estimated Remaining Effort:** 4–6 hours for full GPU FFT + integration + testing

---

## Current Architecture

```
src/metal_fft.metal       ← MSL GPU FFT kernels (forward + inverse, complete)
src/metal_fft.rs          ← Rust wrapper + fallback API (scaffold ready)
src/stft.rs               ← STFT dispatch (ready for GPU path addition)
src/istft.rs              ← iSTFT dispatch (ready for GPU path addition)
```

All files compile cleanly with zero regressions.

---

## Remaining Implementation Steps

### Step 1: Full Metal Compute Dispatch (2–3 hours)

**Location:** `src/metal_fft.rs`

Replace the placeholder error path with full Metal API:

**What to implement:**
1. Metal device instantiation (`MTLCreateSystemDefaultDevice`)
2. Command queue creation
3. MSL shader source loading & compilation
4. Compute pipeline state creation (forward + inverse)
5. Persistent buffer pool (grow-on-demand strategy)
6. Command encoder setup & dispatch
7. GPU-to-CPU readback

**Key decisions:**
- Use `metal` crate or raw `objc` FFI? → Recommend `objc` + `core-foundation` for lower-level control
- Buffer pooling strategy? → Shared memory with 1.5x headroom (as documented)
- Error handling? → Propagate Metal errors, but fallback to CPU always available

**Code template (pseudocode):**
```rust
fn fft_forward_dispatch(buffer: &mut [Complex<f32>], n: usize) -> Result<(), String> {
    // 1. Get/allocate GPU buffer
    // 2. Copy buffer to GPU (shared memory pointer)
    // 3. Encode compute command with pipeline
    // 4. Dispatch thread-groups
    // 5. Wait for completion
    // 6. Read back to CPU
    Ok(())
}
```

**Test command:**
```bash
cargo check  # Should still compile
cargo test --lib metal_fft  # No tests yet, but structure ready
```

### Step 2: STFT/iSTFT GPU Dispatch Integration (1–2 hours)

**Locations:** `src/stft.rs`, `src/istft.rs`

Add AppleGpu match arms to dispatch wrappers:

**What to implement:**
1. Add GPU path to `stft_power` dispatch
2. Add GPU path to `stft_complex` dispatch
3. Add GPU path to `istft_f32` dispatch
4. Add workload threshold gating (env var: `IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD`)
5. Log/debug output for dispatch decisions (optional)

**Pattern (from Phase16):**
```rust
#[pyfunction]
pub fn stft_power<'py>(...) -> PyResult<...> {
    match resolved_rust_device() {
        RustDevice::Cpu => stft_power_cpu(...),
        RustDevice::AppleGpu => {
            // Try GPU first, fallback to CPU on failure
            match stft_power_gpu(...) {
                Ok(_) => Ok(result),
                Err(_) => stft_power_cpu(...),  // Graceful fallback
            }
        }
        RustDevice::Auto => {
            if work_size >= fft_gpu_threshold() {
                match stft_power_gpu(...) {
                    Ok(_) => Ok(result),
                    Err(_) => stft_power_cpu(...),
                }
            } else {
                stft_power_cpu(...)
            }
        }
    }
}

fn stft_power_gpu<'py>(...) -> PyResult<...> {
    // New GPU path:
    // 1. Validate n_fft is power-of-2 and ≤ 2048
    // 2. For each frame, call metal_fft::fft_forward_with_fallback()
    // 3. Process results (windowing, magnitude computation)
    Ok(output)
}
```

**Test command:**
```bash
pytest tests/test_phase4_istft_and_db.py -k stft -p no:warnings  # Should still pass
```

### Step 3: Numerical Validation Tests (1–2 hours)

**New file:** `tests/test_metal_fft_dispatch.py`

**Test cases:**
1. FFT size power-of-2 validation
2. Output size verification
3. CPU vs GPU numerical parity (allow f32 tolerance)
4. Edge cases (small FFT, large FFT, boundary FFT sizes)
5. Batch correctness (multiple frames produce independent results)
6. Fallback verification (GPU failure → CPU path works)

**Example test:**
```python
def test_stft_power_gpu_matches_cpu():
    y = np.random.randn(8000).astype(np.float32)
    
    os.environ["IRON_LIBROSA_RUST_DEVICE"] = "cpu"
    out_cpu = _rust_ext.stft_power(y, n_fft=1024, hop_length=512)
    
    os.environ["IRON_LIBROSA_RUST_DEVICE"] = "apple-gpu"
    out_gpu = _rust_ext.stft_power(y, n_fft=1024, hop_length=512)
    
    # Allow f32 accumulation differences
    np.testing.assert_allclose(out_cpu, out_gpu, rtol=1e-5, atol=1e-5)
```

### Step 4: Threshold Tuning & Environment Variable (30 min)

**Location:** `src/stft.rs` (add helper function)

```rust
fn fft_gpu_work_threshold() -> usize {
    std::env::var("IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000_000)  // Default: ~100M operations
}
```

Workload estimate: `n_frames * n_fft * log2(n_fft)`

### Step 5: Benchmark Harness (1 hour)

**New file or extend:** `Benchmarks/scripts/benchmark_phase17_gpu_fft.py`

**Measurements:**
- CPU vs GPU STFT time across FFT sizes (512, 1024, 2048, 4096)
- Audio lengths (1 sec, 10 sec, 60 sec)
- Speedup ratio calculation
- JSON output for review

**Expected results:**
- Small audio (<4 sec): overhead-dominated, ~0.8–1.2x
- Medium audio (4–30 sec): breakeven to modest, ~1.0–2.0x
- Large audio (>30 sec): strong gains, ~2–5x (depending on GPU model)

### Step 6: Documentation Update (30 min)

Update: `Development_docs/PHASE17_METAL_FFT_FOUNDATION.md`

Add:
- Final implementation notes
- Performance profile (actual measured speedups)
- Tuning recommendations
- Known limitations (max 2048-point, GPU fallback behavior)

---

## Validation Checkpoints

After each step, run:

```bash
# Compilation
cargo check
cargo build --release

# Unit tests
pytest tests/test_phase4_istft_and_db.py -p no:warnings

# Device override tests (new)
pytest tests/test_metal_fft_dispatch.py -p no:warnings

# Full regression gate (optional, but recommended)
pytest -q tests/ --tb=short 2>&1 | tail -5
```

---

## Success Criteria for Phase 17 Completion

- ✅ Metal GPU FFT kernels compile and execute (Step 1)
- ✅ STFT/iSTFT dispatch routes to GPU path correctly (Step 2)
- ✅ CPU and GPU paths produce numerically identical results (Step 3)
- ✅ Workload threshold gating works (Step 4)
- ✅ Benchmark harness runs and generates reports (Step 5)
- ✅ All existing tests still pass (zero regressions)
- ✅ Documentation updated with performance data (Step 6)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Metal shader compilation fails | Use pre-compiled binary or MSL source embedding |
| GPU dispatch hangs | Add timeout/watchdog; fallback to CPU |
| Numerical divergence > tolerance | Relax f32 tolerance or adjust kernel precision |
| Performance below CPU baseline | Tune threshold, investigate cache behavior |
| Compatibility issues on older macOS | Test on multiple macOS versions; graceful degradation |

---

## Environment Variables for Tuning

```bash
# Force GPU (even for small workloads)
export IRON_LIBROSA_RUST_DEVICE=apple-gpu

# Force CPU
export IRON_LIBROSA_RUST_DEVICE=cpu

# Auto-select with custom threshold (default ~100M ops)
export IRON_LIBROSA_FFT_GPU_WORK_THRESHOLD=200000000

# Debug: See dispatch decisions
export RUST_LOG=debug  # (if logging added)
```

---

## Timeline

| Task | Est. Time | Cumulative |
|------|-----------|-----------|
| Metal dispatch impl | 2–3h | 2–3h |
| STFT/iSTFT integration | 1–2h | 3–5h |
| Tests & validation | 1–2h | 4–7h |
| Threshold tuning | 30min | 4.5–7.5h |
| Benchmarking | 1h | 5.5–8.5h |
| Documentation | 30min | 6–9h |
| **Total** | | **~6–9 hours** |

---

## Next Session Handoff

To continue Phase 17 from here:

1. Open `src/metal_fft.rs`
2. Implement `fft_forward_dispatch()` with full Metal API
3. Follow Step 1 → Step 6 checklist above
4. Use provided test patterns and benchmark template
5. Reference Phase16 GPU acceleration (mel.rs) for pattern consistency

**Key reference file:** `/Users/kenjohnson/Dev/Rust/iron-librosa/src/mel.rs` (lines ~120–250 show complete GPU pattern)

---

**Status:** Foundation ready for full implementation. No blockers identified.

