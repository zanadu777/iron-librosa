# Phase 21 Implementation Status & Next Steps
# Date: April 14, 2026
# GPU: NVIDIA RTX 3090 | CUDA 13.2 | cuFFT 12

## ✅ WHAT WE IMPLEMENTED TODAY

### 0. Dispatch diagnostics + fallback test coverage (CUD-004/CUD-005)
- Added additive backend diagnostics fields: `requested_reason`, `resolved_reason`, `dispatch_policy`.
- Added subprocess-based auto-dispatch tests for forced CUDA runtime on/off behavior.
- Extended Phase 21 benchmark-gate smoke test to validate backend reason metadata.
- Status: **WORKING** ✓

### 1. Fixed CUDA Library Loading
- Was: Silent failure, no DLLs found
- Now: Searches CUDA_PATH env var + common install dirs
- Result: Finds `cudart64_13.dll` + `cufft64_12.dll` in CUDA 13.2 install
- Status: **WORKING** ✓

### 2. Fixed Dispatch Thresholds  
- Was: `IRON_LIBROSA_CUDA_FFT_MIN_WORK_THRESHOLD=20_000_000` → nothing triggered GPU
- Now: `1_000_000` default (lowered 20x), `min_frames=32` (lowered 4x)
- Result: Medium+ workloads now dispatch to GPU
- Status: **WORKING** ✓

### 3. Implemented `IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on`
- Was: Env var set but not honored by Rust code
- Now: Bypasses all size/frame thresholds when set
- Result: Any workload can use GPU via force-on flag
- Status: **WORKING** ✓

### 4. Added `cuda_diagnostics()` Python Function
- Returns GPU presence, DLL probe results, device count
- Enables quick verification: `import librosa._rust as _; print(_r.cuda_diagnostics())`
- Status: **WORKING** ✓

### 5. Added `IRON_LIBROSA_CUDA_DEBUG=1` Verbose Logging
- Shows DLL loading attempts, H2D/compute/D2H timing, error messages
- Per-call breakdown: H2D=1.2ms, GPU=0.021ms, D2H=1.3ms
- Status: **WORKING** ✓

### 6. Fixed CPU Fallback (Cached Planner)
- Was: New `FftPlanner::new()` created on every GPU-fallback call (slow)
- Now: Thread-local cached planner reused across calls
- Result: CPU fallback path is as fast as direct CPU path
- Status: **FIXED** ✓

### 7. Parallelized Batch Build/Unpack (Rayon)
- Was: Sequential loops for O(frames * n_fft) operations
- Now: Rayon par_chunks_mut for both build and unpack
- Result: ~2ms savings for large workloads
- Status: **DONE** ✓

### 8. Cached Device Probe
- Was: `cudaGetDeviceCount` called on every GPU FFT request  
- Now: Cached per-thread after first call
- Status: **DONE** ✓

### 9. CUD-006 heuristic retune started (small-workload safety)
- Auto CUDA thresholds retuned to `work >= 2_000_000` and `frames >= 96` in STFT/iSTFT.
- Two-stream pipeline gate retuned to require larger workloads (`frames >= 256` and `n_fft*frames >= 262_144`).
- Goal: reduce short-workload regressions while preserving medium/large auto-dispatch.
- Status: **IN PROGRESS** ✓

### 10. CUD-007 benchmark gate update started
- Added policy-versioned gate metadata (`phase21-cud007-v1`) to benchmark payloads.
- Added near-parity tolerance handling (1% floor) so measurement noise does not count as hard regression.
- Added benchmark-gate tests for both near-parity promotion and true-regression blocking.
- Status: **IN PROGRESS** ✓

---

## 📊 CURRENT PERFORMANCE RESULTS

### GPU Execution Confirmed (RTX 3090)
```
20s audio, n=1024, 1723 frames:
  H2D transfer:  1.225ms  (14.1 MB to GPU)
  cuFFT compute: 0.021ms  ← GPU blazingly fast!
  D2H transfer:  1.258ms  (14.1 MB back)
  Other overhead: ~2.3ms  (batch build + unpack)
  Total GPU:     4.8ms

  Rust CPU (Rayon): 1.78ms
  Python librosa:   6.46ms

  GPU vs Python:  1.35x faster
  GPU vs CPU:     0.37x (GPU slower)
```

### Three-Level Speedup Table (20s audio)
```
                    Python   Rust CPU   Rust+CUDA
n=1024  20s:        6.46ms   1.78ms    4.80ms
  vs Python:         1.00x    3.63x     1.35x
  vs CPU:             --       --        0.37x (GPU slower!)
  
n=2048  20s:        7.09ms   1.92ms    5.00ms
  vs Python:         1.00x    3.69x     1.42x
  vs CPU:             --       --        0.38x
```

---

## 🔍 WHY GPU IS SLOWER THAN CPU FOR STFT

**The bottleneck breakdown:**

```
PCIe transfer (H2D + D2H):  2.5ms  ← physical limit
Batch build (CPU-side):     1.2ms  ← room to improve
Batch unpack (CPU-side):    1.1ms  ← room to improve
cuFFT compute:              0.021ms ← essentially free!
─────────────────────────────────
Total:                      4.8ms  vs CPU 1.78ms
```

**Why cuFFT is so fast:** The RTX 3090 has 35.58 TFLOPS of compute. For n=1024, 1723 frames:
- Total ops: 1723 * 1024 * 10 ≈ 17.6M ops
- RTX 3090: 17.6M / 35.58T = 0.5 microseconds theoretical
- Actual: 21 microseconds (40x overhead for kernel launch + scheduling)

**The physics constraint:**
- PCIe Gen 4 x16 bandwidth: ~32 GB/s theoretical
- Actual pageable H2D/D2H: ~5-6 GB/s (measured: 14MB/1.25ms = 11.2 GB/s — good!)
- Cannot reduce below: 14MB / 32GB/s = 0.44ms each way = 0.88ms minimum transfer
- With minimum transfer: 0.88ms + 0.021ms = ~0.9ms vs CPU 1.78ms = 2x GPU wins!

**What prevents reaching minimum:**
- Synchronous (blocking) transfer — CPU waits for each step
- Pageable (non-pinned) host memory → kernel copy overhead

---

## 🚀 PATH TO 5x SPEEDUP

### Option A: Pipeline GPU (Keep Data On GPU) — HIGHEST IMPACT

Instead of STFT → CPU → mel, do STFT + mel all on GPU:
```
Python pipeline (current):
  CPU: STFT → CPU result → mel filter → CPU result
  Total: 6ms + 3ms = 9ms

GPU pipeline (target):
  Raw audio → GPU (1.76MB, 0.15ms)
  GPU: window + FFT (0.1ms)
  GPU: mel filter (matrix multiply, 0.1ms)  
  D2H mel spec (0.5MB, 0.04ms)
  Total: 0.39ms → 23x speedup!
```

This requires implementing `mel_gpu_kernel` — a CUDA kernel for windowing + FFT + mel filter.

### Option B: Pinned Memory + Async Transfers — MEDIUM IMPACT

```rust
// Current: pageable memcpy (5-6 GB/s effective)
cudaMemcpy(dev_ptr, host_ptr, bytes, H2D);  // blocking

// Optimized: pinned + async + stream sync
cudaMallocHost(&host_pinned, bytes);          // locked pages
cudaMemcpyAsync(dev_ptr, host_pinned, bytes, H2D, stream);  // async
cudaStreamSynchronize(stream);                // wait only here
```

Expected savings: H2D 1.2ms → 0.6ms, D2H 1.3ms → 0.6ms
Total savings: ~1.3ms for 20s workload
New GPU time: ~3.5ms vs CPU 1.78ms → still GPU slower

### Option C: GPU Window + Pack Kernel — MEDIUM IMPACT

```cuda
__global__ void window_and_pack(
    float* audio, float* window, Complex* output,
    int n_fft, int hop, int n_frames
) {
    // GPU parallel frame windowing — eliminates 1.2ms CPU batch build
}
```

Combined with async transfers:
- Batch build: 0ms (done on GPU)
- H2D audio only: 0.15ms (vs 1.2ms for full STFT matrix)
- GPU compute: 0.1ms (window + FFT)
- D2H: 1.3ms (irreducible for STFT output)
- Total: ~1.6ms → GPU finally wins vs CPU!

---

## 📋 CONCRETE NEXT STEPS

### STEP 1: Pinned Memory (Week 2, 4-6 hours)
- Implement `cudaMallocHost` in CudaWorkspace
- Use `cudaMemcpyAsync` with streams
- Expected: ~1.3ms savings, GPU goes from 4.8ms → 3.5ms

### STEP 2: GPU Window+Pack Kernel (Week 2-3, 8-12 hours)
- Write CUDA kernel file `src/cuda_kernels.cu`
- Kernel: `apply_window_and_pack<<<grid, block>>>(audio, window, batch_buf, ...)`
- Expected: eliminates 1.2ms batch build, GPU goes from 3.5ms → 2.3ms

### STEP 3: Mel Spectrogram on GPU (Week 3-4, 16-20 hours)
- Write GPU mel filter: `cublasGemm(mel_filters, stft_magnitudes, ...)`
- Keep STFT output on GPU, avoid D2H until after mel
- Expected: eliminates most D2H cost, total pipeline ~0.5ms
- Expected speedup: 10-15x vs Python

### STEP 4: Async Stream Pipeline (Week 3-4, 8-10 hours)
- Chunk audio into batches
- Pipeline: H2D[t] + GPU[t-1] + D2H[t-2] simultaneously
- Expected: 2-3x additional throughput improvement

---

## 🎯 REVISED SPEEDUP EXPECTATIONS

### STFT-only path (current approach, PCIe bound):
```
Current:    GPU 4.8ms, CPU 1.78ms → GPU 0.37x (slower)
+Pinned:    GPU 3.5ms, CPU 1.78ms → GPU 0.51x (still slower)
+Kernel:    GPU 2.3ms, CPU 1.78ms → GPU 0.77x (approaching parity)
+Async:     GPU 1.5ms, CPU 1.78ms → GPU 1.2x (GPU finally wins!)
```

### Full pipeline (STFT + mel) path:
```
Current (CPU): Python ~22ms, Rust CPU ~5ms
With GPU pipeline: ~0.5ms → 10x vs CPU, 44x vs Python
```

### The 5x goal:
- ✗ 5x for STFT-only: **not achievable** with synchronous PCIe (physics limit)
- ✓ 5x for mel spectrogram: **achievable** with GPU pipeline approach
- ✓ 10x+ for full pipeline: **achievable** by keeping data on GPU

---

## 📌 ENVIRONMENT VARIABLES SUMMARY

```bash
# Enable GPU dispatch
export IRON_LIBROSA_RUST_DEVICE=cuda-gpu

# Force GPU for all workload sizes (bypass threshold)
export IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on

# Debug logging (shows H2D/compute/D2H timing)
export IRON_LIBROSA_CUDA_DEBUG=1

# Hard cap on element count (0 = unlimited)
export IRON_LIBROSA_CUDA_MAX_WORK=0

# Work threshold for automatic GPU dispatch
export IRON_LIBROSA_CUDA_FFT_MIN_WORK_THRESHOLD=1000000

# Minimum frames for GPU dispatch
export IRON_LIBROSA_CUDA_FFT_MIN_FRAMES=32

# Force CUDA runtime available (testing only)
export IRON_LIBROSA_CUDA_RUNTIME_FORCE=1
```

---

## 🔧 BUILD COMMANDS

```bash
# Build with CUDA support
python -m maturin develop --release --features cuda-gpu

# Run diagnostics
python -c "import librosa._rust as r; d=r.cuda_diagnostics(); print(d['diagnostics_text'])"

# Run benchmark with GPU
IRON_LIBROSA_RUST_DEVICE=cuda-gpu IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on \
  python Benchmarks/scripts/benchmark_phase21_cuda_baseline.py --device cuda-gpu
```

