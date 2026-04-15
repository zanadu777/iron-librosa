// Phase 21 â€” CUDA FFT wrapper for STFT/iSTFT.
//
// Mirrors the public API of src/metal_fft.rs so stft.rs / istft.rs can dispatch
// with identical call sites.
//
// ENVIRONMENT VARIABLES
//   IRON_LIBROSA_RUST_DEVICE=cuda-gpu        enable CUDA dispatch (opt-in)
//   IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL=force-on   bypass work-size gates
//   IRON_LIBROSA_CUDA_MAX_WORK=<n>           hard upper cap on element count
//   IRON_LIBROSA_CUDA_FFT_BATCH_CHUNK_SIZE=<n>  chunk size for batched ops
//   IRON_LIBROSA_CUDA_RUNTIME_FORCE=1        pretend CUDA is available (test)
//   IRON_LIBROSA_CUDA_DEBUG=1                verbose per-call logging to stderr
//   IRON_LIBROSA_CUDA_USE_PINNED_STAGING=0   disable pinned host staging for diagnostics
//
// FEATURE GATE
//   Cargo.toml: cuda-gpu = []
//   Build:      maturin develop --release --features cuda-gpu

use rustfft::num_complex::Complex;

// â”€â”€ Debug logging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn cuda_debug_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        matches!(
            std::env::var("IRON_LIBROSA_CUDA_DEBUG")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

#[inline]
fn cdbg(msg: &str) {
    if cuda_debug_enabled() {
        eprintln!("[CUDA] {}", msg);
    }
}

fn cuda_window_pack_helper_path_env() -> Option<&'static str> {
    option_env!("IRON_LIBROSA_CUDA_WINDOW_PACK_HELPER")
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

pub fn cuda_window_pack_helper_built() -> bool {
    cfg!(has_cuda_window_pack_kernel) && cuda_window_pack_helper_path_env().is_some()
}

pub fn cuda_window_pack_helper_path() -> Option<&'static str> {
    cuda_window_pack_helper_path_env()
}

pub fn cuda_fused_mel_helper_built() -> bool {
    cuda_window_pack_helper_built()
}

pub fn cuda_fused_mel_helper_path() -> Option<&'static str> {
    cuda_window_pack_helper_path()
}

#[inline]
fn c2c_required_headroom(bytes: usize) -> usize {
    bytes.saturating_mul(2)
}

#[inline]
fn split_io_required_headroom(in_bytes: usize, out_bytes: usize) -> usize {
    in_bytes
        .saturating_add(out_bytes)
        .saturating_mul(2)
}

#[inline]
fn window_pack_required_headroom(
    signal_bytes: usize,
    window_bytes: usize,
    fft_in_bytes: usize,
    out_bytes: usize,
) -> usize {
    signal_bytes
        .saturating_add(window_bytes)
        .saturating_add(fft_in_bytes)
        .saturating_add(out_bytes)
        .saturating_mul(2)
}

#[inline]
fn stft_mel_required_headroom(
    signal_bytes: usize,
    window_bytes: usize,
    fft_in_bytes: usize,
    fft_out_bytes: usize,
    mel_basis_bytes: usize,
    mel_out_bytes: usize,
) -> usize {
    signal_bytes
        .saturating_add(window_bytes)
        .saturating_add(fft_in_bytes)
        .saturating_add(fft_out_bytes)
        .saturating_add(mel_basis_bytes)
        .saturating_add(mel_out_bytes)
        .saturating_mul(2)
}

#[inline]
unsafe fn copy_bytes_to_staging(src: *const std::ffi::c_void, dst: *mut std::ffi::c_void, bytes: usize) {
    std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes);
}

#[inline]
unsafe fn copy_bytes_from_staging(src: *const std::ffi::c_void, dst: *mut std::ffi::c_void, bytes: usize) {
    std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, bytes);
}

// â”€â”€ Feature-gated GPU implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(feature = "cuda-gpu")]
pub mod cuda_fft_impl {
    use std::cell::RefCell;
    use std::collections::hash_map::DefaultHasher;
    use std::ffi::c_void;
    use std::hash::Hasher;
    use std::mem;
    use std::ptr;
    use std::sync::Arc;

    use libloading::Library;
    use rustfft::num_complex::Complex;

    type CudaError = i32;
    type CufftResult = i32;
    type CufftHandle = i32;

    const CUDA_SUCCESS: CudaError = 0;
    const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
    const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;

    const CUFFT_SUCCESS: CufftResult = 0;
    const CUFFT_C2C: i32 = 0x29;
    const CUFFT_R2C: i32 = 0x2a;
    const CUFFT_C2R: i32 = 0x2c;
    const CUFFT_FORWARD: i32 = -1;
    const CUFFT_INVERSE: i32 = 1;

    type CudaGetDeviceCountFn = unsafe extern "system" fn(*mut i32) -> CudaError;
    type CudaMallocFn = unsafe extern "system" fn(*mut *mut c_void, usize) -> CudaError;
    type CudaFreeFn = unsafe extern "system" fn(*mut c_void) -> CudaError;
    type CudaMallocHostFn = unsafe extern "system" fn(*mut *mut c_void, usize) -> CudaError;
    type CudaFreeHostFn = unsafe extern "system" fn(*mut c_void) -> CudaError;
    type CudaMemGetInfoFn = unsafe extern "system" fn(*mut usize, *mut usize) -> CudaError;
    type CudaStreamCreateFn = unsafe extern "system" fn(*mut *mut c_void) -> CudaError;
    type CudaStreamDestroyFn = unsafe extern "system" fn(*mut c_void) -> CudaError;
    type CudaMemcpyAsyncFn = unsafe extern "system" fn(
        *mut c_void,
        *const c_void,
        usize,
        i32,
        *mut c_void,
    ) -> CudaError;
    type CudaStreamSynchronizeFn = unsafe extern "system" fn(*mut c_void) -> CudaError;

    type CufftPlanManyFn = unsafe extern "C" fn(
        *mut CufftHandle,
        i32,
        *const i32,
        *const i32,
        i32,
        i32,
        *const i32,
        i32,
        i32,
        i32,
        i32,
    ) -> CufftResult;
    type CufftExecC2CFn =
        unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void, i32) -> CufftResult;
    type CufftExecR2CFn =
        unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void) -> CufftResult;
    type CufftExecC2RFn =
        unsafe extern "C" fn(CufftHandle, *mut c_void, *mut c_void) -> CufftResult;
    type CufftDestroyFn = unsafe extern "C" fn(CufftHandle) -> CufftResult;
    type LaunchWindowAndPackF32Fn = unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        *mut c_void,
    ) -> CudaError;
    type LaunchWindowAndPackBatchF32Fn = unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        i32,
        *mut c_void,
    ) -> CudaError;
    type LaunchMelProjectPowerF32Fn = unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        *mut c_void,
    ) -> CudaError;
    type LaunchMelProjectPowerBatchF32Fn = unsafe extern "C" fn(
        *const c_void,
        *const c_void,
        *mut c_void,
        i32,
        i32,
        i32,
        i32,
        *mut c_void,
    ) -> CudaError;

    #[derive(Clone)]
    struct CudaApi {
        _lib: Arc<Library>,
        cuda_get_device_count: CudaGetDeviceCountFn,
        cuda_malloc: CudaMallocFn,
        cuda_free: CudaFreeFn,
        cuda_malloc_host: CudaMallocHostFn,
        cuda_free_host: CudaFreeHostFn,
        cuda_mem_get_info: CudaMemGetInfoFn,
        cuda_stream_create: CudaStreamCreateFn,
        cuda_stream_destroy: CudaStreamDestroyFn,
        cuda_memcpy_async: CudaMemcpyAsyncFn,
        cuda_stream_synchronize: CudaStreamSynchronizeFn,
    }

    #[derive(Clone)]
    struct CufftApi {
        _lib: Arc<Library>,
        cufft_plan_many: CufftPlanManyFn,
        cufft_exec_c2c: CufftExecC2CFn,
        cufft_exec_r2c: CufftExecR2CFn,
        cufft_exec_c2r: CufftExecC2RFn,
        cufft_destroy: CufftDestroyFn,
        cufft_set_stream:
            unsafe extern "C" fn(CufftHandle, *mut c_void) -> CufftResult,
    }

    #[derive(Clone)]
    struct CudaWindowPackApi {
        _lib: Arc<Library>,
        launch_window_and_pack_f32: LaunchWindowAndPackF32Fn,
        launch_window_and_pack_batch_f32: LaunchWindowAndPackBatchF32Fn,
        launch_mel_project_power_f32: LaunchMelProjectPowerF32Fn,
        launch_mel_project_power_batch_f32: LaunchMelProjectPowerBatchF32Fn,
    }

    thread_local! {
        static TL_CUDA_API:  RefCell<Option<Result<CudaApi,  String>>> = const { RefCell::new(None) };
        static TL_CUFFT_API: RefCell<Option<Result<CufftApi, String>>> = const { RefCell::new(None) };
        static TL_CUDA_WINDOW_PACK_API: RefCell<Option<Result<Option<CudaWindowPackApi>, String>>> = const { RefCell::new(None) };
        static TL_CUDA_WS:   RefCell<Option<CudaWorkspace>>          = const { RefCell::new(None) };
    }

    #[inline]
    fn fingerprint_f32_slice(data: &[f32]) -> (usize, u64) {
        let bytes = data.len().saturating_mul(mem::size_of::<f32>());
        if bytes == 0 {
            return (0, 0);
        }
        let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };

        // For smaller payloads, keep full hashing for exactness.
        // For larger payloads, sampled fingerprinting avoids hashing MBs per call.
        if bytes <= 64 * 1024 {
            let mut hasher = DefaultHasher::new();
            hasher.write(raw);
            return (bytes, hasher.finish());
        }

        // Lightweight sampled fingerprint (FNV-1a style mixing).
        const FNV_OFFSET: u64 = 0xcbf29ce484222325;
        const FNV_PRIME: u64 = 0x100000001b3;
        let mut h = FNV_OFFSET;

        #[inline]
        fn mix_u64(mut h: u64, v: u64) -> u64 {
            h ^= v;
            h.wrapping_mul(FNV_PRIME)
        }

        h = mix_u64(h, bytes as u64);
        let len = data.len();
        h = mix_u64(h, data[0].to_bits() as u64);
        h = mix_u64(h, data[len / 2].to_bits() as u64);
        h = mix_u64(h, data[len - 1].to_bits() as u64);

        // Spread byte-level probes across the payload.
        const N_SAMPLES: usize = 16;
        for i in 0..N_SAMPLES {
            let pos = i.saturating_mul(bytes.saturating_sub(1)) / (N_SAMPLES - 1);
            h = mix_u64(h, raw[pos] as u64);
        }

        (bytes, h)
    }

    #[inline]
    fn quick_sig_f32_slice(data: &[f32]) -> (usize, u64) {
        let bytes = data.len().saturating_mul(mem::size_of::<f32>());
        if data.is_empty() {
            return (bytes, 0);
        }
        let len = data.len();
        let first = data[0].to_bits() as u64;
        let mid = data[len / 2].to_bits() as u64;
        let last = data[len - 1].to_bits() as u64;
        let sig = bytes as u64 ^ first.rotate_left(13) ^ mid.rotate_left(29) ^ last.rotate_left(47);
        (bytes, sig)
    }

    #[inline]
    fn fingerprint_f32_slice_with_identity_cache(
        data: &[f32],
        last_ptr: &mut usize,
        last_bytes: &mut usize,
        last_sig: &mut u64,
        last_hash: &mut u64,
    ) -> (usize, u64) {
        let (bytes, sig) = quick_sig_f32_slice(data);
        let ptr = data.as_ptr() as usize;
        if *last_ptr == ptr && *last_bytes == bytes && *last_sig == sig {
            return (bytes, *last_hash);
        }
        let (_, hash) = fingerprint_f32_slice(data);
        *last_ptr = ptr;
        *last_bytes = bytes;
        *last_sig = sig;
        *last_hash = hash;
        (bytes, hash)
    }

    struct CudaWorkspace {
        cuda: CudaApi,
        cufft: CufftApi,
        dev_ptr: *mut c_void,
        dev_r2c_in_ptr: *mut c_void,
        dev_r2c_out_ptr: *mut c_void,
        dev_signal_ptr: *mut c_void,
        dev_window_ptr: *mut c_void,
        dev_mel_basis_ptr: *mut c_void,
        dev_mel_out_ptr: *mut c_void,
        dev_c2r_in_ptr: *mut c_void,
        dev_c2r_out_ptr: *mut c_void,
        host_ptr: *mut c_void,
        host_input_ptr: *mut c_void,
        host_output_ptr: *mut c_void,
        host_input_ptr_b: *mut c_void,
        host_output_ptr_b: *mut c_void,
        stream: *mut c_void,
        capacity_bytes: usize,
        r2c_in_capacity_bytes: usize,
        r2c_out_capacity_bytes: usize,
        signal_capacity_bytes: usize,
        signal_resident_bytes: usize,
        signal_resident_hash: u64,
        signal_last_ptr: usize,
        signal_last_bytes: usize,
        signal_last_sig: u64,
        signal_last_hash: u64,
        window_capacity_bytes: usize,
        window_resident_bytes: usize,
        window_resident_hash: u64,
        window_last_ptr: usize,
        window_last_bytes: usize,
        window_last_sig: u64,
        window_last_hash: u64,
        mel_basis_capacity_bytes: usize,
        mel_basis_resident_bytes: usize,
        mel_basis_resident_hash: u64,
        mel_basis_last_ptr: usize,
        mel_basis_last_bytes: usize,
        mel_basis_last_sig: u64,
        mel_basis_last_hash: u64,
        mel_out_capacity_bytes: usize,
        c2r_in_capacity_bytes: usize,
        c2r_out_capacity_bytes: usize,
        host_capacity_bytes: usize,
        host_io_capacity_bytes: usize,
        host_io_capacity_b_bytes: usize,
        plan: CufftHandle,
        plan_r2c: CufftHandle,
        plan_c2r: CufftHandle,
        plan_n: usize,
        plan_batch: usize,
        plan_r2c_n: usize,
        plan_r2c_batch: usize,
        plan_c2r_n: usize,
        plan_c2r_batch: usize,
        stft_cache_valid: bool,
        stft_cache_signal_bytes: usize,
        stft_cache_window_bytes: usize,
        stft_cache_signal_hash: u64,
        stft_cache_window_hash: u64,
        stft_cache_n: usize,
        stft_cache_hop: usize,
        stft_cache_frames: usize,
        stft_cache_channels: usize,
        stft_cache_samples_per_channel: usize,
        // ── Two-stream pipeline ──────────────────────────────────────
        /// Second CUDA stream (stream B); stream A == `stream` above.
        stream_b: *mut c_void,
        /// Second device buffer for ping-pong between stream A/B.
        dev_ptr_b: *mut c_void,
        capacity_b_bytes: usize,
        /// cuFFT C2C plan bound to stream_b (batch == chunk_size).
        plan_b: CufftHandle,
        plan_b_n: usize,
        plan_b_batch: usize,
    }

    impl CudaWorkspace {
        fn new(cuda: CudaApi, cufft: CufftApi) -> Self {
            Self {
                cuda,
                cufft,
                dev_ptr: ptr::null_mut(),
                dev_r2c_in_ptr: ptr::null_mut(),
                dev_r2c_out_ptr: ptr::null_mut(),
                dev_signal_ptr: ptr::null_mut(),
                dev_window_ptr: ptr::null_mut(),
                dev_mel_basis_ptr: ptr::null_mut(),
                dev_mel_out_ptr: ptr::null_mut(),
                dev_c2r_in_ptr: ptr::null_mut(),
                dev_c2r_out_ptr: ptr::null_mut(),
                host_ptr: ptr::null_mut(),
                host_input_ptr: ptr::null_mut(),
                host_output_ptr: ptr::null_mut(),
                host_input_ptr_b: ptr::null_mut(),
                host_output_ptr_b: ptr::null_mut(),
                stream: ptr::null_mut(),
                capacity_bytes: 0,
                r2c_in_capacity_bytes: 0,
                r2c_out_capacity_bytes: 0,
                signal_capacity_bytes: 0,
                signal_resident_bytes: 0,
                signal_resident_hash: 0,
                signal_last_ptr: 0,
                signal_last_bytes: 0,
                signal_last_sig: 0,
                signal_last_hash: 0,
                window_capacity_bytes: 0,
                window_resident_bytes: 0,
                window_resident_hash: 0,
                window_last_ptr: 0,
                window_last_bytes: 0,
                window_last_sig: 0,
                window_last_hash: 0,
                mel_basis_capacity_bytes: 0,
                mel_basis_resident_bytes: 0,
                mel_basis_resident_hash: 0,
                mel_basis_last_ptr: 0,
                mel_basis_last_bytes: 0,
                mel_basis_last_sig: 0,
                mel_basis_last_hash: 0,
                mel_out_capacity_bytes: 0,
                c2r_in_capacity_bytes: 0,
                c2r_out_capacity_bytes: 0,
                host_capacity_bytes: 0,
                host_io_capacity_bytes: 0,
                host_io_capacity_b_bytes: 0,
                plan: 0,
                plan_r2c: 0,
                plan_c2r: 0,
                plan_n: 0,
                plan_batch: 0,
                plan_r2c_n: 0,
                plan_r2c_batch: 0,
                plan_c2r_n: 0,
                plan_c2r_batch: 0,
                stft_cache_valid: false,
                stft_cache_signal_bytes: 0,
                stft_cache_window_bytes: 0,
                stft_cache_signal_hash: 0,
                stft_cache_window_hash: 0,
                stft_cache_n: 0,
                stft_cache_hop: 0,
                stft_cache_frames: 0,
                stft_cache_channels: 0,
                stft_cache_samples_per_channel: 0,
                // two-stream pipeline
                stream_b: ptr::null_mut(),
                dev_ptr_b: ptr::null_mut(),
                capacity_b_bytes: 0,
                plan_b: 0,
                plan_b_n: 0,
                plan_b_batch: 0,
            }
        }

        fn signal_fingerprint(&mut self, signal: &[f32]) -> (usize, u64) {
            fingerprint_f32_slice_with_identity_cache(
                signal,
                &mut self.signal_last_ptr,
                &mut self.signal_last_bytes,
                &mut self.signal_last_sig,
                &mut self.signal_last_hash,
            )
        }

        fn window_fingerprint(&mut self, window: &[f32]) -> (usize, u64) {
            fingerprint_f32_slice_with_identity_cache(
                window,
                &mut self.window_last_ptr,
                &mut self.window_last_bytes,
                &mut self.window_last_sig,
                &mut self.window_last_hash,
            )
        }

        fn mel_basis_fingerprint(&mut self, mel_basis: &[f32]) -> (usize, u64) {
            fingerprint_f32_slice_with_identity_cache(
                mel_basis,
                &mut self.mel_basis_last_ptr,
                &mut self.mel_basis_last_bytes,
                &mut self.mel_basis_last_sig,
                &mut self.mel_basis_last_hash,
            )
        }

        fn ensure_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.capacity_bytes >= bytes && !self.dev_ptr.is_null() {
                return Ok(());
            }
            if !self.dev_ptr.is_null() {
                unsafe {
                    check_cuda((self.cuda.cuda_free)(self.dev_ptr), "cudaFree(realloc)")?;
                }
                self.dev_ptr = ptr::null_mut();
                self.capacity_bytes = 0;
            }
            super::cdbg(&format!("cudaMalloc {} bytes", bytes));
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc)(
                        &mut self.dev_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMalloc",
                )?;
            }
            self.capacity_bytes = bytes;
            Ok(())
        }

        fn ensure_r2c_capacity(&mut self, in_bytes: usize, out_bytes: usize) -> Result<(), String> {
            if self.r2c_in_capacity_bytes < in_bytes || self.dev_r2c_in_ptr.is_null() {
                if !self.dev_r2c_in_ptr.is_null() {
                    unsafe {
                        check_cuda((self.cuda.cuda_free)(self.dev_r2c_in_ptr), "cudaFree(r2c in realloc)")?;
                    }
                    self.dev_r2c_in_ptr = ptr::null_mut();
                    self.r2c_in_capacity_bytes = 0;
                }
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_malloc)(&mut self.dev_r2c_in_ptr as *mut *mut c_void, in_bytes),
                        "cudaMalloc(r2c in)",
                    )?;
                }
                self.r2c_in_capacity_bytes = in_bytes;
                self.stft_cache_valid = false;
            }

            if self.r2c_out_capacity_bytes < out_bytes || self.dev_r2c_out_ptr.is_null() {
                if !self.dev_r2c_out_ptr.is_null() {
                    unsafe {
                        check_cuda((self.cuda.cuda_free)(self.dev_r2c_out_ptr), "cudaFree(r2c out realloc)")?;
                    }
                    self.dev_r2c_out_ptr = ptr::null_mut();
                    self.r2c_out_capacity_bytes = 0;
                }
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_malloc)(
                            &mut self.dev_r2c_out_ptr as *mut *mut c_void,
                            out_bytes,
                        ),
                        "cudaMalloc(r2c out)",
                    )?;
                }
                self.r2c_out_capacity_bytes = out_bytes;
                self.stft_cache_valid = false;
            }
            Ok(())
        }

        fn stft_cache_matches(
            &self,
            signal_bytes: usize,
            signal_hash: u64,
            window_bytes: usize,
            window_hash: u64,
            n: usize,
            hop_length: usize,
            n_frames: usize,
            n_channels: usize,
            n_samples_per_channel: usize,
        ) -> bool {
            self.stft_cache_valid
                && self.stft_cache_signal_bytes == signal_bytes
                && self.stft_cache_signal_hash == signal_hash
                && self.stft_cache_window_bytes == window_bytes
                && self.stft_cache_window_hash == window_hash
                && self.stft_cache_n == n
                && self.stft_cache_hop == hop_length
                && self.stft_cache_frames == n_frames
                && self.stft_cache_channels == n_channels
                && self.stft_cache_samples_per_channel == n_samples_per_channel
                && !self.dev_r2c_out_ptr.is_null()
        }

        fn update_stft_cache(
            &mut self,
            signal_bytes: usize,
            signal_hash: u64,
            window_bytes: usize,
            window_hash: u64,
            n: usize,
            hop_length: usize,
            n_frames: usize,
            n_channels: usize,
            n_samples_per_channel: usize,
        ) {
            self.stft_cache_valid = true;
            self.stft_cache_signal_bytes = signal_bytes;
            self.stft_cache_signal_hash = signal_hash;
            self.stft_cache_window_bytes = window_bytes;
            self.stft_cache_window_hash = window_hash;
            self.stft_cache_n = n;
            self.stft_cache_hop = hop_length;
            self.stft_cache_frames = n_frames;
            self.stft_cache_channels = n_channels;
            self.stft_cache_samples_per_channel = n_samples_per_channel;
        }

        fn ensure_signal_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.signal_capacity_bytes >= bytes && !self.dev_signal_ptr.is_null() {
                return Ok(());
            }
            if !self.dev_signal_ptr.is_null() {
                unsafe {
                    check_cuda((self.cuda.cuda_free)(self.dev_signal_ptr), "cudaFree(signal realloc)")?;
                }
                self.dev_signal_ptr = ptr::null_mut();
                self.signal_capacity_bytes = 0;
                self.signal_resident_bytes = 0;
                self.signal_resident_hash = 0;
            }
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc)(
                        &mut self.dev_signal_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMalloc(signal)",
                )?;
            }
            self.signal_capacity_bytes = bytes;
            Ok(())
        }

        fn upload_signal_if_needed(
            &mut self,
            signal: &[f32],
            use_pinned_staging: bool,
        ) -> Result<usize, String> {
            let (bytes, hash) = self.signal_fingerprint(signal);
            self.ensure_signal_capacity(bytes)?;
            if self.signal_resident_bytes == bytes && self.signal_resident_hash == hash {
                super::cdbg(&format!("signal resident cache hit ({} bytes)", bytes));
                return Ok(0);
            }

            super::cdbg(&format!("signal resident cache miss -> upload {} bytes", bytes));
            unsafe {
                if use_pinned_staging {
                    super::copy_bytes_to_staging(
                        signal.as_ptr() as *const c_void,
                        self.host_input_ptr,
                        bytes,
                    );
                }
                check_cuda(
                    (self.cuda.cuda_memcpy_async)(
                        self.dev_signal_ptr,
                        if use_pinned_staging {
                            self.host_input_ptr as *const c_void
                        } else {
                            signal.as_ptr() as *const c_void
                        },
                        bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        self.stream,
                    ),
                    "cudaMemcpyAsync signal H2D(resident)",
                )?;
            }

            self.signal_resident_bytes = bytes;
            self.signal_resident_hash = hash;
            Ok(bytes)
        }

        fn signal_upload_bytes_if_needed(&mut self, signal: &[f32]) -> usize {
            let (bytes, hash) = self.signal_fingerprint(signal);
            if !self.dev_signal_ptr.is_null()
                && self.signal_capacity_bytes >= bytes
                && self.signal_resident_bytes == bytes
                && self.signal_resident_hash == hash
            {
                0
            } else {
                bytes
            }
        }

        fn ensure_window_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.window_capacity_bytes >= bytes && !self.dev_window_ptr.is_null() {
                return Ok(());
            }
            if !self.dev_window_ptr.is_null() {
                unsafe {
                    check_cuda((self.cuda.cuda_free)(self.dev_window_ptr), "cudaFree(window realloc)")?;
                }
                self.dev_window_ptr = ptr::null_mut();
                self.window_capacity_bytes = 0;
                self.window_resident_bytes = 0;
                self.window_resident_hash = 0;
            }
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc)(
                        &mut self.dev_window_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMalloc(window)",
                )?;
            }
            self.window_capacity_bytes = bytes;
            Ok(())
        }

        fn ensure_mel_basis_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.mel_basis_capacity_bytes >= bytes && !self.dev_mel_basis_ptr.is_null() {
                return Ok(());
            }
            if !self.dev_mel_basis_ptr.is_null() {
                unsafe {
                    check_cuda((self.cuda.cuda_free)(self.dev_mel_basis_ptr), "cudaFree(mel basis realloc)")?;
                }
                self.dev_mel_basis_ptr = ptr::null_mut();
                self.mel_basis_capacity_bytes = 0;
                self.mel_basis_resident_bytes = 0;
                self.mel_basis_resident_hash = 0;
            }
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc)(
                        &mut self.dev_mel_basis_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMalloc(mel basis)",
                )?;
            }
            self.mel_basis_capacity_bytes = bytes;
            Ok(())
        }

        fn window_upload_bytes_if_needed(&mut self, window: &[f32]) -> usize {
            let (bytes, hash) = self.window_fingerprint(window);
            if !self.dev_window_ptr.is_null()
                && self.window_capacity_bytes >= bytes
                && self.window_resident_bytes == bytes
                && self.window_resident_hash == hash
            {
                0
            } else {
                bytes
            }
        }

        fn mel_basis_upload_bytes_if_needed(&mut self, mel_basis: &[f32]) -> usize {
            let (bytes, hash) = self.mel_basis_fingerprint(mel_basis);
            if !self.dev_mel_basis_ptr.is_null()
                && self.mel_basis_capacity_bytes >= bytes
                && self.mel_basis_resident_bytes == bytes
                && self.mel_basis_resident_hash == hash
            {
                0
            } else {
                bytes
            }
        }

        fn upload_window_if_needed(&mut self, window: &[f32]) -> Result<usize, String> {
            let (bytes, hash) = self.window_fingerprint(window);
            self.ensure_window_capacity(bytes)?;
            if self.window_resident_bytes == bytes && self.window_resident_hash == hash {
                super::cdbg(&format!("window resident cache hit ({} bytes)", bytes));
                return Ok(0);
            }
            super::cdbg(&format!("window resident cache miss -> upload {} bytes", bytes));
            unsafe {
                check_cuda(
                    (self.cuda.cuda_memcpy_async)(
                        self.dev_window_ptr,
                        window.as_ptr() as *const c_void,
                        bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        self.stream,
                    ),
                    "cudaMemcpyAsync window H2D(resident)",
                )?;
            }
            self.window_resident_bytes = bytes;
            self.window_resident_hash = hash;
            Ok(bytes)
        }

        fn upload_mel_basis_if_needed(&mut self, mel_basis: &[f32]) -> Result<usize, String> {
            let (bytes, hash) = self.mel_basis_fingerprint(mel_basis);
            self.ensure_mel_basis_capacity(bytes)?;
            if self.mel_basis_resident_bytes == bytes && self.mel_basis_resident_hash == hash {
                super::cdbg(&format!("mel basis resident cache hit ({} bytes)", bytes));
                return Ok(0);
            }
            super::cdbg(&format!("mel basis resident cache miss -> upload {} bytes", bytes));
            unsafe {
                check_cuda(
                    (self.cuda.cuda_memcpy_async)(
                        self.dev_mel_basis_ptr,
                        mel_basis.as_ptr() as *const c_void,
                        bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        self.stream,
                    ),
                    "cudaMemcpyAsync mel basis H2D(resident)",
                )?;
            }
            self.mel_basis_resident_bytes = bytes;
            self.mel_basis_resident_hash = hash;
            Ok(bytes)
        }

        fn ensure_mel_out_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.mel_out_capacity_bytes >= bytes && !self.dev_mel_out_ptr.is_null() {
                return Ok(());
            }
            if !self.dev_mel_out_ptr.is_null() {
                unsafe {
                    check_cuda((self.cuda.cuda_free)(self.dev_mel_out_ptr), "cudaFree(mel out realloc)")?;
                }
                self.dev_mel_out_ptr = ptr::null_mut();
                self.mel_out_capacity_bytes = 0;
            }
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc)(
                        &mut self.dev_mel_out_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMalloc(mel out)",
                )?;
            }
            self.mel_out_capacity_bytes = bytes;
            Ok(())
        }

        fn ensure_host_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.host_capacity_bytes >= bytes && !self.host_ptr.is_null() {
                return Ok(());
            }
            super::cdbg(&format!("cudaMallocHost {} bytes", bytes));
            let mut new_host_ptr = ptr::null_mut();
            let old_host_ptr = self.host_ptr;
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc_host)(
                        &mut new_host_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMallocHost",
                )?;
            }
            self.host_ptr = new_host_ptr;
            self.host_capacity_bytes = bytes;
            if !old_host_ptr.is_null() {
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_free_host)(old_host_ptr),
                        "cudaFreeHost(realloc)",
                    )?;
                }
            }
            Ok(())
        }

        fn ensure_host_io_capacity(&mut self, bytes: usize) -> Result<(), String> {
            if self.host_io_capacity_bytes >= bytes
                && !self.host_input_ptr.is_null()
                && !self.host_output_ptr.is_null()
            {
                return Ok(());
            }
            let mut new_input_ptr = ptr::null_mut();
            let mut new_output_ptr = ptr::null_mut();
            let old_input_ptr = self.host_input_ptr;
            let old_output_ptr = self.host_output_ptr;
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc_host)(
                        &mut new_input_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMallocHost(input)",
                )?;
                if let Err(err) = check_cuda(
                    (self.cuda.cuda_malloc_host)(
                        &mut new_output_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMallocHost(output)",
                ) {
                    let _ = (self.cuda.cuda_free_host)(new_input_ptr);
                    return Err(err);
                }
            }
            self.host_input_ptr = new_input_ptr;
            self.host_output_ptr = new_output_ptr;
            self.host_io_capacity_bytes = bytes;
            if !old_input_ptr.is_null() {
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_free_host)(old_input_ptr),
                        "cudaFreeHost(input realloc)",
                    )?;
                }
            }
            if !old_output_ptr.is_null() {
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_free_host)(old_output_ptr),
                        "cudaFreeHost(output realloc)",
                    )?;
                }
            }
            Ok(())
        }

        fn ensure_host_io_capacity_b(&mut self, bytes: usize) -> Result<(), String> {
            if self.host_io_capacity_b_bytes >= bytes
                && !self.host_input_ptr_b.is_null()
                && !self.host_output_ptr_b.is_null()
            {
                return Ok(());
            }
            let mut new_input_ptr = ptr::null_mut();
            let mut new_output_ptr = ptr::null_mut();
            let old_input_ptr = self.host_input_ptr_b;
            let old_output_ptr = self.host_output_ptr_b;
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc_host)(
                        &mut new_input_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMallocHost(input B)",
                )?;
                if let Err(err) = check_cuda(
                    (self.cuda.cuda_malloc_host)(
                        &mut new_output_ptr as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMallocHost(output B)",
                ) {
                    let _ = (self.cuda.cuda_free_host)(new_input_ptr);
                    return Err(err);
                }
            }
            self.host_input_ptr_b = new_input_ptr;
            self.host_output_ptr_b = new_output_ptr;
            self.host_io_capacity_b_bytes = bytes;
            if !old_input_ptr.is_null() {
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_free_host)(old_input_ptr),
                        "cudaFreeHost(input B realloc)",
                    )?;
                }
            }
            if !old_output_ptr.is_null() {
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_free_host)(old_output_ptr),
                        "cudaFreeHost(output B realloc)",
                    )?;
                }
            }
            Ok(())
        }

        fn gpu_available_bytes(&self) -> Result<usize, String> {
            let mut free = 0usize;
            let mut total = 0usize;
            unsafe {
                check_cuda(
                    (self.cuda.cuda_mem_get_info)(
                        &mut free as *mut usize,
                        &mut total as *mut usize,
                    ),
                    "cudaMemGetInfo",
                )?;
            }
            Ok(free)
        }

        fn should_use_gpu_for_c2c(&self, bytes: usize) -> Result<bool, String> {
            let free = self.gpu_available_bytes()?;
            Ok(free >= super::c2c_required_headroom(bytes))
        }

        fn should_use_gpu_for_split_io(
            &self,
            in_bytes: usize,
            out_bytes: usize,
        ) -> Result<bool, String> {
            let free = self.gpu_available_bytes()?;
            Ok(free >= super::split_io_required_headroom(in_bytes, out_bytes))
        }

        fn should_use_gpu_for_window_pack(
            &self,
            signal_bytes: usize,
            window_upload_bytes: usize,
            fft_in_bytes: usize,
            out_bytes: usize,
        ) -> Result<bool, String> {
            let free = self.gpu_available_bytes()?;
            Ok(
                free
                    >= super::window_pack_required_headroom(
                        signal_bytes,
                        window_upload_bytes,
                        fft_in_bytes,
                        out_bytes,
                    ),
            )
        }

        fn should_use_gpu_for_stft_mel(
            &self,
            signal_bytes: usize,
            window_upload_bytes: usize,
            fft_in_bytes: usize,
            fft_out_bytes: usize,
            mel_basis_upload_bytes: usize,
            mel_out_bytes: usize,
        ) -> Result<bool, String> {
            let free = self.gpu_available_bytes()?;
            Ok(
                free
                    >= super::stft_mel_required_headroom(
                        signal_bytes,
                        window_upload_bytes,
                        fft_in_bytes,
                        fft_out_bytes,
                        mel_basis_upload_bytes,
                        mel_out_bytes,
                    ),
            )
        }

        fn ensure_plan(&mut self, n: usize, n_frames: usize) -> Result<(), String> {
            if self.plan != 0 && self.plan_n == n && self.plan_batch == n_frames {
                return Ok(());
            }
            if self.plan != 0 {
                super::cdbg(&format!(
                    "cufftDestroy old plan (n={}, batch={})",
                    self.plan_n, self.plan_batch
                ));
                unsafe {
                    check_cufft(
                        (self.cufft.cufft_destroy)(self.plan),
                        "cufftDestroy(replan)",
                    )?;
                }
                self.plan = 0;
            }
            let n_i32 =
                i32::try_from(n).map_err(|_| "n too large for cuFFT".to_string())?;
            let batch =
                i32::try_from(n_frames).map_err(|_| "batch too large for cuFFT".to_string())?;
            super::cdbg(&format!("cufftPlanMany n={} batch={}", n, n_frames));
            unsafe {
                check_cufft(
                    (self.cufft.cufft_plan_many)(
                        &mut self.plan as *mut CufftHandle,
                        1,
                        &n_i32 as *const i32,
                        ptr::null(),
                        1,
                        n_i32,
                        ptr::null(),
                        1,
                        n_i32,
                        CUFFT_C2C,
                        batch,
                    ),
                    "cufftPlanMany",
                )?;
                check_cufft(
                    (self.cufft.cufft_set_stream)(self.plan, self.stream),
                    "cufftSetStream",
                )?;
            }
            self.plan_n = n;
            self.plan_batch = n_frames;
            Ok(())
        }

        fn ensure_plan_r2c(&mut self, n: usize, n_frames: usize) -> Result<(), String> {
            if self.plan_r2c != 0 && self.plan_r2c_n == n && self.plan_r2c_batch == n_frames {
                return Ok(());
            }
            if self.plan_r2c != 0 {
                unsafe {
                    check_cufft(
                        (self.cufft.cufft_destroy)(self.plan_r2c),
                        "cufftDestroy(replan r2c)",
                    )?;
                }
                self.plan_r2c = 0;
            }
            let n_i32 = i32::try_from(n).map_err(|_| "n too large for cuFFT R2C".to_string())?;
            let n_bins = n / 2 + 1;
            let n_bins_i32 = i32::try_from(n_bins)
                .map_err(|_| "n_bins too large for cuFFT R2C".to_string())?;
            let batch = i32::try_from(n_frames)
                .map_err(|_| "batch too large for cuFFT R2C".to_string())?;
            unsafe {
                check_cufft(
                    (self.cufft.cufft_plan_many)(
                        &mut self.plan_r2c as *mut CufftHandle,
                        1,
                        &n_i32 as *const i32,
                        ptr::null(),
                        1,
                        n_i32,
                        ptr::null(),
                        1,
                        n_bins_i32,
                        CUFFT_R2C,
                        batch,
                    ),
                    "cufftPlanMany(R2C)",
                )?;
                check_cufft(
                    (self.cufft.cufft_set_stream)(self.plan_r2c, self.stream),
                    "cufftSetStream(R2C)",
                )?;
            }
            self.plan_r2c_n = n;
            self.plan_r2c_batch = n_frames;
            Ok(())
        }

        fn ensure_c2r_capacity(&mut self, in_bytes: usize, out_bytes: usize) -> Result<(), String> {
            if self.c2r_in_capacity_bytes < in_bytes || self.dev_c2r_in_ptr.is_null() {
                if !self.dev_c2r_in_ptr.is_null() {
                    unsafe {
                        check_cuda((self.cuda.cuda_free)(self.dev_c2r_in_ptr), "cudaFree(c2r in realloc)")?;
                    }
                    self.dev_c2r_in_ptr = ptr::null_mut();
                    self.c2r_in_capacity_bytes = 0;
                }
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_malloc)(
                            &mut self.dev_c2r_in_ptr as *mut *mut c_void,
                            in_bytes,
                        ),
                        "cudaMalloc(c2r in)",
                    )?;
                }
                self.c2r_in_capacity_bytes = in_bytes;
            }

            if self.c2r_out_capacity_bytes < out_bytes || self.dev_c2r_out_ptr.is_null() {
                if !self.dev_c2r_out_ptr.is_null() {
                    unsafe {
                        check_cuda((self.cuda.cuda_free)(self.dev_c2r_out_ptr), "cudaFree(c2r out realloc)")?;
                    }
                    self.dev_c2r_out_ptr = ptr::null_mut();
                    self.c2r_out_capacity_bytes = 0;
                }
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_malloc)(
                            &mut self.dev_c2r_out_ptr as *mut *mut c_void,
                            out_bytes,
                        ),
                        "cudaMalloc(c2r out)",
                    )?;
                }
                self.c2r_out_capacity_bytes = out_bytes;
            }
            Ok(())
        }

        fn ensure_plan_c2r(&mut self, n: usize, n_frames: usize) -> Result<(), String> {
            if self.plan_c2r != 0 && self.plan_c2r_n == n && self.plan_c2r_batch == n_frames {
                return Ok(());
            }
            if self.plan_c2r != 0 {
                unsafe {
                    check_cufft(
                        (self.cufft.cufft_destroy)(self.plan_c2r),
                        "cufftDestroy(replan c2r)",
                    )?;
                }
                self.plan_c2r = 0;
            }
            let n_i32 = i32::try_from(n).map_err(|_| "n too large for cuFFT C2R".to_string())?;
            let n_bins = n / 2 + 1;
            let n_bins_i32 = i32::try_from(n_bins)
                .map_err(|_| "n_bins too large for cuFFT C2R".to_string())?;
            let batch = i32::try_from(n_frames)
                .map_err(|_| "batch too large for cuFFT C2R".to_string())?;

            unsafe {
                check_cufft(
                    (self.cufft.cufft_plan_many)(
                        &mut self.plan_c2r as *mut CufftHandle,
                        1,
                        &n_i32 as *const i32,
                        ptr::null(),
                        1,
                        n_bins_i32,
                        ptr::null(),
                        1,
                        n_i32,
                        CUFFT_C2R,
                        batch,
                    ),
                    "cufftPlanMany(C2R)",
                )?;
                check_cufft(
                    (self.cufft.cufft_set_stream)(self.plan_c2r, self.stream),
                    "cufftSetStream(C2R)",
                )?;
            }

            self.plan_c2r_n = n;
            self.plan_c2r_batch = n_frames;
            Ok(())
        }

        fn ensure_stream(&mut self) -> Result<(), String> {
            if !self.stream.is_null() {
                return Ok(());
            }
            unsafe {
                check_cuda(
                    (self.cuda.cuda_stream_create)(
                        &mut self.stream as *mut *mut c_void,
                    ),
                    "cudaStreamCreate",
                )?;
            }
            Ok(())
        }

        // ── Two-stream helpers ────────────────────────────────────────

        fn ensure_stream_b(&mut self) -> Result<(), String> {
            if !self.stream_b.is_null() {
                return Ok(());
            }
            unsafe {
                check_cuda(
                    (self.cuda.cuda_stream_create)(
                        &mut self.stream_b as *mut *mut c_void,
                    ),
                    "cudaStreamCreate(B)",
                )?;
            }
            Ok(())
        }

        fn ensure_capacity_b(&mut self, bytes: usize) -> Result<(), String> {
            if self.capacity_b_bytes >= bytes && !self.dev_ptr_b.is_null() {
                return Ok(());
            }
            if !self.dev_ptr_b.is_null() {
                unsafe {
                    check_cuda(
                        (self.cuda.cuda_free)(self.dev_ptr_b),
                        "cudaFree(B realloc)",
                    )?;
                }
                self.dev_ptr_b = ptr::null_mut();
                self.capacity_b_bytes = 0;
            }
            super::cdbg(&format!("cudaMalloc(B) {} bytes", bytes));
            unsafe {
                check_cuda(
                    (self.cuda.cuda_malloc)(
                        &mut self.dev_ptr_b as *mut *mut c_void,
                        bytes,
                    ),
                    "cudaMalloc(B)",
                )?;
            }
            self.capacity_b_bytes = bytes;
            Ok(())
        }

        fn ensure_plan_b(&mut self, n: usize, n_frames: usize) -> Result<(), String> {
            if self.plan_b != 0 && self.plan_b_n == n && self.plan_b_batch == n_frames {
                return Ok(());
            }
            if self.plan_b != 0 {
                unsafe {
                    check_cufft(
                        (self.cufft.cufft_destroy)(self.plan_b),
                        "cufftDestroy(B replan)",
                    )?;
                }
                self.plan_b = 0;
            }
            let n_i32 =
                i32::try_from(n).map_err(|_| "n too large for cuFFT plan B".to_string())?;
            let batch =
                i32::try_from(n_frames).map_err(|_| "batch too large for cuFFT plan B".to_string())?;
            super::cdbg(&format!("cufftPlanMany(B) n={} batch={}", n, n_frames));
            unsafe {
                check_cufft(
                    (self.cufft.cufft_plan_many)(
                        &mut self.plan_b as *mut CufftHandle,
                        1,
                        &n_i32 as *const i32,
                        ptr::null(),
                        1,
                        n_i32,
                        ptr::null(),
                        1,
                        n_i32,
                        CUFFT_C2C,
                        batch,
                    ),
                    "cufftPlanMany(B)",
                )?;
                check_cufft(
                    (self.cufft.cufft_set_stream)(self.plan_b, self.stream_b),
                    "cufftSetStream(B)",
                )?;
            }
            self.plan_b_n = n;
            self.plan_b_batch = n_frames;
            Ok(())
        }
    }

    impl Drop for CudaWorkspace {
        fn drop(&mut self) {
            if self.plan != 0 {
                unsafe {
                    let _ = (self.cufft.cufft_destroy)(self.plan);
                }
            }
            if self.plan_r2c != 0 {
                unsafe {
                    let _ = (self.cufft.cufft_destroy)(self.plan_r2c);
                }
            }
            if self.plan_c2r != 0 {
                unsafe {
                    let _ = (self.cufft.cufft_destroy)(self.plan_c2r);
                }
            }
            if !self.dev_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_ptr);
                }
            }
            if !self.dev_r2c_in_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_r2c_in_ptr);
                }
            }
            if !self.dev_r2c_out_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_r2c_out_ptr);
                }
            }
            if !self.dev_signal_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_signal_ptr);
                }
            }
            if !self.dev_window_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_window_ptr);
                }
            }
            if !self.dev_mel_basis_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_mel_basis_ptr);
                }
            }
            if !self.dev_mel_out_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_mel_out_ptr);
                }
            }
            if !self.dev_c2r_in_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_c2r_in_ptr);
                }
            }
            if !self.dev_c2r_out_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_c2r_out_ptr);
                }
            }
            if !self.host_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free_host)(self.host_ptr);
                }
            }
            if !self.host_input_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free_host)(self.host_input_ptr);
                }
            }
            if !self.host_output_ptr.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free_host)(self.host_output_ptr);
                }
            }
            if !self.host_input_ptr_b.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free_host)(self.host_input_ptr_b);
                }
            }
            if !self.host_output_ptr_b.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free_host)(self.host_output_ptr_b);
                }
            }
            if !self.stream.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_stream_destroy)(self.stream);
                }
            }
            // two-stream pipeline cleanup
            if self.plan_b != 0 {
                unsafe {
                    let _ = (self.cufft.cufft_destroy)(self.plan_b);
                }
            }
            if !self.dev_ptr_b.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_free)(self.dev_ptr_b);
                }
            }
            if !self.stream_b.is_null() {
                unsafe {
                    let _ = (self.cuda.cuda_stream_destroy)(self.stream_b);
                }
            }
        }
    }

    // ── DLL loading ────────────────────────────────────────────────────────────────

    /// Try each name, also searching CUDA_PATH/bin and common Windows install dirs.
    pub fn load_first_dll(names: &[&str]) -> Result<Arc<Library>, String> {
        // Build ordered search dir list:
        //  1. System PATH (plain name)
        //  2. $CUDA_PATH\bin and $CUDA_PATH\bin\x64
        //  3. Hard-coded CUDA Toolkit install root (newest version first)
        let mut search_dirs: Vec<std::path::PathBuf> = Vec::new();

        if let Ok(cp) = std::env::var("CUDA_PATH") {
            let p = std::path::Path::new(&cp);
            search_dirs.push(p.join("bin"));
            search_dirs.push(p.join("bin").join("x64"));
        }

        let cuda_root = std::path::Path::new(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        );
        if cuda_root.exists() {
            if let Ok(rd) = std::fs::read_dir(cuda_root) {
                let mut versions: Vec<_> = rd
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
                    .filter(|e| e.file_name().to_string_lossy().starts_with('v'))
                    .map(|e| e.path())
                    .collect();
                versions.sort_unstable_by(|a, b| b.cmp(a)); // newest first
                for v in versions {
                    search_dirs.push(v.join("bin"));
                    search_dirs.push(v.join("bin").join("x64"));
                }
            }
        }

        let mut tried: Vec<String> = Vec::new();
        for name in names {
            // (a) plain name — works when CUDA bin is on PATH
            super::cdbg(&format!("trying '{}'", name));
            match unsafe { Library::new(name) } {
                Ok(lib) => {
                    super::cdbg(&format!("  OK (PATH): {}", name));
                    return Ok(Arc::new(lib));
                }
                Err(e) => {
                    super::cdbg(&format!("  not on PATH: {}", e));
                    tried.push(name.to_string());
                }
            }
            // (b) full path through each search dir
            for dir in &search_dirs {
                let full = dir.join(name);
                super::cdbg(&format!("trying '{}'", full.display()));
                match unsafe { Library::new(&full) } {
                    Ok(lib) => {
                        super::cdbg(&format!("  OK: {}", full.display()));
                        return Ok(Arc::new(lib));
                    }
                    Err(e) => {
                        super::cdbg(&format!("  not found: {}", e));
                        tried.push(full.to_string_lossy().into_owned());
                    }
                }
            }
        }
        Err(format!(
            "unable to load CUDA DLL after {} attempts (tip: set CUDA_PATH or add CUDA bin to PATH)",
            tried.len()
        ))
    }

    fn load_cuda_api() -> Result<CudaApi, String> {
        super::cdbg("loading CUDA runtime...");
        let lib = load_first_dll(&[
            "cudart64_13.dll",   // CUDA 13.x
            "cudart64_130.dll",  // alt naming
            "cudart64_12.dll",   // CUDA 12.x
            "cudart64_11080.dll",// CUDA 11.8
            "cudart64_110.dll",  // CUDA 11.x
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so.11.0",
            "libcudart.so",
        ])?;
        super::cdbg("CUDA runtime loaded â€” resolving symbols...");
        unsafe {
            let cuda_get_device_count = *lib
                .get::<CudaGetDeviceCountFn>(b"cudaGetDeviceCount")
                .map_err(|e| format!("cudaGetDeviceCount: {}", e))?;
            let cuda_malloc = *lib
                .get::<CudaMallocFn>(b"cudaMalloc")
                .map_err(|e| format!("cudaMalloc: {}", e))?;
            let cuda_free = *lib
                .get::<CudaFreeFn>(b"cudaFree")
                .map_err(|e| format!("cudaFree: {}", e))?;
            let cuda_malloc_host = *lib
                .get::<CudaMallocHostFn>(b"cudaMallocHost")
                .map_err(|e| format!("cudaMallocHost: {}", e))?;
            let cuda_free_host = *lib
                .get::<CudaFreeHostFn>(b"cudaFreeHost")
                .map_err(|e| format!("cudaFreeHost: {}", e))?;
            let cuda_mem_get_info = *lib
                .get::<CudaMemGetInfoFn>(b"cudaMemGetInfo")
                .map_err(|e| format!("cudaMemGetInfo: {}", e))?;
            let cuda_stream_create = *lib
                .get::<CudaStreamCreateFn>(b"cudaStreamCreate")
                .map_err(|e| format!("cudaStreamCreate: {}", e))?;
            let cuda_stream_destroy = *lib
                .get::<CudaStreamDestroyFn>(b"cudaStreamDestroy")
                .map_err(|e| format!("cudaStreamDestroy: {}", e))?;
            let cuda_memcpy_async = *lib
                .get::<CudaMemcpyAsyncFn>(b"cudaMemcpyAsync")
                .map_err(|e| format!("cudaMemcpyAsync: {}", e))?;
            let cuda_stream_synchronize = *lib
                .get::<CudaStreamSynchronizeFn>(b"cudaStreamSynchronize")
                .map_err(|e| format!("cudaStreamSynchronize: {}", e))?;
            super::cdbg("CUDA runtime symbols OK");
            Ok(CudaApi {
                _lib: lib,
                cuda_get_device_count,
                cuda_malloc,
                cuda_free,
                cuda_malloc_host,
                cuda_free_host,
                cuda_mem_get_info,
                cuda_stream_create,
                cuda_stream_destroy,
                cuda_memcpy_async,
                cuda_stream_synchronize,
            })
        }
    }

    fn load_cufft_api() -> Result<CufftApi, String> {
        super::cdbg("loading cuFFT...");
        // DLL versioning note (Windows):
        //   CUDA 13.x toolkit ships  cufft64_12.dll  (cuFFT API v12)
        //   CUDA 12.x toolkit ships  cufft64_11.dll  (cuFFT API v11)
        //   CUDA 11.x toolkit ships  cufft64_10.dll  (cuFFT API v10)
        let lib = load_first_dll(&[
            "cufft64_12.dll",   // CUDA 13.x  ← this system
            "cufft64_11.dll",   // CUDA 12.x
            "cufft64_10.dll",   // CUDA 11.x
            "libcufft.so.12",
            "libcufft.so.11",
            "libcufft.so.10",
            "libcufft.so",
        ])?;
        super::cdbg("cuFFT loaded â€” resolving symbols...");
        unsafe {
            let cufft_plan_many = *lib
                .get::<CufftPlanManyFn>(b"cufftPlanMany")
                .map_err(|e| format!("cufftPlanMany: {}", e))?;
            let cufft_exec_c2c = *lib
                .get::<CufftExecC2CFn>(b"cufftExecC2C")
                .map_err(|e| format!("cufftExecC2C: {}", e))?;
            let cufft_exec_r2c = *lib
                .get::<CufftExecR2CFn>(b"cufftExecR2C")
                .map_err(|e| format!("cufftExecR2C: {}", e))?;
            let cufft_exec_c2r = *lib
                .get::<CufftExecC2RFn>(b"cufftExecC2R")
                .map_err(|e| format!("cufftExecC2R: {}", e))?;
            let cufft_destroy = *lib
                .get::<CufftDestroyFn>(b"cufftDestroy")
                .map_err(|e| format!("cufftDestroy: {}", e))?;
            let cufft_set_stream = *lib
                .get::<unsafe extern "C" fn(CufftHandle, *mut c_void) -> CufftResult>(
                    b"cufftSetStream",
                )
                .map_err(|e| format!("cufftSetStream: {}", e))?;
            super::cdbg("cuFFT symbols OK");
            Ok(CufftApi {
                _lib: lib,
                cufft_plan_many,
                cufft_exec_c2c,
                cufft_exec_r2c,
                cufft_exec_c2r,
                cufft_destroy,
                cufft_set_stream,
            })
        }
    }

    fn get_cuda_api() -> Result<CudaApi, String> {
        TL_CUDA_API.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(load_cuda_api());
            }
            match cell.borrow().as_ref().unwrap() {
                Ok(api) => Ok(api.clone()),
                Err(e) => Err(e.clone()),
            }
        })
    }

    fn get_cufft_api() -> Result<CufftApi, String> {
        TL_CUFFT_API.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(load_cufft_api());
            }
            match cell.borrow().as_ref().unwrap() {
                Ok(api) => Ok(api.clone()),
                Err(e) => Err(e.clone()),
            }
        })
    }

    fn cuda_window_pack_helper_path() -> Option<&'static str> {
        option_env!("IRON_LIBROSA_CUDA_WINDOW_PACK_HELPER")
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }

    fn load_cuda_window_pack_api() -> Result<Option<CudaWindowPackApi>, String> {
        let Some(path) = cuda_window_pack_helper_path() else {
            return Ok(None);
        };
        let helper_path = std::path::Path::new(path);
        if !helper_path.exists() {
            super::cdbg(&format!(
                "CUDA window-pack helper path missing; falling back: {}",
                helper_path.display()
            ));
            return Ok(None);
        }

        super::cdbg(&format!(
            "loading CUDA window-pack helper: {}",
            helper_path.display()
        ));
        let lib = unsafe { Library::new(helper_path) }
            .map_err(|e| format!("cuda window-pack helper load failed: {}", e))?;
        unsafe {
            let launch_window_and_pack_f32 = *lib
                .get::<LaunchWindowAndPackF32Fn>(b"launch_window_and_pack_f32")
                .map_err(|e| format!("launch_window_and_pack_f32: {}", e))?;
            let launch_window_and_pack_batch_f32 = *lib
                .get::<LaunchWindowAndPackBatchF32Fn>(b"launch_window_and_pack_batch_f32")
                .map_err(|e| format!("launch_window_and_pack_batch_f32: {}", e))?;
            let launch_mel_project_power_f32 = *lib
                .get::<LaunchMelProjectPowerF32Fn>(b"launch_mel_project_power_f32")
                .map_err(|e| format!("launch_mel_project_power_f32: {}", e))?;
            let launch_mel_project_power_batch_f32 = *lib
                .get::<LaunchMelProjectPowerBatchF32Fn>(b"launch_mel_project_power_batch_f32")
                .map_err(|e| format!("launch_mel_project_power_batch_f32: {}", e))?;
            Ok(Some(CudaWindowPackApi {
                _lib: Arc::new(lib),
                launch_window_and_pack_f32,
                launch_window_and_pack_batch_f32,
                launch_mel_project_power_f32,
                launch_mel_project_power_batch_f32,
            }))
        }
    }

    fn get_cuda_window_pack_api() -> Result<Option<CudaWindowPackApi>, String> {
        TL_CUDA_WINDOW_PACK_API.with(|cell| {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(load_cuda_window_pack_api());
            }
            match cell.borrow().as_ref().unwrap() {
                Ok(api) => Ok(api.clone()),
                Err(e) => Err(e.clone()),
            }
        })
    }

    fn cuda_profile_enabled() -> bool {
        static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *EN.get_or_init(|| {
            matches!(
                std::env::var("IRON_LIBROSA_CUDA_PROFILE")
                    .unwrap_or_default()
                    .trim()
                    .to_ascii_lowercase()
                    .as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    }

    /// Emit a profile line to stderr when IRON_LIBROSA_CUDA_PROFILE=1.
    /// Format: [CUDA_PROFILE] op=<op> n=<n> frames=<frames> h2d_kb=<f> kernel_ms=<f> d2h_kb=<f> sync_ms=<f> total_ms=<f> plan_created=<bool>
    fn emit_profile(
        op: &str,
        n: usize,
        frames: usize,
        h2d_bytes: usize,
        kernel_ms: f64,
        d2h_bytes: usize,
        sync_ms: f64,
        total_ms: f64,
        plan_created: bool,
    ) {
        if !cuda_profile_enabled() {
            return;
        }
        eprintln!(
            "[CUDA_PROFILE] op={} n={} frames={} h2d_kb={:.1} kernel_ms={:.3} d2h_kb={:.1} sync_ms={:.3} total_ms={:.3} plan_created={}",
            op,
            n,
            frames,
            h2d_bytes as f64 / 1024.0,
            kernel_ms,
            d2h_bytes as f64 / 1024.0,
            sync_ms,
            total_ms,
            plan_created,
        );
    }

    fn check_cuda(code: CudaError, ctx: &str) -> Result<(), String> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(format!("CUDA error {} in {}", code, ctx))
        }
    }

    fn check_cufft(code: CufftResult, ctx: &str) -> Result<(), String> {
        if code == CUFFT_SUCCESS {
            Ok(())
        } else {
            Err(format!("cuFFT error {} in {}", code, ctx))
        }
    }

    unsafe fn sync_stream_and_copy_from_staging(
        cuda: &CudaApi,
        stream: *mut c_void,
        staging_src: *const c_void,
        dst: *mut c_void,
        bytes: usize,
        ctx: &str,
    ) -> Result<(), String> {
        check_cuda((cuda.cuda_stream_synchronize)(stream), ctx)?;
        super::copy_bytes_from_staging(staging_src, dst, bytes);
        Ok(())
    }

    fn cuda_runtime_available(cuda: &CudaApi) -> bool {
        // Cache per-thread; device count doesn't change during a session.
        thread_local! {
            static CACHED: std::cell::RefCell<Option<bool>> =
                const { std::cell::RefCell::new(None) };
        }
        CACHED.with(|cell| {
            if let Some(v) = *cell.borrow() {
                return v;
            }
            let mut count = 0i32;
            let ok = unsafe {
                (cuda.cuda_get_device_count)(&mut count) == CUDA_SUCCESS && count > 0
            };
            super::cdbg(&format!("cudaGetDeviceCount → count={} ok={}", count, ok));
            *cell.borrow_mut() = Some(ok);
            ok
        })
    }

    fn cuda_max_work_inner() -> Option<usize> {
        std::env::var("IRON_LIBROSA_CUDA_MAX_WORK")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
    }

    fn cuda_use_pinned_staging() -> bool {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ENABLED.get_or_init(|| {
            !matches!(
                std::env::var("IRON_LIBROSA_CUDA_USE_PINNED_STAGING")
                    .unwrap_or_default()
                    .trim()
                    .to_ascii_lowercase()
                    .as_str(),
                "0" | "false" | "no" | "off"
            )
        })
    }

    fn run_cufft_c2c_inplace(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
        direction: i32,
        dir_label: &str,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_frames == 0 {
            return Ok(());
        }
        let total = n
            .checked_mul(n_frames)
            .ok_or("batched FFT size overflow")?;
        if buffer.len() < total {
            return Err(format!(
                "buffer len {} < n*n_frames {}",
                buffer.len(),
                total
            ));
        }

        // Optional hard cap
        if let Some(max_work) = cuda_max_work_inner() {
            if total > max_work {
                return Err(format!(
                    "CUDA work {} > IRON_LIBROSA_CUDA_MAX_WORK {}; CPU fallback",
                    total, max_work
                ));
            }
        }

        super::cdbg(&format!(
            "run_cufft_c2c_inplace dir={} n={} frames={} total_elems={}",
            dir_label, n, n_frames, total
        ));

        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;

        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let bytes = total
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("byte-size overflow")?;

        let t0 = std::time::Instant::now();
        let result = TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }

            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            // Refresh API references (DLL may reload)
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();

            ws.ensure_stream()?;
            ws.ensure_capacity(bytes)?;
            let use_pinned_staging = cuda_use_pinned_staging();
            if !ws.should_use_gpu_for_c2c(bytes)? {
                return Err("CUDA GPU memory headroom insufficient; CPU fallback".to_string());
            }
            if use_pinned_staging {
                ws.ensure_host_io_capacity(bytes)?;
                unsafe {
                    super::copy_bytes_to_staging(
                        buffer.as_ptr() as *const c_void,
                        ws.host_input_ptr,
                        bytes,
                    );
                }
            }

            let plan_was_created = ws.plan == 0 || ws.plan_n != n || ws.plan_batch != n_frames;
            ws.ensure_plan(n, n_frames)?;

            super::cdbg(&format!("  H2D copy {} bytes...", bytes));
            let t_h2d = std::time::Instant::now();
            unsafe {
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        ws.dev_ptr,
                        if use_pinned_staging {
                            ws.host_input_ptr as *const c_void
                        } else {
                            buffer.as_ptr() as *const c_void
                        },
                        bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync H2D",
                )?;
            }
            let h2d_elapsed_ms = t_h2d.elapsed().as_secs_f64() * 1000.0;
            super::cdbg(&format!("  H2D done in {:.3}ms", h2d_elapsed_ms));

            super::cdbg("  cufftExecC2C...");
            let t_gpu = std::time::Instant::now();
            unsafe {
                check_cufft(
                    (ws.cufft.cufft_exec_c2c)(ws.plan, ws.dev_ptr, ws.dev_ptr, direction),
                    "cufftExecC2C",
                )?;
            }
            let kernel_elapsed_ms = t_gpu.elapsed().as_secs_f64() * 1000.0;
            super::cdbg(&format!("  GPU done in {:.3}ms", kernel_elapsed_ms));

            super::cdbg(&format!("  D2H copy {} bytes...", bytes));
            let t_d2h = std::time::Instant::now();
            unsafe {
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        if use_pinned_staging {
                            ws.host_output_ptr
                        } else {
                            buffer.as_mut_ptr() as *mut c_void
                        },
                        ws.dev_ptr as *const c_void,
                        bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync D2H",
                )?;
                check_cuda(
                    (ws.cuda.cuda_stream_synchronize)(ws.stream),
                    "cudaStreamSynchronize",
                )?;
                if use_pinned_staging {
                    super::copy_bytes_from_staging(
                        ws.host_output_ptr as *const c_void,
                        buffer.as_mut_ptr() as *mut c_void,
                        bytes,
                    );
                }
            }
            let sync_elapsed_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;
            super::cdbg(&format!("  D2H done in {:.3}ms", sync_elapsed_ms));

            emit_profile(
                "c2c",
                n,
                n_frames,
                bytes,       // h2d_bytes
                kernel_elapsed_ms,
                bytes,       // d2h_bytes
                sync_elapsed_ms,
                t0.elapsed().as_secs_f64() * 1000.0,
                plan_was_created,
            );

            Ok(())
        });

        if let Err(ref e) = result {
            super::cdbg(&format!("  ERROR: {}", e));
        } else {
            super::cdbg(&format!(
                "  total GPU round-trip {:.3}ms",
                t0.elapsed().as_secs_f64() * 1000.0
            ));
        }
        result
    }

    /// Forward batched C2C FFT via cuFFT.
    pub fn fft_forward_batched_gpu(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        run_cufft_c2c_inplace(buffer, n, n_frames, CUFFT_FORWARD, "FWD")
    }

    /// Inverse batched C2C FFT via cuFFT.
    pub fn fft_inverse_batched_gpu(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        run_cufft_c2c_inplace(buffer, n, n_frames, CUFFT_INVERSE, "INV")
    }

    // ── Two-stream pipelined C2C FFT ──────────────────────────────────────────

    /// Overlapped two-stream pipeline: enqueues H2D → FFT → D2H for each chunk
    /// on alternating streams A/B so that stream B's H2D transfer overlaps with
    /// stream A's FFT kernel, halving the effective latency for large batches.
    ///
    /// Caller guarantees:
    ///  * `buffer.len() == n_chunks * chunk_size * n`
    ///  * `n_chunks >= 2`  (otherwise single-stream is faster)
    fn run_cufft_c2c_two_stream_pipeline(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_chunks: usize,
        chunk_size: usize,
        direction: i32,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_chunks == 0 {
            return Ok(());
        }

        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;
        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let chunk_elems = chunk_size
            .checked_mul(n)
            .ok_or("chunk element count overflow")?;
        let chunk_bytes = chunk_elems
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("chunk byte-size overflow")?;

        let total_elems = chunk_elems
            .checked_mul(n_chunks)
            .ok_or("total element count overflow")?;
        if buffer.len() < total_elems {
            return Err(format!(
                "buffer too small for two-stream pipeline: {} < {}",
                buffer.len(),
                total_elems
            ));
        }

        super::cdbg(&format!(
            "two-stream pipeline: n={} chunk_size={} n_chunks={} chunk_bytes={}",
            n, chunk_size, n_chunks, chunk_bytes
        ));

        TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }
            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();
            let use_pinned_staging = cuda_use_pinned_staging();

            // Ensure both streams, device buffers, and C2C plans.
            ws.ensure_stream()?;        // stream A
            ws.ensure_stream_b()?;      // stream B
            ws.ensure_capacity(chunk_bytes)?;   // dev_ptr  (A)
            ws.ensure_capacity_b(chunk_bytes)?; // dev_ptr_b (B)
            if use_pinned_staging {
                ws.ensure_host_io_capacity(chunk_bytes)?;
                ws.ensure_host_io_capacity_b(chunk_bytes)?;
            }
            ws.ensure_plan(n, chunk_size)?;     // plan A, bound to stream A
            ws.ensure_plan_b(n, chunk_size)?;   // plan B, bound to stream B

            // Pipeline loop.  For each chunk we pick stream A (even) or B (odd),
            // enqueue H2D → cufftExecC2C → D2H all on the same stream so GPU
            // ordering is guaranteed, but the two streams execute concurrently.
            let buf_ptr = buffer.as_mut_ptr();
            let mut pending_a: Option<usize> = None;
            let mut pending_b: Option<usize> = None;
            for chunk_idx in 0..n_chunks {
                let buf_off = chunk_idx * chunk_elems;

                let (dev_buf, stream, plan, host_in, host_out, pending_slot, sync_ctx) = if chunk_idx % 2 == 0 {
                    (
                        ws.dev_ptr,
                        ws.stream,
                        ws.plan,
                        ws.host_input_ptr,
                        ws.host_output_ptr,
                        &mut pending_a,
                        "cudaStreamSynchronize A (two-stream staged)",
                    )
                } else {
                    (
                        ws.dev_ptr_b,
                        ws.stream_b,
                        ws.plan_b,
                        ws.host_input_ptr_b,
                        ws.host_output_ptr_b,
                        &mut pending_b,
                        "cudaStreamSynchronize B (two-stream staged)",
                    )
                };

                unsafe {
                    if use_pinned_staging {
                        if let Some(prev_chunk_idx) = pending_slot.take() {
                            let prev_off = prev_chunk_idx * chunk_elems;
                            sync_stream_and_copy_from_staging(
                                &ws.cuda,
                                stream,
                                host_out as *const c_void,
                                buf_ptr.add(prev_off) as *mut c_void,
                                chunk_bytes,
                                sync_ctx,
                            )?;
                        }
                        super::copy_bytes_to_staging(
                            buf_ptr.add(buf_off) as *const c_void,
                            host_in,
                            chunk_bytes,
                        );
                    }
                    // H2D (async; host returns immediately)
                    check_cuda(
                        (ws.cuda.cuda_memcpy_async)(
                            dev_buf,
                            if use_pinned_staging {
                                host_in as *const c_void
                            } else {
                                buf_ptr.add(buf_off) as *const c_void
                            },
                            chunk_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            stream,
                        ),
                        "cudaMemcpyAsync H2D (two-stream)",
                    )?;
                    // FFT (async on same stream → starts after H2D on GPU)
                    check_cufft(
                        (ws.cufft.cufft_exec_c2c)(plan, dev_buf, dev_buf, direction),
                        "cufftExecC2C (two-stream)",
                    )?;
                    // D2H (async on same stream → starts after FFT on GPU)
                    check_cuda(
                        (ws.cuda.cuda_memcpy_async)(
                            if use_pinned_staging {
                                host_out
                            } else {
                                buf_ptr.add(buf_off) as *mut c_void
                            },
                            dev_buf as *const c_void,
                            chunk_bytes,
                            CUDA_MEMCPY_DEVICE_TO_HOST,
                            stream,
                        ),
                        "cudaMemcpyAsync D2H (two-stream)",
                    )?;
                    if use_pinned_staging {
                        *pending_slot = Some(chunk_idx);
                    }
                }
            }

            // Wait for both streams to drain before returning to caller.
            unsafe {
                if use_pinned_staging {
                    if let Some(prev_chunk_idx) = pending_a.take() {
                        let prev_off = prev_chunk_idx * chunk_elems;
                        sync_stream_and_copy_from_staging(
                            &ws.cuda,
                            ws.stream,
                            ws.host_output_ptr as *const c_void,
                            buf_ptr.add(prev_off) as *mut c_void,
                            chunk_bytes,
                            "cudaStreamSynchronize A (two-stream staged final)",
                        )?;
                    }
                    if let Some(prev_chunk_idx) = pending_b.take() {
                        let prev_off = prev_chunk_idx * chunk_elems;
                        sync_stream_and_copy_from_staging(
                            &ws.cuda,
                            ws.stream_b,
                            ws.host_output_ptr_b as *const c_void,
                            buf_ptr.add(prev_off) as *mut c_void,
                            chunk_bytes,
                            "cudaStreamSynchronize B (two-stream staged final)",
                        )?;
                    }
                } else {
                    check_cuda(
                        (ws.cuda.cuda_stream_synchronize)(ws.stream),
                        "cudaStreamSynchronize A (two-stream)",
                    )?;
                    check_cuda(
                        (ws.cuda.cuda_stream_synchronize)(ws.stream_b),
                        "cudaStreamSynchronize B (two-stream)",
                    )?;
                }
            }

            super::cdbg("two-stream pipeline complete");
            Ok(())
        })
    }

    /// Exposed forward path for the two-stream pipeline.
    pub fn fft_forward_two_stream_pipeline(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_chunks: usize,
        chunk_size: usize,
    ) -> Result<(), String> {
        run_cufft_c2c_two_stream_pipeline(buffer, n, n_chunks, chunk_size, CUFFT_FORWARD)
    }

    /// Exposed inverse path for the two-stream pipeline.
    pub fn fft_inverse_two_stream_pipeline(
        buffer: &mut [Complex<f32>],
        n: usize,
        n_chunks: usize,
        chunk_size: usize,
    ) -> Result<(), String> {
        run_cufft_c2c_two_stream_pipeline(buffer, n, n_chunks, chunk_size, CUFFT_INVERSE)
    }

    pub fn fft_forward_r2c_batched_gpu(
        input: &[f32],
        output: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_frames == 0 {
            return Ok(());
        }
        let total = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
        if input.len() < total {
            return Err(format!("input len {} < n*n_frames {}", input.len(), total));
        }
        let n_bins = n / 2 + 1;
        let out_total = n_bins
            .checked_mul(n_frames)
            .ok_or("batched R2C output size overflow")?;
        if output.len() < out_total {
            return Err(format!(
                "output len {} < n_bins*n_frames {}",
                output.len(),
                out_total
            ));
        }

        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;
        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let in_bytes = total
            .checked_mul(mem::size_of::<f32>())
            .ok_or("R2C in byte-size overflow")?;
        let out_bytes = out_total
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("R2C out byte-size overflow")?;

        TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }

            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();

            ws.ensure_stream()?;
            let use_pinned_staging = cuda_use_pinned_staging();
            if !ws.should_use_gpu_for_split_io(in_bytes, out_bytes)? {
                return Err("CUDA GPU memory headroom insufficient for R2C; CPU fallback".to_string());
            }
            ws.ensure_r2c_capacity(in_bytes, out_bytes)?;
            if use_pinned_staging {
                ws.ensure_host_io_capacity(in_bytes.max(out_bytes))?;
            }
            let plan_was_created = ws.plan_r2c == 0 || ws.plan_r2c_n != n || ws.plan_r2c_batch != n_frames;
            ws.ensure_plan_r2c(n, n_frames)?;

            let t0 = std::time::Instant::now();
            unsafe {
                if use_pinned_staging {
                    super::copy_bytes_to_staging(
                        input.as_ptr() as *const c_void,
                        ws.host_input_ptr,
                        in_bytes,
                    );
                }
                let t_h2d = std::time::Instant::now();
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        ws.dev_r2c_in_ptr,
                        if use_pinned_staging {
                            ws.host_input_ptr as *const c_void
                        } else {
                            input.as_ptr() as *const c_void
                        },
                        in_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync R2C H2D",
                )?;
                let h2d_ms = t_h2d.elapsed().as_secs_f64() * 1000.0;

                let t_kernel = std::time::Instant::now();
                check_cufft(
                    (ws.cufft.cufft_exec_r2c)(ws.plan_r2c, ws.dev_r2c_in_ptr, ws.dev_r2c_out_ptr),
                    "cufftExecR2C",
                )?;
                let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;

                let t_d2h = std::time::Instant::now();
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        if use_pinned_staging {
                            ws.host_output_ptr
                        } else {
                            output.as_mut_ptr() as *mut c_void
                        },
                        ws.dev_r2c_out_ptr as *const c_void,
                        out_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync R2C D2H",
                )?;
                check_cuda(
                    (ws.cuda.cuda_stream_synchronize)(ws.stream),
                    "cudaStreamSynchronize(R2C)",
                )?;
                let sync_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;

                if use_pinned_staging {
                    super::copy_bytes_from_staging(
                        ws.host_output_ptr as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                        out_bytes,
                    );
                }

                let _ = h2d_ms; // used in profile below
                emit_profile(
                    "r2c",
                    n,
                    n_frames,
                    in_bytes,
                    kernel_ms,
                    out_bytes,
                    sync_ms,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    plan_was_created,
                );
            }
            Ok(())
        })
    }

    pub fn fft_forward_r2c_from_signal_gpu(
        y_padded: &[f32],
        window: &[f32],
        hop_length: usize,
        output: &mut [Complex<f32>],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_frames == 0 {
            return Ok(());
        }
        if window.len() != n {
            return Err(format!("window len {} != n {}", window.len(), n));
        }
        let needed_samples = n
            .checked_add((n_frames.saturating_sub(1)).saturating_mul(hop_length))
            .ok_or("signal size overflow")?;
        if y_padded.len() < needed_samples {
            return Err(format!(
                "y_padded len {} < required {}",
                y_padded.len(),
                needed_samples
            ));
        }

        let total = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
        let n_bins = n / 2 + 1;
        let signal_samples = needed_samples;
        let out_total = n_bins
            .checked_mul(n_frames)
            .ok_or("batched R2C output size overflow")?;
        if output.len() < out_total {
            return Err(format!(
                "output len {} < n_bins*n_frames {}",
                output.len(),
                out_total
            ));
        }

        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;
        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let in_bytes = total
            .checked_mul(mem::size_of::<f32>())
            .ok_or("R2C in byte-size overflow")?;
        let signal_bytes = signal_samples
            .checked_mul(mem::size_of::<f32>())
            .ok_or("R2C signal byte-size overflow")?;
        let out_bytes = out_total
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("R2C out byte-size overflow")?;

        TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }

            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();
            let use_pinned_staging = cuda_use_pinned_staging();
            let kernel_api = match get_cuda_window_pack_api() {
                Ok(api) => api,
                Err(err) => {
                    super::cdbg(&format!(
                        "CUDA window-pack helper unavailable; using CPU pack fallback: {}",
                        err
                    ));
                    None
                }
            };

            ws.ensure_stream()?;
            let signal_upload_bytes = if kernel_api.is_some() {
                ws.signal_upload_bytes_if_needed(y_padded)
            } else {
                0
            };
            let window_upload_bytes = if kernel_api.is_some() {
                ws.window_upload_bytes_if_needed(window)
            } else {
                0
            };
            let enough_gpu_memory = if kernel_api.is_some() {
                ws.should_use_gpu_for_window_pack(
                    signal_upload_bytes,
                    window_upload_bytes,
                    in_bytes,
                    out_bytes,
                )?
            } else {
                ws.should_use_gpu_for_split_io(in_bytes, out_bytes)?
            };
            if !enough_gpu_memory {
                return Err("CUDA GPU memory headroom insufficient for fused R2C; CPU fallback".to_string());
            }
            ws.ensure_r2c_capacity(in_bytes, out_bytes)?;
            let plan_was_created = ws.plan_r2c == 0 || ws.plan_r2c_n != n || ws.plan_r2c_batch != n_frames;
            ws.ensure_plan_r2c(n, n_frames)?;

            if let Some(kernel_api) = kernel_api {
                ws.ensure_signal_capacity(signal_bytes)?;
                if use_pinned_staging {
                    ws.ensure_host_io_capacity(signal_bytes.max(out_bytes))?;
                }

                let t0 = std::time::Instant::now();
                unsafe {
                    let uploaded_signal_bytes =
                        ws.upload_signal_if_needed(y_padded, use_pinned_staging)?;
                    let uploaded_window_bytes = ws.upload_window_if_needed(window)?;
                    let t_kernel = std::time::Instant::now();
                    check_cuda(
                        (kernel_api.launch_window_and_pack_f32)(
                            ws.dev_signal_ptr as *const c_void,
                            ws.dev_window_ptr as *const c_void,
                            ws.dev_r2c_in_ptr,
                            i32::try_from(n).map_err(|_| "n too large for window-pack kernel".to_string())?,
                            i32::try_from(hop_length).map_err(|_| "hop_length too large for window-pack kernel".to_string())?,
                            i32::try_from(n_frames).map_err(|_| "n_frames too large for window-pack kernel".to_string())?,
                            ws.stream,
                        ),
                        "launch_window_and_pack_f32",
                    )?;
                    check_cufft(
                        (ws.cufft.cufft_exec_r2c)(ws.plan_r2c, ws.dev_r2c_in_ptr, ws.dev_r2c_out_ptr),
                        "cufftExecR2C(fused kernel)",
                    )?;
                    let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
                    let t_d2h = std::time::Instant::now();
                    check_cuda(
                        (ws.cuda.cuda_memcpy_async)(
                            if use_pinned_staging {
                                ws.host_output_ptr
                            } else {
                                output.as_mut_ptr() as *mut c_void
                            },
                            ws.dev_r2c_out_ptr as *const c_void,
                            out_bytes,
                            CUDA_MEMCPY_DEVICE_TO_HOST,
                            ws.stream,
                        ),
                        "cudaMemcpyAsync R2C D2H(fused kernel)",
                    )?;
                    check_cuda(
                        (ws.cuda.cuda_stream_synchronize)(ws.stream),
                        "cudaStreamSynchronize(R2C fused kernel)",
                    )?;
                    let sync_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;
                    if use_pinned_staging {
                        super::copy_bytes_from_staging(
                            ws.host_output_ptr as *const c_void,
                            output.as_mut_ptr() as *mut c_void,
                            out_bytes,
                        );
                    }
                    emit_profile(
                        "r2c_fused_kernel",
                        n,
                        n_frames,
                        uploaded_signal_bytes + uploaded_window_bytes,
                        kernel_ms,
                        out_bytes,
                        sync_ms,
                        t0.elapsed().as_secs_f64() * 1000.0,
                        plan_was_created,
                    );
                }
            } else {
                ws.ensure_host_capacity(in_bytes)?;
                ws.ensure_host_io_capacity(in_bytes.max(out_bytes))?;

                let host = unsafe {
                    std::slice::from_raw_parts_mut(ws.host_ptr as *mut f32, total)
                };

                {
                    use rayon::prelude::*;
                    host.par_chunks_mut(n)
                        .enumerate()
                        .for_each(|(frame_idx, frame_slice)| {
                            let start = frame_idx * hop_length;
                            for k in 0..n {
                                frame_slice[k] = y_padded[start + k] * window[k];
                            }
                        });
                }

                let t0 = std::time::Instant::now();
                unsafe {
                    let t_h2d = std::time::Instant::now();
                    check_cuda(
                        (ws.cuda.cuda_memcpy_async)(
                            ws.dev_r2c_in_ptr,
                            ws.host_ptr as *const c_void,
                            in_bytes,
                            CUDA_MEMCPY_HOST_TO_DEVICE,
                            ws.stream,
                        ),
                        "cudaMemcpyAsync R2C(H2D fused)",
                    )?;
                    let h2d_ms = t_h2d.elapsed().as_secs_f64() * 1000.0;
                    let t_kernel = std::time::Instant::now();
                    check_cufft(
                        (ws.cufft.cufft_exec_r2c)(ws.plan_r2c, ws.dev_r2c_in_ptr, ws.dev_r2c_out_ptr),
                        "cufftExecR2C(fused)",
                    )?;
                    let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
                    let t_d2h = std::time::Instant::now();
                    check_cuda(
                        (ws.cuda.cuda_memcpy_async)(
                            ws.host_output_ptr,
                            ws.dev_r2c_out_ptr as *const c_void,
                            out_bytes,
                            CUDA_MEMCPY_DEVICE_TO_HOST,
                            ws.stream,
                        ),
                        "cudaMemcpyAsync R2C D2H(fused)",
                    )?;
                    check_cuda(
                        (ws.cuda.cuda_stream_synchronize)(ws.stream),
                        "cudaStreamSynchronize(R2C fused)",
                    )?;
                    let sync_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;
                    super::copy_bytes_from_staging(
                        ws.host_output_ptr as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                        out_bytes,
                    );
                    let _ = h2d_ms;
                    emit_profile(
                        "r2c_cpu_pack",
                        n,
                        n_frames,
                        in_bytes,
                        kernel_ms,
                        out_bytes,
                        sync_ms,
                        t0.elapsed().as_secs_f64() * 1000.0,
                        plan_was_created,
                    );
                }
            }
            Ok(())
        })
    }

    pub fn stft_mel_power_f32_from_signal_gpu(
        y_padded: &[f32],
        window: &[f32],
        hop_length: usize,
        mel_basis: &[f32],
        output: &mut [f32],
        n: usize,
        n_frames: usize,
        n_mels: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_frames == 0 {
            return Ok(());
        }
        if window.len() != n {
            return Err(format!("window len {} != n {}", window.len(), n));
        }
        let n_bins = n / 2 + 1;
        let expected_mel_basis = n_mels
            .checked_mul(n_bins)
            .ok_or("mel basis size overflow")?;
        if mel_basis.len() < expected_mel_basis {
            return Err(format!(
                "mel_basis len {} < n_mels*n_bins {}",
                mel_basis.len(),
                expected_mel_basis
            ));
        }
        let expected_output = n_mels
            .checked_mul(n_frames)
            .ok_or("mel output size overflow")?;
        if output.len() < expected_output {
            return Err(format!(
                "output len {} < n_mels*n_frames {}",
                output.len(),
                expected_output
            ));
        }

        let needed_samples = n
            .checked_add((n_frames.saturating_sub(1)).saturating_mul(hop_length))
            .ok_or("signal size overflow")?;
        if y_padded.len() < needed_samples {
            return Err(format!(
                "y_padded len {} < required {}",
                y_padded.len(),
                needed_samples
            ));
        }

        let kernel_api = get_cuda_window_pack_api()?
            .ok_or_else(|| "CUDA fused mel helper unavailable".to_string())?;
        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;
        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let fft_in_elems = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
        let fft_out_elems = n_bins
            .checked_mul(n_frames)
            .ok_or("batched R2C output size overflow")?;

        let signal_bytes = needed_samples
            .checked_mul(mem::size_of::<f32>())
            .ok_or("signal byte-size overflow")?;
        let window_bytes = n
            .checked_mul(mem::size_of::<f32>())
            .ok_or("window byte-size overflow")?;
        let fft_in_bytes = fft_in_elems
            .checked_mul(mem::size_of::<f32>())
            .ok_or("fft input byte-size overflow")?;
        let fft_out_bytes = fft_out_elems
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("fft output byte-size overflow")?;
        let mel_out_bytes = expected_output
            .checked_mul(mem::size_of::<f32>())
            .ok_or("mel output byte-size overflow")?;

        TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }

            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();
            let use_pinned_staging = cuda_use_pinned_staging();

            ws.ensure_stream()?;
            let (signal_fp_bytes, signal_hash) = ws.signal_fingerprint(y_padded);
            if signal_fp_bytes != signal_bytes {
                return Err("signal byte-size mismatch while fingerprinting".to_string());
            }
            let (window_fp_bytes, window_hash) = ws.window_fingerprint(window);
            if window_fp_bytes != window_bytes {
                return Err("window byte-size mismatch while fingerprinting".to_string());
            }
            let signal_upload_bytes = ws.signal_upload_bytes_if_needed(y_padded);
            let window_upload_bytes = ws.window_upload_bytes_if_needed(window);
            let mel_basis_upload_bytes = ws.mel_basis_upload_bytes_if_needed(mel_basis);
            if !ws.should_use_gpu_for_stft_mel(
                signal_upload_bytes,
                window_upload_bytes,
                fft_in_bytes,
                fft_out_bytes,
                mel_basis_upload_bytes,
                mel_out_bytes,
            )? {
                return Err("CUDA GPU memory headroom insufficient for fused STFT+mel; CPU fallback".to_string());
            }
            ws.ensure_signal_capacity(signal_bytes)?;
            ws.ensure_r2c_capacity(fft_in_bytes, fft_out_bytes)?;
            ws.ensure_mel_out_capacity(mel_out_bytes)?;
            ws.ensure_plan_r2c(n, n_frames)?;
            if use_pinned_staging {
                ws.ensure_host_io_capacity(signal_bytes.max(mel_out_bytes))?;
            }
            let stft_cache_hit = ws.stft_cache_matches(
                signal_bytes,
                signal_hash,
                window_fp_bytes,
                window_hash,
                n,
                hop_length,
                n_frames,
                1,
                needed_samples,
            );
            if stft_cache_hit {
                super::cdbg("stft resident cache hit (single fused mel)");
            }

            let t0 = std::time::Instant::now();
            unsafe {
                let uploaded_signal_bytes = if stft_cache_hit {
                    0
                } else {
                    ws.upload_signal_if_needed(y_padded, use_pinned_staging)?
                };
                let uploaded_window_bytes = if stft_cache_hit {
                    0
                } else {
                    ws.upload_window_if_needed(window)?
                };
                let uploaded_mel_basis_bytes = ws.upload_mel_basis_if_needed(mel_basis)?;
                let t_kernel = std::time::Instant::now();
                if !stft_cache_hit {
                    check_cuda(
                        (kernel_api.launch_window_and_pack_f32)(
                            ws.dev_signal_ptr as *const c_void,
                            ws.dev_window_ptr as *const c_void,
                            ws.dev_r2c_in_ptr,
                            i32::try_from(n).map_err(|_| "n too large for window-pack kernel".to_string())?,
                            i32::try_from(hop_length).map_err(|_| "hop_length too large for window-pack kernel".to_string())?,
                            i32::try_from(n_frames).map_err(|_| "n_frames too large for window-pack kernel".to_string())?,
                            ws.stream,
                        ),
                        "launch_window_and_pack_f32(fused STFT+mel)",
                    )?;
                    check_cufft(
                        (ws.cufft.cufft_exec_r2c)(ws.plan_r2c, ws.dev_r2c_in_ptr, ws.dev_r2c_out_ptr),
                        "cufftExecR2C(fused STFT+mel)",
                    )?;
                    ws.update_stft_cache(
                        signal_bytes,
                        signal_hash,
                        window_fp_bytes,
                        window_hash,
                        n,
                        hop_length,
                        n_frames,
                        1,
                        needed_samples,
                    );
                }
                check_cuda(
                    (kernel_api.launch_mel_project_power_f32)(
                        ws.dev_r2c_out_ptr as *const c_void,
                        ws.dev_mel_basis_ptr as *const c_void,
                        ws.dev_mel_out_ptr,
                        i32::try_from(n_bins).map_err(|_| "n_bins too large for mel kernel".to_string())?,
                        i32::try_from(n_frames).map_err(|_| "n_frames too large for mel kernel".to_string())?,
                        i32::try_from(n_mels).map_err(|_| "n_mels too large for mel kernel".to_string())?,
                        ws.stream,
                    ),
                    "launch_mel_project_power_f32",
                )?;
                let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
                let t_d2h = std::time::Instant::now();
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        if use_pinned_staging {
                            ws.host_output_ptr
                        } else {
                            output.as_mut_ptr() as *mut c_void
                        },
                        ws.dev_mel_out_ptr as *const c_void,
                        mel_out_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync mel D2H(fused STFT+mel)",
                )?;
                check_cuda(
                    (ws.cuda.cuda_stream_synchronize)(ws.stream),
                    "cudaStreamSynchronize(fused STFT+mel)",
                )?;
                let sync_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;
                if use_pinned_staging {
                    super::copy_bytes_from_staging(
                        ws.host_output_ptr as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                        mel_out_bytes,
                    );
                }
                emit_profile(
                    if stft_cache_hit {
                        "stft_mel_fused_stft_cached"
                    } else {
                        "stft_mel_fused"
                    },
                    n,
                    n_frames,
                    uploaded_signal_bytes + uploaded_window_bytes + uploaded_mel_basis_bytes,
                    kernel_ms,
                    mel_out_bytes,
                    sync_ms,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    false,
                );
            }

            Ok(())
        })
    }

    pub fn stft_mel_power_f32_from_signal_batch_gpu(
        y_padded_batch: &[f32],
        n_channels: usize,
        n_samples_per_channel: usize,
        window: &[f32],
        hop_length: usize,
        mel_basis: &[f32],
        output: &mut [f32],
        n: usize,
        n_frames: usize,
        n_mels: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_channels == 0 {
            return Ok(());
        }
        if n_frames == 0 {
            return Ok(());
        }
        if window.len() != n {
            return Err(format!("window len {} != n {}", window.len(), n));
        }

        let expected_signal = n_channels
            .checked_mul(n_samples_per_channel)
            .ok_or("batched signal size overflow")?;
        if y_padded_batch.len() < expected_signal {
            return Err(format!(
                "y_padded_batch len {} < expected {}",
                y_padded_batch.len(),
                expected_signal
            ));
        }

        let n_bins = n / 2 + 1;
        let expected_mel_basis = n_mels
            .checked_mul(n_bins)
            .ok_or("mel basis size overflow")?;
        if mel_basis.len() < expected_mel_basis {
            return Err(format!(
                "mel_basis len {} < n_mels*n_bins {}",
                mel_basis.len(),
                expected_mel_basis
            ));
        }
        let expected_output = n_channels
            .checked_mul(n_mels)
            .and_then(|v| v.checked_mul(n_frames))
            .ok_or("batched mel output size overflow")?;
        if output.len() < expected_output {
            return Err(format!(
                "output len {} < n_channels*n_mels*n_frames {}",
                output.len(),
                expected_output
            ));
        }

        let total_fft_frames = n_channels
            .checked_mul(n_frames)
            .ok_or("batched frame count overflow")?;
        let fft_in_elems = n.checked_mul(total_fft_frames).ok_or("batched FFT size overflow")?;
        let fft_out_elems = n_bins
            .checked_mul(total_fft_frames)
            .ok_or("batched R2C output size overflow")?;

        let kernel_api = get_cuda_window_pack_api()?
            .ok_or_else(|| "CUDA fused mel helper unavailable".to_string())?;
        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;
        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let signal_bytes = expected_signal
            .checked_mul(mem::size_of::<f32>())
            .ok_or("signal byte-size overflow")?;
        let window_bytes = n
            .checked_mul(mem::size_of::<f32>())
            .ok_or("window byte-size overflow")?;
        let fft_in_bytes = fft_in_elems
            .checked_mul(mem::size_of::<f32>())
            .ok_or("fft input byte-size overflow")?;
        let fft_out_bytes = fft_out_elems
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("fft output byte-size overflow")?;
        let mel_out_bytes = expected_output
            .checked_mul(mem::size_of::<f32>())
            .ok_or("mel output byte-size overflow")?;

        TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }

            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();
            let use_pinned_staging = cuda_use_pinned_staging();

            ws.ensure_stream()?;
            let (signal_fp_bytes, signal_hash) = ws.signal_fingerprint(y_padded_batch);
            if signal_fp_bytes != signal_bytes {
                return Err("batched signal byte-size mismatch while fingerprinting".to_string());
            }
            let (window_fp_bytes, window_hash) = ws.window_fingerprint(window);
            if window_fp_bytes != window_bytes {
                return Err("window byte-size mismatch while fingerprinting".to_string());
            }
            let signal_upload_bytes = ws.signal_upload_bytes_if_needed(y_padded_batch);
            let window_upload_bytes = ws.window_upload_bytes_if_needed(window);
            let mel_basis_upload_bytes = ws.mel_basis_upload_bytes_if_needed(mel_basis);
            if !ws.should_use_gpu_for_stft_mel(
                signal_upload_bytes,
                window_upload_bytes,
                fft_in_bytes,
                fft_out_bytes,
                mel_basis_upload_bytes,
                mel_out_bytes,
            )? {
                return Err(
                    "CUDA GPU memory headroom insufficient for fused STFT+mel batch; CPU fallback"
                        .to_string(),
                );
            }

            ws.ensure_signal_capacity(signal_bytes)?;
            ws.ensure_r2c_capacity(fft_in_bytes, fft_out_bytes)?;
            ws.ensure_mel_out_capacity(mel_out_bytes)?;
            ws.ensure_plan_r2c(n, total_fft_frames)?;
            if use_pinned_staging {
                ws.ensure_host_io_capacity(signal_bytes.max(mel_out_bytes))?;
            }
            let stft_cache_hit = ws.stft_cache_matches(
                signal_bytes,
                signal_hash,
                window_fp_bytes,
                window_hash,
                n,
                hop_length,
                n_frames,
                n_channels,
                n_samples_per_channel,
            );
            if stft_cache_hit {
                super::cdbg("stft resident cache hit (batch fused mel)");
            }

            let t0 = std::time::Instant::now();
            unsafe {
                let uploaded_signal_bytes = if stft_cache_hit {
                    0
                } else {
                    ws.upload_signal_if_needed(y_padded_batch, use_pinned_staging)?
                };
                let uploaded_window_bytes = if stft_cache_hit {
                    0
                } else {
                    ws.upload_window_if_needed(window)?
                };
                let uploaded_mel_basis_bytes = ws.upload_mel_basis_if_needed(mel_basis)?;
                let t_kernel = std::time::Instant::now();
                if !stft_cache_hit {
                    check_cuda(
                        (kernel_api.launch_window_and_pack_batch_f32)(
                            ws.dev_signal_ptr as *const c_void,
                            ws.dev_window_ptr as *const c_void,
                            ws.dev_r2c_in_ptr,
                            i32::try_from(n_samples_per_channel)
                                .map_err(|_| "n_samples too large for batch window-pack kernel".to_string())?,
                            i32::try_from(n).map_err(|_| "n too large for batch window-pack kernel".to_string())?,
                            i32::try_from(hop_length)
                                .map_err(|_| "hop_length too large for batch window-pack kernel".to_string())?,
                            i32::try_from(n_frames)
                                .map_err(|_| "n_frames too large for batch window-pack kernel".to_string())?,
                            i32::try_from(n_channels)
                                .map_err(|_| "n_channels too large for batch window-pack kernel".to_string())?,
                            ws.stream,
                        ),
                        "launch_window_and_pack_batch_f32(fused STFT+mel)",
                    )?;
                    check_cufft(
                        (ws.cufft.cufft_exec_r2c)(ws.plan_r2c, ws.dev_r2c_in_ptr, ws.dev_r2c_out_ptr),
                        "cufftExecR2C(fused STFT+mel batch)",
                    )?;
                    ws.update_stft_cache(
                        signal_bytes,
                        signal_hash,
                        window_fp_bytes,
                        window_hash,
                        n,
                        hop_length,
                        n_frames,
                        n_channels,
                        n_samples_per_channel,
                    );
                }
                check_cuda(
                    (kernel_api.launch_mel_project_power_batch_f32)(
                        ws.dev_r2c_out_ptr as *const c_void,
                        ws.dev_mel_basis_ptr as *const c_void,
                        ws.dev_mel_out_ptr,
                        i32::try_from(n_bins)
                            .map_err(|_| "n_bins too large for batch mel kernel".to_string())?,
                        i32::try_from(n_frames)
                            .map_err(|_| "n_frames too large for batch mel kernel".to_string())?,
                        i32::try_from(n_mels)
                            .map_err(|_| "n_mels too large for batch mel kernel".to_string())?,
                        i32::try_from(n_channels)
                            .map_err(|_| "n_channels too large for batch mel kernel".to_string())?,
                        ws.stream,
                    ),
                    "launch_mel_project_power_batch_f32",
                )?;
                let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;
                let t_d2h = std::time::Instant::now();
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        if use_pinned_staging {
                            ws.host_output_ptr
                        } else {
                            output.as_mut_ptr() as *mut c_void
                        },
                        ws.dev_mel_out_ptr as *const c_void,
                        mel_out_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync mel D2H(fused STFT+mel batch)",
                )?;
                check_cuda(
                    (ws.cuda.cuda_stream_synchronize)(ws.stream),
                    "cudaStreamSynchronize(fused STFT+mel batch)",
                )?;
                let sync_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;
                if use_pinned_staging {
                    super::copy_bytes_from_staging(
                        ws.host_output_ptr as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                        mel_out_bytes,
                    );
                }
                emit_profile(
                    if stft_cache_hit {
                        "stft_mel_fused_batch_stft_cached"
                    } else {
                        "stft_mel_fused_batch"
                    },
                    n,
                    total_fft_frames,
                    uploaded_signal_bytes + uploaded_window_bytes + uploaded_mel_basis_bytes,
                    kernel_ms,
                    mel_out_bytes,
                    sync_ms,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    false,
                );
            }

            Ok(())
        })
    }

    pub fn fft_inverse_c2r_batched_gpu(
        input: &[Complex<f32>],
        output: &mut [f32],
        n: usize,
        n_frames: usize,
    ) -> Result<(), String> {
        if n == 0 {
            return Err("FFT size must be > 0".to_string());
        }
        if n_frames == 0 {
            return Ok(());
        }
        let n_bins = n / 2 + 1;
        let in_total = n_bins
            .checked_mul(n_frames)
            .ok_or("batched C2R input size overflow")?;
        if input.len() < in_total {
            return Err(format!("input len {} < n_bins*n_frames {}", input.len(), in_total));
        }
        let out_total = n
            .checked_mul(n_frames)
            .ok_or("batched C2R output size overflow")?;
        if output.len() < out_total {
            return Err(format!("output len {} < n*n_frames {}", output.len(), out_total));
        }

        let cuda = get_cuda_api()?;
        let cufft = get_cufft_api()?;
        if !cuda_runtime_available(&cuda) {
            return Err("CUDA runtime unavailable (no devices)".to_string());
        }

        let in_bytes = in_total
            .checked_mul(mem::size_of::<Complex<f32>>())
            .ok_or("C2R in byte-size overflow")?;
        let out_bytes = out_total
            .checked_mul(mem::size_of::<f32>())
            .ok_or("C2R out byte-size overflow")?;

        TL_CUDA_WS.with(|cell| -> Result<(), String> {
            if cell.borrow().is_none() {
                *cell.borrow_mut() = Some(CudaWorkspace::new(cuda.clone(), cufft.clone()));
            }

            let mut ws_ref = cell.borrow_mut();
            let ws = ws_ref.as_mut().unwrap();
            ws.cuda = cuda.clone();
            ws.cufft = cufft.clone();

            ws.ensure_stream()?;
            let use_pinned_staging = cuda_use_pinned_staging();
            if !ws.should_use_gpu_for_split_io(in_bytes, out_bytes)? {
                return Err("CUDA GPU memory headroom insufficient for C2R; CPU fallback".to_string());
            }
            ws.ensure_c2r_capacity(in_bytes, out_bytes)?;
            if use_pinned_staging {
                ws.ensure_host_io_capacity(in_bytes.max(out_bytes))?;
            }
            let plan_was_created =
                ws.plan_c2r == 0 || ws.plan_c2r_n != n || ws.plan_c2r_batch != n_frames;
            ws.ensure_plan_c2r(n, n_frames)?;

            let t0 = std::time::Instant::now();
            unsafe {
                if use_pinned_staging {
                    super::copy_bytes_to_staging(
                        input.as_ptr() as *const c_void,
                        ws.host_input_ptr,
                        in_bytes,
                    );
                }
                let t_h2d = std::time::Instant::now();
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        ws.dev_c2r_in_ptr,
                        if use_pinned_staging {
                            ws.host_input_ptr as *const c_void
                        } else {
                            input.as_ptr() as *const c_void
                        },
                        in_bytes,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync C2R H2D",
                )?;
                let h2d_ms = t_h2d.elapsed().as_secs_f64() * 1000.0;

                let t_kernel = std::time::Instant::now();
                check_cufft(
                    (ws.cufft.cufft_exec_c2r)(ws.plan_c2r, ws.dev_c2r_in_ptr, ws.dev_c2r_out_ptr),
                    "cufftExecC2R",
                )?;
                let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;

                let t_d2h = std::time::Instant::now();
                check_cuda(
                    (ws.cuda.cuda_memcpy_async)(
                        if use_pinned_staging {
                            ws.host_output_ptr
                        } else {
                            output.as_mut_ptr() as *mut c_void
                        },
                        ws.dev_c2r_out_ptr as *const c_void,
                        out_bytes,
                        CUDA_MEMCPY_DEVICE_TO_HOST,
                        ws.stream,
                    ),
                    "cudaMemcpyAsync C2R D2H",
                )?;
                check_cuda(
                    (ws.cuda.cuda_stream_synchronize)(ws.stream),
                    "cudaStreamSynchronize(C2R)",
                )?;
                let sync_ms = t_d2h.elapsed().as_secs_f64() * 1000.0;
                if use_pinned_staging {
                    super::copy_bytes_from_staging(
                        ws.host_output_ptr as *const c_void,
                        output.as_mut_ptr() as *mut c_void,
                        out_bytes,
                    );
                }

                let _ = h2d_ms; // keep in debug/prof context while profiling by op breakdown
                emit_profile(
                    "c2r",
                    n,
                    n_frames,
                    in_bytes,
                    kernel_ms,
                    out_bytes,
                    sync_ms,
                    t0.elapsed().as_secs_f64() * 1000.0,
                    plan_was_created,
                );
            }
            Ok(())
        })
    }

    /// Diagnostic probe: returns (dll_name, loaded, error_msg) for each candidate.
    pub fn probe_dlls() -> Vec<(String, bool, String)> {
        let cudart_candidates = [
            "cudart64_130.dll", "cudart64_13.dll", "cudart64_12.dll",
            "cudart64_11080.dll", "cudart64_110.dll",
            "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0", "libcudart.so",
        ];
        let cufft_candidates = [
            "cufft64_11.dll", "cufft64_13.dll", "cufft64_12.dll", "cufft64_10.dll",
            "libcufft.so.11", "libcufft.so.10", "libcufft.so",
        ];
        let mut results = Vec::new();
        for name in cudart_candidates.iter().chain(cufft_candidates.iter()) {
            match unsafe { libloading::Library::new(name) } {
                Ok(_) => results.push((name.to_string(), true, String::new())),
                Err(e) => results.push((name.to_string(), false, e.to_string())),
            }
        }
        results
    }

    /// Return device count from CUDA runtime, or error string.
    pub fn device_count_str() -> String {
        match load_cuda_api() {
            Err(e) => format!("cudart load failed: {}", e),
            Ok(api) => {
                let mut count = 0i32;
                let code = unsafe { (api.cuda_get_device_count)(&mut count) };
                if code == 0 {
                    format!("{}", count)
                } else {
                    format!("cudaGetDeviceCount error {}", code)
                }
            }
        }
    }

    pub fn fused_stft_mel_power_f32_gpu(
        y_padded: &[f32],
        window: &[f32],
        hop_length: usize,
        mel_basis: &[f32],
        output: &mut [f32],
        n: usize,
        n_frames: usize,
        n_mels: usize,
    ) -> Result<(), String> {
        stft_mel_power_f32_from_signal_gpu(
            y_padded,
            window,
            hop_length,
            mel_basis,
            output,
            n,
            n_frames,
            n_mels,
        )
    }

    pub fn fused_stft_mel_power_batch_f32_gpu(
        y_padded_batch: &[f32],
        n_channels: usize,
        n_samples_per_channel: usize,
        window: &[f32],
        hop_length: usize,
        mel_basis: &[f32],
        output: &mut [f32],
        n: usize,
        n_frames: usize,
        n_mels: usize,
    ) -> Result<(), String> {
        stft_mel_power_f32_from_signal_batch_gpu(
            y_padded_batch,
            n_channels,
            n_samples_per_channel,
            window,
            hop_length,
            mel_basis,
            output,
            n,
            n_frames,
            n_mels,
        )
    }
}

// ── Non-feature-gated stub (when cuda-gpu not compiled) ───────────────────────

#[cfg(not(feature = "cuda-gpu"))]
pub mod cuda_fft_impl {
    use rustfft::num_complex::Complex;

    pub fn fft_forward_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fft_inverse_batched_gpu(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fft_forward_r2c_batched_gpu(
        _input: &[f32],
        _output: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fft_inverse_c2r_batched_gpu(
        _input: &[Complex<f32>],
        _output: &mut [f32],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fft_forward_r2c_from_signal_gpu(
        _y_padded: &[f32],
        _window: &[f32],
        _hop_length: usize,
        _output: &mut [Complex<f32>],
        _n: usize,
        _n_frames: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fft_forward_two_stream_pipeline(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_chunks: usize,
        _chunk_size: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fft_inverse_two_stream_pipeline(
        _buffer: &mut [Complex<f32>],
        _n: usize,
        _n_chunks: usize,
        _chunk_size: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn probe_dlls() -> Vec<(String, bool, String)> { Vec::new() }

    pub fn device_count_str() -> String {
        "cuda-gpu feature not compiled".to_string()
    }

    pub fn fused_stft_mel_power_f32_gpu(
        _y_padded: &[f32],
        _window: &[f32],
        _hop_length: usize,
        _mel_basis: &[f32],
        _output: &mut [f32],
        _n: usize,
        _n_frames: usize,
        _n_mels: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }

    pub fn fused_stft_mel_power_batch_f32_gpu(
        _y_padded_batch: &[f32],
        _n_channels: usize,
        _n_samples_per_channel: usize,
        _window: &[f32],
        _hop_length: usize,
        _mel_basis: &[f32],
        _output: &mut [f32],
        _n: usize,
        _n_frames: usize,
        _n_mels: usize,
    ) -> Result<(), String> {
        Err("cuda-gpu feature not compiled in — rebuild with --features cuda-gpu".to_string())
    }
}

pub fn fused_stft_mel_power_f32_gpu(
    y_padded: &[f32],
    window: &[f32],
    hop_length: usize,
    mel_basis: &[f32],
    output: &mut [f32],
    n: usize,
    n_frames: usize,
    n_mels: usize,
) -> Result<(), String> {
    cuda_fft_impl::fused_stft_mel_power_f32_gpu(
        y_padded,
        window,
        hop_length,
        mel_basis,
        output,
        n,
        n_frames,
        n_mels,
    )
}

pub fn fused_stft_mel_power_batch_f32_gpu(
    y_padded_batch: &[f32],
    n_channels: usize,
    n_samples_per_channel: usize,
    window: &[f32],
    hop_length: usize,
    mel_basis: &[f32],
    output: &mut [f32],
    n: usize,
    n_frames: usize,
    n_mels: usize,
) -> Result<(), String> {
    cuda_fft_impl::fused_stft_mel_power_batch_f32_gpu(
        y_padded_batch,
        n_channels,
        n_samples_per_channel,
        window,
        hop_length,
        mel_basis,
        output,
        n,
        n_frames,
        n_mels,
    )
}

// ── Public diagnostics (always available) ─────────────────────────────────────

/// Collect CUDA diagnostics info as a human-readable multi-line string.
pub fn cuda_diagnostics_info() -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "cuda-gpu feature compiled: {}\n",
        cfg!(feature = "cuda-gpu")
    ));
    out.push_str(&format!(
        "cuda window-pack helper built: {}\n",
        cuda_window_pack_helper_built()
    ));
    if let Some(path) = cuda_window_pack_helper_path() {
        out.push_str(&format!("cuda window-pack helper path: {}\n", path));
    }
    out.push_str(&format!(
        "cuda fused-mel helper built: {}\n",
        cuda_fused_mel_helper_built()
    ));

    let smi = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output();
    match smi {
        Ok(o) if o.status.success() => {
            out.push_str(&format!(
                "nvidia-smi: OK\n{}\n",
                String::from_utf8_lossy(&o.stdout).trim()
            ));
        }
        Ok(o) => {
            out.push_str(&format!(
                "nvidia-smi: exit {}\n",
                o.status.code().unwrap_or(-1)
            ));
        }
        Err(e) => out.push_str(&format!("nvidia-smi: not found ({})\n", e)),
    }

    out.push_str("DLL probes:\n");
    for (name, ok, err) in cuda_fft_impl::probe_dlls() {
        if ok {
            out.push_str(&format!("  [OK] {}\n", name));
        } else {
            let e = &err[..err.len().min(80)];
            out.push_str(&format!("  [--] {}  ({})\n", name, e));
        }
    }

    out.push_str(&format!("device count: {}\n", cuda_fft_impl::device_count_str()));
    out
}

// ── Efficient CPU fallback via thread-local cached planner ────────────────────

use std::cell::RefCell;
use std::sync::Arc;

thread_local! {
    static TL_FB_PLANNER: RefCell<rustfft::FftPlanner<f32>> =
        RefCell::new(rustfft::FftPlanner::new());
    static TL_FB_FWD: RefCell<Option<(usize, Arc<dyn rustfft::Fft<f32>>)>> =
        const { RefCell::new(None) };
    static TL_FB_INV: RefCell<Option<(usize, Arc<dyn rustfft::Fft<f32>>)>> =
        const { RefCell::new(None) };
}

fn cpu_fallback_forward_parallel(buffer: &mut [Complex<f32>], n: usize) {
    use rayon::prelude::*;
    let fft = TL_FB_FWD.with(|cell| {
        let mut cached = cell.borrow_mut();
        if cached.as_ref().map(|(sz, _)| *sz) != Some(n) {
            let plan = TL_FB_PLANNER.with(|p| p.borrow_mut().plan_fft_forward(n));
            *cached = Some((n, plan));
        }
        cached.as_ref().unwrap().1.clone()
    });
    buffer.par_chunks_mut(n).for_each(|chunk| fft.process(chunk));
}

fn cpu_fallback_inverse_parallel(buffer: &mut [Complex<f32>], n: usize) {
    use rayon::prelude::*;
    let fft = TL_FB_INV.with(|cell| {
        let mut cached = cell.borrow_mut();
        if cached.as_ref().map(|(sz, _)| *sz) != Some(n) {
            let plan = TL_FB_PLANNER.with(|p| p.borrow_mut().plan_fft_inverse(n));
            *cached = Some((n, plan));
        }
        cached.as_ref().unwrap().1.clone()
    });
    buffer.par_chunks_mut(n).for_each(|chunk| fft.process(chunk));
}

// ── Chunk-size heuristic ──────────────────────────────────────────────────────

/// Whether the two-stream overlap pipeline is enabled (default: on).
/// Set `IRON_LIBROSA_CUDA_TWO_STREAM=0` to disable.
fn cuda_two_stream_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("IRON_LIBROSA_CUDA_TWO_STREAM")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "no" | "off"
        )
    })
}

fn cuda_two_stream_min_frames() -> usize {
    static MIN_FRAMES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_FRAMES.get_or_init(|| {
        std::env::var("IRON_LIBROSA_CUDA_TWO_STREAM_MIN_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(256)
    })
}

fn cuda_two_stream_min_elems() -> usize {
    static MIN_ELEMS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN_ELEMS.get_or_init(|| {
        std::env::var("IRON_LIBROSA_CUDA_TWO_STREAM_MIN_ELEMS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(262_144)
    })
}

fn two_stream_meets_workload_gate(
    n_fft: usize,
    n_frames: usize,
    min_frames: usize,
    min_elems: usize,
) -> bool {
    if n_fft == 0 {
        return false;
    }
    n_frames >= min_frames && n_fft.saturating_mul(n_frames) >= min_elems
}

fn should_use_two_stream_pipeline(n_fft: usize, n_frames: usize) -> bool {
    cuda_two_stream_enabled()
        && two_stream_meets_workload_gate(
            n_fft,
            n_frames,
            cuda_two_stream_min_frames(),
            cuda_two_stream_min_elems(),
        )
}

fn cuda_chunk_size(n_fft: usize, n_frames: usize) -> usize {
    if let Ok(s) = std::env::var("IRON_LIBROSA_CUDA_FFT_BATCH_CHUNK_SIZE") {
        if let Ok(v) = s.trim().parse::<usize>() {
            if v > 0 {
                return v.min(n_frames).max(1);
            }
        }
    }
    let _ = n_fft;
    n_frames.max(1) // send the full batch to GPU in one shot
}

fn cuda_max_work_host_guard() -> Option<usize> {
    std::env::var("IRON_LIBROSA_CUDA_MAX_WORK")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|v| *v > 0)
}

// ── Public high-level API ─────────────────────────────────────────────────────

/// Attempt batched forward FFT on CUDA; fall back to CPU on any error.
pub fn fft_forward_batched_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let total = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
    if buffer.len() < total {
        return Err(format!("buffer too small: {} < {}", buffer.len(), total));
    }

    let under_limit = cuda_max_work_host_guard()
        .map(|mx| total <= mx)
        .unwrap_or(true);

    if under_limit {
        match cuda_fft_impl::fft_forward_batched_gpu(buffer, n, n_frames) {
            Ok(()) => {
                cdbg(&format!("fwd GPU OK  n={} frames={}", n, n_frames));
                return Ok(());
            }
            Err(e) => {
                cdbg(&format!("fwd GPU err '{}' → CPU fallback", e));
            }
        }
    } else {
        cdbg("fwd: over max_work → CPU fallback");
    }

    cpu_fallback_forward_parallel(buffer, n);
    Ok(())
}

/// Attempt batched inverse FFT on CUDA; fall back to CPU on any error.
pub fn fft_inverse_batched_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let total = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
    if buffer.len() < total {
        return Err(format!("buffer too small: {} < {}", buffer.len(), total));
    }

    let under_limit = cuda_max_work_host_guard()
        .map(|mx| total <= mx)
        .unwrap_or(true);

    if under_limit {
        match cuda_fft_impl::fft_inverse_batched_gpu(buffer, n, n_frames) {
            Ok(()) => {
                cdbg(&format!("inv GPU OK  n={} frames={}", n, n_frames));
                return Ok(());
            }
            Err(e) => {
                cdbg(&format!("inv GPU err '{}' → CPU fallback", e));
            }
        }
    }

    cpu_fallback_inverse_parallel(buffer, n);
    Ok(())
}

/// Attempt batched forward R2C FFT on CUDA; fall back to CPU complex FFT on error.
/// Input layout: frame-major real buffer with shape (n_frames, n).
/// Output layout: frame-major complex buffer with shape (n_frames, n/2+1).
pub fn fft_forward_real_batched_with_fallback(
    input: &[f32],
    output: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let total = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
    if input.len() < total {
        return Err(format!("input too small: {} < {}", input.len(), total));
    }
    let n_bins = n / 2 + 1;
    let out_total = n_bins
        .checked_mul(n_frames)
        .ok_or("batched R2C output size overflow")?;
    if output.len() < out_total {
        return Err(format!("output too small: {} < {}", output.len(), out_total));
    }

    let under_limit = cuda_max_work_host_guard()
        .map(|mx| total <= mx)
        .unwrap_or(true);
    if under_limit {
        match cuda_fft_impl::fft_forward_r2c_batched_gpu(input, output, n, n_frames) {
            Ok(()) => return Ok(()),
            Err(e) => cdbg(&format!("r2c GPU err '{}' → CPU fallback", e)),
        }
    }

    let mut tmp = vec![Complex::<f32>::default(); total];
    for (idx, x) in input.iter().take(total).enumerate() {
        tmp[idx] = Complex::new(*x, 0.0);
    }
    cpu_fallback_forward_parallel(&mut tmp, n);
    for frame in 0..n_frames {
        let in_off = frame * n;
        let out_off = frame * n_bins;
        output[out_off..(out_off + n_bins)].copy_from_slice(&tmp[in_off..(in_off + n_bins)]);
    }
    Ok(())
}

/// Chunked batched forward FFT — mirrors metal_fft.rs API.
/// Uses a two-stream overlap pipeline by default when n_frames >= 4,
/// falling back to the single-stream per-chunk loop on any error.
pub fn fft_forward_batched_chunked_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }

    // Two-stream pipeline: split the batch in half so stream B's H2D transfer
    // overlaps with stream A's FFT kernel.  Requires at least 2 full chunks.
    if should_use_two_stream_pipeline(n, n_frames) {
        let chunk = (n_frames / 2).max(1);
        let n_full_chunks = n_frames / chunk;
        let remainder    = n_frames % chunk;
        if n_full_chunks >= 2 {
            let full_frames = n_full_chunks * chunk;
            match cuda_fft_impl::fft_forward_two_stream_pipeline(
                &mut buffer[..full_frames * n],
                n,
                n_full_chunks,
                chunk,
            ) {
                Ok(()) => {
                    if remainder > 0 {
                        fft_forward_batched_with_fallback(
                            &mut buffer[full_frames * n..],
                            n,
                            remainder,
                        )?;
                    }
                    return Ok(());
                }
                Err(e) => {
                    cdbg(&format!("two-stream fwd err '{}' → single-stream fallback", e));
                }
            }
        }
    }

    // Fallback: existing single-stream per-chunk loop.
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);
    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let start = frame_start * n;
        let end = start + frames * n;
        fft_forward_batched_with_fallback(&mut buffer[start..end], n, frames)?;
        frame_start += frames;
    }
    Ok(())
}

/// Chunked batched forward R2C FFT — frame-major input/output, mirrors CUDA chunk policy.
pub fn fft_forward_real_batched_chunked_with_fallback(
    input: &[f32],
    output: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);
    let n_bins = n / 2 + 1;
    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let in_start = frame_start * n;
        let in_end = in_start + frames * n;
        let out_start = frame_start * n_bins;
        let out_end = out_start + frames * n_bins;
        fft_forward_real_batched_with_fallback(
            &input[in_start..in_end],
            &mut output[out_start..out_end],
            n,
            frames,
        )?;
        frame_start += frames;
    }
    Ok(())
}

/// Fused STFT-prep path: build frame-windowed real input from signal + window and run batched R2C.
pub fn fft_forward_real_from_signal_chunked_with_fallback(
    y_padded: &[f32],
    window: &[f32],
    hop_length: usize,
    output: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    if window.len() != n {
        return Err(format!("window len {} != n {}", window.len(), n));
    }
    let n_bins = n / 2 + 1;
    let out_total = n_bins
        .checked_mul(n_frames)
        .ok_or("batched R2C output size overflow")?;
    if output.len() < out_total {
        return Err(format!("output too small: {} < {}", output.len(), out_total));
    }

    let required_samples = n
        .checked_add((n_frames.saturating_sub(1)).saturating_mul(hop_length))
        .ok_or("signal size overflow")?;
    if y_padded.len() < required_samples {
        return Err(format!(
            "y_padded too small: {} < {}",
            y_padded.len(),
            required_samples
        ));
    }

    let total = n.checked_mul(n_frames).ok_or("batched FFT size overflow")?;
    let under_limit = cuda_max_work_host_guard()
        .map(|mx| total <= mx)
        .unwrap_or(true);
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);

    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let y_start = frame_start * hop_length;
        let y_end = y_start + (frames - 1) * hop_length + n;
        let out_start = frame_start * n_bins;
        let out_end = out_start + frames * n_bins;

        if under_limit {
            match cuda_fft_impl::fft_forward_r2c_from_signal_gpu(
                &y_padded[y_start..y_end],
                window,
                hop_length,
                &mut output[out_start..out_end],
                n,
                frames,
            ) {
                Ok(()) => {}
                Err(e) => {
                    cdbg(&format!("fused r2c GPU err '{}' → CPU fallback", e));
                    let mut tmp_in = vec![0.0f32; n * frames];
                    for f in 0..frames {
                        let s = y_start + f * hop_length;
                        let off = f * n;
                        for k in 0..n {
                            tmp_in[off + k] = y_padded[s + k] * window[k];
                        }
                    }
                    fft_forward_real_batched_with_fallback(
                        &tmp_in,
                        &mut output[out_start..out_end],
                        n,
                        frames,
                    )?;
                }
            }
        } else {
            let mut tmp_in = vec![0.0f32; n * frames];
            for f in 0..frames {
                let s = y_start + f * hop_length;
                let off = f * n;
                for k in 0..n {
                    tmp_in[off + k] = y_padded[s + k] * window[k];
                }
            }
            fft_forward_real_batched_with_fallback(
                &tmp_in,
                &mut output[out_start..out_end],
                n,
                frames,
            )?;
        }

        frame_start += frames;
    }

    Ok(())
}

/// Attempt batched inverse C2R FFT on CUDA; CPU fallback reconstructs full spectrum then inverse FFT.
pub fn fft_inverse_real_batched_with_fallback(
    input: &[Complex<f32>],
    output: &mut [f32],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let n_bins = n / 2 + 1;
    let in_total = n_bins.checked_mul(n_frames).ok_or("batched C2R input size overflow")?;
    if input.len() < in_total {
        return Err(format!("input too small: {} < {}", input.len(), in_total));
    }
    let out_total = n.checked_mul(n_frames).ok_or("batched C2R output size overflow")?;
    if output.len() < out_total {
        return Err(format!("output too small: {} < {}", output.len(), out_total));
    }

    let under_limit = cuda_max_work_host_guard()
        .map(|mx| out_total <= mx)
        .unwrap_or(true);
    if under_limit {
        match cuda_fft_impl::fft_inverse_c2r_batched_gpu(input, output, n, n_frames) {
            Ok(()) => return Ok(()),
            Err(e) => cdbg(&format!("c2r GPU err '{}' → CPU fallback", e)),
        }
    }

    let mut tmp = vec![Complex::<f32>::default(); out_total];
    for frame in 0..n_frames {
        let in_off = frame * n_bins;
        let out_off = frame * n;
        tmp[out_off..(out_off + n_bins)].copy_from_slice(&input[in_off..(in_off + n_bins)]);
        if n % 2 == 0 {
            for i in 1..n / 2 {
                tmp[out_off + (n - i)] = tmp[out_off + i].conj();
            }
        } else {
            for i in 1..(n + 1) / 2 {
                tmp[out_off + (n - i)] = tmp[out_off + i].conj();
            }
        }
    }
    cpu_fallback_inverse_parallel(&mut tmp, n);
    for (idx, z) in tmp.iter().enumerate().take(out_total) {
        output[idx] = z.re;
    }
    Ok(())
}

/// Chunked batched inverse C2R FFT.
pub fn fft_inverse_real_batched_chunked_with_fallback(
    input: &[Complex<f32>],
    output: &mut [f32],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);
    let n_bins = n / 2 + 1;
    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let in_start = frame_start * n_bins;
        let in_end = in_start + frames * n_bins;
        let out_start = frame_start * n;
        let out_end = out_start + frames * n;
        fft_inverse_real_batched_with_fallback(
            &input[in_start..in_end],
            &mut output[out_start..out_end],
            n,
            frames,
        )?;
        frame_start += frames;
    }
    Ok(())
}

/// Chunked batched inverse FFT — mirrors metal_fft.rs API.
/// Uses the two-stream overlap pipeline by default when n_frames >= 4;
/// falls back to single-stream per-chunk loop on any CUDA error.
pub fn fft_inverse_batched_chunked_with_fallback(
    buffer: &mut [Complex<f32>],
    n: usize,
    n_frames: usize,
) -> Result<(), String> {
    if n == 0 {
        return Err("FFT size must be > 0".to_string());
    }
    if n_frames == 0 {
        return Ok(());
    }

    // Two-stream pipeline: split batch in half (same rationale as forward path).
    if should_use_two_stream_pipeline(n, n_frames) {
        let chunk = (n_frames / 2).max(1);
        let n_full_chunks = n_frames / chunk;
        let remainder    = n_frames % chunk;
        if n_full_chunks >= 2 {
            let full_frames = n_full_chunks * chunk;
            match cuda_fft_impl::fft_inverse_two_stream_pipeline(
                &mut buffer[..full_frames * n],
                n,
                n_full_chunks,
                chunk,
            ) {
                Ok(()) => {
                    if remainder > 0 {
                        fft_inverse_batched_with_fallback(
                            &mut buffer[full_frames * n..],
                            n,
                            remainder,
                        )?;
                    }
                    return Ok(());
                }
                Err(e) => {
                    cdbg(&format!("two-stream inv err '{}' → single-stream fallback", e));
                }
            }
        }
    }

    // Fallback: existing single-stream per-chunk loop.
    let chunk = cuda_chunk_size(n, n_frames).min(n_frames).max(1);
    let mut frame_start = 0usize;
    while frame_start < n_frames {
        let frames = (n_frames - frame_start).min(chunk);
        let start = frame_start * n;
        let end = start + frames * n;
        fft_inverse_batched_with_fallback(&mut buffer[start..end], n, frames)?;
        frame_start += frames;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c2c_required_headroom_basic() {
        assert_eq!(c2c_required_headroom(0), 0);
        assert_eq!(c2c_required_headroom(64), 128);
        assert_eq!(c2c_required_headroom(1024), 2048);
    }

    #[test]
    fn test_c2c_required_headroom_saturates() {
        assert_eq!(c2c_required_headroom(usize::MAX), usize::MAX);
        assert_eq!(c2c_required_headroom(usize::MAX / 2 + 1), usize::MAX);
    }

    #[test]
    fn test_split_io_required_headroom_basic() {
        assert_eq!(split_io_required_headroom(0, 0), 0);
        assert_eq!(split_io_required_headroom(128, 64), 384);
        assert_eq!(split_io_required_headroom(1024, 2048), 6144);
    }

    #[test]
    fn test_split_io_required_headroom_saturates() {
        assert_eq!(split_io_required_headroom(usize::MAX, 1), usize::MAX);
        assert_eq!(split_io_required_headroom(usize::MAX / 2, usize::MAX / 2), usize::MAX);
    }

    #[test]
    fn test_window_pack_required_headroom_basic() {
        assert_eq!(window_pack_required_headroom(128, 64, 256, 512), 1920);
    }

    #[test]
    fn test_window_pack_required_headroom_saturates() {
        assert_eq!(
            window_pack_required_headroom(usize::MAX, 1, 1, 1),
            usize::MAX
        );
    }

    #[test]
    fn test_copy_bytes_staging_round_trip() {
        let src = [1u8, 2, 3, 4, 5, 6];
        let mut staging = [0u8; 6];
        let mut dst = [0u8; 6];

        unsafe {
            copy_bytes_to_staging(
                src.as_ptr() as *const std::ffi::c_void,
                staging.as_mut_ptr() as *mut std::ffi::c_void,
                src.len(),
            );
            copy_bytes_from_staging(
                staging.as_ptr() as *const std::ffi::c_void,
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                staging.len(),
            );
        }

        assert_eq!(src, dst);
    }

    #[test]
    fn test_two_stream_workload_gate_requires_frames_and_work() {
        assert!(!two_stream_meets_workload_gate(1024, 128, 256, 262_144));
        assert!(!two_stream_meets_workload_gate(256, 512, 256, 262_144));
        assert!(two_stream_meets_workload_gate(1024, 256, 256, 262_144));
    }

    #[test]
    fn test_two_stream_workload_gate_handles_zero_fft() {
        assert!(!two_stream_meets_workload_gate(0, 1000, 256, 262_144));
    }
}

