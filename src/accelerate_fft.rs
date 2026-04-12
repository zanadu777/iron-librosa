// Accelerate framework FFT wrapper for STFT/iSTFT on macOS.
//
// Uses vDSP_ctoz / vDSP_fft / vDSP_ztoc for efficient in-place complex FFT.
// Fallback to rustfft if on non-macOS platforms.

use rustfft::num_complex::Complex;
use std::cell::RefCell;

#[cfg(target_os = "macos")]
pub mod accelerate_impl {
    use super::*;
    use std::os::raw::{c_int, c_uint};

    // ── vDSP function bindings ────────────────────────────────────────────
    // These are minimal FFI declarations to Accelerate framework.
    extern "C" {
        // FFT setup and destruction
        pub fn vDSP_create_fftsetup(log2n: c_uint, radix: c_int) -> *mut std::ffi::c_void;
        pub fn vDSP_destroy_fftsetup(setup: *mut std::ffi::c_void);

        // Complex FFT forward transform (in-place)
        pub fn vDSP_fft_zip(
            setup: *const std::ffi::c_void,
            signal: *mut vDSP_Complex,
            stride: c_int,
            log2n: c_uint,
            direction: c_int,
        );

        // Stride conversion: interleaved <-> split
        pub fn vDSP_ctoz(
            c: *const Complex<f32>,
            stride2: c_int,
            z: *mut vDSP_Split,
            stride: c_int,
            count: c_uint,
        );
        pub fn vDSP_ztoc(
            z: *const vDSP_Split,
            stride: c_int,
            c: *mut Complex<f32>,
            stride2: c_int,
            count: c_uint,
        );
    }

    // vDSP_Complex: interleaved real/imag pair
    #[repr(C)]
    pub struct vDSP_Complex {
        pub real: f32,
        pub imag: f32,
    }

    // vDSP_Split: split real/imag arrays
    #[repr(C)]
    pub struct vDSP_Split {
        pub realp: *mut f32,
        pub imagp: *mut f32,
    }

    const FFT_FORWARD: c_int = 1;
    const FFT_INVERSE: c_int = -1;
    const FFT_RADIX_2: c_int = 2;

    /// Wrapper for Accelerate-powered in-place FFT.
    pub fn fft_forward_inplace(buffer: &mut [Complex<f32>], log2n: u32) -> Result<(), String> {
        let n = 1 << log2n;
        if buffer.len() < n {
            return Err(format!(
                "Buffer size {} < required {}",
                buffer.len(),
                n
            ));
        }

        unsafe {
            // Create setup
            let setup = vDSP_create_fftsetup(log2n as c_uint, FFT_RADIX_2);
            if setup.is_null() {
                return Err("vDSP_create_fftsetup failed".to_string());
            }

            // In-place FFT (stride=1 for packed buffer)
            vDSP_fft_zip(
                setup,
                buffer.as_mut_ptr() as *mut vDSP_Complex,
                1,
                log2n as c_uint,
                FFT_FORWARD,
            );

            vDSP_destroy_fftsetup(setup);
        }
        Ok(())
    }

    /// Wrapper for Accelerate-powered in-place inverse FFT.
    pub fn fft_inverse_inplace(buffer: &mut [Complex<f32>], log2n: u32) -> Result<(), String> {
        let n = 1 << log2n;
        if buffer.len() < n {
            return Err(format!(
                "Buffer size {} < required {}",
                buffer.len(),
                n
            ));
        }

        unsafe {
            let setup = vDSP_create_fftsetup(log2n as c_uint, FFT_RADIX_2);
            if setup.is_null() {
                return Err("vDSP_create_fftsetup failed".to_string());
            }

            vDSP_fft_zip(
                setup,
                buffer.as_mut_ptr() as *mut vDSP_Complex,
                1,
                log2n as c_uint,
                FFT_INVERSE,
            );

            vDSP_destroy_fftsetup(setup);
        }
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub mod accelerate_impl {
    use super::*;

    /// Fallback: use rustfft on non-macOS
    pub fn fft_forward_inplace(buffer: &mut [Complex<f32>], log2n: u32) -> Result<(), String> {
        let n = 1 << log2n;
        if buffer.len() < n {
            return Err(format!(
                "Buffer size {} < required {}",
                buffer.len(),
                n
            ));
        }

        use rustfft::FftPlanner;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        fft.process(&mut buffer[..n]);
        Ok(())
    }

    pub fn fft_inverse_inplace(buffer: &mut [Complex<f32>], log2n: u32) -> Result<(), String> {
        let n = 1 << log2n;
        if buffer.len() < n {
            return Err(format!(
                "Buffer size {} < required {}",
                buffer.len(),
                n
            ));
        }

        use rustfft::FftPlanner;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_inverse(n);
        fft.process(&mut buffer[..n]);
        Ok(())
    }
}

// ── Thread-local setup cache (reserved for future optimization) ──────────────
// Currently unused but kept for forward compatibility with setup caching if needed.

#[allow(dead_code)]
pub struct AccelerateFftCache {
    last_log2n: Option<u32>,
}

#[allow(dead_code)]
impl AccelerateFftCache {
    fn new() -> Self {
        AccelerateFftCache { last_log2n: None }
    }
}

/// High-level wrapper: perform in-place FFT using Accelerate (macOS) or rustfft fallback.
pub fn fft_forward(buffer: &mut [Complex<f32>], n: usize) -> Result<(), String> {
    if n == 0 || (n & (n - 1)) != 0 {
        return Err("FFT size must be a power of 2".to_string());
    }

    let log2n = (n as u32).trailing_zeros();
    accelerate_impl::fft_forward_inplace(buffer, log2n)
}




