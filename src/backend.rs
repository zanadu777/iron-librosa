use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::OnceLock;

const RUST_DEVICE_ENV: &str = "IRON_LIBROSA_RUST_DEVICE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustDevice {
    Auto,
    Cpu,
    AppleGpu,
    /// Phase 21: CUDA GPU path (PC/NVIDIA). Feature-gated under `cuda-gpu`.
    /// Opt-in only until benchmark-validated. Falls back to CPU on any error.
    CudaGpu,
}

impl RustDevice {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::AppleGpu => "apple-gpu",
            Self::CudaGpu => "cuda-gpu",
        }
    }
}

fn parse_rust_device(raw: &str) -> RustDevice {
    match raw.trim().to_ascii_lowercase().as_str() {
        "cpu" => RustDevice::Cpu,
        "apple-gpu" | "apple_gpu" | "gpu" | "metal" => RustDevice::AppleGpu,
        "cuda-gpu" | "cuda_gpu" | "cuda" | "nvidia" => RustDevice::CudaGpu,
        _ => RustDevice::Auto,
    }
}

pub fn requested_rust_device() -> RustDevice {
    match std::env::var(RUST_DEVICE_ENV) {
        Ok(value) => parse_rust_device(&value),
        Err(_) => RustDevice::Auto,
    }
}

pub fn apple_gpu_feature_enabled() -> bool {
    cfg!(feature = "apple-gpu")
}

/// Phase 21: Returns true when the `cuda-gpu` feature is compiled in.
/// This is always false until a PC build with `--features cuda-gpu` is produced.
pub fn cuda_gpu_feature_enabled() -> bool {
    cfg!(feature = "cuda-gpu")
}

pub fn apple_gpu_runtime_available() -> bool {
    apple_gpu_runtime_available_impl()
}

/// Phase 21: Runtime probe for CUDA availability.
/// Returns false until cuFFT FFI is wired in Phase 21 implementation.
pub fn cuda_gpu_runtime_available() -> bool {
    cuda_gpu_runtime_available_impl()
}

#[cfg(target_os = "macos")]
fn apple_gpu_runtime_available_impl() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // Conservative runtime probe: require Metal framework to exist.
        std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists()
    })
}

#[cfg(not(target_os = "macos"))]
fn apple_gpu_runtime_available_impl() -> bool {
    false
}

/// Phase 21: CUDA availability probe.
/// Always returns false in the stub phase; will be replaced with cuInit() probe
/// once cuFFT FFI is implemented.
fn cuda_gpu_runtime_available_impl() -> bool {
    false
}

pub fn resolved_rust_device() -> RustDevice {
    match requested_rust_device() {
        RustDevice::Cpu => RustDevice::Cpu,
        RustDevice::AppleGpu => {
            if apple_gpu_feature_enabled() && apple_gpu_runtime_available() {
                RustDevice::AppleGpu
            } else {
                RustDevice::Cpu
            }
        }
        RustDevice::CudaGpu => {
            if cuda_gpu_feature_enabled() && cuda_gpu_runtime_available() {
                RustDevice::CudaGpu
            } else {
                RustDevice::Cpu
            }
        }
        RustDevice::Auto => {
            if apple_gpu_feature_enabled() && apple_gpu_runtime_available() {
                RustDevice::AppleGpu
            } else if cuda_gpu_feature_enabled() && cuda_gpu_runtime_available() {
                RustDevice::CudaGpu
            } else {
                RustDevice::Cpu
            }
        }
    }
}

#[pyfunction]
pub fn rust_backend_info(py: Python<'_>) -> PyResult<PyObject> {
    let requested = requested_rust_device();
    let resolved = resolved_rust_device();
    let info = PyDict::new_bound(py);
    info.set_item("env_var", RUST_DEVICE_ENV)?;
    info.set_item("requested", requested.as_str())?;
    info.set_item("resolved", resolved.as_str())?;
    info.set_item("apple_gpu_feature_enabled", apple_gpu_feature_enabled())?;
    info.set_item("apple_gpu_runtime_available", apple_gpu_runtime_available())?;
    info.set_item("cuda_gpu_feature_enabled", cuda_gpu_feature_enabled())?;
    info.set_item("cuda_gpu_runtime_available", cuda_gpu_runtime_available())?;
    Ok(info.into_py(py))
}

