use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::process::Command;
use std::sync::OnceLock;

const RUST_DEVICE_ENV: &str = "IRON_LIBROSA_RUST_DEVICE";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RustDevice {
    Auto,
    Cpu,
    AppleGpu,
    /// Phase 21: CUDA GPU path (PC/NVIDIA). Feature-gated under `cuda-gpu`.
    /// Auto-selected on CUDA-capable systems when runtime heuristics say it is profitable.
    /// Falls back to CPU on any error.
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

fn requested_rust_device_with_reason() -> (RustDevice, &'static str) {
    match std::env::var(RUST_DEVICE_ENV) {
        Ok(value) => {
            let parsed = parse_rust_device(&value);
            let reason = if parsed == RustDevice::Auto {
                "env_unrecognized_fallback_auto"
            } else {
                "env_override"
            };
            (parsed, reason)
        }
        Err(_) => (RustDevice::Auto, "env_not_set_default_auto"),
    }
}

pub fn apple_gpu_feature_enabled() -> bool {
    cfg!(feature = "apple-gpu")
}

/// Phase 21: Returns true when the `cuda-gpu` feature is compiled in.
pub fn cuda_gpu_feature_enabled() -> bool {
    cfg!(feature = "cuda-gpu")
}

pub fn apple_gpu_runtime_available() -> bool {
    apple_gpu_runtime_available_impl()
}

/// Phase 21: Runtime probe for CUDA availability.
/// Uses a conservative host-side probe so auto mode remains safe for drop-in use.
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
/// Keeps behavior conservative: only report CUDA available when both the feature
/// is compiled and the host can see an NVIDIA device via `nvidia-smi -L`.
fn cuda_gpu_runtime_available_impl() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // Optional override for local diagnostics.
        if let Ok(raw) = std::env::var("IRON_LIBROSA_CUDA_RUNTIME_FORCE") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "1" | "true" | "yes" | "on" => return true,
                "0" | "false" | "no" | "off" => return false,
                _ => {}
            }
        }

        // Conservative runtime probe: require nvidia-smi and a successful query.
        Command::new("nvidia-smi")
            .arg("-L")
            .output()
            .map(|out| out.status.success())
            .unwrap_or(false)
    })
}

pub fn resolved_rust_device() -> RustDevice {
    resolved_rust_device_with_reason().0
}

fn resolved_rust_device_with_reason() -> (RustDevice, &'static str) {
    match requested_rust_device() {
        RustDevice::Cpu => (RustDevice::Cpu, "requested_cpu"),
        RustDevice::AppleGpu => {
            if apple_gpu_feature_enabled() && apple_gpu_runtime_available() {
                (RustDevice::AppleGpu, "requested_apple_gpu_available")
            } else {
                (RustDevice::Cpu, "requested_apple_gpu_unavailable_fallback_cpu")
            }
        }
        RustDevice::CudaGpu => {
            if cuda_gpu_feature_enabled() && cuda_gpu_runtime_available() {
                (RustDevice::CudaGpu, "requested_cuda_gpu_available")
            } else {
                (RustDevice::Cpu, "requested_cuda_gpu_unavailable_fallback_cpu")
            }
        }
        RustDevice::Auto => {
            if apple_gpu_feature_enabled() && apple_gpu_runtime_available() {
                (RustDevice::AppleGpu, "auto_selected_apple_gpu")
            } else if cuda_gpu_feature_enabled() && cuda_gpu_runtime_available() {
                (RustDevice::CudaGpu, "auto_selected_cuda_gpu")
            } else {
                (RustDevice::Cpu, "auto_fallback_cpu")
            }
        }
    }
}

#[pyfunction]
pub fn rust_backend_info(py: Python<'_>) -> PyResult<PyObject> {
    let (requested, requested_reason) = requested_rust_device_with_reason();
    let (resolved, resolved_reason) = resolved_rust_device_with_reason();
    let info = PyDict::new_bound(py);
    info.set_item("env_var", RUST_DEVICE_ENV)?;
    info.set_item("requested", requested.as_str())?;
    info.set_item("requested_reason", requested_reason)?;
    info.set_item("resolved", resolved.as_str())?;
    info.set_item("resolved_reason", resolved_reason)?;
    info.set_item("dispatch_policy", "auto_first_cpu_fallback")?;
    info.set_item("apple_gpu_feature_enabled", apple_gpu_feature_enabled())?;
    info.set_item("apple_gpu_runtime_available", apple_gpu_runtime_available())?;
    info.set_item("cuda_gpu_feature_enabled", cuda_gpu_feature_enabled())?;
    info.set_item("cuda_gpu_runtime_available", cuda_gpu_runtime_available())?;
    Ok(info.into_py(py))
}

/// Return a dict with CUDA diagnostics useful for debugging GPU dispatch.
///
/// Keys:
///   cuda_feature_enabled  (bool)   — built with --features cuda-gpu
///   cuda_runtime_available (bool)  — nvidia-smi found + returned success
///   device_count           (str)   — "1" or error string
///   nvidia_smi_ok          (bool)  — nvidia-smi -L exit 0
///   dll_probes             (list of [name, ok, err])
///   diagnostics_text       (str)   — human-readable summary
#[pyfunction]
pub fn cuda_diagnostics(py: Python<'_>) -> PyResult<PyObject> {
    let info = PyDict::new_bound(py);
    info.set_item("cuda_feature_enabled", cuda_gpu_feature_enabled())?;
    info.set_item("cuda_runtime_available", cuda_gpu_runtime_available())?;
    info.set_item("device_count", crate::cuda_fft::cuda_fft_impl::device_count_str())?;
    info.set_item(
        "cuda_window_pack_helper_built",
        crate::cuda_fft::cuda_window_pack_helper_built(),
    )?;
    info.set_item(
        "cuda_window_pack_helper_path",
        crate::cuda_fft::cuda_window_pack_helper_path().unwrap_or(""),
    )?;
    info.set_item(
        "cuda_fused_mel_helper_built",
        crate::cuda_fft::cuda_fused_mel_helper_built(),
    )?;
    info.set_item(
        "cuda_fused_mel_helper_path",
        crate::cuda_fft::cuda_fused_mel_helper_path().unwrap_or(""),
    )?;

    let smi_ok = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    info.set_item("nvidia_smi_ok", smi_ok)?;

    // DLL probe list → Python list of 3-tuples
    let probes = pyo3::types::PyList::empty_bound(py);
    for (name, ok, err) in crate::cuda_fft::cuda_fft_impl::probe_dlls() {
        let t = pyo3::types::PyTuple::new_bound(py, &[
            name.into_py(py),
            ok.into_py(py),
            err.into_py(py),
        ]);
        probes.append(t)?;
    }
    info.set_item("dll_probes", probes)?;
    info.set_item("diagnostics_text", crate::cuda_fft::cuda_diagnostics_info())?;
    Ok(info.into_py(py))
}

