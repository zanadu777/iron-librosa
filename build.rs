use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn push_if_exists(paths: &mut Vec<PathBuf>, p: PathBuf) {
    if p.exists() {
        paths.push(p);
    }
}

fn windows_cuda_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(base) = env::var(var) {
            push_if_exists(&mut out, Path::new(&base).join("lib").join("x64"));
        }
    }

    let default_root = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
    if let Ok(entries) = fs::read_dir(default_root) {
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        versions.sort();
        versions.reverse();
        for v in versions {
            push_if_exists(&mut out, v.join("lib").join("x64"));
        }
    }

    out
}

fn unix_cuda_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(base) = env::var(var) {
            push_if_exists(&mut out, Path::new(&base).join("lib64"));
        }
    }

    push_if_exists(&mut out, PathBuf::from("/usr/local/cuda/lib64"));
    push_if_exists(&mut out, PathBuf::from("/usr/lib/x86_64-linux-gnu"));

    out
}

fn windows_nvcc_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(base) = env::var(var) {
            push_if_exists(&mut out, Path::new(&base).join("bin").join("nvcc.exe"));
        }
    }

    let default_root = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
    if let Ok(entries) = fs::read_dir(default_root) {
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        versions.sort();
        versions.reverse();
        for v in versions {
            push_if_exists(&mut out, v.join("bin").join("nvcc.exe"));
        }
    }

    out.push(PathBuf::from("nvcc.exe"));
    out
}

fn windows_msvc_cl_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    let vswhere = PathBuf::from(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe");
    if vswhere.exists() {
        if let Ok(output) = Command::new(&vswhere)
            .args([
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-find",
                r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
            ])
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                for line in stdout.lines() {
                    let path = PathBuf::from(line.trim());
                    if path.exists() {
                        out.push(path);
                    }
                }
            }
        }
    }

    out
}

fn unix_nvcc_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();

    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(base) = env::var(var) {
            push_if_exists(&mut out, Path::new(&base).join("bin").join("nvcc"));
        }
    }

    push_if_exists(&mut out, PathBuf::from("/usr/local/cuda/bin/nvcc"));
    out.push(PathBuf::from("nvcc"));
    out
}

fn cuda_helper_file_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "cuda_window_pack.dll"
    } else if cfg!(target_os = "macos") {
        "libcuda_window_pack.dylib"
    } else {
        "libcuda_window_pack.so"
    }
}

fn cuda_helper_arch() -> String {
    env::var("IRON_LIBROSA_CUDA_HELPER_ARCH")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "native".to_string())
}

fn try_build_cuda_window_pack_helper(out_dir: &Path) -> Option<PathBuf> {
    let source = Path::new("src").join("cuda_kernels.cu");
    if !source.exists() {
        println!("cargo:warning=CUDA helper source not found: {}", source.display());
        return None;
    }

    let helper_path = out_dir.join(cuda_helper_file_name());
    let helper_arch = cuda_helper_arch();
    let nvcc_candidates = if cfg!(target_os = "windows") {
        windows_nvcc_candidates()
    } else {
        unix_nvcc_candidates()
    };

    let host_compiler_dirs = if cfg!(target_os = "windows") {
        let mut dirs: Vec<Option<PathBuf>> = vec![None];
        for cl in windows_msvc_cl_candidates() {
            dirs.push(cl.parent().map(Path::to_path_buf));
        }
        dirs
    } else {
        vec![None]
    };

    for nvcc in nvcc_candidates {
        for host_dir in &host_compiler_dirs {
            let mut cmd = Command::new(&nvcc);
            cmd.arg("--shared")
                .arg("-arch")
                .arg(&helper_arch)
                .arg("-o")
                .arg(&helper_path)
                .arg(&source);

            if let Some(host_dir) = host_dir {
                cmd.arg("-ccbin").arg(host_dir);
            }

            if cfg!(target_os = "windows") {
                cmd.arg("-Xcompiler").arg("/MD");
            } else {
                cmd.arg("-Xcompiler").arg("-fPIC");
            }

            match cmd.output() {
                Ok(output) if output.status.success() => {
                    println!(
                        "cargo:warning=Built CUDA window-pack helper with {}{} arch={}",
                        nvcc.display(),
                        host_dir
                            .as_ref()
                            .map(|p| format!(" (ccbin={})", p.display()))
                            .unwrap_or_default(),
                        helper_arch
                    );
                    return Some(helper_path);
                }
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).trim().replace('\n', " | ");
                    let stderr = String::from_utf8_lossy(&output.stderr).trim().replace('\n', " | ");
                    println!(
                        "cargo:warning=nvcc helper build failed via {}{} arch={}: stdout='{}' stderr='{}'",
                        nvcc.display(),
                        host_dir
                            .as_ref()
                            .map(|p| format!(" (ccbin={})", p.display()))
                            .unwrap_or_default(),
                        helper_arch,
                        stdout,
                        stderr
                    );
                }
                Err(err) => {
                    println!(
                        "cargo:warning=Unable to invoke nvcc candidate {}{}: {}",
                        nvcc.display(),
                        host_dir
                            .as_ref()
                            .map(|p| format!(" (ccbin={})", p.display()))
                            .unwrap_or_default(),
                        err
                    );
                }
            }
        }
    }

    println!(
        "cargo:warning=CUDA window-pack helper not built; fused GPU pack path will fall back"
    );
    None
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=IRON_LIBROSA_CUDA_HELPER_ARCH");
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");
    println!("cargo:rustc-check-cfg=cfg(has_cuda_window_pack_kernel)");

    if env::var("CARGO_FEATURE_CUDA_GPU").is_err() {
        println!("cargo:rustc-env=IRON_LIBROSA_CUDA_WINDOW_PACK_HELPER=");
        return;
    }

    let mut candidates = if cfg!(target_os = "windows") {
        windows_cuda_candidates()
    } else {
        unix_cuda_candidates()
    };

    candidates.dedup();
    for path in candidates {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    if let Some(helper_path) = try_build_cuda_window_pack_helper(&out_dir) {
        println!("cargo:rustc-cfg=has_cuda_window_pack_kernel");
        println!(
            "cargo:rustc-env=IRON_LIBROSA_CUDA_WINDOW_PACK_HELPER={}",
            helper_path.display()
        );
    } else {
        println!("cargo:rustc-env=IRON_LIBROSA_CUDA_WINDOW_PACK_HELPER=");
    }

    // CUDA/cuFFT are loaded dynamically at runtime in src/cuda_fft.rs.
}

