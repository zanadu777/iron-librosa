"""
Three-level speedup benchmark: Python librosa vs Rust CPU vs Rust+CUDA.
Run: python scripts/benchmark_three_level.py
"""
import os, time, json, sys
import numpy as np

os.environ['IRON_LIBROSA_RUST_DEVICE'] = 'cuda-gpu'
os.environ['IRON_LIBROSA_ENABLE_CUDA_FFT_EXPERIMENTAL'] = 'force-on'

import librosa._rust as _rust
import librosa as lr

sr = 22050
n_runs = 5

# Warmup
for n in [512, 1024, 2048]:
    y_w = np.zeros(sr, dtype=np.float32)
    _rust.stft_complex(y_w, n, n // 4, True, None)

results = {'date': '2026-04-14', 'gpu': 'RTX 3090', 'cuda': '13.2', 'rows': []}

configs = [
    (1, 512, 128), (5, 512, 128), (10, 512, 128), (20, 512, 128),
    (1, 1024, 256), (5, 1024, 256), (10, 1024, 256), (20, 1024, 256),
]

print('=== THREE-LEVEL SPEEDUP: Python vs Rust CPU vs Rust+CUDA ===')
print(f'GPU: RTX 3090 | CUDA 13.2 | Date: 2026-04-14')
print()
print(f"{'Workload':18s} {'Python':>9s} {'RustCPU':>9s} {'RustGPU':>9s}  {'cpu/py':>7s} {'gpu/py':>7s} {'gpu/cpu':>8s}")
print('-' * 76)

for dur, n_fft, hop in configs:
    y = np.random.randn(sr * dur).astype(np.float32)

    # Python librosa
    times_py = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        lr.stft(y, n_fft=n_fft, hop_length=hop)
        times_py.append((time.perf_counter() - t0) * 1000)

    # Rust CPU
    os.environ['IRON_LIBROSA_RUST_DEVICE'] = 'cpu'
    times_cpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _rust.stft_complex(y, n_fft, hop, True, None)
        times_cpu.append((time.perf_counter() - t0) * 1000)

    # Rust GPU
    os.environ['IRON_LIBROSA_RUST_DEVICE'] = 'cuda-gpu'
    times_gpu = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _rust.stft_complex(y, n_fft, hop, True, None)
        times_gpu.append((time.perf_counter() - t0) * 1000)

    py_ms  = min(times_py)
    cpu_ms = min(times_cpu)
    gpu_ms = min(times_gpu)

    label = f"{dur}s n={n_fft}"
    print(f"{label:18s} {py_ms:8.2f}ms {cpu_ms:8.2f}ms {gpu_ms:8.2f}ms  "
          f"{py_ms/cpu_ms:6.2f}x {py_ms/gpu_ms:6.2f}x {cpu_ms/gpu_ms:7.2f}x")

    results['rows'].append({
        'dur': dur, 'n_fft': n_fft, 'hop': hop,
        'python_ms': round(py_ms, 3),
        'cpu_ms': round(cpu_ms, 3),
        'gpu_ms': round(gpu_ms, 3),
        'cpu_vs_py': round(py_ms / cpu_ms, 2),
        'gpu_vs_py': round(py_ms / gpu_ms, 2),
        'gpu_vs_cpu': round(cpu_ms / gpu_ms, 2),
    })

out_path = 'Benchmarks/results/phase21_stft_three_level_2026-04-14.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f'Results saved: {out_path}')

