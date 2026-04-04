import numpy as np, time, librosa, iron_librosa
from librosa._rust_bridge import _rust_ext

def bench(fn, n=20):
    fn(); ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return min(ts)*1e3

np.random.seed(7)
sr = 22050
n_fft, hop, n_mels = 2048, 512, 128

print("STFT kernel benchmark (raw, no Python pipeline overhead)")
print(f"  {'case':<30} {'librosa.stft':>14} {'rust stft_power':>16} {'speedup':>9}")
print("-"*76)
for dur in [1, 5, 15, 30]:
    y = np.random.randn(sr * dur).astype(np.float32)
    y_c = np.ascontiguousarray(y)
    mb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # pre-built

    # Unpatched Python STFT pipeline (librosa.stft is never Rust-patched)
    t_py = bench(lambda: np.abs(librosa.stft(y_c, n_fft=n_fft, hop_length=hop))**2)
    # Rust parallel STFT
    t_rs = bench(lambda: _rust_ext.stft_power(y_c, n_fft, hop, True))

    label = f"{dur}s ({sr*dur//1000}k samples)"
    print(f"  {label:<30} {t_py:>14.2f} ms {t_rs:>14.2f} ms {t_py/t_rs:>8.2f}x")

print()
print("Full melspectrogram pipeline (STFT + mel GEMM), pre-built mel_basis")
print(f"  {'case':<30} {'python STFT+GEMM':>16} {'rust STFT+GEMM':>16} {'speedup':>9}")
print("-"*76)
for dur in [1, 5, 15, 30]:
    y = np.random.randn(sr * dur).astype(np.float32)
    y_c = np.ascontiguousarray(y)
    mb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mb_c = np.ascontiguousarray(mb)

    def py_pipeline():
        S = np.abs(librosa.stft(y_c, n_fft=n_fft, hop_length=hop))**2
        return mb_c.dot(S)

    def rs_pipeline():
        S = _rust_ext.stft_power(y_c, n_fft, hop, True)
        return mb_c.dot(S)

    t_py = bench(py_pipeline)
    t_rs = bench(rs_pipeline)
    label = f"{dur}s ({sr*dur//1000}k samples)"
    print(f"  {label:<30} {t_py:>16.2f} ms {t_rs:>14.2f} ms {t_py/t_rs:>8.2f}x")
