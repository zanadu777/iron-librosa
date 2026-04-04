"""Quick Phase 11 benchmark: spectral_contrast Rust dispatch on mono/stereo/quad."""

import time
import numpy as np
import librosa
import benchmark_phase5_spectral as bench


def timeit(fn, runs=6):
    fn()
    vals = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return min(vals)


def main():
    np.random.seed(2051)
    sr = 22050
    n_fft = 2048
    n_bins = n_fft // 2 + 1
    print("Phase 11 quick benchmark: spectral_contrast multichannel")
    print(f"{'case':<10} {'rust_ms':<12} {'python_ms':<12} {'speedup':<10}")
    for n_frames in [300, 800, 1600]:
        print(f"\nframes={n_frames}")
        cases = [
            ("mono", np.abs(np.random.randn(n_bins, n_frames).astype(np.float32))),
            ("stereo", np.abs(np.random.randn(2, n_bins, n_frames).astype(np.float32))),
            ("quad", np.abs(np.random.randn(4, n_bins, n_frames).astype(np.float32))),
        ]
        for label, s in cases:
            with bench._force_python_fallback(False):
                t_r = timeit(lambda: librosa.feature.spectral_contrast(S=s, sr=sr))
            with bench._force_python_fallback(True):
                t_p = timeit(lambda: librosa.feature.spectral_contrast(S=s, sr=sr))
            print(f"{label:<10} {t_r*1e3:<12.3f} {t_p*1e3:<12.3f} {t_p/t_r:<10.2f}x")


if __name__ == "__main__":
    main()


