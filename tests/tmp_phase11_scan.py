import time
import numpy as np
import librosa
import benchmark_phase5_spectral as bench
import librosa.feature.spectral as spectral_mod


def timeit(fn, runs=4):
    fn()
    vals = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return min(vals)


def run_case(ch, n_bins, n_frames, mode="auto"):
    shape = (n_bins, n_frames) if ch == 1 else (ch, n_bins, n_frames)
    s = np.abs(np.random.randn(*shape).astype(np.float32))

    prev_mode = spectral_mod._CONTRAST_RUST_MODE
    spectral_mod._CONTRAST_RUST_MODE = mode
    with bench._force_python_fallback(False):
        tr = timeit(lambda: librosa.feature.spectral_contrast(S=s, sr=22050))
    with bench._force_python_fallback(True):
        tp = timeit(lambda: librosa.feature.spectral_contrast(S=s, sr=22050))
    spectral_mod._CONTRAST_RUST_MODE = prev_mode
    return tp / tr


np.random.seed(2051)
for ch in [1, 2, 4, 8]:
    for n_frames in [300, 800, 1600]:
        sp_auto = run_case(ch, 1025, n_frames, mode="auto")
        sp_rust = run_case(ch, 1025, n_frames, mode="rust")
        print(
            f"ch={ch}, frames={n_frames}, speedup_auto={sp_auto:.2f}x, speedup_rust={sp_rust:.2f}x"
        )

