"""Phase 10A detailed benchmark: compare scipy.ndimage median filters vs Rust-backed HPSS."""

import time

import numpy as np

import librosa
from scipy.ndimage import median_filter


def _run_scipy_case(S: np.ndarray, *, win_harm=31, win_perc=31, repeats=5) -> float:
    """Benchmark scipy.ndimage.median_filter directly."""
    # Warmup
    harm_shape = [1] * S.ndim
    harm_shape[-1] = win_harm
    perc_shape = [1] * S.ndim
    perc_shape[-2] = win_perc
    _ = median_filter(S, size=harm_shape, mode="reflect")
    _ = median_filter(S, size=perc_shape, mode="reflect")

    t0 = time.perf_counter()
    for _ in range(repeats):
        harm = median_filter(S, size=harm_shape, mode="reflect")
        perc = median_filter(S, size=perc_shape, mode="reflect")
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def _run_hpss_case(S: np.ndarray, *, kernel_size=(31, 31), power=2.0, margin=1.0, repeats=5) -> float:
    """Benchmark librosa.decompose.hpss (includes Rust acceleration)."""
    # Warmup
    librosa.decompose.hpss(S, kernel_size=kernel_size, power=power, margin=margin)

    t0 = time.perf_counter()
    for _ in range(repeats):
        librosa.decompose.hpss(S, kernel_size=kernel_size, power=power, margin=margin)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main() -> None:
    rng = np.random.default_rng(9102026)

    cases = [
        ("small-real", np.abs(rng.standard_normal((513, 200))).astype(np.float32), 17, 31),
        ("medium-real", np.abs(rng.standard_normal((1025, 600))).astype(np.float32), 31, 31),
        ("stereo-real", np.abs(rng.standard_normal((2, 1025, 600))).astype(np.float32), 31, 31),
    ]

    print("Phase 10A HPSS: scipy.ndimage median filters vs Rust-backed HPSS")
    print("=" * 80)
    print(f"{'case':<15} {'shape':<15} {'scipy_sec':<15} {'hpss_sec':<15} {'speedup':<10}")
    print("-" * 80)

    for name, S, win_harm, win_perc in cases:
        scipy_sec = _run_scipy_case(S, win_harm=win_harm, win_perc=win_perc)
        hpss_sec = _run_hpss_case(S, kernel_size=(win_harm, win_perc))
        speedup = scipy_sec / hpss_sec if hpss_sec > 0 else 0
        print(f"{name:<15} {str(S.shape):<15} {scipy_sec:<15.6f} {hpss_sec:<15.6f} {speedup:<10.2f}x")

    print("=" * 80)
    print("Note: HPSS includes median filtering + masking; scipy shows just median_filter time")


if __name__ == "__main__":
    main()

