"""Phase 10A kernel benchmark: scipy median filters vs Rust HPSS median kernels."""

import time

import numpy as np
from scipy.ndimage import median_filter

import librosa._rust as _rust


def _run_scipy_case(S: np.ndarray, *, win_harm: int, win_perc: int, repeats: int = 5) -> float:
    harm_shape = [1, win_harm]
    perc_shape = [win_perc, 1]
    median_filter(S, size=harm_shape, mode="reflect")
    median_filter(S, size=perc_shape, mode="reflect")

    t0 = time.perf_counter()
    for _ in range(repeats):
        median_filter(S, size=harm_shape, mode="reflect")
        median_filter(S, size=perc_shape, mode="reflect")
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def _run_rust_case(S: np.ndarray, *, win_harm: int, win_perc: int, repeats: int = 5) -> float:
    if S.dtype == np.float32:
        harm = _rust.median_filter_harmonic_f32
        perc = _rust.median_filter_percussive_f32
    elif S.dtype == np.float64:
        harm = _rust.median_filter_harmonic_f64
        perc = _rust.median_filter_percussive_f64
    else:
        raise TypeError(f"Unsupported dtype for Rust kernel benchmark: {S.dtype}")

    harm(S, win_harm)
    perc(S, win_perc)

    t0 = time.perf_counter()
    for _ in range(repeats):
        harm(S, win_harm)
        perc(S, win_perc)
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def main() -> None:
    rng = np.random.default_rng(9102026)
    cases = [
        ("small-f32", np.abs(rng.standard_normal((513, 200))).astype(np.float32), 17, 31),
        ("medium-f32", np.abs(rng.standard_normal((1025, 600))).astype(np.float32), 31, 31),
        ("medium-f64", np.abs(rng.standard_normal((1025, 600))).astype(np.float64), 31, 31),
    ]

    print("Phase 10A HPSS kernels: scipy median filters vs Rust kernels")
    print("case\tshape\tdtype\tscipy_sec\trust_sec\tspeedup")
    for name, S, win_harm, win_perc in cases:
        scipy_sec = _run_scipy_case(S, win_harm=win_harm, win_perc=win_perc)
        rust_sec = _run_rust_case(S, win_harm=win_harm, win_perc=win_perc)
        speedup = scipy_sec / rust_sec if rust_sec > 0 else float("inf")
        print(f"{name}\t{S.shape}\t{S.dtype}\t{scipy_sec:.6f}\t{rust_sec:.6f}\t{speedup:.2f}x")


if __name__ == "__main__":
    main()

