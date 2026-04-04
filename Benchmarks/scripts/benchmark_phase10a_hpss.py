"""Phase 10A benchmark: librosa.decompose.hpss (Rust dispatch when eligible)."""

import time

import numpy as np

import librosa


def _run_case(S: np.ndarray, *, kernel_size=(31, 31), power=2.0, margin=1.0, repeats=5) -> float:
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
        ("small-real", np.abs(rng.standard_normal((513, 200))).astype(np.float32), (17, 31), 2.0, 1.0),
        ("medium-real", np.abs(rng.standard_normal((1025, 600))).astype(np.float32), (31, 31), 2.0, 1.0),
        ("stereo-real", np.abs(rng.standard_normal((2, 1025, 600))).astype(np.float32), (31, 31), 2.0, 1.0),
        ("medium-complex", (rng.standard_normal((1025, 600)) + 1j * rng.standard_normal((1025, 600))).astype(np.complex64), (31, 31), 2.0, 1.0),
    ]

    print("Phase 10A HPSS benchmark (Rust dispatch when eligible)")
    print("case\tshape\tkernel\tpower\tmargin\tmean_sec")
    for name, S, kernel_size, power, margin in cases:
        sec = _run_case(S, kernel_size=kernel_size, power=power, margin=margin)
        print(f"{name}\t{S.shape}\t{kernel_size}\t{power}\t{margin}\t{sec:.6f}")


if __name__ == "__main__":
    main()

