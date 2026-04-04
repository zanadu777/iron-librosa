"""Break down HPSS time into median filters vs masking/other work."""

import time

import numpy as np
from scipy.ndimage import median_filter

import librosa


def _time_call(fn, repeats=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    t1 = time.perf_counter()
    return (t1 - t0) / repeats


def _breakdown_case(S: np.ndarray, *, kernel_size=(31, 31), power=2.0, margin=1.0, repeats=5):
    if np.iscomplexobj(S):
        mag, phase = librosa.magphase(S)
    else:
        mag = S
        phase = 1

    if isinstance(kernel_size, (tuple, list)):
        win_harm, win_perc = kernel_size
    else:
        win_harm = win_perc = kernel_size

    if isinstance(margin, (tuple, list)):
        margin_harm, margin_perc = margin
    else:
        margin_harm = margin_perc = margin

    harm_shape = [1] * mag.ndim
    harm_shape[-1] = win_harm
    perc_shape = [1] * mag.ndim
    perc_shape[-2] = win_perc
    split_zeros = margin_harm == 1 and margin_perc == 1

    t_harm = _time_call(lambda: median_filter(mag, size=harm_shape, mode="reflect"), repeats=repeats)
    harm = median_filter(mag, size=harm_shape, mode="reflect")

    t_perc = _time_call(lambda: median_filter(mag, size=perc_shape, mode="reflect"), repeats=repeats)
    perc = median_filter(mag, size=perc_shape, mode="reflect")

    t_mask = _time_call(
        lambda: (
            librosa.util.softmask(harm, perc * margin_harm, power=power, split_zeros=split_zeros),
            librosa.util.softmask(perc, harm * margin_perc, power=power, split_zeros=split_zeros),
        ),
        repeats=repeats,
    )

    t_end_to_end = _time_call(
        lambda: librosa.decompose.hpss(S, kernel_size=kernel_size, power=power, margin=margin),
        repeats=repeats,
    )

    return {
        "harm": t_harm,
        "perc": t_perc,
        "mask": t_mask,
        "end_to_end": t_end_to_end,
        "median_total": t_harm + t_perc,
        "other": max(0.0, t_end_to_end - (t_harm + t_perc + t_mask)),
    }


def main():
    rng = np.random.default_rng(9102026)
    cases = [
        ("small-real", np.abs(rng.standard_normal((513, 200))).astype(np.float32), (17, 31)),
        ("medium-real", np.abs(rng.standard_normal((1025, 600))).astype(np.float32), (31, 31)),
        ("medium-complex", (rng.standard_normal((1025, 600)) + 1j * rng.standard_normal((1025, 600))).astype(np.complex64), (31, 31)),
    ]

    print("HPSS timing breakdown")
    print("case\tharm_sec\tperc_sec\tmedian_total\tmask_sec\tother_sec\tend_to_end")
    for name, S, kernel_size in cases:
        stats = _breakdown_case(S, kernel_size=kernel_size)
        print(
            f"{name}\t{stats['harm']:.6f}\t{stats['perc']:.6f}\t{stats['median_total']:.6f}\t{stats['mask']:.6f}\t{stats['other']:.6f}\t{stats['end_to_end']:.6f}"
        )


if __name__ == "__main__":
    main()



