[![librosa logo](docs/img/iron-librosa_logo_text.png)](https://librosa.org/)

# iron-librosa

Rust-accelerated music and audio analysis with broad `librosa` API compatibility.

`iron-librosa` is an active acceleration project built around the `librosa` ecosystem.
It is **not** a one-off fork for a single bug fix or feature patch: the goal is to
progressively move hot-path analysis and transform kernels into Rust while preserving
Python-facing behavior and compatibility.

This repository currently contains:

- the Python `librosa` package interface used for compatibility,
- the Rust extension module exposed as `librosa._rust`, and
- the convenience namespace package `iron_librosa` for explicit accelerated imports.

[![License](https://img.shields.io/pypi/l/librosa.svg)](https://github.com/zanadu777/iron-librosa/blob/main/LICENSE.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

[![CI](https://github.com/zanadu777/iron-librosa/actions/workflows/ci.yml/badge.svg)](https://github.com/zanadu777/iron-librosa/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zanadu777/iron-librosa/branch/main/graph/badge.svg)](https://codecov.io/gh/zanadu777/iron-librosa)
[![Docs](https://github.com/zanadu777/iron-librosa/actions/workflows/docs.yml/badge.svg)](https://github.com/zanadu777/iron-librosa/actions/workflows/docs.yml)

> Upstream lineage: this work builds on the `librosa` project and keeps upstream
> documentation/citation links where relevant, while extending the implementation
> with Rust acceleration.

#  Table of Contents

- [Documentation](#documentation)
- [Project Status](#project-status)
- [Rust Coverage Snapshot](#rust-coverage-snapshot)
- [Phase Commit Policy](#phase-commit-policy)
- [Phase 13 References](#phase-13-references)
- [Installation](#installation)
  - [Using PyPI](#using-pypi)
  - [Using Anaconda](#using-anaconda)
  - [Building From Source](#building-from-source)
  - [Hints for Installation](#hints-for-the-installation)
    - [`soundfile`](#soundfile)
    - [`audioread`](#audioread-and-mp3-support)
      - [Linux (`apt get`)](#linux-apt-get)
      - [Linux (`yum`)](#linux-yum)
      - [Mac](#mac)
      - [Windows](#windows)
- [Performance guard](#performance-guard)
- [Discussion](#discussion)
- [Citing](#citing)

---

## Documentation


See https://librosa.org/doc/ for a complete reference manual and introductory tutorials.

The [advanced example gallery](https://librosa.org/doc/latest/advanced.html) should give you a quick sense of the kinds
of things that librosa can do.

---

[Back To Top ↥](#iron-librosa)

## Project Status

- **Current phase:** Phase 13 complete (CQT/VQT seam integrated, opt-in)
- **Recent completion:** Rust CQT/VQT dense projection seam implemented and parity-validated
- **Near-term objective:** keep the seam benchmark-driven and promote only after a faster GEMM pass
- **Project direction:** progressively expand Rust coverage across high-value analysis paths
- **Tracking doc:** `Development_docs/PHASE13_CQT_VQT_COMPLETION_REPORT.md`

---

[Back To Top ↥](#iron-librosa)

## Rust Coverage Snapshot

- **Current acceleration coverage:** ~70% of hot-path operations
- **Implemented modules:** STFT/ISTFT, phase vocoder, mel, onset, tuning, chroma, spectral utilities
- **Phase 13 target:** delivered as an opt-in CQT/VQT acceleration seam with parity hardening
- **Coverage roadmap:** `Development_docs/LIBROSA_RUST_COVERAGE_ROADMAP.md`

---

[Back To Top ↥](#iron-librosa)

## Phase Commit Policy

At the end of each phase, create a dedicated commit containing only files owned by that phase's scope, validation, and docs.

Minimum checklist before a phase-end commit:

1. Update the phase status document in `Development_docs/`
2. Run the validation commands listed in the phase plan
3. Capture benchmark/parity artifacts when required by exit criteria
4. Commit with a phase-scoped message such as `phase-13: <summary>`

---

[Back To Top ↥](#iron-librosa)

## Phase 13 References

- Completion report: `Development_docs/PHASE13_CQT_VQT_COMPLETION_REPORT.md`
- Spike plan: `Development_docs/PHASE13_CQT_VQT_SPIKE.md`
- Next actions: `Development_docs/NEXT_ACTIONS_PHASE13_PLANNING.md`
- Coverage quick reference: `Development_docs/RUST_COVERAGE_QUICK_REFERENCE.md`
- Coverage roadmap: `Development_docs/LIBROSA_RUST_COVERAGE_ROADMAP.md`

---

[Back To Top ↥](#iron-librosa)

## Installation


### Using PyPI

For this repository/project configuration, the package name is `iron-librosa`.
If you are consuming a published wheel for this project, install it with:
```
python -m pip install iron-librosa
```

### Using Anaconda

A dedicated conda package is not documented in this repository yet.
For now, prefer source/editable installs for development work:
```
python -m pip install -e .
```

### Building from source

To build `iron-librosa` from source, say 
```
python setup.py build
```
Then, to install it, say 
```
python setup.py install
```
If all went well, you should be able to execute the following commands from a python console:
```
import librosa
librosa.show_versions()
```
This should print out a description of your software environment, along with the installed versions of other packages used by librosa.

📝 OS X users should follow the installation guide given below.

Alternatively, you can download or clone the repository and use `pip` to handle dependencies:

```
unzip iron-librosa.zip
python -m pip install -e .
```
or

```
git clone https://github.com/zanadu777/iron-librosa.git
cd iron-librosa
python -m pip install -e .
```

By calling `pip list` you should see `iron-librosa` now as an installed package:
```
iron-librosa (0.x.x, /path/to/iron-librosa)
```

---

[Back To Top ↥](#iron-librosa)

### Hints for the Installation

`librosa` uses `soundfile` and `audioread` to load audio files.

📝 Note that older releases of `soundfile` (prior to 0.11) do not support MP3, which will cause librosa to fall back on the `audioread` library.

### `soundfile`

If you're using `conda` to install librosa, then audio encoding dependencies will be handled automatically.

If you're using `pip` on a Linux environment, you may need to install `libsndfile`
manually.  Please refer to the [SoundFile installation documentation](https://python-soundfile.readthedocs.io/#installation) for details.

### `audioread` and MP3 support

To fuel `audioread` with more audio-decoding power (e.g., for reading MP3 files),
you may need to install either *ffmpeg* or *GStreamer*.

📝*Note that on some platforms, `audioread` needs at least one of the programs to work properly.*

If you are using Anaconda, install *ffmpeg* by calling

```
conda install -c conda-forge ffmpeg
```

If you are not using Anaconda, here are some common commands for different operating systems:

- ####  Linux (`apt-get`): 

```
apt-get install ffmpeg
```
or
 
```
apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly
```
- #### Linux (`yum`):
```
yum install ffmpeg
```
or


```
yum install gstreamer1.0-plugins-base gstreamer1.0-plugins-ugly
```

- #### Mac: 
```
brew install ffmpeg
```
or

```
brew install gstreamer
```

- #### Windows: 

download ffmpeg binaries from this [website](https://www.gyan.dev/ffmpeg/builds/) or gstreamer binaries from this [website](https://gstreamer.freedesktop.org/)

For GStreamer, you also need to install the Python bindings with 

```
python -m pip install pygobject
```

---

[Back To Top ↥](#iron-librosa)

## Performance guard

This repository includes a lightweight MFCC performance guard script:

```bash
python -u Benchmarks/scripts/benchmark_guard.py
```

Useful options:

```bash
python -u Benchmarks/scripts/benchmark_guard.py --runs 10
python -u Benchmarks/scripts/benchmark_guard.py --review-threshold 1.5
python -u Benchmarks/scripts/benchmark_guard.py --review-threshold 1.5 --fail-on-review-required
```

Policy: any measured speedup below `1.5x` is automatically flagged as
`AUTO-REVIEW REQUIRED`.

Validate benchmark artifact schema (`meta`, `auto_review_cases`, `rows`):

```bash
python -u Benchmarks/scripts/validate_benchmark_payloads.py --paths Benchmarks/results/*.json
```

A manual GitHub Actions workflow is also available at
`.github/workflows/perf-guard.yml` (`Actions` -> `Performance Guard` -> `Run workflow`).

---

[Back To Top ↥](#iron-librosa)

## Discussion


Please direct non-development questions and discussion topics to our web forum at
https://groups.google.com/forum/#!forum/librosa

---

[Back To Top ↥](#iron-librosa)

## Citing


If you want to cite librosa in a scholarly work, there are two ways to do it.

- If you are using the library for your work, for the sake of reproducibility, please cite
  the version you used as indexed at Zenodo:

    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.591533.svg)](https://doi.org/10.5281/zenodo.591533)

  From librosa version 0.10.2 or later, you can also use `librosa.cite()`
  to get the DOI link for any version of librosa.

- If you wish to cite librosa for its design, motivation, etc., please cite the paper
  published at SciPy 2015:

    McFee, Brian, Colin Raffel, Dawen Liang, Daniel PW Ellis, Matt McVicar, Eric Battenberg, and Oriol Nieto. "librosa: Audio and music signal analysis in python." In Proceedings of the 14th python in science conference, pp. 18-25. 2015.

---

[Back To Top ↥](#iron-librosa)
