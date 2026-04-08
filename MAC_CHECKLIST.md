# macOS Validation Checklist

Use this checklist after cloning the repository on a Mac to validate a clean build/test run.

## Known baseline

- Linux/WSL baseline run passed:
  - `13980 passed, 18 skipped, 565 xfailed, 1 xpassed`
  - coverage artifact generated: `coverage.xml`
- Rust toolchain is required to build extension components.

## 1) Fresh clone and branch

```bash
git clone <your-repo-url>
cd iron-librosa
git checkout <your-branch-name>
```

## 2) Install OS dependency

```bash
brew install libsamplerate
```

## 3) Create Python environment

```bash
python3 -m venv .venv-mac
source .venv-mac/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## 4) Ensure Rust is available

```bash
rustc --version
cargo --version
```

If Rust is missing:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
```

## 5) Build/install project (editable)

```bash
python -m pip install -e '.[tests]'
```

## 6) Run tests

```bash
PYTHONPATH=Benchmarks/scripts pytest
```

Optional full test entrypoint:

```bash
python -u run_full_tests.py
```

## 7) Pass criteria

- Editable install succeeds without Rust build/import errors.
- `pytest` completes without unexpected failures.
- Any platform-specific skips/xfails are acceptable if consistent with CI.

## 8) If failures occur

Capture and share:

```bash
python --version
rustc --version
cargo --version
python -m pip freeze | sed -n '1,120p'
PYTHONPATH=Benchmarks/scripts pytest -q --maxfail=20
```

Include GitHub Actions links for `CI` and `lint_python` runs from the same commit.

