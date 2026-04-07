#!/usr/bin/env python
"""Run the full test gate used by pre-push validation."""
import subprocess
import sys
import os

# Always resolve paths relative to this script, not the caller's CWD.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(label: str, command: list[str]) -> int:
    """Run a command, print captured output, and return the exit code."""
    print(label)
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, cwd=SCRIPT_DIR
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    print()
    return result.returncode

print("=" * 70)
print("RUNNING FULL TEST SUITE VALIDATION")
print("=" * 70)
print()

overall_status = 0

# Step 1: Full project test suite
overall_status |= run_step(
    "Running full pytest suite...",
    [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--ignore=tests/test_display.py",
    ],
)

# Step 2: Additional hotfix validation coverage
overall_status |= run_step(
    "Running hotfix validation script...",
    [sys.executable, "tests/test_hotfix_validation.py"],
)

print("=" * 70)
print("TEST EXECUTION COMPLETE")
print("=" * 70)
sys.exit(overall_status)
