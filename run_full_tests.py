#!/usr/bin/env python
"""
Run pytest programmatically to test the hotfix validation.
"""
import subprocess
import sys

print("=" * 70)
print("RUNNING FULL TEST SUITE VALIDATION")
print("=" * 70)
print()

# Test 1: Run test_core.py
print("Running test_core.py...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_core.py", "-q", "-o", "addopts="],
    capture_output=True,
    text=True
)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print()

# Test 2: Run test_multichannel.py with focus on mel operations
print("Running test_multichannel.py with melspectrogram focus...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_multichannel.py", "-k", "mel", "-q", "-o", "addopts="],
    capture_output=True,
    text=True
)
lines = result.stdout.split('\n')
print('\n'.join(lines[-20:]))  # Last 20 lines
print()

# Test 3: Run validation script
print("Running hotfix validation script...")
result = subprocess.run(
    [sys.executable, "test_hotfix_validation.py"],
    capture_output=True,
    text=True
)
print(result.stdout)

print("=" * 70)
print("TEST EXECUTION COMPLETE")
print("=" * 70)

