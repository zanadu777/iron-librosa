#!/usr/bin/env python
"""
Phase-Vocoder Promotion Validation
Run post-promotion to confirm all tests pass and dispatch is correctly enabled.
"""
import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a shell command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True)
    if result.returncode == 0:
        print(f"✓ PASS: {description}")
        return True
    else:
        print(f"✗ FAIL: {description}")
        return False


def main():
    os.chdir(os.path.dirname(__file__) or ".")

    print("\n" + "="*60)
    print("PHASE-VOCODER PROMOTION VALIDATION")
    print("="*60)

    all_pass = True

    # Test 1: Parity validation
    all_pass &= run_command(
        "python test_phase_vocoder_parity.py",
        "Parity test (numerics match Python reference)"
    )

    # Test 2: Dispatch tests (default behavior)
    all_pass &= run_command(
        "python -m pytest tests/test_features.py::test_phase_vocoder_dispatch_prefers_rust_by_default -v",
        "Dispatch test (Rust called by default)"
    )

    # Test 3: Fallback test
    all_pass &= run_command(
        "python -m pytest tests/test_features.py::test_phase_vocoder_dispatch_fallback_with_prefer_rust_false -v",
        "Fallback test (Python used when prefer_rust=False)"
    )

    # Test 4: Per-channel test
    all_pass &= run_command(
        "python -m pytest tests/test_features.py::test_phase_vocoder_dispatch_opt_in_calls_rust_per_channel -v",
        "Per-channel dispatch test"
    )

    # Test 5: Parity kernel test
    all_pass &= run_command(
        "python -m pytest tests/test_features.py::test_phase_vocoder_rust_kernel_matches_reference_loop -v",
        "Kernel parity test (strict numeric matching)"
    )

    # Test 6: Full phase_vocoder test suite
    all_pass &= run_command(
        "python -m pytest tests/test_features.py -k phase_vocoder -v --tb=short",
        "All phase_vocoder tests"
    )

    # Test 7: Multichannel tests
    all_pass &= run_command(
        "python -m pytest tests/test_multichannel.py::test_phase_vocoder -v",
        "Multichannel phase_vocoder test"
    )

    print("\n" + "="*60)
    if all_pass:
        print("✓✓✓ ALL PROMOTION VALIDATION TESTS PASSED ✓✓✓")
        print("="*60)
        print("\nPhase-vocoder Rust dispatch is ready for production use.")
        print("Dispatch is enabled by default (prefer_rust=True).")
        print("Use prefer_rust=False to force Python fallback if needed.")
        return 0
    else:
        print("✗✗✗ SOME VALIDATION TESTS FAILED ✗✗✗")
        print("="*60)
        print("\nPlease review failures above and rerun after fixes.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

