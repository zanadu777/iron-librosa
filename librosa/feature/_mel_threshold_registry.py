"""Built-in mel projection auto-dispatch thresholds by CPU profile.

This table is intentionally small and conservative. Use
`IRON_LIBROSA_MEL_THRESHOLD_FILE` for machine-local calibration data.
"""

# Profile key format is user-defined. Recommended:
#   <os>-<arch>-<blas>
# Example:
#   windows-amd64-mkl
MEL_WORK_THRESHOLDS = {
    # Measured on this host.
    "windows-amd64-openblas": 201_226_955,
    # Conservative fallbacks until calibrated values are collected.
    # 0 disables Rust auto-dispatch for the profile.
    "windows-amd64-mkl": 0,
    "linux-x86_64-openblas": 0,
    "darwin-arm64-accelerate": 0,
}

