#!/usr/bin/env python
"""Backend info and device resolution checks for Rust dispatch policy."""

import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rust_backend_info_keys_present():
    info = _rust_ext.rust_backend_info()

    assert "env_var" in info
    assert "requested" in info
    assert "resolved" in info
    assert "apple_gpu_feature_enabled" in info
    assert "apple_gpu_runtime_available" in info


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rust_backend_info_resolution_policy(monkeypatch):
    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "apple-gpu")
    info = _rust_ext.rust_backend_info()

    expected = (
        "apple-gpu"
        if info["apple_gpu_feature_enabled"] and info["apple_gpu_runtime_available"]
        else "cpu"
    )
    assert info["requested"] == "apple-gpu"
    assert info["resolved"] == expected

