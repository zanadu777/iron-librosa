#!/usr/bin/env python
"""Backend info and device resolution checks for Rust dispatch policy."""

import pytest

from librosa._rust_bridge import RUST_AVAILABLE, _rust_ext


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rust_backend_info_keys_present():
    info = _rust_ext.rust_backend_info()

    assert "env_var" in info
    assert "requested" in info
    assert "requested_reason" in info
    assert "resolved" in info
    assert "resolved_reason" in info
    assert "dispatch_policy" in info
    assert "apple_gpu_feature_enabled" in info
    assert "apple_gpu_runtime_available" in info
    assert "cuda_gpu_feature_enabled" in info
    assert "cuda_gpu_runtime_available" in info
    assert info["dispatch_policy"] == "auto_first_cpu_fallback"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rust_backend_info_default_request_is_auto(monkeypatch):
    monkeypatch.delenv("IRON_LIBROSA_RUST_DEVICE", raising=False)
    info = _rust_ext.rust_backend_info()

    assert info["requested"] == "auto"
    assert info["resolved"] in {"cpu", "apple-gpu", "cuda-gpu"}


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rust_backend_info_cpu_override_resolves_cpu(monkeypatch):
    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cpu")
    info = _rust_ext.rust_backend_info()

    assert info["requested"] == "cpu"
    assert info["resolved"] == "cpu"


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


@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust extension not available")
def test_rust_backend_info_cuda_resolution_policy(monkeypatch):
    monkeypatch.setenv("IRON_LIBROSA_RUST_DEVICE", "cuda-gpu")
    info = _rust_ext.rust_backend_info()

    expected = (
        "cuda-gpu"
        if info["cuda_gpu_feature_enabled"] and info["cuda_gpu_runtime_available"]
        else "cpu"
    )
    assert info["requested"] == "cuda-gpu"
    assert info["resolved"] == expected


