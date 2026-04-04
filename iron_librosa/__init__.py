"""
iron_librosa — drop-in replacement for librosa
===============================================

This package re-exports the entire ``librosa`` public API so that::

    import iron_librosa as librosa

works as a transparent, Rust-accelerated replacement.

All existing code that uses ``import librosa`` continues to work
unchanged; swap the import alias to start benefiting from the Rust
acceleration layer automatically.
"""
from __future__ import annotations

# Re-export the full librosa namespace at the top level.
# lazy_loader is already used by librosa itself, so this import is cheap.
import lazy_loader as lazy

# Expose every public symbol that librosa exposes.
import librosa as _librosa  # noqa: F401

from librosa import __version__  # noqa: F401

# Provide sub-module access via this package as well.
def __getattr__(name: str):
    return getattr(_librosa, name)

def __dir__():
    return dir(_librosa)
