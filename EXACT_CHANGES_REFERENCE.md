# Exact Changes Made - Line-by-Line Reference

## File 1: librosa/_rust_bridge.py

### Change 1: Added environment variable to __all__ exports (line 38-40)
```python
__all__ = [
    "_rust_ext",
    "RUST_AVAILABLE",
    "RUST_EXTENSION_AVAILABLE",  # ← NEW
    "FORCE_NUMPY_MEL",
    ...
]
```

### Change 2: Create RUST_EXTENSION_AVAILABLE (line 82-83)
```python
# Keep extension availability separate from dispatch policy.
RUST_EXTENSION_AVAILABLE: bool = _rust_ext is not None
```

### Change 3: Add environment-gated RUST_AVAILABLE (line 85-88)
```python
# Global dispatch gate: default to safe NumPy/SciPy parity in CI and production.
# Set IRON_LIBROSA_RUST_DISPATCH=1 to enable Rust accelerated dispatch paths.
_rust_dispatch = os.getenv("IRON_LIBROSA_RUST_DISPATCH", "0").strip().lower()
RUST_AVAILABLE: bool = RUST_EXTENSION_AVAILABLE and _rust_dispatch in {"1", "true", "yes", "on"}
```

---

## File 2: librosa/core/convert.py

### Change 1: hz_to_mel dimensional guard (lines 1212-1219)
**BEFORE:**
```python
frequencies = np.asanyarray(frequencies)

if RUST_AVAILABLE and hasattr(_rust_ext, "hz_to_mel"):
    return _rust_ext.hz_to_mel(frequencies, htk=htk)

if htk:
    ...
```

**AFTER:**
```python
frequencies = np.asanyarray(frequencies)

# --- iron-librosa: Rust acceleration ---
if (
    RUST_AVAILABLE
    and hasattr(_rust_ext, "hz_to_mel")
    and frequencies.ndim == 1  # ← NEW GUARD
):
    freq_rust = np.ascontiguousarray(frequencies, dtype=np.float64)  # ← NEW
    return _rust_ext.hz_to_mel(freq_rust, htk=htk)
# --- end Rust acceleration ---

if htk:
    ...
```

**Changes:**
- Added `frequencies.ndim == 1` check (dimensional guard)
- Added `np.ascontiguousarray(frequencies, dtype=np.float64)` conversion
- Added explanatory comments

### Change 2: mel_to_hz dimensional guard (lines 1297-1300)
**BEFORE:**
```python
mels = np.asanyarray(mels)

if RUST_AVAILABLE and hasattr(_rust_ext, "mel_to_hz"):
    return _rust_ext.mel_to_hz(mels, htk=htk)

if htk:
    ...
```

**AFTER:**
```python
mels = np.asanyarray(mels)

# --- iron-librosa: Rust acceleration ---
if RUST_AVAILABLE and hasattr(_rust_ext, "mel_to_hz") and mels.ndim == 1:  # ← NEW GUARD
    mels_rust = np.ascontiguousarray(mels, dtype=np.float64)  # ← NEW
    return _rust_ext.mel_to_hz(mels_rust, htk=htk)
# --- end Rust acceleration ---

if htk:
    ...
```

**Changes:**
- Added `mels.ndim == 1` check (dimensional guard)
- Added `np.ascontiguousarray(mels, dtype=np.float64)` conversion
- Added explanatory comments

---

## Summary of Changes

| File | Lines | Type | Effect |
|------|-------|------|--------|
| `_rust_bridge.py` | 38-40 | Addition | Export new constants |
| `_rust_bridge.py` | 82-83 | Addition | Create RUST_EXTENSION_AVAILABLE |
| `_rust_bridge.py` | 85-88 | Addition | Environment-gated RUST_AVAILABLE |
| `convert.py` | 1212-1219 | Modification | Guard hz_to_mel with ndim==1 |
| `convert.py` | 1297-1300 | Modification | Guard mel_to_hz with ndim==1 |

---

## Verification

### To verify hz_to_mel changes:
```bash
grep -n "frequencies.ndim == 1" librosa/core/convert.py
# Output: 1215:        and frequencies.ndim == 1
```

### To verify mel_to_hz changes:
```bash
grep -n "mels.ndim == 1" librosa/core/convert.py
# Output: 1297:    if RUST_AVAILABLE and hasattr(_rust_ext, "mel_to_hz") and mels.ndim == 1:
```

### To verify environment gating:
```bash
grep -n "IRON_LIBROSA_RUST_DISPATCH" librosa/_rust_bridge.py
# Output: 87:_rust_dispatch = os.getenv("IRON_LIBROSA_RUST_DISPATCH", "0").strip().lower()
```

---

## No Other Files Modified

The hotfix is minimal and focused:
- ✅ Only modified 2 files
- ✅ Only 5 logical changes
- ✅ No impact on test files
- ✅ No dependencies added
- ✅ No API changes

This ensures maximum safety and minimal risk of unintended side effects.

