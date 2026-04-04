import sys
try:
    from librosa._rust_bridge import _rust_ext
    f32_ok = hasattr(_rust_ext, 'spectral_flatness_f32')
    f64_ok = hasattr(_rust_ext, 'spectral_flatness_f64')
    print(f"spectral_flatness_f32: {f32_ok}")
    print(f"spectral_flatness_f64: {f64_ok}")
    if f32_ok and f64_ok:
        import numpy as np
        S = np.abs(np.random.randn(513, 10).astype(np.float32))
        out = _rust_ext.spectral_flatness_f32(S, 1e-10, 2.0)
        print(f"output shape: {out.shape}, dtype: {out.dtype}")
        print(f"output range: [{out.min():.4f}, {out.max():.4f}]")
        print("OK")
    else:
        print("MISSING SYMBOLS")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

