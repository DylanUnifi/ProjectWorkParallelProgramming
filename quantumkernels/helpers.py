import numpy as np
import os


def ensure_numpy(a, dtype=None):
    """Converts to a C-contiguous ndarray, respecting dtype if provided."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(a, order="C")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def tile_ranges(n, tile):
    """Generate (start, end) pairs for tiling."""
    for s in range(0, n, tile):
        e = min(s + tile, n)
        yield s, e


def normalize_diag_inplace(K: np.ndarray):
    """Normalize so that diag(K)=1 for square Gram matrices."""
    if K.shape[0] != K.shape[1]:
        return
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)


def setup_cupy():
    """Initialize CuPy memory pool and includes for RawModule."""
    import cupy as cp
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    _ = cp.ones((1,), dtype=cp.float32)
    del _
    cp.cuda.runtime.deviceSynchronize()
    os.environ.setdefault("CUPY_NVRTC_OPTIONS", "-I/usr/local/cuda/include")
    for key in ("CPATH", "CPLUS_INCLUDE_PATH"):
        os.environ[key] = "/usr/local/cuda/include:" + os.environ.get(key, "")
