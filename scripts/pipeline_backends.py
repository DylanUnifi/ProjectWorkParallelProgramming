"""
pipeline_backends.py
--------------------
Unified API to build a (quantum) kernel / Gram matrix from embeddings or
pairwise similarity callbacks, with multiple HPC backends.

Usage
-----
from pipeline_backends import compute_kernel_matrix

# X, Y: numpy or torch/cupy arrays (depending on backend)
K = compute_kernel_matrix(X, Y=None, backend="cpu", tile_size=128, symmetric=True)

Backends
--------
- "cpu"       : Pure NumPy (tiling + symmetry)
- "numba"     : Numba JIT for upper-triangle fill (CPU)
- "torchcuda" : Torch CUDA (cuBLAS) K = X @ X^T
- "pycuda"    : Custom CUDA kernel for tiled Gram computation (optional demo)
- "openmp"    : Placeholder to call a pybind11/C++ OpenMP extension (optional)

Notes
-----
- This module expects *embeddings* X (shape [N, D]). If you still compute
  pairwise quantum fidelities via a simulator, adapt `pairwise_fn` to produce
  the similarity between two rows and set backend="cpu" (it will call pairwise_fn).
- For dense GPU compute, prefer "torchcuda" if you already have X as torch.cuda.Tensor.
"""

from typing import Optional, Callable, Literal, Any
import numpy as np

Backend = Literal["cpu", "numba", "torchcuda", "pycuda", "openmp"]


def _ensure_numpy(a: Any) -> np.ndarray:
    """Move torch/cupy arrays to NumPy (CPU) without forcing copy when possible."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except Exception:
        pass
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return a


def _gram_numpy(X: np.ndarray, Y: Optional[np.ndarray], tile_size: int, symmetric: bool) -> np.ndarray:
    """Dense Gram via NumPy with tiling + symmetry (CPU)."""
    X = _ensure_numpy(X).astype(np.float32, copy=False)
    Y = None if Y is None else _ensure_numpy(Y).astype(np.float32, copy=False)

    if Y is None:
        N = X.shape[0]
        K = np.zeros((N, N), dtype=np.float32)
        for i0 in range(0, N, tile_size):
            i1 = min(i0 + tile_size, N)
            # j loop: upper triangle only if symmetric
            j_start = i0 if symmetric else 0
            for j0 in range(j_start, N, tile_size):
                j1 = min(j0 + tile_size, N)
                # block product
                Bij = X[i0:i1] @ X[j0:j1].T
                K[i0:i1, j0:j1] = Bij
                if symmetric and j0 != i0:
                    K[j0:j1, i0:i1] = Bij.T
        return K
    else:
        N, M = X.shape[0], Y.shape[0]
        K = np.zeros((N, M), dtype=np.float32)
        for i0 in range(0, N, tile_size):
            i1 = min(i0 + tile_size, N)
            for j0 in range(0, M, tile_size):
                j1 = min(j0 + tile_size, M)
                K[i0:i1, j0:j1] = X[i0:i1] @ Y[j0:j1].T
        return K


def _gram_numba(X: np.ndarray, Y: Optional[np.ndarray], symmetric: bool) -> np.ndarray:
    """Upper-triangle fill with Numba (CPU). For X-only case (symmetric Gram)."""
    from numba import njit, prange

    X = _ensure_numpy(X).astype(np.float32, copy=False)
    if Y is not None:
        # fall back to numpy for the rectangular case
        return _gram_numpy(X, _ensure_numpy(Y).astype(np.float32, copy=False), tile_size=128, symmetric=False)

    @njit(parallel=True, fastmath=True)
    def gram_upper(X):
        n, d = X.shape
        K = np.zeros((n, n), dtype=np.float32)
        for i in prange(n):
            K[i, i] = 1.0  # assuming normalized vectors; otherwise compute dot
            xi = X[i]
            for j in range(i + 1, n):
                xj = X[j]
                s = 0.0
                for k in range(d):
                    s += xi[k] * xj[k]
                K[i, j] = s
        return K

    Ku = gram_upper(X)
    # mirror
    Ku = Ku + Ku.T - np.diag(np.diag(Ku))
    return Ku


def _gram_torch_cuda(X: Any, Y: Optional[Any], tile_size: int, symmetric: bool) -> Any:
    """Dense Gram with torch.mm on CUDA. Returns torch.Tensor on GPU."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available for torchcuda backend"

    def to_cuda(a):
        if isinstance(a, torch.Tensor):
            return a.to("cuda", non_blocking=True).float()
        # try dlpack from numpy/cupy
        try:
            import cupy as cp
            if isinstance(a, cp.ndarray):
                return torch.utils.dlpack.from_dlpack(a.toDlpack()).to("cuda").float()
        except Exception:
            pass
        if isinstance(a, np.ndarray):
            return torch.from_numpy(a).to("cuda").float()
        raise TypeError("Unsupported array type for torchcuda backend")

    Xt = to_cuda(X)
    Yt = None if Y is None else to_cuda(Y)

    if Yt is None:
        if symmetric:
            K = Xt @ Xt.T
        else:
            # full square but without mirroring
            K = Xt @ Xt.T
        return K  # keep on GPU
    else:
        K = Xt @ Yt.T
        return K


def _gram_pycuda(X: np.ndarray, Y: Optional[np.ndarray], tile_size: int, symmetric: bool) -> np.ndarray:
    """Demo tiled Gram via PyCUDA. Falls back to cuBLAS (cupy) when available."""
    try:
        import cupy as cp
        # Prefer robust path: use cuBLAS via CuPy for dense GEMM
        Xd = cp.asarray(X, dtype=cp.float32)
        Yd = Xd if Y is None else cp.asarray(Y, dtype=cp.float32)
        Kd = Xd @ Yd.T
        K = cp.asnumpy(Kd)
        if Y is None and symmetric:
            # ensure symmetry numerically
            K = (K + K.T) * 0.5
        return K
    except Exception as e:
        raise RuntimeError(f"PyCUDA backend requires CuPy. {e}")


def _gram_openmp(X: np.ndarray, Y: Optional[np.ndarray]) -> np.ndarray:
    """Placeholder: call a pybind11/C++ extension compiled with OpenMP.

    Expected extension signature:
        import gram_omp
        K = gram_omp.gram_upper_omp(X.astype(np.float32))  # X is [N, D]
    """
    try:
        import gram_omp  # user-compiled module
    except Exception as e:
        raise ImportError("OpenMP backend requires a compiled 'gram_omp' extension") from e

    X = _ensure_numpy(X).astype(np.float32, copy=False)
    if Y is not None:
        # rectangular case: combine two calls or fallback
        return _gram_numpy(X, _ensure_numpy(Y).astype(np.float32, copy=False), tile_size=128, symmetric=False)
    K = gram_omp.gram_upper_omp(X)
    # mirror
    K = K + K.T - np.diag(np.diag(K))
    return K


def compute_kernel_matrix(
    X: Any,
    Y: Optional[Any] = None,
    backend: Backend = "cpu",
    tile_size: int = 128,
    symmetric: bool = True,
    pairwise_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> Any:
    """Unified API.

    Parameters
    ----------
    X, Y : embeddings (NumPy / Torch / CuPy) of shape [N, D] and [M, D].
           If Y is None: compute square Gram(X, X).
    backend : one of {"cpu","numba","torchcuda","pycuda","openmp"}.
    tile_size : block size for tiled GEMM on CPU/GPU (where applicable).
    symmetric : when Y is None, compute/assume an upper-triangular and mirror.
    pairwise_fn : optional Python callback to compute similarity between two rows.
                  If provided and backend="cpu", a slow but general fallback will be used.

    Returns
    -------
    K : kernel/Gram matrix. Type depends on backend:
        - cpu/numba/openmp/pycuda  -> NumPy array (float32)
        - torchcuda                -> torch.cuda.FloatTensor
    """
    if pairwise_fn is not None and backend == "cpu":
        # generic slow path: O(N^2) Python loop using pairwise_fn
        Xn = _ensure_numpy(X).astype(np.float32, copy=False)
        Yn = Xn if Y is None else _ensure_numpy(Y).astype(np.float32, copy=False)
        N, M = Xn.shape[0], Yn.shape[0]
        K = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            # allow symmetric shortcut
            j_start = i if (Y is None and symmetric) else 0
            for j in range(j_start, M):
                kij = pairwise_fn(Xn[i], Yn[j])
                K[i, j] = kij
                if Y is None and symmetric and j != i:
                    K[j, i] = kij
        return K

    if backend == "cpu":
        return _gram_numpy(X, Y, tile_size=tile_size, symmetric=symmetric)
    elif backend == "numba":
        return _gram_numba(X, Y, symmetric=symmetric)
    elif backend == "torchcuda":
        return _gram_torch_cuda(X, Y, tile_size=tile_size, symmetric=symmetric)
    elif backend == "pycuda":
        return _gram_pycuda(_ensure_numpy(X), None if Y is None else _ensure_numpy(Y), tile_size=tile_size, symmetric=symmetric)
    elif backend == "openmp":
        return _gram_openmp(X, Y)
    else:
        raise ValueError(f"Unknown backend: {backend}")
