"""
quantum_kernel.py (UNIFIED)
---------------------------
Single entry point to compute kernel/Gram matrices with multiple backends.
Backends are implemented in `pipeline_backends.py`.

Usage
-----
from quantum_kernel import compute_kernel_matrix

# X, Y can be NumPy arrays, torch.Tensors (CPU/GPU), or CuPy arrays
K = compute_kernel_matrix(X, Y=None, backend="cpu", tile_size=128, symmetric=True)

Notes
-----
- Prefer backend="torchcuda" when X is already on GPU (torch.cuda).
- If you still need a generic pairwise callback (e.g., fidelity from a simulator),
  pass `pairwise_fn` and keep backend="cpu" (it will use a Python loop; slower).
- This module replaces the old `quantum_kernel_parallel.py`.
"""

from __future__ import annotations
from typing import Optional, Callable, Any, Literal
import warnings

# Delegate heavy lifting to the unified backend API
try:
    from scripts.pipeline_backends import compute_kernel_matrix as _compute_backend
except Exception as e:
    raise ImportError("pipeline_backends.py not found. Please add it to your project path.") from e

Backend = Literal["cpu", "numba", "torchcuda", "pycuda", "openmp"]


def compute_kernel_matrix(
    X: Any,
    Y: Optional[Any] = None,
    backend: Backend = "cpu",
    tile_size: int = 128,
    symmetric: bool = True,
    pairwise_fn: Optional[Callable[[Any, Any], float]] = None,
    processes: Optional[int] = None,
    **kwargs,
):
    """Unified API to build a kernel/Gram matrix.
    
    Parameters
    ----------
    X, Y : embeddings (NumPy / Torch / CuPy) of shape [N, D] and [M, D].
           If Y is None, computes a square Gram(X, X). If Y is provided, computes X @ Y^T.
    backend : {"cpu","numba","torchcuda","pycuda","openmp"} backend to use.
    tile_size : block size for tiled implementations where applicable.
    symmetric : whether to exploit symmetry when Y is None.
    pairwise_fn : optional callback (slow path) to compute similarity between two rows when
                  you don't have vector embeddings (e.g., on-the-fly simulator).
    processes : reserved for compatibility; multiprocessing is handled inside backends when needed.
    **kwargs : forwarded to backend (e.g., stream configs in custom impls).
    """
    if processes is not None:
        warnings.warn("`processes` is ignored in the unified API; backends handle parallelism internally.",
                      RuntimeWarning)

    return _compute_backend(
        X, Y=Y, backend=backend, tile_size=tile_size, symmetric=symmetric, pairwise_fn=pairwise_fn
    )


# Convenience alias for compatibility with old imports
compute_qkernel = compute_kernel_matrix
