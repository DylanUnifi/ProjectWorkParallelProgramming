import os
import numpy as np
from tqdm import tqdm

from .helpers import ensure_numpy, normalize_diag_inplace, setup_cupy
from .torch_backends import gram_torch_stream
from .cupy_rawkernels import autotune_tiles, get_full_kernel_with_macros, get_lower_kernel_with_macros
from .gram_cpu import gram_pennylane_angles_mp


def compute_kernel_matrix(
        X, Y=None, *, weights, device_name="lightning.qubit",
        tile_size=64, symmetric=True, n_workers=0,
        dtype=None, return_dtype=None, gram_backend="auto",
        progress=False, desc="Gram",
        angle_scale=1.0, re_embed_between_layers=False,
        embed_mode="ryrz", normalize=False, jitter=0.0,
        state_tile=8192, tile_m="auto", tile_n="auto", tile_k="auto"):
    """
    Fidelity kernel between quantum states prepared from angles X(/Y).
    """

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if dtype in ("float32", "float64"):
        float_dt = np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
    else:
        float_dt = np.float32 if "gpu" in device_name.lower() else np.float64

    if return_dtype in ("float32", "float64"):
        ret_dt = np.dtype(np.float32) if return_dtype == "float32" else np.dtype(np.float64)
    else:
        ret_dt = float_dt

    # --- CUDA STATES BACKEND ---
    if gram_backend == "cuda_states":
        import cupy as cp
        try:
            setup_cupy()
        except Exception:
            pass

        A = ensure_numpy(X, dtype=float_dt)
        B = A if Y is None else ensure_numpy(Y, dtype=float_dt)
        n, nq = A.shape
        m = B.shape[0]
        w = ensure_numpy(weights, dtype=float_dt)
        if w.ndim != 2 or w.shape[1] != nq:
            raise ValueError(f"`weights` must be [n_layers, n_qubits={nq}]")
        dim = 1 << nq

        auto_tiles = (tile_m == "auto" or tile_n == "auto" or tile_k == "auto")
        if auto_tiles:
            tm, tn, tk = autotune_tiles(dim, bi=min(state_tile, n),
                                        bj=min(state_tile, m),
                                        symmetric=(symmetric and (B is A)))
        else:
            tm, tn, tk = int(tile_m), int(tile_n), int(tile_k)

        k_func = get_lower_kernel_with_macros(tm, tn, tk) if (symmetric and (B is A)) \
            else get_full_kernel_with_macros(tm, tn, tk)

        from .torch_backends import build_states_block_torch_cuda, torch_cuda_to_cupy
        K_cp = cp.empty((n, m), dtype=cp.float32)

        j_tiles = list(range(0, m, state_tile))
        sb_cache = {}
        for j0 in j_tiles:
            j1 = min(j0 + state_tile, m)
            SB_th = build_states_block_torch_cuda(
                B[j0:j1], w, device_name,
                angle_scale=angle_scale,
                re_embed_between_layers=re_embed_between_layers,
                embed_mode=embed_mode,
            )
            sb_cache[(j0, j1)] = torch_cuda_to_cupy(SB_th)

        for i0 in range(0, n, state_tile):
            i1 = min(i0 + state_tile, n)
            SA_th = build_states_block_torch_cuda(
                A[i0:i1], w, device_name,
                angle_scale=angle_scale,
                re_embed_between
