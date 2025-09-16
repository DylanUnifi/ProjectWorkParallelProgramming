# pipeline_backends.py — PennyLane kernels with RBF-like controls (angle_scale, re-embed, normalize)
from typing import Optional, Any, Tuple, Callable
import os
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

# =====================================================================
# Helpers
# =====================================================================
def _ensure_numpy(a: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Converts to a C-contiguous ndarray, respecting dtype if provided."""
    try:
        import torch  # type: ignore
        if isinstance(a, torch.Tensor):  # type: ignore
            a = a.detach().cpu().numpy()  # type: ignore
    except Exception:
        pass
    arr = np.asarray(a, order="C")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr

def _tile_ranges(n: int, tile: int):
    for s in range(0, n, tile):
        e = min(s + tile, n)
        yield s, e

def _normalize_diag_inplace(K: np.ndarray):
    """Normalize so that diag(K)=1 for square Gram matrices."""
    if K.shape[0] != K.shape[1]:
        return
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)

# =====================================================================
# Worker globals (multiprocessing path)
# =====================================================================
_pl_w: Optional[np.ndarray] = None
_pl_nq: Optional[int] = None
_pl_device: Optional[str] = None
_pl_qnode: Optional[Callable] = None
_pl_float_dtype: Optional[np.dtype] = None
_pl_complex_dtype: Optional[np.dtype] = None
_pl_angle_scale: float = 1.0
_pl_re_embed: bool = False
_pl_embed_mode: str = "ryrz"  # "ry" | "ryrz" | "angle"

def _pl_worker_init(w_local: np.ndarray, device_name: str, nq: int,
                    float_dtype_str: str = "float64",
                    angle_scale: float = 1.0,
                    re_embed_between_layers: bool = False,
                    embed_mode: str = "ryrz"):
    """Initializer called once per worker process."""
    global _pl_w, _pl_nq, _pl_device, _pl_qnode, _pl_float_dtype, _pl_complex_dtype
    global _pl_angle_scale, _pl_re_embed, _pl_embed_mode
    _pl_w = _ensure_numpy(w_local, np.float32 if float_dtype_str == "float32" else np.float64)
    _pl_nq = int(nq)
    _pl_device = str(device_name)
    _pl_qnode = None
    _pl_float_dtype = np.dtype(np.float32) if float_dtype_str == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if float_dtype_str == "float32" else np.dtype(np.complex128)
    _pl_angle_scale = float(angle_scale)
    _pl_re_embed = bool(re_embed_between_layers)
    _pl_embed_mode = str(embed_mode)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    w = _pl_w
    if w.ndim != 2 or w.shape[1] != _pl_nq:
        raise ValueError(f"_pl_w must have shape (L, n_qubits={_pl_nq}); got {w.shape}.")

def _pl_get_qnode():
    """Create the qnode used in this process worker with embedding controls."""
    global _pl_qnode
    if _pl_qnode is None:
        import pennylane as qml
        dev = qml.device(
            _pl_device, wires=_pl_nq, shots=None, c_dtype=_pl_complex_dtype
        )

        def _embed(theta):
            s = _pl_angle_scale
            if _pl_embed_mode == "angle":
                qml.AngleEmbedding(s * theta[:_pl_nq], wires=range(_pl_nq), rotation="Y", normalize=False)
            else:
                for i in range(_pl_nq):
                    qml.RY(s * theta[i], wires=i)
                    if _pl_embed_mode == "ryrz":
                        qml.RZ(s * theta[i], wires=i)

        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row):
            theta = qml.math.asarray(theta_row, dtype=_pl_float_dtype)
            if theta.shape[0] < _pl_nq:
                raise ValueError(f"theta has length {theta.shape[0]} < n_qubits={_pl_nq}")
            if _pl_re_embed:
                L = _pl_w.shape[0]
                for l in range(L):
                    _embed(theta)
                    qml.templates.BasicEntanglerLayers(_pl_w[l:l+1], wires=range(_pl_nq))
            else:
                _embed(theta)
                qml.templates.BasicEntanglerLayers(_pl_w, wires=range(_pl_nq))
            return qml.state()

        _pl_qnode = _state
    return _pl_qnode

def _pl_states_for_rows(rows: list[int], mat: np.ndarray) -> np.ndarray:
    qnode = _pl_get_qnode()
    out = np.empty((len(rows), 1 << _pl_nq), dtype=_pl_complex_dtype)
    for t, idx in enumerate(rows):
        out[t] = qnode(mat[idx])
    return out

# =====================================================================
# Ultra-fast GPU path (Torch stream)
# =====================================================================
def _gram_torch_stream(
    a_np: np.ndarray,
    b_np: Optional[np.ndarray],
    *,
    weights_np: np.ndarray,
    device_name: str,
    tile_size: int,
    symmetric: bool,
    float_dt: np.dtype,
    ret_dt: np.dtype,
    angle_scale: float,
    re_embed_between_layers: bool,
    embed_mode: str,
) -> np.ndarray:
    import torch as th
    import pennylane as qml

    assert "gpu" in device_name.lower(), "torch_stream requires a GPU device (e.g. 'lightning.gpu')."
    nq = a_np.shape[1]
    n = a_np.shape[0]
    m = n if b_np is None else b_np.shape[0]

    t_float = th.float32 if float_dt == np.float32 else th.float64
    t_complex = th.complex64 if float_dt == np.float32 else th.complex128
    t_ret = th.float32 if ret_dt == np.float32 else th.float64

    a = th.from_numpy(np.ascontiguousarray(a_np)).to("cuda", dtype=t_float, non_blocking=True)
    b = a if b_np is None else th.from_numpy(np.ascontiguousarray(b_np)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(weights_np)).to("cuda", dtype=t_float, non_blocking=True)

    dev = qml.device(
        device_name,
        wires=nq, shots=None,
        c_dtype=(np.complex64 if float_dt == np.float32 else np.complex128),
    )

    def _embed(v):
        s = float(angle_scale)
        if embed_mode == "angle":
            qml.AngleEmbedding(s * v[:nq], wires=range(nq), rotation="Y", normalize=False)
        else:
            for i in range(nq):
                qml.RY(s * v[i], wires=i)
                if embed_mode == "ryrz":
                    qml.RZ(s * v[i], wires=i)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(theta_row: th.Tensor) -> th.Tensor:
        if re_embed_between_layers:
            L = w.shape[0]
            for l in range(L):
                _embed(theta_row)
                qml.templates.BasicEntanglerLayers(w[l:l+1], wires=range(nq))
        else:
            _embed(theta_row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try:
        from torch import vmap  # type: ignore
        _has_vmap = True
    except Exception:
        _has_vmap = False

    def build_states(x_block: th.Tensor) -> th.Tensor:
        if _has_vmap and x_block.ndim == 2:
            return vmap(_state)(x_block).to(dtype=t_complex)
        states = [_state(x_block[t]) for t in range(x_block.shape[0])]
        return th.stack(states, dim=0).to(dtype=t_complex)

    k = th.empty((n, m), device="cuda", dtype=t_ret)

    with th.no_grad():
        outer_iter = range(0, n, tile_size)
        if tqdm is not None:
            outer_iter = tqdm(outer_iter, desc="Gram (torch): outer", leave=False)
        for i0 in outer_iter:
            i1 = min(i0 + tile_size, n)
            sa_x = build_states(a[i0:i1])
            inner_iter = range(0 if not (symmetric and b is a) else i0, m, tile_size)
            if tqdm is not None:
                inner_iter = tqdm(inner_iter, desc=f"Gram (torch): inner[{i0}:{i1}]", leave=False)
            for j0 in inner_iter:
                j1 = min(j0 + tile_size, m)
                sb_x = sa_x if (b is a and j0 == i0) else build_states(b[j0:j1])
                g = sa_x @ sb_x.conj().transpose(0, 1)
                k_blk = (g.abs() ** 2).to(dtype=t_ret)
                k[i0:i1, j0:j1] = k_blk
                if symmetric and b is a and j0 > i0:
                    k[j0:j1, i0:i1] = k_blk.transpose(0, 1)
            del sa_x
            th.cuda.empty_cache()

    return k.detach().cpu().numpy().astype(ret_dt, copy=False)

# =====================================================================
# CPU / generic path (NumPy or multiprocessing)
# =====================================================================
def _gram_pennylane_angles_mp(
    A: Any,
    B: Optional[Any] = None,
    *,
    weights: np.ndarray,
    device_name: str = "lightning.qubit",
    tile_size: int = 64,
    symmetric: bool = True,
    n_workers: int = 0,
    dtype: Optional[str] = None,
    return_dtype: Optional[str] = None,
    progress: bool = False,
    desc: str = "Gram",
    angle_scale: float = 1.0,
    re_embed_between_layers: bool = False,
    embed_mode: str = "ryrz",
) -> np.ndarray:
    """Kernel de fidélité PennyLane, parallèle (multiprocessing), tuilé (CPU)."""
    import multiprocessing as mp
    import pennylane as qml  # noqa: F401

    def _resolve_float_dtype() -> np.dtype:
        if dtype in ("float32", "float64"):
            return np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
        src_dtypes: list[np.dtype] = []
        try: src_dtypes.append(np.asarray(A).dtype)
        except Exception: pass
        try: src_dtypes.append(np.asarray(weights).dtype)
        except Exception: pass
        if any(dt == np.float32 for dt in src_dtypes):
            return np.dtype(np.float32)
        return np.dtype(np.float64)

    float_dt = _resolve_float_dtype()

    def _resolve_return_dtype() -> np.dtype:
        if return_dtype in ("float32", "float64"):
            return np.dtype(np.float32) if return_dtype == "float32" else np.dtype(np.float64)
        return float_dt

    ret_dt = _resolve_return_dtype()

    A = _ensure_numpy(A, dtype=float_dt)
    B = A if B is None else _ensure_numpy(B, dtype=float_dt)
    n, nq = A.shape
    m = B.shape[0]

    w = _ensure_numpy(weights, dtype=float_dt)
    if w.ndim != 2 or w.shape[1] != nq:
        raise ValueError(f"`weights` must be [n_layers, n_qubits={nq}]")

    def _chunk_indices(n_items: int, chunk: int) -> list[list[int]]:
        return [list(range(s, min(s + chunk, n_items))) for s in range(0, n_items, chunk)]

    rows_a = _chunk_indices(n, max(1, tile_size))
    rows_b = rows_a if (B is A) else _chunk_indices(m, max(1, tile_size))

    if n_workers is None or n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)

    # initargs for worker
    initargs = (w, device_name, nq, "float32" if float_dt == np.float32 else "float64",
                float(angle_scale), bool(re_embed_between_layers), str(embed_mode))

    if n_workers == 1:
        _pl_worker_init(*initargs)
        sa = np.concatenate([_pl_states_for_rows(rs, A) for rs in rows_a], axis=0)
        if B is A:
            sb = sa
        else:
            sb = np.concatenate([_pl_states_for_rows(rs, B) for rs in rows_b], axis=0)
    else:
        ctx = mp.get_context("spawn")
        from functools import partial
        with ctx.Pool(
            processes=n_workers,
            initializer=_pl_worker_init,
            initargs=initargs,
        ) as pool:
            funcA = partial(_pl_states_for_rows, mat=A)
            sa = np.concatenate(list(pool.imap(funcA, rows_a, chunksize=1)), axis=0)
            if B is A:
                sb = sa
            else:
                funcB = partial(_pl_states_for_rows, mat=B)
                sb = np.concatenate(list(pool.imap(funcB, rows_b, chunksize=1)), axis=0)

    k = np.empty((n, m), dtype=ret_dt)
    outer_iter = list(_tile_ranges(n, tile_size))
    if tqdm is not None and progress:
        outer_iter = tqdm(outer_iter, desc=f"{desc}: outer", leave=False)

    for i0, i1 in outer_iter:
        sa_blk = np.ascontiguousarray(sa[i0:i1])
        inner_iter = list(_tile_ranges(m, tile_size))
        j_start = 0 if not (symmetric and (B is A)) else i0
        if tqdm is not None and progress:
            inner_iter = tqdm(inner_iter, desc=f"{desc}: inner[{i0}:{i1}]", leave=False)

        for j0, j1 in inner_iter:
            if j0 < j_start:
                continue
            sb_blk = np.ascontiguousarray(sb[j0:j1])
            g = sa_blk @ sb_blk.conj().T
            mag2 = (np.abs(g) ** 2).astype(ret_dt, copy=False)
            k[i0:i1, j0:j1] = mag2
            if symmetric and (B is A) and (j0 > i0):
                k[j0:j1, i0:i1] = mag2.T

    return k

# =====================================================================
# Public API
# =====================================================================
def compute_kernel_matrix(
    X: Any,
    Y: Optional[Any] = None,
    *,
    weights: np.ndarray,
    device_name: str = "lightning.qubit",
    tile_size: int = 64,
    symmetric: bool = True,
    n_workers: int = 0,
    dtype: Optional[str] = None,
    return_dtype: Optional[str] = None,
    gram_backend: str = "auto",
    progress: bool = False,
    desc: str = "Gram",
    # RBF-like controls
    angle_scale: float = 1.0,
    re_embed_between_layers: bool = False,
    embed_mode: str = "ryrz",
    normalize: bool = False,
    jitter: float = 0.0,
    # kept for compat (ignored here)
    state_tile: int = 128, tile_m: int | str = "auto", tile_n: int | str = "auto", tile_k: int | str = "auto",
) -> np.ndarray:
    """
    Fidelity kernel between quantum states prepared from angles X(/Y).
    - angle_scale: multiplicative factor on input angles (γ-like control)
    - re_embed_between_layers: re-apply data embedding between entangler layers
    - embed_mode: 'ry' | 'ryrz' | 'angle'
    - normalize: enforce diag(K)=1 (only for square K); jitter adds tiny value on diag before normalization
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # resolve dtypes
    if dtype in ("float32", "float64"):
        float_dt = np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
    else:
        float_dt = np.float32 if "gpu" in device_name.lower() else np.float64
    if return_dtype in ("float32", "float64"):
        ret_dt = np.dtype(np.float32) if return_dtype == "float32" else np.dtype(np.float64)
    else:
        ret_dt = float_dt

    # Try fast GPU path with Torch
    if ("gpu" in device_name.lower()) and gram_backend in ("auto", "torch", "cuda_states"):
        try:
            import torch as _th  # type: ignore
            if _th.cuda.is_available():
                A = _ensure_numpy(X, dtype=float_dt)
                B = A if Y is None else _ensure_numpy(Y, dtype=float_dt)
                w = _ensure_numpy(weights, dtype=float_dt)
                K = _gram_torch_stream(
                    A, None if (B is A) else B,
                    weights_np=w,
                    device_name=device_name,
                    tile_size=tile_size,
                    symmetric=symmetric,
                    float_dt=float_dt,
                    ret_dt=ret_dt,
                    angle_scale=angle_scale,
                    re_embed_between_layers=re_embed_between_layers,
                    embed_mode=embed_mode,
                )
            else:
                raise RuntimeError("CUDA not available")
        except Exception:
            # fallback CPU
            K = _gram_pennylane_angles_mp(
                X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
                symmetric=symmetric, n_workers=n_workers, dtype=str(float_dt).replace("float", "float"),
                return_dtype=str(ret_dt).replace("float", "float"), progress=progress, desc=desc,
                angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
            )
    else:
        # CPU path
        K = _gram_pennylane_angles_mp(
            X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
            symmetric=symmetric, n_workers=n_workers, dtype=str(float_dt).replace("float", "float"),
            return_dtype=str(ret_dt).replace("float", "float"), progress=progress, desc=desc,
            angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
        )

    if normalize and (Y is None):
        if jitter and jitter > 0:
            K = K + float(jitter) * np.eye(K.shape[0], dtype=K.dtype)
        _normalize_diag_inplace(K)

    return K
