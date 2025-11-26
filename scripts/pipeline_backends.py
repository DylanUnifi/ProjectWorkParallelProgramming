# pipeline_backends.py — PennyLane, multiprocessing-safe (spawn-friendly)
from typing import Optional, Any
import os
import numpy as np

# tqdm (optional)
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # fallback: no bars if tqdm is absent


# ---------- helpers ----------
def _ensure_numpy(a: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Converts to a C-contiguous ndarray, respecting dtype if provided."""
    try:
        import torch  # type: ignore
        if isinstance(a, torch.Tensor):  # type: ignore
            a = a.detach().cpu().numpy()  # type: ignore
    except ImportError:
        pass

    arr = np.asarray(a, order="C")
    if dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr


def _tile_ranges(n: int, tile: int):
    for s in range(0, n, tile):
        e = min(s + tile, n)
        yield s, e


# ---------- GLOBALS for worker processes ----------
_pl_w: Optional[np.ndarray] = None
_pl_nq: Optional[int] = None
_pl_device: Optional[str] = None
_pl_qnode = None  # cached qnode per process
_pl_float_dtype: Optional[np.dtype] = None   # np.float32 / np.float64
_pl_complex_dtype: Optional[np.dtype] = None # np.complex64 / np.complex128


def _pl_worker_init(w_local: np.ndarray, device_name: str, nq: int,
                    float_dtype_str: str = "float64"):
    """Initializer called once per worker process."""
    global _pl_w, _pl_nq, _pl_device, _pl_qnode, _pl_float_dtype, _pl_complex_dtype
    _pl_w = _ensure_numpy(w_local, np.float32 if float_dtype_str == "float32" else np.float64)
    _pl_nq = int(nq)
    _pl_device = str(device_name)
    _pl_qnode = None
    _pl_float_dtype = np.dtype(np.float32) if float_dtype_str == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if float_dtype_str == "float32" else np.dtype(np.complex128)
    # avoid BLAS over-parallelism in each worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _pl_get_qnode():
    """Creates/hides the qnode in this process worker (dtype compliant with the device)."""
    global _pl_qnode
    if _pl_qnode is None:
        import pennylane as qml
        try:
            from pennylane.typing import TensorLike as PLTensorLike  # PL ≥ 0.30
        except ImportError:  # pragma: no cover
            from typing import Any as PLTensorLike  # fallback typings only

        dev = qml.device(
            _pl_device,
            wires=_pl_nq,
            shots=None,
            c_dtype=_pl_complex_dtype,  # précision interne du statevector
        )

        # spell-checker: disable-next-line
        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row: PLTensorLike):
            # Normalise dtype/format (float32/float64)
            theta = qml.math.asarray(theta_row, dtype=_pl_float_dtype)
            # encodage: RY+RZ puis couches d'entanglement (W)
            for i in range(_pl_nq):
                qml.RY(theta[i], wires=i)
                qml.RZ(theta[i], wires=i)
            qml.templates.BasicEntanglerLayers(_pl_w, wires=range(_pl_nq))
            return qml.state()

        _pl_qnode = _state
    return _pl_qnode


def _pl_states_for_rows(rows: list[int], mat: np.ndarray) -> np.ndarray:
    """Builds state vectors for the requested indices (picklable)."""
    from typing import Callable, cast
    try:
        from pennylane.typing import TensorLike as PLTensorLike  # PL ≥ 0.30
    except ImportError:  # fallback typings only
        from typing import Any as PLTensorLike  # type: ignore

    qnode = _pl_get_qnode()
    qnode_typed = cast(Callable[[PLTensorLike], np.ndarray], qnode)

    out = np.empty((len(rows), 1 << _pl_nq), dtype=_pl_complex_dtype)
    for t, idx in enumerate(rows):
        theta_row = cast(PLTensorLike, mat[idx])  # np.ndarray -> TensorLike (typing)
        out[t] = qnode_typed(theta_row)
    return out


# ---------- “All Torch” GPU path: states + GEMM 100% CUDA ----------
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
) -> np.ndarray:
    """Ultra-fast path: states + GEMM on GPU via Torch, streamed in blocks."""
    import torch as th
    import pennylane as qml

    assert "gpu" in device_name.lower(), "torch_stream: device GPU requis"

    nq = a_np.shape[1]
    n = a_np.shape[0]
    m = n if b_np is None else b_np.shape[0]

    # dtypes Torch
    t_float = th.float32 if float_dt == np.float32 else th.float64
    t_complex = th.complex64 if float_dt == np.float32 else th.complex128
    t_ret = th.float32 if ret_dt == np.float32 else th.float64

    # GPU data
    a = th.from_numpy(a_np).to("cuda", dtype=t_float, non_blocking=True)
    b = a if b_np is None else th.from_numpy(b_np).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(weights_np).to("cuda", dtype=t_float, non_blocking=True)

    dev = qml.device(
        device_name,
        wires=nq,
        shots=None,
        c_dtype=(np.complex64 if float_dt == np.float32 else np.complex128),
    )

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(theta_row: th.Tensor) -> th.Tensor:
        for i in range(nq):
            qml.RY(theta_row[i], wires=i)
            qml.RZ(theta_row[i], wires=i)
        qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()  # torch.complex{64,128}

    def build_states(x_block: th.Tensor) -> th.Tensor:
        # We keep it simple and robust; vmap possible.
        states = [_state(x_block[t]) for t in range(x_block.shape[0])]
        return th.stack(states, dim=0).to(dtype=t_complex)  # [blk, 2^nq]

    k = th.empty((n, m), device="cuda", dtype=t_ret)

    with th.no_grad():
        # progress bar (outer)
        outer_iter = range(0, n, tile_size)
        if tqdm is not None:
            outer_iter = tqdm(outer_iter, desc="Gram (torch): outer", leave=False)
        for i0 in outer_iter:
            i1 = min(i0 + tile_size, n)
            sa_x = build_states(a[i0:i1])  # [bi, 2^nq]

            inner_iter = range(0 if not (symmetric and b is a) else i0, m, tile_size)
            if tqdm is not None:
                inner_iter = tqdm(inner_iter, desc=f"Gram (torch): inner[{i0}:{i1}]", leave=False)

            for j0 in inner_iter:
                j1 = min(j0 + tile_size, m)
                sb_x = sa_x if (b is a and j0 == i0) else build_states(b[j0:j1])

                g = sa_x @ sb_x.conj().transpose(0, 1)   # [bi, bj]
                k_blk = (g.abs() ** 2).to(dtype=t_ret)    # [bi, bj]
                k[i0:i1, j0:j1] = k_blk

                if symmetric and b is a and j0 > i0:
                    k[j0:j1, i0:i1] = k_blk.transpose(0, 1)

            # releases as it flows
            del sa_x
            th.cuda.empty_cache()

    return k.detach().cpu().numpy().astype(ret_dt, copy=False)


# ---------- main API ----------
def _gram_pennylane_angles_mp(
    A: Any,
    B: Optional[Any] = None,
    *,
    weights: np.ndarray,
    device_name: str = "lightning.qubit",
    tile_size: int = 64,
    symmetric: bool = True,
    n_workers: int = 0,
    dtype: Optional[str] = None,          # "float32", "float64" ou None/auto (calcul)
    return_dtype: Optional[str] = None,    # "float32", "float64" ou None/auto (sortie K)
    gram_backend: str = "auto",            # "auto" | "numpy" | "cupy" | "torch"
    progress: bool = False,                # progress bars
    desc: str = "Gram",                    # bar prefix
) -> np.ndarray:
    """Kernel de fidélité PennyLane, parallèle (multiprocessing), tuilé."""
    import multiprocessing as mp
    from functools import partial

    # --- choice of floating dtype (calculation): returns np.dtype
    def _resolve_float_dtype() -> np.dtype:
        if dtype in ("float32", "float64"):
            return np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
        if "gpu" in device_name.lower():
            return np.dtype(np.float32)
        # si entrées déjà en float32, on respecte
        src_dtypes: list[np.dtype] = []
        try:
            src_dtypes.append(np.asarray(A).dtype)
        except (TypeError, ValueError):
            pass
        try:
            src_dtypes.append(np.asarray(weights).dtype)
        except (TypeError, ValueError):
            pass
        if any(dt == np.float32 for dt in src_dtypes):
            return np.dtype(np.float32)
        return np.dtype(np.float64)

    float_dt = _resolve_float_dtype()
    complex_dt = np.dtype(np.complex64) if float_dt == np.float32 else np.dtype(np.complex128)

    # --- output dtype K
    def _resolve_return_dtype() -> np.dtype:
        if return_dtype in ("float32", "float64"):
            return np.dtype(np.float32) if return_dtype == "float32" else np.dtype(np.float64)
        return float_dt  # default: identical to calculation

    ret_dt = _resolve_return_dtype()

    # --- data cast
    A = _ensure_numpy(A, dtype=float_dt)
    B = A if B is None else _ensure_numpy(B, dtype=float_dt)
    n, nq = A.shape
    m = B.shape[0]

    w = _ensure_numpy(weights, dtype=float_dt)
    if w.ndim != 2 or w.shape[1] != nq:
        raise ValueError(f"`weights` must be [n_layers, n_qubits={nq}]")

    # --- if GPU + torch available + backend torch/auto => Torch streaming path
    is_gpu = "gpu" in device_name.lower()
    if is_gpu and gram_backend in ("torch", "auto"):
        try:
            import torch as _th  # type: ignore
            if _th.cuda.is_available():
                return _gram_torch_stream(
                    A, None if (B is A) else B,
                    weights_np=w,
                    device_name=device_name,
                    tile_size=tile_size,
                    symmetric=symmetric,
                    float_dt=float_dt,
                    ret_dt=ret_dt,
                )
        except ImportError:
            pass  # no torch → continue with existing paths

    # --- index chunking
    def _chunk_indices(n_items: int, chunk: int) -> list[list[int]]:
        return [list(range(s, min(s + chunk, n_items))) for s in range(0, n_items, chunk)]

    rows_a: list[list[int]] = _chunk_indices(n, max(1, tile_size))
    rows_b: list[list[int]] = rows_a if (B is A) else _chunk_indices(m, max(1, tile_size))

    # --- workers management (force 1 on GPU)
    if n_workers is None or n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)
    if is_gpu:
        n_workers = 1

    # --- Choose implementation for matmul (numpy/cupy/torch) — CPU => numpy forced
    matmul_mode = "numpy"
    cp = None
    th = None
    torch_device = None

    if not is_gpu:
        matmul_mode = "numpy"
    else:
        if gram_backend == "numpy":
            matmul_mode = "numpy"
        elif gram_backend in ("cupy", "auto"):
            try:
                import cupy as _cp  # type: ignore
                cp = _cp
                matmul_mode = "cupy"
            except ImportError:
                matmul_mode = "numpy"
        elif gram_backend == "torch":
            try:
                import torch as _th  # type: ignore
                th = _th
                if th.cuda.is_available():
                    torch_device = th.device("cuda")
                    matmul_mode = "torch"
                else:
                    matmul_mode = "numpy"
            except ImportError:
                matmul_mode = "numpy"

    # --- States production (progress include)
    if n_workers == 1:
        _pl_worker_init(w, device_name, nq,
                        "float32" if float_dt == np.float32 else "float64")

        iterA = rows_a
        pbarA = None
        if progress and tqdm is not None:
            pbarA = tqdm(total=len(rows_a), desc=f"{desc}: states A", leave=False)
        sa_parts = []
        for rs in iterA:
            sa_parts.append(_pl_states_for_rows(rs, A))
            if pbarA is not None:
                pbarA.update(1)
        if pbarA is not None:
            pbarA.close()
        sa = np.concatenate(sa_parts, axis=0)

        if B is A:
            sb = sa
        else:
            iterB = rows_b
            pbarB = None
            if progress and tqdm is not None:
                pbarB = tqdm(total=len(rows_b), desc=f"{desc}: states B", leave=False)
            sb_parts = []
            for rs in iterB:
                sb_parts.append(_pl_states_for_rows(rs, B))
                if pbarB is not None:
                    pbarB.update(1)
            if pbarB is not None:
                pbarB.close()
            sb = np.concatenate(sb_parts, axis=0)

    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=_pl_worker_init,
            initargs=(w, device_name, nq, "float32" if float_dt == np.float32 else "float64"),
        ) as pool:
            # A (progress via IMAP, which streams and preserves order)
            funcA = partial(_pl_states_for_rows, mat=A)
            iterA = pool.imap(funcA, rows_a, chunksize=1)
            pbarA = None
            if progress and tqdm is not None:
                pbarA = tqdm(total=len(rows_a), desc=f"{desc}: states A", leave=False)
            sa_parts = []
            for part in iterA:
                sa_parts.append(part)
                if pbarA is not None:
                    pbarA.update(1)
            if pbarA is not None:
                pbarA.close()
            sa = np.concatenate(sa_parts, axis=0)

            # B
            if B is A:
                sb = sa
            else:
                funcB = partial(_pl_states_for_rows, mat=B)
                iterB = pool.imap(funcB, rows_b, chunksize=1)
                pbarB = None
                if progress and tqdm is not None:
                    pbarB = tqdm(total=len(rows_b), desc=f"{desc}: states B", leave=False)
                sb_parts = []
                for part in iterB:
                    sb_parts.append(part)
                    if pbarB is not None:
                        pbarB.update(1)
                if pbarB is not None:
                    pbarB.close()
                sb = np.concatenate(sb_parts, axis=0)

    # --- K = |SA @ SB^H|^2 (tiled) -> output dtype = ret_dt
    k = np.empty((n, m), dtype=ret_dt)

    outer_iter = list(_tile_ranges(n, tile_size))
    if progress and tqdm is not None:
        outer_iter = tqdm(outer_iter, desc=f"{desc}: outer", leave=False)

    for i0, i1 in outer_iter:
        sa_blk = sa[i0:i1]

        inner_iter = list(_tile_ranges(m, tile_size))
        if progress and tqdm is not None:
            inner_iter = tqdm(inner_iter, desc=f"{desc}: inner[{i0}:{i1}]", leave=False)

        if matmul_mode == "cupy":
            sb_cache = None  # on convertit bloc par bloc
            sa_x = cp.asarray(sa_blk)

        elif matmul_mode == "torch":
            sb_cache = None
            sa_x = th.from_numpy(sa_blk).to(torch_device)

        for j0, j1 in inner_iter:
            sb_blk = sb[j0:j1]

            if matmul_mode == "cupy":
                sb_x = cp.asarray(sb_blk)
                g = sa_x @ cp.conj(sb_x).T
                mag2 = cp.abs(g) ** 2
                k[i0:i1, j0:j1] = cp.asnumpy(mag2).astype(ret_dt, copy=False)

            elif matmul_mode == "torch":
                sb_x = th.from_numpy(sb_blk).to(torch_device)
                g = sa_x @ th.conj(sb_x).T
                mag2 = th.abs(g) ** 2
                k[i0:i1, j0:j1] = mag2.detach().cpu().numpy().astype(ret_dt, copy=False)

            else:
                g = sa_blk @ sb_blk.conj().T
                mag2 = (np.abs(g) ** 2).astype(ret_dt, copy=False)
                k[i0:i1, j0:j1] = mag2

    if symmetric and (B is A):
        k = (k + k.T) * np.array(0.5, dtype=ret_dt)
    return k


def compute_kernel_matrix(
    X: Any,
    Y: Optional[Any] = None,
    *,
    weights: np.ndarray,
    device_name: str = "lightning.qubit",
    tile_size: int = 64,
    symmetric: bool = True,
    n_workers: int = 0,
    dtype: Optional[str] = None,          # "float32", "float64" ou None/auto (computation)
    return_dtype: Optional[str] = None,    # "float32", "float64" ou None/auto (output)
    gram_backend: str = "auto",            # "auto" | "numpy" | "cupy" | "torch"
    progress: bool = False,
    desc: str = "Gram",
) -> np.ndarray:
    """Public API (PennyLane only). X,Y: angles [N, nq], [M, nq]."""
    # limit BLAS threads in the parent process as well
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    return _gram_pennylane_angles_mp(
        X, Y,
        weights=weights,
        device_name=device_name,
        tile_size=tile_size,
        symmetric=symmetric,
        n_workers=n_workers,
        dtype=dtype,
        return_dtype=return_dtype,
        gram_backend=gram_backend,
        progress=progress,
        desc=desc,
    )
