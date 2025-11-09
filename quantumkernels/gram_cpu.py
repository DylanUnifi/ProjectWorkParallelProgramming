import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from functools import partial

from .helpers import ensure_numpy, tile_ranges
from .pl_worker import pl_worker_init, pl_states_for_rows


def gram_pennylane_angles_mp(
        A, B=None, *, weights, device_name="lightning.qubit",
        tile_size=64, symmetric=True, n_workers=0,
        dtype=None, return_dtype=None, progress=False,
        desc="Gram", angle_scale=1.0,
        re_embed_between_layers=False, embed_mode="ryrz"):
    """Kernel de fidélité PennyLane, parallèle (multiprocessing), tuilé (CPU)."""

    def _resolve_float_dtype():
        if dtype in ("float32", "float64"):
            return np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
        src_dtypes = [np.asarray(A).dtype, np.asarray(weights).dtype]
        if any(dt == np.float32 for dt in src_dtypes):
            return np.float32
        return np.float64

    float_dt = _resolve_float_dtype()
    ret_dt = np.dtype(return_dtype) if return_dtype in ("float32", "float64") else float_dt

    A = ensure_numpy(A, dtype=float_dt)
    B = A if B is None else ensure_numpy(B, dtype=float_dt)
    n, nq = A.shape
    m = B.shape[0]
    w = ensure_numpy(weights, dtype=float_dt)
    if w.ndim != 2 or w.shape[1] != nq:
        raise ValueError(f"`weights` must be [n_layers, n_qubits={nq}]")

    def _chunk_indices(n_items, chunk):
        return [list(range(s, min(s + chunk, n_items))) for s in range(0, n_items, chunk)]

    rows_a = _chunk_indices(n, max(1, tile_size))
    rows_b = rows_a if (B is A) else _chunk_indices(m, max(1, tile_size))

    if n_workers is None or n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)

    initargs = (w, device_name, nq, "float32" if float_dt == np.float32 else "float64",
                angle_scale, re_embed_between_layers, embed_mode)

    if n_workers == 1:
        pl_worker_init(*initargs)
        sa = np.concatenate([pl_states_for_rows(rs, A) for rs in rows_a], axis=0)
        sb = sa if B is A else np.concatenate([pl_states_for_rows(rs, B) for rs in rows_b], axis=0)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers,
                      initializer=pl_worker_init,
                      initargs=initargs) as pool:
            funcA = partial(pl_states_for_rows, mat=A)
            sa = np.concatenate(list(pool.imap(funcA, rows_a, chunksize=1)), axis=0)
            if B is A:
                sb = sa
            else:
                funcB = partial(pl_states_for_rows, mat=B)
                sb = np.concatenate(list(pool.imap(funcB, rows_b, chunksize=1)), axis=0)

    k = np.empty((n, m), dtype=ret_dt)
    outer_iter = list(tile_ranges(n, tile_size))
    if tqdm is not None and progress:
        outer_iter = tqdm(outer_iter, desc=f"{desc}: outer", leave=False)

    for i0, i1 in outer_iter:
        sa_blk = np.ascontiguousarray(sa[i0:i1])
        inner_iter = list(tile_ranges(m, tile_size))
        j_start = 0 if not (symmetric and (B is A)) else i0
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
