# pipeline_backends.py — PennyLane-only, multiprocessing-safe (spawn-friendly)
from typing import Optional, Any
import os
import numpy as np

# ---------- helpers ----------
def _ensure_numpy(a: Any, dtype=np.float64) -> np.ndarray:
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().astype(dtype, copy=False)
    except Exception:
        pass
    return np.asarray(a, dtype=dtype, order="C")

def _tile_ranges(n: int, tile: int):
    for s in range(0, n, tile):
        e = min(s + tile, n)
        yield s, e

# ---------- GLOBALS for worker processes ----------
_PL_W: Optional[np.ndarray] = None
_PL_NQ: Optional[int] = None
_PL_DEVICE: Optional[str] = None
_PL_QNODE = None  # cached qnode per process

def _pl_worker_init(W_local: np.ndarray, device_name: str, nq: int):
    """Initializer called once per worker process."""
    global _PL_W, _PL_NQ, _PL_DEVICE, _PL_QNODE
    _PL_W = np.asarray(W_local, dtype=np.float64)
    _PL_NQ = int(nq)
    _PL_DEVICE = str(device_name)
    _PL_QNODE = None
    # evite la sur-parallélisation BLAS dans chaque worker
    os.environ.setdefault("OMP_NUM_THREADS", "1")

def _pl_get_qnode():
    """Lazy-create & cache the qnode in this worker process."""
    global _PL_QNODE
    if _PL_QNODE is None:
        import pennylane as qml
        dev = qml.device(_PL_DEVICE, wires=_PL_NQ, shots=None)

        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row: np.ndarray):
            # encodage: RY+RZ puis entangler avec W
            for i in range(_PL_NQ):
                qml.RY(theta_row[i], wires=i)
                qml.RZ(theta_row[i], wires=i)
            qml.templates.BasicEntanglerLayers(_PL_W, wires=range(_PL_NQ))
            return qml.state()
        _PL_QNODE = _state
    return _PL_QNODE

def _pl_states_for_rows(rows: list[int], MAT: np.ndarray) -> np.ndarray:
    """Top-level worker function (picklable): build statevectors for given row indices."""
    qnode = _pl_get_qnode()
    out = np.empty((len(rows), 1 << _PL_NQ), dtype=np.complex128)
    for t, idx in enumerate(rows):
        out[t] = qnode(MAT[idx])
    return out

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
) -> np.ndarray:
    """Parallel PennyLane kernel (fidelity) with multiprocessing (spawn-safe)."""
    import multiprocessing as mp

    A = _ensure_numpy(A, dtype=np.float64)
    B = A if B is None else _ensure_numpy(B, dtype=np.float64)
    n, nq = A.shape
    m = B.shape[0]

    W = np.asarray(weights, dtype=np.float64)
    if W.ndim != 2 or W.shape[1] != nq:
        raise ValueError(f"`weights` must be [n_layers, n_qubits={nq}]")

    # chunks of row indices
    def _chunk_indices(N, chunk):
        return [list(range(s, min(s+chunk, N))) for s in range(0, N, chunk)]

    rows_A = _chunk_indices(n, max(1, tile_size))
    rows_B = rows_A if (B is A) else _chunk_indices(m, max(1, tile_size))

    if n_workers is None or n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)

    # special-case: single process (debug or very small)
    if n_workers == 1:
        _pl_worker_init(W, device_name, nq)
        SA = np.concatenate([_pl_states_for_rows(rs, A) for rs in rows_A], axis=0)
        SB = SA if (B is A) else np.concatenate([_pl_states_for_rows(rs, B) for rs in rows_B], axis=0)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers,
                      initializer=_pl_worker_init,
                      initargs=(W, device_name, nq)) as pool:
            SA_parts = pool.starmap(_pl_states_for_rows, [(rs, A) for rs in rows_A])
            SA = np.concatenate(SA_parts, axis=0)
            if B is A:
                SB = SA
            else:
                SB_parts = pool.starmap(_pl_states_for_rows, [(rs, B) for rs in rows_B])
                SB = np.concatenate(SB_parts, axis=0)

    # K = |SA @ SB^H|^2 (tuilé pour RAM)
    K = np.empty((n, m), dtype=np.float64)
    for i0, i1 in _tile_ranges(n, tile_size):
        SA_blk = SA[i0:i1]
        for j0, j1 in _tile_ranges(m, tile_size):
            SB_blk = SB[j0:j1]
            G = SA_blk @ SB_blk.conj().T
            K[i0:i1, j0:j1] = (np.abs(G) ** 2).real

    if symmetric and (B is A):
        K = 0.5 * (K + K.T)
    return K

def compute_kernel_matrix(
    X: Any,
    Y: Optional[Any] = None,
    *,
    weights: np.ndarray,
    device_name: str = "lightning.qubit",
    tile_size: int = 64,
    symmetric: bool = True,
    n_workers: int = 0,
) -> np.ndarray:
    """Public API (PennyLane only). X,Y are angles [N, nq], [M, nq]."""
    # limiter les threads BLAS dans le process parent aussi
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return _gram_pennylane_angles_mp(
        X, Y,
        weights=weights,
        device_name=device_name,
        tile_size=tile_size,
        symmetric=symmetric,
        n_workers=n_workers,
    )
