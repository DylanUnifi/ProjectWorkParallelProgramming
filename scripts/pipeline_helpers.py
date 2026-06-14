"""Shared helper functions for kernel pipeline backends."""

from typing import Any, Optional

import numpy as np


def _ensure_numpy(a: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
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


def _tile_ranges(n: int, tile: int):
    for s in range(0, n, tile):
        e = min(s + tile, n)
        yield s, e


def _normalize_diag_inplace(K: np.ndarray):
    if K.shape[0] != K.shape[1]:
        return
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)


def _normalize_cross_inplace(K: np.ndarray, diag_left: np.ndarray, diag_right: np.ndarray):
    if K.size == 0:
        return
    dtype = K.dtype if np.issubdtype(K.dtype, np.floating) else np.float64
    d_left = np.sqrt(np.clip(np.asarray(diag_left, dtype=dtype), 1e-12, None))
    d_right = np.sqrt(np.clip(np.asarray(diag_right, dtype=dtype), 1e-12, None))
    K /= d_left[:, None] * d_right[None, :]


def _create_qml_device(device_name: Any, wires: Any, c_dtype=None):
    """Create the requested PennyLane device without switching to a different backend."""
    import pennylane as qml

    if c_dtype is not None:
        try:
            return qml.device(device_name, wires=wires, shots=None, c_dtype=c_dtype)
        except Exception:
            pass
    return qml.device(device_name, wires=wires, shots=None)


def _compute_self_kernel_diag(
        X: Any, *, weights: np.ndarray,
        device_name: str, tile_size: int, symmetric: bool,
        n_workers: int, dtype: str, return_dtype: str,
        gram_backend: str, angle_scale: float, re_embed_between_layers: bool,
        embed_mode: str, jitter: float, state_tile: int,
        tile_m: Any, tile_n: Any, tile_k: Any, autotune: bool,
        precompute_all_states: bool, vram_fraction: float,
        dynamic_batch: bool, num_streams: int, learn_tiles: bool,
        profile_memory: bool, use_cuda_graphs: bool, verbose_profile: bool,
        use_pinned_memory: bool, use_cuda_streams: bool,
        use_amp: bool, use_compile: bool, tensorcore_precision: str
) -> np.ndarray:
    f_dt = np.float32 if dtype == "float32" else np.float64
    r_dt = np.float32 if return_dtype == "float32" else np.float64
    x_np = _ensure_numpy(X, f_dt)
    nq = int(x_np.shape[1])
    w_np = _ensure_numpy(weights, f_dt)

    if gram_backend == "cuda_states" and nq >= 14 and dtype == "float32":
        f_dt = np.float64
        x_np = x_np.astype(f_dt, copy=False)
        w_np = _ensure_numpy(weights, f_dt)

    block_size = int(state_tile) if int(state_tile) > 0 else int(tile_size)
    block_size = max(1, block_size)

    def _finalize_diag(norm_sq):
        norm_sq = np.asarray(norm_sq, dtype=np.float64)
        return (norm_sq * norm_sq).astype(r_dt, copy=False)

    def _diag_from_gpu_states(normalize_states: bool):
        import cupy as cp

        from scripts.pipeline_compute import (
            _build_states_block_torch_cuda,
            _normalize_state_tile_cp,
            _torch_cuda_to_cupy,
        )

        diag = np.empty(x_np.shape[0], dtype=r_dt)
        for i0, i1 in _tile_ranges(x_np.shape[0], block_size):
            s_th = _build_states_block_torch_cuda(
                x_np[i0:i1], w_np, device_name, angle_scale, re_embed_between_layers, embed_mode,
                progress=False, desc="SelfDiag"
            )
            s_cp = _torch_cuda_to_cupy(s_th)
            if normalize_states:
                s_cp = _normalize_state_tile_cp(s_cp)
            norm_sq = cp.sum(cp.abs(s_cp) ** 2, axis=1)
            diag[i0:i1] = _finalize_diag(cp.asnumpy(norm_sq))
            del s_cp, s_th
        cp.get_default_memory_pool().free_all_blocks()
        return diag

    def _diag_from_cpu_states():
        chunk_size = max(1, int(tile_size))
        chunks = [list(range(s, min(s + chunk_size, x_np.shape[0]))) for s in range(0, x_np.shape[0], chunk_size)]
        initargs = (
            w_np,
            device_name,
            nq,
            "float32" if f_dt == np.float32 else "float64",
            angle_scale,
            re_embed_between_layers,
            embed_mode,
        )

        if n_workers is None or n_workers <= 1:
            from scripts.pipeline_compute import _pl_states_for_rows, _pl_worker_init

            _pl_worker_init(*initargs)
            states_blocks = (_pl_states_for_rows(rows, x_np) for rows in chunks)
        else:
            import multiprocessing as mp
            from functools import partial

            from scripts.pipeline_compute import _pl_states_for_rows, _pl_worker_init

            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=n_workers, initializer=_pl_worker_init, initargs=initargs)
            states_blocks = pool.imap(partial(_pl_states_for_rows, mat=x_np), chunks)

        diag_blocks = []
        try:
            for states in states_blocks:
                norm_sq = np.sum(np.abs(states) ** 2, axis=1)
                diag_blocks.append(_finalize_diag(norm_sq))
        finally:
            if n_workers is not None and n_workers > 1:
                pool.close()
                pool.join()

        if not diag_blocks:
            return np.empty((0,), dtype=r_dt)
        return np.concatenate(diag_blocks, axis=0)

    if gram_backend == "cuda_states":
        return _diag_from_gpu_states(normalize_states=(nq >= 12))

    if gram_backend in ["torch", "auto"] and "gpu" in str(device_name):
        try:
            return _diag_from_gpu_states(normalize_states=False)
        except Exception:
            if gram_backend == "torch":
                raise

    return _diag_from_cpu_states()