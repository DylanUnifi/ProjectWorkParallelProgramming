"""Kernel computation and state generation helpers for pipeline backends."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import math
import os
import time

import numpy as np
from tqdm import tqdm

from scripts.pipeline_helpers import _create_qml_device, _ensure_numpy, _tile_ranges
from scripts.pipeline_gpu_optimizations import _compute_optimal_state_tile

CUDA_TEMPLATE = r"""
extern "C" __global__
void cgemm_abs2_os_full(const double2* __restrict__ A,
                        const double2* __restrict__ B,
                        double* __restrict__ C,
                        const int M, const int N, const int K_dim,
                        const int lda, const int ldb, const int ldc)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= M || j >= N) return;

    double sum_real = 0.0;
    double sum_imag = 0.0;
    const int row_a = i * lda;
    const int row_b = j * ldb;

    for (int k = 0; k < K_dim; ++k) {
        const double2 a_val = A[row_a + k];
        const double2 b_val = B[row_b + k];
        
        const double a_r = a_val.x;
        const double a_i = a_val.y;
        const double b_r = b_val.x;
        const double b_i = b_val.y;

        sum_real += (a_r * b_r + a_i * b_i);
        sum_imag += (a_i * b_r - a_r * b_i);
    }

    C[i * ldc + j] = (sum_real * sum_real) + (sum_imag * sum_imag);
}

extern "C" __global__
void cgemm_abs2_os_lower(const double2* __restrict__ A,
                         const double2* __restrict__ B,
                         double* __restrict__ C,
                         const int M, const int N, const int K_dim,
                         const int lda, const int ldb, const int ldc)
{
    if (blockIdx.x > blockIdx.y) return;

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= M || j >= N) return;
    if (M == N && j > i) return;

    double sum_real = 0.0;
    double sum_imag = 0.0;
    const int row_a = i * lda;
    const int row_b = j * ldb;

    for (int k = 0; k < K_dim; ++k) {
        const double2 a_val = A[row_a + k];
        const double2 b_val = B[row_b + k];
        
        const double a_r = a_val.x;
        const double a_i = a_val.y;
        const double b_r = b_val.x;
        const double b_i = b_val.y;

        sum_real += (a_r * b_r + a_i * b_i);
        sum_imag += (a_i * b_r - a_r * b_i);
    }

    C[i * ldc + j] = (sum_real * sum_real) + (sum_imag * sum_imag);
}
"""

_PL_W = None
_PL_NQ = None
_PL_DEVICE = None
_PL_QNODE = None
_PL_FLOAT_DTYPE = None
_PL_COMPLEX_DTYPE = None
_PL_ANGLE_SCALE = 1.0
_PL_RE_EMBED = False
_PL_EMBED_MODE = "ryrz"

_RAWKERNEL_NAME_ALIASES = {
    "cgemm_abs2_tiled_full": "cgemm_abs2_os_full",
    "cgemm_abs2_tiled_lower": "cgemm_abs2_os_lower",
}

_RAWKERNEL_CACHE = {}
_AUTOTUNE_CACHE_FILE = ".cuda_kernel_autotune.json"
_AUTOTUNE_CACHE: Dict[str, Tuple[int, int, int]] = {}
_COMPUTE_STREAM = None


def _round_to_pow2(x):
    return 2 ** int(np.ceil(np.log2(max(1, x))))


def _get_kernel(tm, tn, tk, name, double):
    import cupy as cp

    kernel_name = _RAWKERNEL_NAME_ALIASES.get(name, name)
    key = (kernel_name,)
    if key in _RAWKERNEL_CACHE:
        return _RAWKERNEL_CACHE[key]

    fn = cp.RawKernel(CUDA_TEMPLATE, kernel_name, options=("--std=c++14",))
    _RAWKERNEL_CACHE[key] = fn
    return fn


def _launch_output_stationary_kernel(kernel_fn, a_tile, b_tile, block_x, block_y, out_tile=None):
    import cupy as cp

    a_contig = cp.ascontiguousarray(a_tile, dtype=cp.complex128)
    b_contig = cp.ascontiguousarray(b_tile, dtype=cp.complex128)

    if a_contig.ndim != 2 or b_contig.ndim != 2:
        raise ValueError("Output-stationary Gram kernel expects 2D input tiles.")
    if a_contig.shape[1] != b_contig.shape[1]:
        raise ValueError("Input tiles must share the same state dimension.")

    bi = int(a_contig.shape[0])
    bj = int(b_contig.shape[0])
    k_dim = int(a_contig.shape[1])
    block_x = max(1, int(block_x))
    block_y = max(1, int(block_y))

    if block_x * block_y > 1024:
        raise ValueError(f"Invalid CUDA block shape ({block_x}, {block_y}): exceeds 1024 threads.")

    if out_tile is None:
        out_tile = cp.empty((bi, bj), dtype=cp.float64)

    grid = (math.ceil(bj / block_x), math.ceil(bi / block_y), 1)
    block = (block_x, block_y, 1)
    args = (a_contig, b_contig, out_tile, bi, bj, k_dim, k_dim, k_dim, bj)
    kernel_fn(grid, block, args)
    return out_tile, a_contig, b_contig, grid, block, args


def _normalize_state_tile_cp(s_tile):
    import cupy as cp

    norms = cp.linalg.norm(s_tile, axis=1, keepdims=True)
    norms = cp.where(norms > 1e-12, norms, 1.0)
    return s_tile / norms


def _load_autotune_cache():
    global _AUTOTUNE_CACHE
    if os.path.exists(_AUTOTUNE_CACHE_FILE):
        try:
            with open(_AUTOTUNE_CACHE_FILE, "r") as f:
                data = np.load(f, allow_pickle=True)
                _AUTOTUNE_CACHE = {k: tuple(v) for k, v in data.item().items()}
        except Exception:
            try:
                import json

                with open(_AUTOTUNE_CACHE_FILE, "r") as f:
                    data = json.load(f)
                    _AUTOTUNE_CACHE = {k: tuple(v) for k, v in data.items()}
            except Exception:
                pass


def _save_autotune_cache():
    try:
        import json

        with open(_AUTOTUNE_CACHE_FILE, "w") as f:
            json.dump({k: list(v) for k, v in _AUTOTUNE_CACHE.items()}, f, indent=2)
    except Exception:
        pass


def _autotune_kernel_tiles(nq: int, is_double: bool = False,
                           test_size: int = 512, warmup: int = 2, trials: int = 5) -> Tuple[int, int, int]:
    import cupy as cp

    cache_key = f"nq{nq}_{'double' if is_double else 'float'}"
    if cache_key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[cache_key]

    dim = 1 << nq
    dtype_real = cp.float64
    rng = cp.random.default_rng(42)
    SA = rng.random((test_size, dim), dtype=dtype_real) + 1j * rng.random((test_size, dim), dtype=dtype_real)
    SB = rng.random((test_size, dim), dtype=dtype_real) + 1j * rng.random((test_size, dim), dtype=dtype_real)
    K_out = cp.empty((test_size, test_size), dtype=dtype_real)

    if nq >= 14:
        candidates_m_n = [16, 32]
        candidates_k = [16, 32]
    elif nq >= 12:
        candidates_m_n = [16, 32, 64]
        candidates_k = [16, 32, 64]
    else:
        candidates_m_n = [16, 32, 64]
        candidates_k = [16, 32, 64, 128]

    results = []
    for tm in candidates_m_n:
        for tn in candidates_m_n:
            for tk in candidates_k:
                bytes_per_complex = 16 if is_double else 8
                shared_mem = (tm * tk + tn * tk) * bytes_per_complex
                if shared_mem > 48 * 1024:
                    continue

                try:
                    kernel = _get_kernel(tm, tn, tk, "cgemm_abs2_tiled_full", is_double)
                    for _ in range(warmup):
                        _launch_output_stationary_kernel(kernel, SA, SB, tn, tm, out_tile=K_out)
                    cp.cuda.runtime.deviceSynchronize()

                    times = []
                    for _ in range(trials):
                        start = time.perf_counter()
                        _launch_output_stationary_kernel(kernel, SA, SB, tn, tm, out_tile=K_out)
                        cp.cuda.runtime.deviceSynchronize()
                        times.append(time.perf_counter() - start)

                    results.append((np.mean(times), tm, tn, tk))
                except Exception:
                    continue

    if not results:
        return (32, 32, 32)

    results.sort()
    best = results[0]
    best_config = (best[1], best[2], best[3])
    _AUTOTUNE_CACHE[cache_key] = best_config
    _save_autotune_cache()
    return best_config


def _pl_worker_init(w, dev, nq, fdtype, ascale, re_emb, mode):
    global _PL_W, _PL_NQ, _PL_DEVICE, _PL_QNODE, _PL_FLOAT_DTYPE, _PL_COMPLEX_DTYPE, _PL_ANGLE_SCALE, _PL_RE_EMBED, _PL_EMBED_MODE
    os.environ["OMP_NUM_THREADS"] = "1"
    _PL_FLOAT_DTYPE = np.dtype(np.float32) if fdtype == "float32" else np.dtype(np.float64)
    _PL_COMPLEX_DTYPE = np.dtype(np.complex64) if fdtype == "float32" else np.dtype(np.complex128)
    _PL_W = _ensure_numpy(w, _PL_FLOAT_DTYPE)
    _PL_NQ, _PL_DEVICE = int(nq), str(dev)
    _PL_ANGLE_SCALE, _PL_RE_EMBED, _PL_EMBED_MODE = float(ascale), bool(re_emb), str(mode)
    _PL_QNODE = None


def _pl_get_qnode():
    global _PL_QNODE
    if _PL_QNODE is None:
        import pennylane as qml

        dev = _create_qml_device(_PL_DEVICE, wires=_PL_NQ, c_dtype=_PL_COMPLEX_DTYPE)

        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row):
            theta = qml.math.asarray(theta_row, dtype=_PL_FLOAT_DTYPE)

            def _embed(v):
                if _PL_EMBED_MODE == "angle":
                    qml.AngleEmbedding(_PL_ANGLE_SCALE * v[:_PL_NQ], wires=range(_PL_NQ), rotation="Y")
                else:
                    for i in range(_PL_NQ):
                        qml.RY(_PL_ANGLE_SCALE * v[i], wires=i)
                        if _PL_EMBED_MODE == "ryrz":
                            qml.RZ(_PL_ANGLE_SCALE * v[i], wires=i)

            if _PL_RE_EMBED:
                for l in range(_PL_W.shape[0]):
                    _embed(theta)
                    qml.templates.BasicEntanglerLayers(_PL_W[l:l + 1], wires=range(_PL_NQ))
            else:
                _embed(theta)
                qml.templates.BasicEntanglerLayers(_PL_W, wires=range(_PL_NQ))
            return qml.state()

        _PL_QNODE = _state
    return _PL_QNODE


def _pl_states_for_rows(rows, mat):
    qnode = _pl_get_qnode()
    out = np.empty((len(rows), 1 << _PL_NQ), dtype=_PL_COMPLEX_DTYPE)
    for t, idx in enumerate(rows):
        out[t] = qnode(mat[idx])
    return out


def _build_states_sequential_torch(state_fn, x, progress=False, desc="States"):
    import torch as th

    n_rows = int(x.shape[0])
    if n_rows == 0:
        raise ValueError("Cannot build states for an empty batch")

    first_state = state_fn(x[0])
    states = th.empty((n_rows, first_state.shape[-1]), dtype=first_state.dtype, device=first_state.device)
    states[0] = first_state

    iterator = range(1, n_rows)
    if progress and n_rows > 1:
        iterator = tqdm(iterator, total=n_rows - 1, desc=desc, leave=False)

    for idx in iterator:
        states[idx] = state_fn(x[idx])

    return states


def _safe_state_chunk_size(nq: int, dtype, reserve_fraction: float = 0.5) -> int:
    try:
        import torch as th

        device = th.cuda.current_device()
        free_bytes, _total_bytes = th.cuda.mem_get_info(device)
        bytes_per_complex = 8 if dtype == np.float32 else 16
        bytes_per_state = (1 << int(nq)) * bytes_per_complex
        if bytes_per_state <= 0:
            return 64

        usable_bytes = max(0, int(free_bytes * reserve_fraction))
        chunk = usable_bytes // bytes_per_state
        return max(64, min(int(chunk) if chunk > 0 else 64, 256))
    except Exception:
        return 64


def _build_states_block_torch_cuda(x_blk, w_np, dev_name, ascale, re_emb, mode,
                                   progress=False, desc="States"):
    import torch as th
    import pennylane as qml

    nq = int(x_blk.shape[1])
    t_float = th.float32 if x_blk.dtype == np.float32 else th.float64
    t_cplx = th.complex64 if t_float == th.float32 else th.complex128

    x = th.from_numpy(np.ascontiguousarray(x_blk)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)

    dev = _create_qml_device(
        dev_name,
        wires=nq,
        c_dtype=np.complex64 if t_float == th.float32 else np.complex128,
    )

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if mode == "angle":
                qml.AngleEmbedding(ascale * v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(ascale * v[i], wires=i)
                    if mode == "ryrz":
                        qml.RZ(ascale * v[i], wires=i)

        if re_emb:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try:
        states = _state(x)
        if states.ndim != 2 or states.shape[0] != x.shape[0]:
            raise ValueError("Batching not natively supported")
    except Exception:
        if progress and x.shape[0] > 1:
            print(f"Warning: Native batching unavailable for {desc.lower()}, using sequential state construction.")
        chunk_size = _safe_state_chunk_size(nq, t_float)
        if x.shape[0] > chunk_size:
            chunks = []
            for start in range(0, x.shape[0], chunk_size):
                stop = min(start + chunk_size, x.shape[0])
                chunks.append(_build_states_block_torch_cuda(
                    x_blk[start:stop], w_np, dev_name, ascale, re_emb, mode,
                    progress=progress, desc=desc,
                ))
            return th.cat(chunks, dim=0)

        states = _build_states_sequential_torch(_state, x, progress=progress, desc=desc)

    try:
        states = states.to(device="cuda", dtype=t_cplx, non_blocking=False)
    except th.cuda.OutOfMemoryError:
        if x.shape[0] > 1:
            if progress:
                print(f"Warning: CUDA OOM for {desc.lower()} block of {x.shape[0]}, retrying with smaller chunks.")
            chunk_size = max(1, x.shape[0] // 2)
            chunks = []
            for start in range(0, x.shape[0], chunk_size):
                stop = min(start + chunk_size, x.shape[0])
                chunks.append(_build_states_block_torch_cuda(
                    x_blk[start:stop], w_np, dev_name, ascale, re_emb, mode,
                    progress=progress, desc=desc,
                ))
            return th.cat(chunks, dim=0)
        raise

    th.cuda.synchronize()
    return states


def _torch_cuda_to_cupy(t):
    import cupy as cp

    return cp.from_dlpack(t)


def _build_all_states_torch_cuda(x_all, w_np, dev_name, ascale, re_emb, mode,
                                 use_pinned=True, progress=False, desc="States"):
    import torch as th
    import pennylane as qml

    nq = int(x_all.shape[1])
    t_float = th.float32 if x_all.dtype == np.float32 else th.float64
    t_cplx = th.complex64 if t_float == th.float32 else th.complex128

    if use_pinned:
        x = th.from_numpy(np.ascontiguousarray(x_all)).pin_memory().to("cuda", dtype=t_float, non_blocking=True)
        w = th.from_numpy(np.ascontiguousarray(w_np)).pin_memory().to("cuda", dtype=t_float, non_blocking=True)
    else:
        x = th.from_numpy(np.ascontiguousarray(x_all)).to("cuda", dtype=t_float, non_blocking=True)
        w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)

    dev = _create_qml_device(
        dev_name,
        wires=nq,
        c_dtype=np.complex64 if t_float == th.float32 else np.complex128,
    )

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if mode == "angle":
                qml.AngleEmbedding(ascale * v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(ascale * v[i], wires=i)
                    if mode == "ryrz":
                        qml.RZ(ascale * v[i], wires=i)

        if re_emb:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try:
        states = _state(x)
        if states.ndim != 2 or states.shape[0] != x.shape[0]:
            raise ValueError("Batching not natively supported")
    except Exception:
        if progress and x.shape[0] > 1:
            print(f"Warning: Native batching unavailable for {desc.lower()}, using sequential state construction.")
        states = _build_states_sequential_torch(_state, x, progress=progress, desc=desc)

    states = states.to(device="cuda", dtype=t_cplx, non_blocking=False).contiguous()
    th.cuda.synchronize()
    return _torch_cuda_to_cupy(states)


def _get_compute_stream():
    global _COMPUTE_STREAM
    if _COMPUTE_STREAM is None:
        import cupy as cp

        _COMPUTE_STREAM = cp.cuda.Stream(non_blocking=True)
    return _COMPUTE_STREAM


def _dispatch_kernel_async(kernel_fn, grid, block, args, stream=None):
    if stream is None:
        stream = _get_compute_stream()

    with stream:
        kernel_fn(grid, block, args)


def _gram_torch_stream(a_np, b_np, weights_np, device_name, tile_size, symmetric, float_dt, ret_dt, angle_scale, re_embed_between_layers, embed_mode,
                       use_pinned_memory=False, use_cuda_streams=False, use_amp=False, use_compile=False, tensorcore_precision="fp32"):
    import torch as th
    import pennylane as qml

    n, nq = a_np.shape
    m = n if b_np is None else b_np.shape[0]

    tf = th.float32 if float_dt == np.float32 else th.float64
    tc = th.complex64 if float_dt == np.float32 else th.complex128

    if use_pinned_memory and th.cuda.is_available():
        a = th.from_numpy(a_np).pin_memory().to("cuda", dtype=tf, non_blocking=True)
        b = a if b_np is None else th.from_numpy(b_np).pin_memory().to("cuda", dtype=tf, non_blocking=True)
        w = th.from_numpy(weights_np).pin_memory().to("cuda", dtype=tf, non_blocking=True)
    else:
        a = th.from_numpy(a_np).to("cuda", dtype=tf)
        b = a if b_np is None else th.from_numpy(b_np).to("cuda", dtype=tf)
        w = th.from_numpy(weights_np).to("cuda", dtype=tf)

    dev = _create_qml_device(
        device_name,
        wires=nq,
        c_dtype=np.complex64 if float_dt == np.float32 else np.complex128,
    )

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if embed_mode == "angle":
                qml.AngleEmbedding(float(angle_scale) * v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(float(angle_scale) * v[i], wires=i)
                    if embed_mode == "ryrz":
                        qml.RZ(float(angle_scale) * v[i], wires=i)
        if re_embed_between_layers:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try:
        build = lambda x: _state(x).to(dtype=tc)
        _ = build(a[:2])
    except Exception:
        build = lambda x: th.stack([_state(x[i]) for i in range(len(x))]).to(dtype=tc)

    autocast_dtype = None
    if tensorcore_precision == "bf16" and th.cuda.is_bf16_supported():
        autocast_dtype = th.bfloat16
        th.set_float32_matmul_precision('high')
    elif tensorcore_precision == "fp16":
        autocast_dtype = th.float16
    elif tensorcore_precision == "tf32":
        th.set_float32_matmul_precision('medium')

    enable_amp = (autocast_dtype is not None)

    if use_compile and hasattr(th, 'compile'):
        try:
            build = th.compile(build, mode="reduce-overhead")
        except Exception:
            pass

    compute_stream = None
    if use_cuda_streams and th.cuda.is_available():
        compute_stream = th.cuda.Stream()

    k = th.empty((n, m), device="cuda", dtype=tf)

    def compute_kernel_block(i0, i1):
        sa = build(a[i0:i1])
        j_start = i0 if (symmetric and b_np is None) else 0
        for j0 in range(j_start, m, tile_size):
            j1 = min(j0 + tile_size, m)
            sb = sa if (b_np is None and j0 == i0) else build(b[j0:j1])
            res = (sa @ sb.conj().T).abs().square()
            k[i0:i1, j0:j1] = res
            if symmetric and b_np is None and j0 > i0:
                k[j0:j1, i0:i1] = res.T
        del sa

    with th.no_grad():
        with th.amp.autocast(device_type="cuda", enabled=enable_amp, dtype=autocast_dtype):
            for i0 in range(0, n, tile_size):
                i1 = min(i0 + tile_size, n)
                if compute_stream:
                    with th.cuda.stream(compute_stream):
                        compute_kernel_block(i0, i1)
                else:
                    compute_kernel_block(i0, i1)
                th.cuda.empty_cache()

    if compute_stream:
        compute_stream.synchronize()

    return k.cpu().numpy().astype(ret_dt)


def _gram_pennylane_angles_mp(
        A, B, weights, device_name, tile_size, symmetric, n_workers,
        dtype, return_dtype, progress, desc, angle_scale, re_embed_between_layers, embed_mode
):
    import multiprocessing as mp

    f_dt = np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
    r_dt = np.dtype(np.float32) if return_dtype == "float32" else np.dtype(np.float64)

    A = _ensure_numpy(A, f_dt)
    B = A if B is None else _ensure_numpy(B, f_dt)
    n, nq = A.shape
    m = B.shape[0]
    w = _ensure_numpy(weights, f_dt)

    def _chunk(n, c):
        return [list(range(s, min(s + c, n))) for s in range(0, n, c)]

    ra = _chunk(n, max(1, tile_size))
    rb = ra if (B is A) else _chunk(m, max(1, tile_size))

    if n_workers is None or n_workers <= 0:
        n_workers = 1

    initargs = (w, device_name, nq, "float32" if f_dt == np.float32 else "float64", angle_scale, re_embed_between_layers, embed_mode)

    if n_workers == 1:
        _pl_worker_init(*initargs)
        sa = np.concatenate([_pl_states_for_rows(r, A) for r in ra], axis=0)
        sb = sa if (B is A) else np.concatenate([_pl_states_for_rows(r, B) for r in rb], axis=0)
    else:
        ctx = mp.get_context("spawn")
        from functools import partial

        with ctx.Pool(processes=n_workers, initializer=_pl_worker_init, initargs=initargs) as pool:
            sa = np.concatenate(list(pool.imap(partial(_pl_states_for_rows, mat=A), ra)), axis=0)
            sb = sa if (B is A) else np.concatenate(list(pool.imap(partial(_pl_states_for_rows, mat=B), rb)), axis=0)

    k = np.empty((n, m), dtype=r_dt)
    for i0, i1 in _tile_ranges(n, tile_size):
        sa_blk = sa[i0:i1]
        j_start = i0 if (symmetric and (B is A)) else 0
        for j0, j1 in _tile_ranges(m, tile_size):
            if j0 < j_start:
                continue
            sb_blk = sb[j0:j1]
            g = sa_blk @ sb_blk.conj().T
            mag2 = (np.abs(g) ** 2).astype(r_dt)
            k[i0:i1, j0:j1] = mag2
            if symmetric and (B is A) and j0 > i0:
                k[j0:j1, i0:i1] = mag2.T
    return k
