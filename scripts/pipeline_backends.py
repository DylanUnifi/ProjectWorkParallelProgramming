# pipeline_backends.py — Optimized & Synchronized for CUDA Stability
from typing import Optional, Any, Callable, Tuple, List
import os
import sys
import numpy as np
from tqdm import tqdm

# =====================================================================
# Helpers
# =====================================================================
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
    if K.shape[0] != K.shape[1]: return
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)

def _setup_cupy():
    import cupy as cp
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    # Warmup & Sync
    _ = cp.ones((1,), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

    # Robust Include Path for NVRTC
    candidates = [
        os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME"),
        "/usr/local/cuda", "/opt/cuda",
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include")
    ]
    include_path = None
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "include")):
            include_path = os.path.join(path, "include")
            break
        elif path and os.path.exists(path) and path.endswith("include"):
            include_path = path
            break
    
    if include_path:
        opts = f"-I{include_path}"
        existing = os.environ.get("CUPY_NVRTC_OPTIONS", "")
        if include_path not in existing:
            os.environ["CUPY_NVRTC_OPTIONS"] = existing + " " + opts

# =====================================================================
# Worker globals (multiprocessing path)
# =====================================================================
_pl_w = None
_pl_nq = None
_pl_device = None
_pl_qnode = None
_pl_float_dtype = None
_pl_complex_dtype = None
_pl_angle_scale = 1.0
_pl_re_embed = False
_pl_embed_mode = "ryrz"

def _pl_worker_init(w, dev, nq, fdtype, ascale, re_emb, mode):
    global _pl_w, _pl_nq, _pl_device, _pl_float_dtype, _pl_complex_dtype, _pl_angle_scale, _pl_re_embed, _pl_embed_mode
    os.environ["OMP_NUM_THREADS"] = "1"
    _pl_float_dtype = np.dtype(np.float32) if fdtype == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if fdtype == "float32" else np.dtype(np.complex128)
    _pl_w = _ensure_numpy(w, _pl_float_dtype)
    _pl_nq, _pl_device = int(nq), str(dev)
    _pl_angle_scale, _pl_re_embed, _pl_embed_mode = float(ascale), bool(re_emb), str(mode)

def _pl_get_qnode():
    global _pl_qnode
    if _pl_qnode is None:
        import pennylane as qml
        try:
            dev = qml.device(_pl_device, wires=_pl_nq, shots=None, c_dtype=_pl_complex_dtype)
        except:
            # Fallback for older PennyLane versions or if c_dtype not supported
            dev = qml.device(_pl_device, wires=_pl_nq, shots=None)
            
        @qml.qnode(dev, interface=None, diff_method=None)
        def _state(theta_row):
            theta = qml.math.asarray(theta_row, dtype=_pl_float_dtype)
            def _embed(v):
                if _pl_embed_mode == "angle": qml.AngleEmbedding(_pl_angle_scale*v[:_pl_nq], wires=range(_pl_nq), rotation="Y")
                else:
                    for i in range(_pl_nq):
                        qml.RY(_pl_angle_scale*v[i], wires=i)
                        if _pl_embed_mode=="ryrz": qml.RZ(_pl_angle_scale*v[i], wires=i)
            if _pl_re_embed:
                for l in range(_pl_w.shape[0]):
                    _embed(theta)
                    qml.templates.BasicEntanglerLayers(_pl_w[l:l+1], wires=range(_pl_nq))
            else:
                _embed(theta)
                qml.templates.BasicEntanglerLayers(_pl_w, wires=range(_pl_nq))
            return qml.state()
        _pl_qnode = _state
    return _pl_qnode

def _pl_states_for_rows(rows, mat):
    qnode = _pl_get_qnode()
    out = np.empty((len(rows), 1 << _pl_nq), dtype=_pl_complex_dtype)
    for t, idx in enumerate(rows): out[t] = qnode(mat[idx])
    return out

# =====================================================================
# CUDA Kernels (Template)
# =====================================================================
CUDA_TEMPLATE = r"""
#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

extern "C" __global__
void cgemm_abs2_tiled_full(const T_COMPLEX* __restrict__ SA,
                           const T_COMPLEX* __restrict__ SB,
                           T_REAL* __restrict__ K,
                           const int BM, const int BN, const int D,
                           const int lda, const int ldb, const int ldk)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= BM || j >= BN) return;

    __shared__ T_COMPLEX sA[TILE_M][TILE_K];
    __shared__ T_COMPLEX sB[TILE_N][TILE_K];
    
    T_COMPLEX acc = MAKE_COMPLEX(0.0, 0.0);

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : MAKE_COMPLEX(0.0, 0.0);
            }
        }
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                T_COMPLEX v = (k < D) ? SB[j * ldb + k] : MAKE_COMPLEX(0.0, 0.0);
                sB[threadIdx.x][tk] = MAKE_COMPLEX(v.x, -v.y); 
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            T_COMPLEX a = sA[threadIdx.y][tk];
            T_COMPLEX b = sB[threadIdx.x][tk];
            T_REAL rx = a.x * b.x - a.y * b.y;
            T_REAL ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }
    K[i * ldk + j] = acc.x * acc.x + acc.y * acc.y;
}

extern "C" __global__
void cgemm_abs2_tiled_lower(const T_COMPLEX* __restrict__ SA,
                            const T_COMPLEX* __restrict__ SB,
                            T_REAL* __restrict__ K,
                            const int BM, const int BN, const int D,
                            const int lda, const int ldb, const int ldk)
{
    if (blockIdx.x > blockIdx.y) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= BM || j >= BN) return;
    if (BM == BN && j > i) return;

    __shared__ T_COMPLEX sA[TILE_M][TILE_K];
    __shared__ T_COMPLEX sB[TILE_N][TILE_K];
    T_COMPLEX acc = MAKE_COMPLEX(0.0, 0.0);

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : MAKE_COMPLEX(0.0, 0.0);
            }
        }
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                T_COMPLEX v = (k < D) ? SB[j * ldb + k] : MAKE_COMPLEX(0.0, 0.0);
                sB[threadIdx.x][tk] = MAKE_COMPLEX(v.x, -v.y);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            T_COMPLEX a = sA[threadIdx.y][tk];
            T_COMPLEX b = sB[threadIdx.x][tk];
            T_REAL rx = a.x * b.x - a.y * b.y;
            T_REAL ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }
    K[i * ldk + j] = acc.x * acc.x + acc.y * acc.y;
}
"""

_RAWMOD_CACHE = {}
def _get_kernel(tm, tn, tk, name, double):
    import cupy as cp
    key = (tm, tn, tk, name, double)
    if key in _RAWMOD_CACHE: return _RAWMOD_CACHE[key]
    
    t_m = ("-DT_REAL=double", "-DT_COMPLEX=double2", "-DMAKE_COMPLEX=make_double2") if double else \
          ("-DT_REAL=float", "-DT_COMPLEX=float2", "-DMAKE_COMPLEX=make_float2")
    opts = ("--std=c++14", "--use_fast_math") + t_m + (f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    
    mod = cp.RawModule(code=CUDA_TEMPLATE, options=opts, name_expressions=(name,))
    fn = mod.get_function(name)
    _RAWMOD_CACHE[key] = fn
    return fn

# =====================================================================
# State Generation (Torch) -> CuPy
# =====================================================================
def _build_states_block_torch_cuda(x_blk, w_np, dev_name, ascale, re_emb, mode):
    import torch as th
    import pennylane as qml
    
    nq = int(x_blk.shape[1])
    t_float = th.float32 if x_blk.dtype == np.float32 else th.float64
    t_cplx = th.complex64 if t_float == th.float32 else th.complex128
    
    x = th.from_numpy(np.ascontiguousarray(x_blk)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)
    
    try:
        dev = qml.device(dev_name, wires=nq, shots=None, c_dtype=np.complex64 if t_float==th.float32 else np.complex128)
    except:
        dev = qml.device("lightning.gpu", wires=nq, shots=None)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if mode=="angle": qml.AngleEmbedding(ascale*v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(ascale*v[i], wires=i)
                    if mode=="ryrz": qml.RZ(ascale*v[i], wires=i)
        if re_emb:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l+1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try:
        from torch import vmap
        states = vmap(_state)(x)
    except:
        states = th.stack([_state(x[i]) for i in range(x.shape[0])])
    
    # Force le déplacement sur GPU (au cas où la simulation s'est faite sur CPU)
    # et assure le bon type complexe
    states = states.to(device="cuda", dtype=t_cplx, non_blocking=False)
    
    # --- SYNC CORRECTION ---
    th.cuda.synchronize()
    # -----------------------
    
    return states

def _torch_cuda_to_cupy(t):
    import cupy as cp
    return cp.from_dlpack(t)

# =====================================================================
# Main Compute Functions
# =====================================================================
def _gram_torch_stream(a_np, b_np, weights_np, device_name, tile_size, symmetric, float_dt, ret_dt, angle_scale, re_embed_between_layers, embed_mode):
    import torch as th
    import pennylane as qml
    
    n, nq = a_np.shape
    m = n if b_np is None else b_np.shape[0]
    
    tf = th.float32 if float_dt==np.float32 else th.float64
    tc = th.complex64 if float_dt==np.float32 else th.complex128
    
    a = th.from_numpy(a_np).to("cuda", dtype=tf)
    b = a if b_np is None else th.from_numpy(b_np).to("cuda", dtype=tf)
    w = th.from_numpy(weights_np).to("cuda", dtype=tf)
    
    try: dev = qml.device(device_name, wires=nq, shots=None, c_dtype=(np.complex64 if float_dt==np.float32 else np.complex128))
    except: dev = qml.device("lightning.gpu", wires=nq, shots=None)
    
    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(row):
        def _emb(v):
            if embed_mode=="angle": qml.AngleEmbedding(float(angle_scale)*v[:nq], wires=range(nq), rotation="Y")
            else:
                for i in range(nq):
                    qml.RY(float(angle_scale)*v[i], wires=i)
                    if embed_mode=="ryrz": qml.RZ(float(angle_scale)*v[i], wires=i)
        if re_embed_between_layers:
            for l in range(w.shape[0]):
                _emb(row)
                qml.templates.BasicEntanglerLayers(w[l:l+1], wires=range(nq))
        else:
            _emb(row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    try: from torch import vmap; build = lambda x: vmap(_state)(x).to(dtype=tc)
    except: build = lambda x: th.stack([_state(x[i]) for i in range(len(x))]).to(dtype=tc)

    k = th.empty((n, m), device="cuda", dtype=tf)
    
    with th.no_grad():
        for i0 in range(0, n, tile_size):
            i1 = min(i0+tile_size, n)
            sa = build(a[i0:i1])
            j_start = i0 if (symmetric and b_np is None) else 0
            for j0 in range(j_start, m, tile_size):
                j1 = min(j0+tile_size, m)
                sb = sa if (b_np is None and j0==i0) else build(b[j0:j1])
                res = (sa @ sb.conj().T).abs().square()
                k[i0:i1, j0:j1] = res
                if symmetric and b_np is None and j0 > i0:
                    k[j0:j1, i0:i1] = res.T
            del sa
            th.cuda.empty_cache()
            
    return k.cpu().numpy().astype(ret_dt)

def _gram_pennylane_angles_mp(
        A, B, weights, device_name, tile_size, symmetric, n_workers,
        dtype, return_dtype, progress, desc, angle_scale, re_embed_between_layers, embed_mode
):
    import multiprocessing as mp
    
    # Resolve dtypes properly
    f_dt = np.dtype(np.float32) if dtype=="float32" else np.dtype(np.float64)
    r_dt = np.dtype(np.float32) if return_dtype=="float32" else np.dtype(np.float64)
    
    A = _ensure_numpy(A, f_dt)
    B = A if B is None else _ensure_numpy(B, f_dt)
    n, nq = A.shape
    m = B.shape[0]
    w = _ensure_numpy(weights, f_dt)
    
    def _chunk(n, c): return [list(range(s, min(s+c, n))) for s in range(0, n, c)]
    ra = _chunk(n, max(1, tile_size))
    rb = ra if (B is A) else _chunk(m, max(1, tile_size))
    
    if n_workers is None or n_workers <= 0: n_workers = 1
    
    initargs = (w, device_name, nq, "float32" if f_dt==np.float32 else "float64", angle_scale, re_embed_between_layers, embed_mode)
    
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
            if j0 < j_start: continue
            sb_blk = sb[j0:j1]
            g = sa_blk @ sb_blk.conj().T
            mag2 = (np.abs(g)**2).astype(r_dt)
            k[i0:i1, j0:j1] = mag2
            if symmetric and (B is A) and j0 > i0:
                k[j0:j1, i0:i1] = mag2.T
    return k

def compute_kernel_matrix(
        X: Any, Y: Optional[Any] = None, *, weights: np.ndarray,
        device_name: str = "lightning.qubit", tile_size: int = 64, symmetric: bool = True,
        n_workers: int = 0, dtype: str = "float32", return_dtype: str = "float32",
        gram_backend: str = "auto", progress: bool = False, desc: str = "Gram",
        angle_scale: float = 1.0, re_embed_between_layers: bool = False, embed_mode: str = "ryrz",
        normalize: bool = False, jitter: float = 0.0,
        state_tile: int = 8192, tile_m="auto", tile_n="auto", tile_k="auto"
):
    f_dt = np.float32 if dtype=="float32" else np.float64
    r_dt = np.float32 if return_dtype=="float32" else np.float64
    is_double = (f_dt == np.float64)

    if gram_backend == "cuda_states":
        import cupy as cp
        try: _setup_cupy()
        except: pass
        
        A = _ensure_numpy(X, f_dt)
        B = A if Y is None else _ensure_numpy(Y, f_dt)
        w = _ensure_numpy(weights, f_dt)
        n, nq = A.shape
        m = B.shape[0]
        dim = 1 << nq

        tm, tn, tk = (32, 32, 32)
        if tile_m != "auto": tm, tn, tk = int(tile_m), int(tile_n), int(tile_k)
        
        K_cp = cp.empty((n, m), dtype=cp.float64 if is_double else cp.float32)
        
        j_ranges = list(_tile_ranges(m, state_tile))
        b_cache = {}
        
        it_b = tqdm(j_ranges, desc="Cache B", leave=False) if (progress and Y is not None) else j_ranges
        for j0, j1 in it_b:
            s_th = _build_states_block_torch_cuda(B[j0:j1], w, device_name, angle_scale, re_embed_between_layers, embed_mode)
            b_cache[(j0, j1)] = _torch_cuda_to_cupy(s_th)
        
        i_ranges = list(_tile_ranges(n, state_tile))
        it_a = tqdm(i_ranges, desc=desc, leave=False) if progress else i_ranges
        
        for i0, i1 in it_a:
            s_a_th = _build_states_block_torch_cuda(A[i0:i1], w, device_name, angle_scale, re_embed_between_layers, embed_mode)
            s_a_cp = _torch_cuda_to_cupy(s_a_th)
            
            relevant_j = [ (j0, j1) for (j0, j1) in j_ranges if (not symmetric or j1 > i0) ]
            for j0, j1 in relevant_j:
                s_b_cp = b_cache.get((j0, j1))
                if s_b_cp is None: s_b_cp = b_cache[(j0, j1)] = s_a_cp
                
                bi, bj = int(i1-i0), int(j1-j0)
                
                use_lower = (symmetric and (Y is None) and j0==i0)
                k_fn = _get_kernel(tm, tn, tk, "cgemm_abs2_tiled_lower" if use_lower else "cgemm_abs2_tiled_full", is_double)
                
                grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
                block = (tn, tm, 1)
                
                out_tile = cp.empty((bi, bj), dtype=K_cp.dtype)
                k_fn(grid, block, (s_a_cp, s_b_cp, out_tile, bi, bj, dim, dim, dim, bj))
                
                K_cp[i0:i1, j0:j1] = out_tile
                if symmetric and (Y is None) and j0 > i0:
                    K_cp[j0:j1, i0:i1] = out_tile.T
            
            del s_a_cp
            
        K = K_cp.get().astype(r_dt)
        if normalize and Y is None: 
            if jitter > 0: K += jitter * np.eye(n, dtype=K.dtype)
            _normalize_diag_inplace(K)
        return K

    if gram_backend in ["torch", "auto"] and "gpu" in device_name:
        import torch as th
        try:
            return _gram_torch_stream(
                _ensure_numpy(X, f_dt), _ensure_numpy(Y, f_dt) if Y is not None else None,
                weights_np=_ensure_numpy(weights, f_dt), device_name=device_name,
                tile_size=tile_size, symmetric=symmetric, float_dt=f_dt, ret_dt=r_dt,
                angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
            )
        except Exception as e:
            if gram_backend=="torch": raise e
            
    return _gram_pennylane_angles_mp(
        X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
        symmetric=symmetric, n_workers=n_workers, dtype=str(f_dt), return_dtype=str(r_dt),
        progress=progress, desc=desc, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
    )