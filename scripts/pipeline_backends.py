# pipeline_backends.py — PennyLane kernels with RBF-like controls (angle_scale, re-embed, normalize)
# Corrected & Optimized version with Float64 support and robust CUDA paths.

from typing import Optional, Any, Callable, Tuple, List, Dict
import os
import sys
import numpy as np
from tqdm import tqdm

# =====================================================================
# Helpers
# =====================================================================
def _ensure_numpy(a: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Converts to a C-contiguous ndarray, respecting dtype if provided."""
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
    """Normalize so that diag(K)=1 for square Gram matrices."""
    if K.shape[0] != K.shape[1]:
        return
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)


def _setup_cupy():
    """Pool mémoire + includes NVRTC robustes."""
    import cupy as cp
    
    # 1. Setup Memory Pool
    pool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    
    # Warmup
    _ = cp.ones((1,), dtype=cp.float32)
    del _
    cp.cuda.runtime.deviceSynchronize()

    # 2. Robust Include Path Detection for NVRTC
    # Standard paths to check
    candidates = [
        os.environ.get("CUDA_PATH"),
        os.environ.get("CUDA_HOME"),
        "/usr/local/cuda",
        "/opt/cuda",
        # Conda environment fallback
        os.path.join(sys.prefix, "include"),
        os.path.join(sys.prefix, "Library", "include") # Windows Conda
    ]
    
    include_path = None
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "include")):
            include_path = os.path.join(path, "include")
            break
        elif path and os.path.exists(path) and path.endswith("include"):
            include_path = path
            break

    nvrtc_opts = []
    if include_path:
        nvrtc_opts.append(f"-I{include_path}")
    
    # Inject into environment for CuPy
    if nvrtc_opts:
        existing = os.environ.get("CUPY_NVRTC_OPTIONS", "")
        if include_path not in existing:
            os.environ["CUPY_NVRTC_OPTIONS"] = existing + " " + " ".join(nvrtc_opts)


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
_pl_embed_mode: str = "ryrz"


def _pl_worker_init(w_local: np.ndarray, device_name: str, nq: int,
                    float_dtype_str: str = "float64",
                    angle_scale: float = 1.0,
                    re_embed_between_layers: bool = False,
                    embed_mode: str = "ryrz"):
    """Initializer called once per worker process."""
    global _pl_w, _pl_nq, _pl_device, _pl_qnode, _pl_float_dtype, _pl_complex_dtype
    global _pl_angle_scale, _pl_re_embed, _pl_embed_mode
    
    # Set threading vars to 1 to avoid oversubscription in workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    _pl_float_dtype = np.dtype(np.float32) if float_dtype_str == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if float_dtype_str == "float32" else np.dtype(np.complex128)
    
    _pl_w = _ensure_numpy(w_local, _pl_float_dtype)
    _pl_nq = int(nq)
    _pl_device = str(device_name)
    _pl_qnode = None
    _pl_angle_scale = float(angle_scale)
    _pl_re_embed = bool(re_embed_between_layers)
    _pl_embed_mode = str(embed_mode)


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
            # Ensure input matches dtype
            theta = qml.math.asarray(theta_row, dtype=_pl_float_dtype)
            if _pl_re_embed:
                L = _pl_w.shape[0]
                for l in range(L):
                    _embed(theta)
                    qml.templates.BasicEntanglerLayers(_pl_w[l:l + 1], wires=range(_pl_nq))
            else:
                _embed(theta)
                qml.templates.BasicEntanglerLayers(_pl_w, wires=range(_pl_nq))
            return qml.state()

        _pl_qnode = _state
    return _pl_qnode


def _pl_states_for_rows(rows: List[int], mat: np.ndarray) -> np.ndarray:
    qnode = _pl_get_qnode()
    # Pre-allocate output
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

    assert "gpu" in device_name.lower(), "torch_stream requires a GPU device."
    nq = a_np.shape[1]
    n = a_np.shape[0]
    m = n if b_np is None else b_np.shape[0]

    t_float = th.float32 if float_dt == np.float32 else th.float64
    t_complex = th.complex64 if float_dt == np.float32 else th.complex128
    t_ret = th.float32 if ret_dt == np.float32 else th.float64

    # Move data to GPU async
    a = th.from_numpy(np.ascontiguousarray(a_np)).to("cuda", dtype=t_float, non_blocking=True)
    b = a if b_np is None else th.from_numpy(np.ascontiguousarray(b_np)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(weights_np)).to("cuda", dtype=t_float, non_blocking=True)

    # Setup device
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
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
        else:
            _embed(theta_row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    # Vmap detection
    try:
        from torch import vmap
        _has_vmap = True
    except ImportError:
        _has_vmap = False

    def build_states(x_block: th.Tensor) -> th.Tensor:
        if _has_vmap and x_block.ndim == 2:
            return vmap(_state)(x_block).to(dtype=t_complex)
        states = [_state(x_block[t]) for t in range(x_block.shape[0])]
        return th.stack(states, dim=0).to(dtype=t_complex)

    # Output matrix
    k = th.empty((n, m), device="cuda", dtype=t_ret)

    # Compute
    with th.no_grad():
        outer_iter = range(0, n, tile_size)
        if tqdm is not None:
            outer_iter = tqdm(outer_iter, desc="Gram (torch)", leave=False)
            
        for i0 in outer_iter:
            i1 = min(i0 + tile_size, n)
            # Batch state preparation
            sa_x = build_states(a[i0:i1])
            
            inner_start = i0 if (symmetric and b is a) else 0
            for j0 in range(inner_start, m, tile_size):
                j1 = min(j0 + tile_size, m)
                
                if (b is a) and (j0 == i0):
                    sb_x = sa_x
                else:
                    sb_x = build_states(b[j0:j1])
                
                # Matrix multiplication
                g = sa_x @ sb_x.conj().transpose(0, 1)
                k_blk = (g.abs() ** 2).to(dtype=t_ret)
                
                k[i0:i1, j0:j1] = k_blk
                
                if symmetric and (b is a) and (j0 > i0):
                    k[j0:j1, i0:i1] = k_blk.transpose(0, 1)
            
            del sa_x
            # Periodic cache clear helps with fragmentation on smaller GPUs
            if i0 % (tile_size * 4) == 0:
                th.cuda.empty_cache()

    return k.detach().cpu().numpy().astype(ret_dt, copy=False)


# =====================================================================
# ---------- RawKernels cuda_states (Corrected Types) -----------------
# =====================================================================

# Template C++ code to support both float (float2) and double (double2)
CUDA_TEMPLATE_SRC = r"""
#ifndef TILE_M
#define TILE_M 32
#endif
#ifndef TILE_N
#define TILE_N 32
#endif
#ifndef TILE_K
#define TILE_K 32
#endif

// Type definitions via Macros substitution
// T_REAL: float or double
// T_COMPLEX: float2 or double2
// MAKE_COMPLEX: make_float2 or make_double2

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
    
    // Accumulator in high precision if needed, but keeping simple here
    T_COMPLEX acc; 
    acc.x = 0; acc.y = 0;

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        // Load A tile
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : MAKE_COMPLEX(0, 0);
            }
        }
        // Load B tile (conjugated)
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                T_COMPLEX v = (k < D) ? SB[j * ldb + k] : MAKE_COMPLEX(0, 0);
                sB[threadIdx.x][tk] = MAKE_COMPLEX(v.x, -v.y); 
            }
        }
        __syncthreads();

        // Compute tile
        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            T_COMPLEX a = sA[threadIdx.y][tk];
            T_COMPLEX b = sB[threadIdx.x][tk];
            // Complex mul: (ax + i ay)(bx + i by) = (ax bx - ay by) + i(ax by + ay bx)
            T_REAL rx = a.x * b.x - a.y * b.y;
            T_REAL ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }

    T_REAL mag2 = acc.x * acc.x + acc.y * acc.y;
    K[i * ldk + j] = mag2;
}

extern "C" __global__
void cgemm_abs2_tiled_lower(const T_COMPLEX* __restrict__ SA,
                            const T_COMPLEX* __restrict__ SB,
                            T_REAL* __restrict__ K,
                            const int BM, const int BN, const int D,
                            const int lda, const int ldb, const int ldk)
{
    // Symmetric lower triangular optimization
    if (blockIdx.x > blockIdx.y) return;
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= BM || j >= BN) return;
    if (BM == BN && j > i) return;

    __shared__ T_COMPLEX sA[TILE_M][TILE_K];
    __shared__ T_COMPLEX sB[TILE_N][TILE_K];
    T_COMPLEX acc; 
    acc.x = 0; acc.y = 0;

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : MAKE_COMPLEX(0, 0);
            }
        }
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                T_COMPLEX v = (k < D) ? SB[j * ldb + k] : MAKE_COMPLEX(0, 0);
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

    T_REAL mag2 = acc.x * acc.x + acc.y * acc.y;
    K[i * ldk + j] = mag2;
}
"""

_RAWMOD_CACHE = {}

def _get_kernel_with_macros(tm: int, tn: int, tk: int, name: str, use_double: bool):
    """
    Generates and compiles the CUDA kernel for specific tile sizes and precision.
    """
    import cupy as cp
    
    key = (tm, tn, tk, name, use_double)
    if key in _RAWMOD_CACHE:
        return _RAWMOD_CACHE[key]

    # Type definition macros
    if use_double:
        type_macros = ("-DT_REAL=double", "-DT_COMPLEX=double2", "-DMAKE_COMPLEX=make_double2")
    else:
        type_macros = ("-DT_REAL=float", "-DT_COMPLEX=float2", "-DMAKE_COMPLEX=make_float2")

    tile_macros = (f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    
    options = ("--std=c++14", "--use_fast_math") + type_macros + tile_macros
    
    mod = cp.RawModule(code=CUDA_TEMPLATE_SRC,
                       options=options,
                       name_expressions=(name,))
    
    fn = mod.get_function(name)
    _RAWMOD_CACHE[key] = fn
    return fn

def _get_full_kernel(tm, tn, tk, use_double=False):
    return _get_kernel_with_macros(tm, tn, tk, "cgemm_abs2_tiled_full", use_double)

def _get_lower_kernel(tm, tn, tk, use_double=False):
    return _get_kernel_with_macros(tm, tn, tk, "cgemm_abs2_tiled_lower", use_double)


# ---------------- Auto-tuner (Robust for Float/Double) ----------------
def _time_kernel_once(SA_cp, SB_cp, bi, bj, dim, tm, tn, tk, symmetric, use_double):
    import cupy as cp
    
    if symmetric:
        k_func = _get_lower_kernel(tm, tn, tk, use_double)
    else:
        k_func = _get_full_kernel(tm, tn, tk, use_double)
    
    # Output dtype matches input precision
    dtype = cp.float64 if use_double else cp.float32
    
    if symmetric:
        # Symmetric lower only outputs to (bi, bi)
        K_view = cp.empty((bi, bi), dtype=dtype)
        args = (SA_cp, SA_cp, K_view, 
                np.int32(bi), np.int32(bi), np.int32(dim),
                np.int32(dim), np.int32(dim), np.int32(bi))
    else:
        K_view = cp.empty((bi, bj), dtype=dtype)
        args = (SA_cp, SB_cp, K_view, 
                np.int32(bi), np.int32(bj), np.int32(dim),
                np.int32(dim), np.int32(dim), np.int32(bj))

    block = (tn, tm, 1)
    grid_x = (bj + tn - 1) // tn if not symmetric else (bi + tn - 1) // tn
    grid_y = (bi + tm - 1) // tm
    grid = (grid_x, grid_y, 1)

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    k_func(grid, block, args)
    end.record()
    end.synchronize()
    
    return cp.cuda.get_elapsed_time(start, end)

def autotune_tiles(
        dim: int,
        bi: int = 128,
        bj: int = 128,
        *,
        symmetric: bool = True,
        use_double: bool = False,
        candidates: Optional[List[Tuple[int, int, int]]] = None,
        warmup: int = 1,
        iters: int = 3,
        seed: int = 0,
) -> Tuple[int, int, int]:
    """Finds best (TILE_M, TILE_N, TILE_K) for specific precision."""
    import cupy as cp
    rng = np.random.default_rng(seed)
    
    if candidates is None:
        candidates = [
            (32, 32, 32), (64, 16, 32), (16, 64, 32),
            (32, 32, 16), (32, 32, 64),
            (64, 32, 16), (32, 64, 16),
        ]
        
    dtype = np.float64 if use_double else np.float32
    
    # Generate dummy data on GPU
    SA_host = (rng.standard_normal((bi, dim)).astype(dtype) + 
               1j * rng.standard_normal((bi, dim)).astype(dtype))
    SA_cp = cp.asarray(SA_host)
    
    if symmetric:
        SB_cp = SA_cp
    else:
        SB_host = (rng.standard_normal((bj, dim)).astype(dtype) + 
                   1j * rng.standard_normal((bj, dim)).astype(dtype))
        SB_cp = cp.asarray(SB_host)

    def _fits(tm, tn, tk):
        # Calc shared mem usage
        elem_size = 16 if use_double else 8 # complex size in bytes
        bytes_per_tile = (tm * tk + tn * tk) * elem_size
        
        # Check against device limit (conservative 48KB default)
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            sm_lim = int(props.get('sharedMemPerBlockOptin', props.get('sharedMemPerBlock', 48 * 1024)))
        except Exception:
            sm_lim = 48 * 1024
        
        return bytes_per_tile <= sm_lim

    best_ms = float('inf')
    best_cfg = (32, 32, 32)

    for (tm, tn, tk) in candidates:
        if not _fits(tm, tn, tk):
            continue
            
        try:
            # Warmup
            for _ in range(warmup):
                _time_kernel_once(SA_cp, SB_cp, bi, bj, dim, tm, tn, tk, symmetric, use_double)
            
            # Measure
            times = []
            for _ in range(iters):
                times.append(_time_kernel_once(SA_cp, SB_cp, bi, bj, dim, tm, tn, tk, symmetric, use_double))
            
            avg_ms = np.mean(times)
            
            if avg_ms < best_ms:
                best_ms = avg_ms
                best_cfg = (tm, tn, tk)
        except Exception:
            continue
            
    return best_cfg


# =====================================================================
# ---------- États (par blocs) via PennyLane+Torch -> DLPack ----------
# =====================================================================
def _build_states_block_torch_cuda(
    x_blk: np.ndarray,
    w_np: np.ndarray,
    device_name: str,
    *,
    angle_scale: float = 1.0,
    re_embed_between_layers: bool = False,
    embed_mode: str = "ryrz",
) -> "torch.Tensor":
    import torch as th
    import pennylane as qml

    nq = int(x_blk.shape[1])
    
    # Input dtype decides internal precision
    t_float = th.float32 if x_blk.dtype == np.float32 else th.float64
    t_complex = th.complex64 if t_float == th.float32 else th.complex128

    x = th.from_numpy(np.ascontiguousarray(x_blk)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)

    # Normalize device name for PennyLane
    dev_name = device_name
    if "gpu" not in str(device_name).lower():
        # Fallback to lightning.gpu if generic name passed but we are in the CUDA path
        try:
            _ = qml.device("lightning.gpu", wires=nq)
            dev_name = "lightning.gpu"
        except:
            dev_name = "default.qubit" # Should not happen in this path

    # Create device
    c_dtype = np.complex64 if t_float == th.float32 else np.complex128
    dev = qml.device(dev_name, wires=nq, shots=None, c_dtype=c_dtype)

    def _embed(v: "th.Tensor"):
        s = float(angle_scale)
        if embed_mode == "angle":
            qml.AngleEmbedding(s * v[:nq], wires=range(nq), rotation="Y", normalize=False)
        else:
            for i in range(nq):
                qml.RY(s * v[i], wires=i)
                if embed_mode == "ryrz":
                    qml.RZ(s * v[i], wires=i)

    @qml.qnode(dev, interface="torch", diff_method=None)
    def _state(theta_row: "th.Tensor") -> "th.Tensor":
        if re_embed_between_layers:
            L = w.shape[0]
            for l in range(L):
                _embed(theta_row)
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
        else:
            _embed(theta_row)
            qml.templates.BasicEntanglerLayers(w, wires=range(nq))
        return qml.state()

    # Vectorization attempt
    try:
        from torch import vmap
        states = vmap(_state)(x).to(dtype=t_complex)
    except Exception:
        # Fallback loop
        states = th.stack([_state(x[i]) for i in range(x.shape[0])], dim=0).to(dtype=t_complex)

    if states.device.type != "cuda":
        states = states.to("cuda", non_blocking=True)
    return states


def _torch_cuda_to_cupy(t: "torch.Tensor"):
    """Zero-copy Torch CUDA -> CuPy via DLPack with robust fallback."""
    import cupy as cp
    import warnings

    # Sync logical device
    try:
        if hasattr(t, "device") and t.device.type == "cuda":
            dev_id = int(t.device.index or 0)
            cp.cuda.Device(dev_id).use()
    except Exception:
        pass

    # Modern API
    try:
        return cp.from_dlpack(t)
    except Exception:
        pass

    # Legacy API
    try:
        from torch.utils.dlpack import to_dlpack
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cp.from_dlpack(to_dlpack(t))
    except Exception:
        # CPU Fallback
        return cp.asarray(t.detach().cpu().numpy())


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
    """CPU Multiprocessing Kernel."""
    import multiprocessing as mp
    import pennylane as qml

    def _resolve_dtype(d, ref_arrs) -> np.dtype:
        if d in ("float32", "float64"):
            return np.dtype(np.float32) if d == "float32" else np.dtype(np.float64)
        # Auto-detect
        for arr in ref_arrs:
            try:
                if np.asanyarray(arr).dtype == np.float32:
                    return np.dtype(np.float32)
            except: pass
        return np.dtype(np.float64)

    float_dt = _resolve_dtype(dtype, [A, weights])
    ret_dt = _resolve_dtype(return_dtype, []) if return_dtype else float_dt

    A = _ensure_numpy(A, dtype=float_dt)
    B = A if B is None else _ensure_numpy(B, dtype=float_dt)
    n, nq = A.shape
    m = B.shape[0]

    w = _ensure_numpy(weights, dtype=float_dt)

    def _chunk_indices(n_items, chunk):
        return [list(range(s, min(s + chunk, n_items))) for s in range(0, n_items, chunk)]

    rows_a = _chunk_indices(n, max(1, tile_size))
    rows_b = rows_a if (B is A) else _chunk_indices(m, max(1, tile_size))

    if n_workers is None or n_workers <= 0:
        try:
            n_workers = max(1, len(os.sched_getaffinity(0)))
        except:
            n_workers = max(1, mp.cpu_count() - 1)

    initargs = (w, device_name, nq, "float32" if float_dt == np.float32 else "float64",
                float(angle_scale), bool(re_embed_between_layers), str(embed_mode))

    if n_workers == 1:
        _pl_worker_init(*initargs)
        sa = np.concatenate([_pl_states_for_rows(rs, A) for rs in rows_a], axis=0)
        sb = sa if (B is A) else np.concatenate([_pl_states_for_rows(rs, B) for rs in rows_b], axis=0)
    else:
        # Robust spawn context
        ctx = mp.get_context("spawn")
        from functools import partial
        with ctx.Pool(processes=n_workers, initializer=_pl_worker_init, initargs=initargs) as pool:
            funcA = partial(_pl_states_for_rows, mat=A)
            sa = np.concatenate(list(pool.imap(funcA, rows_a)), axis=0)
            if B is A:
                sb = sa
            else:
                funcB = partial(_pl_states_for_rows, mat=B)
                sb = np.concatenate(list(pool.imap(funcB, rows_b)), axis=0)

    # Compute Gram matrix on CPU
    k = np.empty((n, m), dtype=ret_dt)
    
    outer_iter = list(_tile_ranges(n, tile_size))
    if tqdm and progress:
        outer_iter = tqdm(outer_iter, desc=f"{desc} (CPU)", leave=False)

    for i0, i1 in outer_iter:
        sa_blk = sa[i0:i1]
        
        inner_start = i0 if (symmetric and (B is A)) else 0
        for j0, j1 in _tile_ranges(m, tile_size):
            if j0 < inner_start: continue
            
            sb_blk = sb[j0:j1]
            # Dot product state vectors
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
        # --- PARAMS pour cuda_states ---
        state_tile: int = 8192,
        tile_m: int | str = "auto",
        tile_n: int | str = "auto",
        tile_k: int | str = "auto",
) -> np.ndarray:
    """
    Fidelity kernel between quantum states.
    Robust backend selection (CPU/GPU/Float32/Float64).
    """
    # Environment cleanup
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Determine Precisions
    def _get_np_dtype(s, default=np.float64):
        if s == "float32": return np.float32
        if s == "float64": return np.float64
        return default

    # Default logic: GPU prefers float32, CPU prefers float64
    default_prec = np.float32 if "gpu" in device_name.lower() else np.float64
    
    float_dt = _get_np_dtype(dtype, default_prec)
    ret_dt = _get_np_dtype(return_dtype, float_dt)
    
    is_double = (float_dt == np.float64)

    # -------- Backend: cuda_states --------
    if gram_backend == "cuda_states":
        import cupy as cp
        try:
            _setup_cupy()
        except Exception:
            pass

        # Prepare inputs
        A = _ensure_numpy(X, dtype=float_dt)
        B = A if Y is None else _ensure_numpy(Y, dtype=float_dt)
        w = _ensure_numpy(weights, dtype=float_dt)
        
        n, nq = A.shape
        m = B.shape[0]
        dim = 1 << nq

        # Auto-tune tiles
        auto_tiles = (tile_m == "auto")
        if auto_tiles:
            try:
                tm, tn, tk = autotune_tiles(dim,
                                            bi=min(state_tile, n),
                                            bj=min(state_tile, m),
                                            symmetric=(symmetric and (B is A)),
                                            use_double=is_double)
            except Exception:
                tm, tn, tk = (32, 32, 32)
        else:
            tm, tn, tk = int(tile_m), int(tile_n), int(tile_k)

        # Select Kernel (Lower/Full + Double/Float)
        if symmetric and (B is A):
            k_func = _get_lower_kernel(tm, tn, tk, use_double=is_double)
        else:
            k_func = _get_full_kernel(tm, tn, tk, use_double=is_double)

        # Output Buffer
        out_dtype = cp.float64 if is_double else cp.float32
        K_cp = cp.empty((n, m), dtype=out_dtype)

        # Cache SB states
        j_tiles = list(_tile_ranges(m, state_tile))
        sb_cache = {}
        
        iter_b = tqdm(j_tiles, desc="cuda_states: Cache B", leave=False) if (tqdm and progress) else j_tiles
        for (j0, j1) in iter_b:
            SB_th = _build_states_block_torch_cuda(B[j0:j1], w, device_name,
                                                   angle_scale=angle_scale,
                                                   re_embed_between_layers=re_embed_between_layers,
                                                   embed_mode=embed_mode)
            sb_cache[(j0, j1)] = _torch_cuda_to_cupy(SB_th)

        # Loop A tiles
        outer = list(_tile_ranges(n, state_tile))
        iter_a = tqdm(outer, desc="cuda_states: Gram", leave=False) if (tqdm and progress) else outer

        for (i0, i1) in iter_a:
            SA_th = _build_states_block_torch_cuda(A[i0:i1], w, device_name,
                                                   angle_scale=angle_scale,
                                                   re_embed_between_layers=re_embed_between_layers,
                                                   embed_mode=embed_mode)
            SA_cp = _torch_cuda_to_cupy(SA_th)

            inner = j_tiles
            if symmetric and (B is A):
                inner = [(j0, j1) for (j0, j1) in j_tiles if j1 > i0]

            for (j0, j1) in inner:
                SB_cp = sb_cache[(j0, j1)]
                bi, bj = int(i1 - i0), int(j1 - j0)
                K_view = K_cp[i0:i1, j0:j1]
                
                # Launch Kernel
                block = (tn, tm, 1)
                grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
                
                args = (SA_cp, SB_cp, K_view,
                        np.int32(bi), np.int32(bj), np.int32(dim),
                        np.int32(dim), np.int32(dim), np.int32(dim))
                
                k_func(grid, block, args)

                if symmetric and (B is A) and (j0 > i0):
                    K_cp[j0:j1, i0:i1] = K_view.T
            
            del SA_cp

        K = K_cp.get().astype(ret_dt, copy=False)

    # -------- Backend: torch_stream (Fast GPU fallback) --------
    elif ("gpu" in device_name.lower()) and gram_backend in ("auto", "torch"):
        try:
            import torch as _th
            if _th.cuda.is_available():
                A = _ensure_numpy(X, dtype=float_dt)
                B = A if Y is None else _ensure_numpy(Y, dtype=float_dt)
                w = _ensure_numpy(weights, dtype=float_dt)
                K = _gram_torch_stream(
                    A, None if (B is A) else B, weights_np=w, device_name=device_name,
                    tile_size=tile_size, symmetric=symmetric, float_dt=float_dt, ret_dt=ret_dt,
                    angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
                )
            else:
                raise RuntimeError("CUDA missing")
        except Exception:
            # Fallback CPU
            K = _gram_pennylane_angles_mp(
                X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
                symmetric=symmetric, n_workers=n_workers, dtype=str(float_dt), return_dtype=str(ret_dt),
                progress=progress, desc=desc, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
            )
    
    # -------- Backend: CPU --------
    else:
        K = _gram_pennylane_angles_mp(
            X, Y, weights=weights, device_name=device_name, tile_size=tile_size,
            symmetric=symmetric, n_workers=n_workers, dtype=str(float_dt), return_dtype=str(ret_dt),
            progress=progress, desc=desc, angle_scale=angle_scale, re_embed_between_layers=re_embed_between_layers, embed_mode=embed_mode
        )

    # Normalization & Jitter
    if normalize and (Y is None):
        if jitter > 0:
            K = K + jitter * np.eye(K.shape[0], dtype=K.dtype)
        _normalize_diag_inplace(K)
    elif jitter > 0 and (Y is None):
        # Apply jitter even if not normalizing (useful for stability)
        K = K + jitter * np.eye(K.shape[0], dtype=K.dtype)

    return K