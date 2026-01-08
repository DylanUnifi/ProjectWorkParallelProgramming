# pipeline_backends.py — Optimized, Synchronized, Vmap-Free & Numerically Safe
from typing import Optional, Any, Callable, Dict, Tuple, List
import os
import sys
import json
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
    # Clip pour éviter les racines de nombres négatifs minuscules
    d = np.sqrt(np.clip(np.diag(K), 1e-12, None))
    K /= d[:, None] * d[None, :]
    np.fill_diagonal(K, 1.0)

# =====================================================================
# VRAM & Memory Management
# =====================================================================
# Constants for memory calculations
MATRIX_PAIRS_FACTOR = 2  # Factor for A and B matrices in precompute size calculation
BATCH_SYNC_INTERVAL = 32  # Number of tiles between synchronization calls

class PersistentBufferPool:
    """Manages reusable GPU buffers to reduce allocation overhead."""
    def __init__(self):
        self.buffers: Dict[Tuple[tuple, type], Any] = {}
    
    def get_buffer(self, shape: tuple, dtype, device="cuda"):
        """Get or create a buffer with the given shape and dtype."""
        import cupy as cp
        key = (shape, dtype)
        if key not in self.buffers:
            self.buffers[key] = cp.empty(shape, dtype=dtype)
        return self.buffers[key]
    
    def clear(self):
        """Clear all cached buffers."""
        self.buffers.clear()

_BUFFER_POOL = PersistentBufferPool()
_PINNED_BUFFERS: Dict[Tuple[tuple, type], Any] = {}

def _get_pinned_buffer(shape: tuple, dtype):
    """Get or create a pinned host memory buffer."""
    import cupy as cp
    key = (shape, dtype)
    if key not in _PINNED_BUFFERS:
        pinned_pool = cp.cuda.PinnedMemoryPool()
        # Use pinned memory allocator
        with cp.cuda.using_allocator(pinned_pool.malloc):
            _PINNED_BUFFERS[key] = cp.zeros(shape, dtype=dtype)
    return _PINNED_BUFFERS[key]

def _compute_optimal_state_tile(vram_fraction: float = 0.85, nq: int = 6, 
                                 dtype=np.float32, overhead_gb: float = 2.0) -> int:
    """
    Compute optimal state_tile size based on available VRAM.
    
    Args:
        vram_fraction: Maximum fraction of VRAM to use (default 0.85)
        nq: Number of qubits
        dtype: Data type for computations
        overhead_gb: Reserved VRAM for framework overhead in GB
    
    Returns:
        Optimal tile size (number of states)
    """
    try:
        import cupy as cp
        # Get available VRAM
        device = cp.cuda.Device()
        total_vram = device.mem_info[1]  # Total memory in bytes
        available_vram = total_vram * vram_fraction - (overhead_gb * 1024**3)
        
        # Calculate memory per state: dim = 2^nq complex numbers
        dim = 1 << nq
        bytes_per_complex = 8 if dtype == np.float32 else 16  # complex64 or complex128
        bytes_per_state = dim * bytes_per_complex
        
        # Calculate max states that fit in available VRAM
        max_states = int(available_vram / bytes_per_state)
        
        # Round down to nearest power of 2 for efficiency
        tile_size = 2 ** int(np.log2(max_states))
        
        # Ensure minimum and maximum bounds
        tile_size = max(256, min(tile_size, 32768))
        
        return tile_size
    except Exception as e:
        # Fallback to conservative default
        return 8192

def _compute_max_precompute_size(vram_fraction: float = 0.85, nq: int = 6,
                                  dtype=np.float32, overhead_gb: float = 2.0) -> int:
    """
    Determine maximum number of states that can be cached in GPU memory.
    
    Args:
        vram_fraction: Maximum fraction of VRAM to use
        nq: Number of qubits
        dtype: Data type
        overhead_gb: Reserved VRAM in GB
    
    Returns:
        Maximum number of states to cache
    """
    try:
        import cupy as cp
        device = cp.cuda.Device()
        total_vram = device.mem_info[1]
        available_vram = total_vram * vram_fraction - (overhead_gb * 1024**3)
        
        dim = 1 << nq
        bytes_per_complex = 8 if dtype == np.float32 else 16
        bytes_per_state = dim * bytes_per_complex
        
        max_states = int(available_vram / (bytes_per_state * MATRIX_PAIRS_FACTOR))
        
        return max(1024, max_states)
    except Exception:
        return 8192

def _setup_cupy():
    import cupy as cp
    # Use managed memory pool for GPU allocations
    pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
    cp.cuda.set_allocator(pool.malloc)
    
    # Use pinned memory pool for host transfers
    pinned_pool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
    
    # Warm up
    _ = cp.ones((1,), dtype=cp.float32)
    cp.cuda.runtime.deviceSynchronize()

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
# Worker globals
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
    global _pl_w, _pl_nq, _pl_device, _pl_qnode, _pl_float_dtype, _pl_complex_dtype, _pl_angle_scale, _pl_re_embed, _pl_embed_mode
    os.environ["OMP_NUM_THREADS"] = "1"
    _pl_float_dtype = np.dtype(np.float32) if fdtype == "float32" else np.dtype(np.float64)
    _pl_complex_dtype = np.dtype(np.complex64) if fdtype == "float32" else np.dtype(np.complex128)
    _pl_w = _ensure_numpy(w, _pl_float_dtype)
    _pl_nq, _pl_device = int(nq), str(dev)
    _pl_angle_scale, _pl_re_embed, _pl_embed_mode = float(ascale), bool(re_emb), str(mode)
    
    # --- BUG FIX: Reset QNode to force device recreation on param change ---
    _pl_qnode = None 
    # ---------------------------------------------------------------------

def _pl_get_qnode():
    global _pl_qnode
    if _pl_qnode is None:
        import pennylane as qml
        try:
            dev = qml.device(_pl_device, wires=_pl_nq, shots=None, c_dtype=_pl_complex_dtype)
        except:
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
# CUDA Kernels
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
# CUDA Kernel Autotuning
# =====================================================================
_AUTOTUNE_CACHE_FILE = ".cuda_kernel_autotune.json"
_AUTOTUNE_CACHE: Dict[str, Tuple[int, int, int]] = {}

def _load_autotune_cache():
    """Load autotuning results from disk."""
    global _AUTOTUNE_CACHE
    if os.path.exists(_AUTOTUNE_CACHE_FILE):
        try:
            with open(_AUTOTUNE_CACHE_FILE, 'r') as f:
                data = json.load(f)
                _AUTOTUNE_CACHE = {k: tuple(v) for k, v in data.items()}
        except Exception:
            pass

def _save_autotune_cache():
    """Save autotuning results to disk."""
    try:
        with open(_AUTOTUNE_CACHE_FILE, 'w') as f:
            json.dump({k: list(v) for k, v in _AUTOTUNE_CACHE.items()}, f, indent=2)
    except Exception:
        pass

def _autotune_kernel_tiles(nq: int, is_double: bool = False, 
                           test_size: int = 512, warmup: int = 2, trials: int = 5) -> Tuple[int, int, int]:
    """
    Benchmark different TILE_M, TILE_N, TILE_K combinations and return the best.
    
    Args:
        nq: Number of qubits
        is_double: Whether to use double precision
        test_size: Size of test matrices
        warmup: Number of warmup iterations
        trials: Number of benchmark trials
    
    Returns:
        Tuple of (TILE_M, TILE_N, TILE_K) with best performance
    """
    import cupy as cp
    import time
    
    # Check cache first
    cache_key = f"nq{nq}_{'double' if is_double else 'float'}"
    if cache_key in _AUTOTUNE_CACHE:
        return _AUTOTUNE_CACHE[cache_key]
    
    dim = 1 << nq
    dtype_complex = cp.complex128 if is_double else cp.complex64
    dtype_real = cp.float64 if is_double else cp.float32
    
    # Generate test data
    rng = cp.random.default_rng(42)
    SA = rng.random((test_size, dim), dtype=dtype_real) + 1j * rng.random((test_size, dim), dtype=dtype_real)
    SB = rng.random((test_size, dim), dtype=dtype_real) + 1j * rng.random((test_size, dim), dtype=dtype_real)
    K_out = cp.empty((test_size, test_size), dtype=dtype_real)
    
    # Tile size candidates constrained by 48KB shared memory
    # For float2: 48KB / 8 bytes = 6144 elements
    # Constraint: (TILE_M * TILE_K + TILE_N * TILE_K) * bytes_per_complex <= 48KB
    candidates_m_n = [16, 32, 64]
    candidates_k = [16, 32, 64, 128]
    
    results = []
    
    for tm in candidates_m_n:
        for tn in candidates_m_n:
            for tk in candidates_k:
                # Check shared memory constraint
                # sA[TILE_M][TILE_K] + sB[TILE_N][TILE_K]
                bytes_per_complex = 16 if is_double else 8
                shared_mem = (tm * tk + tn * tk) * bytes_per_complex
                if shared_mem > 48 * 1024:  # 48KB limit
                    continue
                
                try:
                    kernel = _get_kernel(tm, tn, tk, "cgemm_abs2_tiled_full", is_double)
                    
                    grid = ((test_size + tn - 1) // tn, (test_size + tm - 1) // tm, 1)
                    block = (tn, tm, 1)
                    
                    # Warmup
                    for _ in range(warmup):
                        kernel(grid, block, (SA, SB, K_out, test_size, test_size, dim, dim, dim, test_size))
                    cp.cuda.runtime.deviceSynchronize()
                    
                    # Benchmark
                    times = []
                    for _ in range(trials):
                        start = time.perf_counter()
                        kernel(grid, block, (SA, SB, K_out, test_size, test_size, dim, dim, dim, test_size))
                        cp.cuda.runtime.deviceSynchronize()
                        times.append(time.perf_counter() - start)
                    
                    avg_time = np.mean(times)
                    results.append((avg_time, tm, tn, tk))
                    
                except Exception:
                    continue
    
    if not results:
        # Fallback to default
        return (32, 32, 32)
    
    # Select best configuration
    results.sort()
    best = results[0]
    best_config = (best[1], best[2], best[3])
    
    # Cache result
    _AUTOTUNE_CACHE[cache_key] = best_config
    _save_autotune_cache()
    
    return best_config

# =====================================================================
# State Generation (Torch) -> CuPy (FIXED & BATCHED)
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

    # --- ATTEMPT NATIVE BATCHING (FASTEST, NO VMAP) ---
    try:
        states = _state(x)
        if states.ndim != 2 or states.shape[0] != x.shape[0]:
            raise ValueError("Batching not natively supported")
    except:
        states = th.stack([_state(x[i]) for i in range(x.shape[0])])
    
    states = states.to(device="cuda", dtype=t_cplx, non_blocking=False)
    
    # --- CRITICAL FIX: SYNC ---
    th.cuda.synchronize()
    # --------------------------
    
    return states

def _torch_cuda_to_cupy(t):
    import cupy as cp
    return cp.from_dlpack(t)

# =====================================================================
# Bulk State Precomputation & Async Dispatch
# =====================================================================
def _build_all_states_torch_cuda(x_all, w_np, dev_name, ascale, re_emb, mode, use_pinned=True):
    """
    Build ALL quantum states in one pass with pinned memory optimization.
    Minimizes torch→cupy handoffs by precomputing entire matrix at once.
    
    Args:
        x_all: Full input data matrix (n_samples × n_qubits)
        w_np: Weights array
        dev_name: Device name
        ascale: Angle scale
        re_emb: Re-embedding flag
        mode: Embedding mode
        use_pinned: Whether to use pinned memory for transfers
    
    Returns:
        CuPy array of all quantum states
    """
    import torch as th
    import pennylane as qml
    
    nq = int(x_all.shape[1])
    t_float = th.float32 if x_all.dtype == np.float32 else th.float64
    t_cplx = th.complex64 if t_float == th.float32 else th.complex128
    
    # Use pinned memory for faster transfers
    if use_pinned:
        x = th.from_numpy(np.ascontiguousarray(x_all)).pin_memory().to("cuda", dtype=t_float, non_blocking=True)
        w = th.from_numpy(np.ascontiguousarray(w_np)).pin_memory().to("cuda", dtype=t_float, non_blocking=True)
    else:
        x = th.from_numpy(np.ascontiguousarray(x_all)).to("cuda", dtype=t_float, non_blocking=True)
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

    # Attempt native batching for all states at once
    try:
        states = _state(x)
        if states.ndim != 2 or states.shape[0] != x.shape[0]:
            raise ValueError("Batching not natively supported")
    except:
        states = th.stack([_state(x[i]) for i in range(x.shape[0])])
    
    states = states.to(device="cuda", dtype=t_cplx, non_blocking=False)
    th.cuda.synchronize()
    
    # Convert to CuPy with zero-copy DLPack
    return _torch_cuda_to_cupy(states)

_COMPUTE_STREAM = None

def _get_compute_stream():
    """Get or create dedicated compute stream for async dispatch."""
    global _COMPUTE_STREAM
    if _COMPUTE_STREAM is None:
        import cupy as cp
        _COMPUTE_STREAM = cp.cuda.Stream(non_blocking=True)
    return _COMPUTE_STREAM

def _dispatch_kernel_async(kernel_fn, grid, block, args, stream=None):
    """
    Dispatch kernel asynchronously without immediate synchronization.
    
    Args:
        kernel_fn: Compiled kernel function
        grid: Grid dimensions
        block: Block dimensions
        args: Kernel arguments
        stream: CUDA stream (uses compute_stream if None)
    """
    import cupy as cp
    if stream is None:
        stream = _get_compute_stream()
    
    with stream:
        kernel_fn(grid, block, args)

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

    try: 
        build = lambda x: _state(x).to(dtype=tc)
        _ = build(a[:2])
    except: 
        build = lambda x: th.stack([_state(x[i]) for i in range(len(x))]).to(dtype=tc)

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
            from functools import partial
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
        state_tile: int = -1, tile_m="auto", tile_n="auto", tile_k="auto",
        autotune: bool = True, precompute_all_states: bool = True, vram_fraction: float = 0.85
):
    f_dt = np.float32 if dtype=="float32" else np.float64
    r_dt = np.float32 if return_dtype=="float32" else np.float64
    is_double = (f_dt == np.float64)

    if gram_backend == "cuda_states":
        import cupy as cp
        try: 
            _setup_cupy()
            _load_autotune_cache()
        except: 
            pass
        
        A = _ensure_numpy(X, f_dt)
        B = A if Y is None else _ensure_numpy(Y, f_dt)
        w = _ensure_numpy(weights, f_dt)
        n, nq = A.shape
        m = B.shape[0]
        dim = 1 << nq

        # OPTIMIZATION 1: VRAM-aware state_tile sizing
        if state_tile == -1:
            state_tile = _compute_optimal_state_tile(vram_fraction, nq, f_dt)
            if progress:
                print(f"📊 Auto-sized state_tile={state_tile} (using {vram_fraction*100:.0f}% VRAM)")
        
        # OPTIMIZATION 3: Kernel autotuning
        if autotune and tile_m == "auto":
            tm, tn, tk = _autotune_kernel_tiles(nq, is_double)
            if progress:
                print(f"🔧 Autotuned kernel tiles: M={tm}, N={tn}, K={tk}")
        else:
            tm, tn, tk = (32, 32, 32)
            if tile_m != "auto": 
                tm, tn, tk = int(tile_m), int(tile_n), int(tile_k)
        
        K_cp = cp.empty((n, m), dtype=cp.float64 if is_double else cp.float32)
        
        # OPTIMIZATION 2: Bulk state precomputation
        max_precompute = _compute_max_precompute_size(vram_fraction, nq, f_dt)
        use_bulk_precompute = precompute_all_states and (max(n, m) <= max_precompute)
        
        if use_bulk_precompute:
            # Precompute ALL states at once to minimize handoffs
            if progress:
                print(f"⚡ Bulk precomputing {n} + {m} states...")
            
            s_a_cp = _build_all_states_torch_cuda(A, w, device_name, angle_scale, 
                                                   re_embed_between_layers, embed_mode, use_pinned=True)
            if Y is None:
                s_b_cp = s_a_cp
            else:
                s_b_cp = _build_all_states_torch_cuda(B, w, device_name, angle_scale,
                                                       re_embed_between_layers, embed_mode, use_pinned=True)
            
            # OPTIMIZATION 4: Async dispatch with batch synchronization
            compute_stream = _get_compute_stream()
            tile_count = 0
            
            i_ranges = list(_tile_ranges(n, state_tile))
            j_ranges = list(_tile_ranges(m, state_tile))
            
            it = tqdm(total=len(i_ranges)*len(j_ranges), desc=desc, leave=False) if progress else None
            
            for i0, i1 in i_ranges:
                s_a_tile = s_a_cp[i0:i1]
                j_start = i0 if (symmetric and Y is None) else 0
                
                for j0, j1 in j_ranges:
                    if j0 < j_start: 
                        if it: it.update(1)
                        continue
                    
                    s_b_tile = s_b_cp[j0:j1]
                    bi, bj = int(i1-i0), int(j1-j0)
                    
                    use_lower = (symmetric and (Y is None) and j0==i0)
                    k_fn = _get_kernel(tm, tn, tk, "cgemm_abs2_tiled_lower" if use_lower else "cgemm_abs2_tiled_full", is_double)
                    
                    grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
                    block = (tn, tm, 1)
                    
                    out_tile = cp.empty((bi, bj), dtype=K_cp.dtype)
                    
                    # Async dispatch
                    _dispatch_kernel_async(k_fn, grid, block, 
                                         (s_a_tile, s_b_tile, out_tile, bi, bj, dim, dim, dim, bj),
                                         stream=compute_stream)
                    
                    K_cp[i0:i1, j0:j1] = out_tile
                    if symmetric and (Y is None) and j0 > i0:
                        K_cp[j0:j1, i0:i1] = out_tile.T
                    
                    tile_count += 1
                    # Batch synchronization
                    if tile_count % BATCH_SYNC_INTERVAL == 0:
                        compute_stream.synchronize()
                    
                    if it: it.update(1)
            
            # Single final synchronization
            compute_stream.synchronize()
            if it: it.close()
            
        else:
            # Fall back to original tiled approach when bulk doesn't fit
            j_ranges = list(_tile_ranges(m, state_tile))
            b_cache = {}
            
            it_b = tqdm(j_ranges, desc="Cache B", leave=False) if (progress and Y is not None) else j_ranges
            for j0, j1 in it_b:
                s_th = _build_states_block_torch_cuda(B[j0:j1], w, device_name, angle_scale, re_embed_between_layers, embed_mode)
                b_cache[(j0, j1)] = _torch_cuda_to_cupy(s_th)
            
            i_ranges = list(_tile_ranges(n, state_tile))
            it_a = tqdm(i_ranges, desc=desc, leave=False) if progress else i_ranges
            
            # OPTIMIZATION 4: Async dispatch
            compute_stream = _get_compute_stream()
            tile_count = 0
            
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
                    
                    # Async dispatch
                    _dispatch_kernel_async(k_fn, grid, block,
                                         (s_a_cp, s_b_cp, out_tile, bi, bj, dim, dim, dim, bj),
                                         stream=compute_stream)
                    
                    K_cp[i0:i1, j0:j1] = out_tile
                    if symmetric and (Y is None) and j0 > i0:
                        K_cp[j0:j1, i0:i1] = out_tile.T
                    
                    tile_count += 1
                    if tile_count % BATCH_SYNC_INTERVAL == 0:
                        compute_stream.synchronize()
                
                del s_a_cp
            
            # Final synchronization
            compute_stream.synchronize()
        
        # OPTIMIZATION 5: Memory cleanup
        K = K_cp.get().astype(r_dt)
        cp.get_default_memory_pool().free_all_blocks()
        
        # --- PROTECTION ANTI-CRASH (NEW) ---
        if not np.all(np.isfinite(K)):
            print("⚠️ Matrice corrompue (NaN/Inf) détectée dans le backend cuda_states. Réparation...")
            K = np.nan_to_num(K, nan=0.0, posinf=1.0, neginf=0.0)
        # -----------------------------------

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