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


def _setup_cupy():
    """Pool mémoire + includes NVRTC utiles pour RawModule."""
    import cupy as cp  # type: ignore
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    _ = cp.ones((1,), dtype=cp.float32);
    del _
    cp.cuda.runtime.deviceSynchronize()
    os.environ.setdefault("CUPY_NVRTC_OPTIONS", "-I/usr/local/cuda/include")
    for key in ("CPATH", "CPLUS_INCLUDE_PATH"):
        os.environ[key] = "/usr/local/cuda/include:" + os.environ.get(key, "")


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
                    qml.templates.BasicEntanglerLayers(_pl_w[l:l + 1], wires=range(_pl_nq))
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
                qml.templates.BasicEntanglerLayers(w[l:l + 1], wires=range(nq))
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
# ---------- RawKernels cuda_states (PL+Torch -> DLPack -> RawKernel) ----------
# =====================================================================
CUDA_STATES_GEMM_FULL_SRC = r"""
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
void cgemm_abs2_tiled_full(const float2* __restrict__ SA,
                           const float2* __restrict__ SB,
                           float* __restrict__ K,
                           const int BM, const int BN, const int D,
                           const int lda, const int ldb, const int ldk)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= BM || j >= BN) return;

    __shared__ float2 sA[TILE_M][TILE_K];
    __shared__ float2 sB[TILE_N][TILE_K];
    float2 acc; acc.x = 0.f; acc.y = 0.f;

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : make_float2(0.f, 0.f);
            }
        }
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                float2 v = (k < D) ? SB[j * ldb + k] : make_float2(0.f, 0.f);
                sB[threadIdx.x][tk] = make_float2(v.x, -v.y); // conj
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            float2 a = sA[threadIdx.y][tk];
            float2 b = sB[threadIdx.x][tk];
            float rx = a.x * b.x - a.y * b.y;
            float ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }

    float mag2 = acc.x * acc.x + acc.y * acc.y;
    K[i * ldk + j] = mag2;
}
""";

CUDA_STATES_GEMM_LOWER_SRC = r"""
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
void cgemm_abs2_tiled_lower(const float2* __restrict__ SA,
                            const float2* __restrict__ SB,
                            float* __restrict__ K,
                            const int BM, const int BN, const int D,
                            const int lda, const int ldb, const int ldk)
{
    if (blockIdx.x > blockIdx.y) return;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= BM || j >= BN) return;
    if (BM == BN && j > i) return;

    __shared__ float2 sA[TILE_M][TILE_K];
    __shared__ float2 sB[TILE_N][TILE_K];
    float2 acc; acc.x = 0.f; acc.y = 0.f;

    for (int k0 = 0; k0 < D; k0 += TILE_K) {
        if (i < BM) {
            for (int tk = threadIdx.x; tk < TILE_K; tk += blockDim.x) {
                int k = k0 + tk;
                sA[threadIdx.y][tk] = (k < D) ? SA[i * lda + k] : make_float2(0.f, 0.f);
            }
        }
        if (j < BN) {
            for (int tk = threadIdx.y; tk < TILE_K; tk += blockDim.y) {
                int k = k0 + tk;
                float2 v = (k < D) ? SB[j * ldb + k] : make_float2(0.f, 0.f);
                sB[threadIdx.x][tk] = make_float2(v.x, -v.y); // conj
            }
        }
        __syncthreads();

        #pragma unroll
        for (int tk = 0; tk < TILE_K; ++tk) {
            float2 a = sA[threadIdx.y][tk];
            float2 b = sB[threadIdx.x][tk];
            float rx = a.x * b.x - a.y * b.y;
            float ry = a.x * b.y + a.y * b.x;
            acc.x += rx; acc.y += ry;
        }
        __syncthreads();
    }

    float mag2 = acc.x * acc.x + acc.y * acc.y;
    K[i * ldk + j] = mag2;
}
""";

# caches RawModule -> fonction, paramétrés par macros
_RAWMOD_FULL_CACHE = {}
_RAWMOD_LOWER_CACHE = {}


# Replace your _get_full_kernel_with_macros and _get_lower_kernel_with_macros
# with these versions (note: no --gpu-architecture/-arch)

def _get_full_kernel_with_macros(tm: int, tn: int, tk: int):
    import cupy as cp
    key = (tm, tn, tk)
    fn = _RAWMOD_FULL_CACHE.get(key)
    if fn is not None:
        return fn
    # CuPy injects --gpu-architecture automatically; do NOT pass it here
    opts = ("--std=c++14", "--use_fast_math",  # optional: fast math
            f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    mod = cp.RawModule(code=CUDA_STATES_GEMM_FULL_SRC,
                       options=opts,
                       name_expressions=("cgemm_abs2_tiled_full",))
    fn = mod.get_function("cgemm_abs2_tiled_full")
    _RAWMOD_FULL_CACHE[key] = fn
    return fn

def _get_lower_kernel_with_macros(tm: int, tn: int, tk: int):
    import cupy as cp
    key = (tm, tn, tk)
    fn = _RAWMOD_LOWER_CACHE.get(key)
    if fn is not None:
        return fn
    # CuPy injects --gpu-architecture automatically; do NOT pass it here
    opts = ("--std=c++14", "--use_fast_math",
            f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    mod = cp.RawModule(code=CUDA_STATES_GEMM_LOWER_SRC,
                       options=opts,
                       name_expressions=("cgemm_abs2_tiled_lower",))
    fn = mod.get_function("cgemm_abs2_tiled_lower")
    _RAWMOD_LOWER_CACHE[key] = fn
    return fn



# ---------------- Auto-tuner des tuiles (RawKernel uniquement) ----------------
def _time_kernel_once_full(SA_cp, SB_cp, bi: int, bj: int, dim: int, tm: int, tn: int, tk: int) -> float:
    import cupy as cp
    k_full = _get_full_kernel_with_macros(tm, tn, tk)
    K_view = cp.empty((bi, bj), dtype=cp.float32)
    block = (tn, tm, 1)
    grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
    args = (SA_cp, SB_cp, K_view, np.int32(bi), np.int32(bj), np.int32(dim),
            np.int32(dim), np.int32(dim), np.int32(bj))
    start = cp.cuda.Event();
    end = cp.cuda.Event()
    start.record();
    k_full(grid, block, args);
    end.record();
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end)  # ms


def _time_kernel_once_lower(SA_cp, bi: int, dim: int, tm: int, tn: int, tk: int) -> float:
    import cupy as cp
    k_lower = _get_lower_kernel_with_macros(tm, tn, tk)
    K_view = cp.empty((bi, bi), dtype=cp.float32)
    block = (tn, tm, 1)
    grid = ((bi + tn - 1) // tn, (bi + tm - 1) // tm, 1)
    args = (SA_cp, SA_cp, K_view, np.int32(bi), np.int32(bi), np.int32(dim),
            np.int32(dim), np.int32(dim), np.int32(bi))
    start = cp.cuda.Event();
    end = cp.cuda.Event()
    start.record();
    k_lower(grid, block, args);
    end.record();
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end)  # ms


def autotune_tiles(
        dim: int,
        bi: int = 128,
        bj: int = 128,
        *,
        symmetric: bool = True,
        candidates: list[tuple[int, int, int]] | None = None,
        warmup: int = 1,
        iters: int = 3,
        seed: int = 0,
) -> tuple[int, int, int]:
    """Essaie plusieurs (TILE_M, TILE_N, TILE_K) et retourne la meilleure config (ms min)."""
    import cupy as cp
    rng = np.random.default_rng(seed)
    if candidates is None:
        candidates = [
            (32, 32, 32), (64, 16, 32), (16, 64, 32),
            (32, 32, 16), (32, 32, 64),
            (64, 32, 16), (32, 64, 16),
        ]
    SA = (rng.standard_normal((bi, dim)).astype(np.float32)
          + 1j * rng.standard_normal((bi, dim)).astype(np.float32))
    SB = (rng.standard_normal((bj, dim)).astype(np.float32)
          + 1j * rng.standard_normal((bj, dim)).astype(np.float32))
    SA_cp = cp.asarray(SA);
    SB_cp = cp.asarray(SB)

    def _fits(tm, tn, tk):
        # Taille en octets des deux tuiles en shared mem (float2 = 8 bytes)
        bytes_per_tile = (tm * tk + tn * tk) * 8

        # CuPy 12/13: récupérer les propriétés du device
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            # priorité: opt-in si dispo, sinon valeur par défaut
            sm_lim = int(props.get('sharedMemPerBlockOptin', props.get('sharedMemPerBlock', 48 * 1024)))
        except Exception:
            # Fallback conservateur (48 KiB)
            sm_lim = 48 * 1024

        return bytes_per_tile <= sm_lim

    best = None
    for (tm, tn, tk) in candidates:
        if not _fits(tm, tn, tk):
            continue
        for _ in range(warmup):
            (_time_kernel_once_lower(SA_cp, min(bi, bj), dim, tm, tn, tk)
             if symmetric else
             _time_kernel_once_full(SA_cp, SB_cp, bi, bj, dim, tm, tn, tk))
        ms = np.mean([
            (_time_kernel_once_lower(SA_cp, min(bi, bj), dim, tm, tn, tk)
             if symmetric else
             _time_kernel_once_full(SA_cp, SB_cp, bi, bj, dim, tm, tn, tk))
            for _ in range(iters)
        ])
        if best is None or ms < best[0]:
            best = (ms, (tm, tn, tk))
    return best[1] if best else (32, 32, 32)


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
    t_float = th.float32 if x_blk.dtype == np.float32 else th.float64
    t_complex = th.complex64 if t_float == th.float32 else th.complex128

    x = th.from_numpy(np.ascontiguousarray(x_blk)).to("cuda", dtype=t_float, non_blocking=True)
    w = th.from_numpy(np.ascontiguousarray(w_np)).to("cuda", dtype=t_float, non_blocking=True)

    dev_name = device_name
    if "gpu" not in str(device_name).lower():
        try:
            _ = qml.device("lightning.gpu", wires=nq, shots=None,
                           c_dtype=(np.complex64 if t_float == th.float32 else np.complex128))
            dev_name = "lightning.gpu"
        except Exception:
            dev_name = device_name

    dev = qml.device(dev_name, wires=nq, shots=None,
                     c_dtype=(np.complex64 if t_float == th.float32 else np.complex128))

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

    try:
        from torch import vmap  # type: ignore
        states = vmap(_state)(x).to(dtype=t_complex)
    except Exception:
        states = th.stack([_state(x[i]) for i in range(x.shape[0])], dim=0).to(dtype=t_complex)

    if states.device.type != "cuda":
        states = states.to("cuda", non_blocking=True)
    return states



def _torch_cuda_to_cupy(t: "torch.Tensor"):
    """Zéro-copie Torch CUDA → CuPy via DLPack, avec fallbacks silencieux."""
    import cupy as cp
    import warnings

    # 1) s’aligner sur le même device logique que le tenseur Torch
    try:
        if hasattr(t, "device") and t.device.type == "cuda":
            dev_id = int(t.device.index or 0)
            cp.cuda.Device(dev_id).use()
    except Exception:
        pass

    # 2) API moderne (CuPy >= 12) : pas de warnings
    try:
        return cp.from_dlpack(t)
    except Exception:
        pass

    # 3) Fallback legacy via capsule DLPack, WARNING masqué
    try:
        from torch.utils.dlpack import to_dlpack
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            # Certains CuPy émettent un VisibleDeprecationWarning ici :
            warnings.filterwarnings("ignore", message=".*fromDlpack.*", category=Warning)
            return cp.from_dlpack(to_dlpack(t))
    except Exception:
        # 4) Dernier recours : copie CPU (lent, mais ne casse pas le bench)
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
    """Kernel de fidélité PennyLane, parallèle (multiprocessing), tuilé (CPU)."""
    import multiprocessing as mp
    import pennylane as qml  # noqa: F401

    def _resolve_float_dtype() -> np.dtype:
        if dtype in ("float32", "float64"):
            return np.dtype(np.float32) if dtype == "float32" else np.dtype(np.float64)
        src_dtypes: list[np.dtype] = []
        try:
            src_dtypes.append(np.asarray(A).dtype)
        except Exception:
            pass
        try:
            src_dtypes.append(np.asarray(weights).dtype)
        except Exception:
            pass
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
        # --- PARAMS pour cuda_states ---
        state_tile: int = 8192,                 # nb d'exemples par bloc d'ÉTATS (streaming)
        tile_m: int | str = "auto",            # macros TILE_M (RawKernel)
        tile_n: int | str = "auto",            # macros TILE_N
        tile_k: int | str = "auto",            # macros TILE_K
) -> np.ndarray:
    """
    Fidelity kernel between quantum states prepared from angles X(/Y).
    - angle_scale: multiplicative factor on input angles (γ-like control)
    - re_embed_between_layers: re-apply data embedding between entangler layers
    - embed_mode: 'ry' | 'ryrz' | 'angle'
    - normalize: enforce diag(K)=1 (only for square K); jitter adds tiny value on diag before normalization

    Public API.
    - 'cuda_states' : états sur GPU via PL+Torch (par blocs) + RawKernel tuilé (recommandé pour SVM K(X,X))
    - 'numpy'/'cupy'/'torch'/'auto' : pipeline PennyLane existant (entanglement OK)
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

    # -------- backend cuda_states --------
    if gram_backend == "cuda_states":
        import cupy as cp
        try:
            _setup_cupy()
        except Exception:
            pass

        calc_dt = np.float32 if (
                dtype is None or dtype == "float32" or "gpu" in device_name.lower()
        ) else np.float64
        ret_dt = np.float32 if (return_dtype is None or return_dtype == "float32") else np.float64

        A = _ensure_numpy(X, dtype=calc_dt)
        B = A if Y is None else _ensure_numpy(Y, dtype=calc_dt)
        n, nq = A.shape
        m = B.shape[0]
        w = _ensure_numpy(weights, dtype=calc_dt)
        if w.ndim != 2 or w.shape[1] != nq:
            raise ValueError(f"`weights` must be [n_layers, n_qubits={nq}]")
        dim = 1 << nq

        # Choose kernel
        auto_tiles = (tile_m == "auto") or (tile_n == "auto") or (tile_k == "auto")
        if auto_tiles:
            try:
                tm, tn, tk = autotune_tiles(dim,
                                            bi=min(state_tile, n),
                                            bj=min(state_tile, m),
                                            symmetric=(symmetric and (B is A)))
            except Exception:
                tm, tn, tk = (32, 32, 32)
        else:
            tm, tn, tk = int(tile_m), int(tile_n), int(tile_k)

        k_func = _get_lower_kernel_with_macros(tm, tn, tk) if (symmetric and (B is A)) \
            else _get_full_kernel_with_macros(tm, tn, tk)

        # Output
        K_cp = cp.empty((n, m), dtype=cp.float32)

        # Precompute and cache SB tiles once
        j_tiles = list(_tile_ranges(m, state_tile))
        sb_cache: dict[tuple[int, int], cp.ndarray] = {}

        jt = tqdm(j_tiles, desc="cuda_states: cache SB", leave=False) if (tqdm is not None and progress) else j_tiles
        for (j0, j1) in jt:
            SB_th = _build_states_block_torch_cuda(
                B[j0:j1], w, device_name,
                angle_scale=angle_scale,
                re_embed_between_layers=re_embed_between_layers,
                embed_mode=embed_mode,
            )
            sb_cache[(j0, j1)] = _torch_cuda_to_cupy(SB_th)  # [bj, dim] complex64

        # Iterate A tiles; for symmetric K(X,X), only j-tiles with j1 > i0
        outer = list(_tile_ranges(n, state_tile))
        ot = tqdm(outer, desc="cuda_states: outer A", leave=False) if (tqdm is not None and progress) else outer

        for (i0, i1) in ot:
            SA_th = _build_states_block_torch_cuda(
                A[i0:i1], w, device_name,
                angle_scale=angle_scale,
                re_embed_between_layers=re_embed_between_layers,
                embed_mode=embed_mode,
            )
            SA_cp = _torch_cuda_to_cupy(SA_th)  # [bi, dim]

            inner = j_tiles
            if symmetric and (B is A):
                inner = [(j0, j1) for (j0, j1) in j_tiles if j1 > i0]

            it = tqdm(inner, desc=f"cuda_states: inner B[{i0}:{i1}]", leave=False) if (
                        tqdm is not None and progress) else inner
            for (j0, j1) in it:
                SB_cp = sb_cache[(j0, j1)]
                bi = int(i1 - i0);
                bj = int(j1 - j0)
                K_view = K_cp[i0:i1, j0:j1]

                # Launch raw kernel (lower/full as chosen)
                block = (tn, tm, 1)  # x=cols, y=rows
                grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
                lda = dim;
                ldb = dim;
                ldk = int(K_view.strides[0] // K_view.itemsize)
                args = (SA_cp, SB_cp, K_view,
                        np.int32(bi), np.int32(bj), np.int32(dim),
                        np.int32(lda), np.int32(ldb), np.int32(ldk))
                k_func(grid, block, args)

                # Mirror upper when symmetric
                if symmetric and (B is A) and (j0 > i0):
                    K_cp[j0:j1, i0:i1] = K_view.T

            del SA_cp  # free earlier

        K = K_cp.get().astype(ret_dt, copy=False)
        if normalize and (Y is None):
            if jitter and jitter > 0:
                K = K + float(jitter) * np.eye(K.shape[0], dtype=K.dtype)
            _normalize_diag_inplace(K)
        return K

    # Try fast GPU path with Torch
    if ("gpu" in device_name.lower()) and gram_backend in ("auto", "torch"):
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
