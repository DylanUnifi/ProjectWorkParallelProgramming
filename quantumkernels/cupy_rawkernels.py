import numpy as np
import cupy as cp


# =====================================================================
# RawKernel CUDA sources
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


# Caches
_RAWMOD_FULL_CACHE = {}
_RAWMOD_LOWER_CACHE = {}


def get_full_kernel_with_macros(tm, tn, tk):
    key = (tm, tn, tk)
    if key in _RAWMOD_FULL_CACHE:
        return _RAWMOD_FULL_CACHE[key]
    opts = ("--std=c++14", "--use_fast_math",
            f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    mod = cp.RawModule(code=CUDA_STATES_GEMM_FULL_SRC,
                       options=opts,
                       name_expressions=("cgemm_abs2_tiled_full",))
    fn = mod.get_function("cgemm_abs2_tiled_full")
    _RAWMOD_FULL_CACHE[key] = fn
    return fn


def get_lower_kernel_with_macros(tm, tn, tk):
    key = (tm, tn, tk)
    if key in _RAWMOD_LOWER_CACHE:
        return _RAWMOD_LOWER_CACHE[key]
    opts = ("--std=c++14", "--use_fast_math",
            f"-DTILE_M={tm}", f"-DTILE_N={tn}", f"-DTILE_K={tk}")
    mod = cp.RawModule(code=CUDA_STATES_GEMM_LOWER_SRC,
                       options=opts,
                       name_expressions=("cgemm_abs2_tiled_lower",))
    fn = mod.get_function("cgemm_abs2_tiled_lower")
    _RAWMOD_LOWER_CACHE[key] = fn
    return fn


# ---------- Autotuning ----------
def autotune_tiles(dim, bi=128, bj=128, *, symmetric=True,
                   candidates=None, warmup=1, iters=3, seed=0):
    """Try several (TILE_M, TILE_N, TILE_K) configs and return the fastest."""
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
    SA_cp = cp.asarray(SA)
    SB_cp = cp.asarray(SB)

    def _fits(tm, tn, tk):
        bytes_per_tile = (tm * tk + tn * tk) * 8
        try:
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            sm_lim = int(props.get('sharedMemPerBlockOptin',
                                   props.get('sharedMemPerBlock', 48 * 1024)))
        except Exception:
            sm_lim = 48 * 1024
        return bytes_per_tile <= sm_lim

    def _time_once(func):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func()
        end.record()
        end.synchronize()
        return cp.cuda.get_elapsed_time(start, end)

    best = None
    for (tm, tn, tk) in candidates:
        if not _fits(tm, tn, tk):
            continue
        kernel = get_lower_kernel_with_macros(tm, tn, tk) if symmetric else get_full_kernel_with_macros(tm, tn, tk)
        K = cp.empty((bi, bj), dtype=cp.float32)
        block = (tn, tm, 1)
        grid = ((bj + tn - 1) // tn, (bi + tm - 1) // tm, 1)
        args = (SA_cp, SB_cp, K, np.int32(bi), np.int32(bj), np.int32(dim),
                np.int32(dim), np.int32(dim), np.int32(bj))
        for _ in range(warmup):
            kernel(grid, block, args)
        ms = np.mean([_time_once(lambda: kernel(grid, block, args))
                      for _ in range(iters)])
        if best is None or ms < best[0]:
            best = (ms, (tm, tn, tk))
    return best[1] if best else (32, 32, 32)
