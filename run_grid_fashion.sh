#!/usr/bin/env bash
set -euo pipefail

CFG="configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml"
PY="python train_hybrid_qcnn_quantumkernel_parallel_patched.py"

LOGDIR="logs_fashion_grid"
mkdir -p "$LOGDIR"

# -------------------------------
# Helpers: set CPU thread env vars
# -------------------------------
set_cpu_threads () {
  local n="$1"
  export OMP_NUM_THREADS="$n"
  export MKL_NUM_THREADS="$n"
  export OPENBLAS_NUM_THREADS="$n"
  export NUMEXPR_NUM_THREADS="$n"
}

# -------------------------------
# 1) CPU (NumPy) — vary threads & tile
# -------------------------------
for T in 1 4 8 16; do
  for TILE in 64 128 256; do
    set_cpu_threads "$T"
    CUDA_VISIBLE_DEVICES=""     $PY --config "$CFG" --backend cpu --tile-size "$TILE"       2>&1 | tee "$LOGDIR/cpu_threads${T}_tile${TILE}.log"
  done
done

# -------------------------------
# 2) Numba — vary threads (tile ignored)
# -------------------------------
for T in 1 4 8 16; do
  export NUMBA_NUM_THREADS="$T"
  CUDA_VISIBLE_DEVICES=""   $PY --config "$CFG" --backend numba --tile-size 128     2>&1 | tee "$LOGDIR/numba_threads${T}.log"
done

# -------------------------------
# 3) OpenMP — vary threads (tile not used)
#     requires compiled gram_omp.*.so
# -------------------------------
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
for T in 1 4 8 16; do
  export OMP_NUM_THREADS="$T"
  CUDA_VISIBLE_DEVICES=""   $PY --config "$CFG" --backend openmp --tile-size 128     2>&1 | tee "$LOGDIR/openmp_threads${T}.log"
done

# -------------------------------
# 4) Torch CUDA — vary TILE & GPU
# -------------------------------
for GPU in 0 1; do
  for TILE in 128 256 512; do
    CUDA_VISIBLE_DEVICES="$GPU"     $PY --config "$CFG" --backend torchcuda --tile-size "$TILE"       2>&1 | tee "$LOGDIR/torchcuda_gpu${GPU}_tile${TILE}.log"
  done
done

# -------------------------------
# 5) PyCUDA — vary TILE & GPU
# -------------------------------
for GPU in 0 1; do
  for TILE in 128 256 512; do
    CUDA_VISIBLE_DEVICES="$GPU"     $PY --config "$CFG" --backend pycuda --tile-size "$TILE"       2>&1 | tee "$LOGDIR/pycuda_gpu${GPU}_tile${TILE}.log"
  done
done

echo "✅ Grid done → $LOGDIR"
