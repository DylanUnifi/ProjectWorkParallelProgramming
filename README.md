# Quantum Machine Learning — Parallel Programming Project

[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20PennyLane%20%7C%20CUDA%20%7C%20OpenMP-green?logo=pytorch)](#)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-PL%20Kernel%20%7C%20CPU%20vs%20GPU-blue)](#)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-logging-orange?logo=weightsandbiases)](#-logging--artifacts)

Hybrid **quantum–classical** pipeline with a focus on **parallel** and **heterogeneous** computation of **quantum kernel matrices**.  
Project developed in the *Parallel Programming* course (University of Florence).

> Core idea: the **Gram matrix** \(K\) (state fidelity kernel) dominates training time. We accelerate it with tiling, multiprocessing and GPU backends (NumPy, CuPy, Torch/CUDA).

---

## Table of Contents
- [Overview](#overview)
- [What’s new](#whats-new)
- [Repo layout](#repo-layout)
- [Setup](#setup)
- [Quick start](#quick-start)
- [Backends & knobs](#backends--knobs)
- [Benchmarks (snapshot)](#benchmarks-snapshot)
- [Datasets → angles](#datasets--angles)
- [Logging & artifacts](#-logging--artifacts)
- [Troubleshooting](#troubleshooting)
- [License & citation](#license--citation)

---

## Overview

We provide two training scripts and a unified kernel backend:

- `scripts/train_hybrid_qcnn_quantumkernel.py` — Hybrid **QCNN + SVM** (sequential kernel).
- `scripts/train_hybrid_qcnn_quantumkernel_patched.py` — same model, **pluggable HPC backends**.
- `scripts/pipeline_backends.py` — unified API to compute \(K\) with:
  - **PennyLane / lightning.qubit** (CPU) and **lightning.gpu** (GPU) devices
  - **Host matmul**: NumPy (CPU), **CuPy** (GPU), **Torch/CUDA** (GPU, *streaming path: states+GEMM entirely on GPU*)
  - **Multiprocessing** (spawn-safe) and **tiling** to bound memory

A dedicated tool, `tools/benchmark_pl_kernel.py`, sweeps configurations and reports **throughput** (Mpairs/s).

---

## What’s new

- **Torch “streaming” GPU path**: state preparation **and** GEMM on GPU (zero host copies), triggered with `--gram-backend torch` and `--device lightning.gpu`.
- **Typed dtypes** end-to-end: `--dtype {float32,float64}` for compute & `--return-dtype` for \(K\).
- **Robust multiprocessing** (spawn) with explicit thread caps (`OMP_NUM_THREADS=1`).
- **Dataset-friendly bench**:
  - `--data-x/--data-y` load `.npy/.npz/.csv`  
  - encode features → angles with `--encoder {minmax,zscore,identity}`, pad/truncate to `nq`
  - symmetric \(K(X,X)\) or cross \(K(X,Y)\)

---

## Repo layout

```
scripts/
  ├─ train_hybrid_qcnn_quantumkernel.py
  ├─ train_hybrid_qcnn_quantumkernel_patched.py
  └─ pipeline_backends.py        # unified kernel API (NumPy/CuPy/Torch, MP-safe)
tools/
  ├─ benchmark_pl_kernel.py      # grid search for kernel throughput
  └─ run_experiments_all.sh
configs/
  └─ config_train_hybrid_qcnn_quantumkernel_*.yaml
results/                          # plots (optional)
```

---

## Setup

### Environment

- **OS**: Linux (tested on Ubuntu 24.04)
- **Python**: 3.11
- **CUDA**: 12.x recommended (for GPU paths)

```bash
# Clone
git clone https://github.com/DylanUnifi/qml-parallel-project.git
cd qml-parallel-project

# Conda env
conda create -n ProjectWork-ParallelProgramming python=3.11 -y
conda activate ProjectWork-ParallelProgramming

# Base deps
pip install -r requirements.txt

# Optional: GPU backends (choose what you need)
# CuPy (CUDA 12.x wheel)
pip install "cupy-cuda12x"
# Torch + CUDA (conda, recommended)
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
# Excel export (if you write .xlsx)
pip install openpyxl
```

> If you prefer conda-forge for CuPy/toolkit:  
> `mamba install -c conda-forge cuda-toolkit=12.1 cupy`

### (Optional) OpenMP C++ extension
If you also use the C++ OpenMP path in your own models:

```bash
cd models/backends
python setup.py build_ext --inplace
```

---

## Quick start

### Train (sequential kernel)
```bash
python scripts/train_hybrid_qcnn_quantumkernel.py   --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml
```

### Train (HPC backends)

**CPU (NumPy tiling):**
```bash
python scripts/train_hybrid_qcnn_quantumkernel_patched.py   --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml   --backend cpu --tile-size 256
```

**GPU (Torch streaming, no host copies):**
```bash
python scripts/train_hybrid_qcnn_quantumkernel_patched.py   --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml   --backend torchcuda --tile-size 2048
```

**GPU (CuPy/cuBLAS matmul):**
```bash
python scripts/train_hybrid_qcnn_quantumkernel_patched.py   --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml   --backend pycuda --tile-size 1024
```

### Stand-alone kernel benchmarks

**CPU vs GPU, grid sweep**
```bash
python tools/benchmark_pl_kernel.py   --samples 1024   --qubits 6 10 12   --tile-size 64 128 256   --workers 1 8 12 16 24 32   --device lightning.qubit lightning.gpu   --symmetric   --layers 2   --dtype float32 --return-dtype float64   --gram-backend auto --repeats 5
```

**Dataset-driven bench**
```bash
python tools/benchmark_pl_kernel.py   --data-x data/features_train.npy   --encoder minmax --enc-alpha 1.0   --qubits 12   --tile-size 512   --workers 8   --device lightning.qubit   --symmetric   --layers 4   --dtype float32 --return-dtype float32   --gram-backend numpy --repeats 3
```

---

## Backends & knobs

- **Devices** (`--device`):
  - `lightning.qubit` (CPU), `lightning.gpu` (GPU)
- **Matmul backends** (`--gram-backend`):
  - `numpy` (CPU)
  - `cupy` (GPU cuBLAS; requires CuPy + CUDA headers)
  - `torch` (GPU, **streaming**: states + GEMM on GPU)
  - `auto` (choose best available)
- **Parallelism / tiling**:
  - `--workers` (processes; GPU forces 1)
  - `--tile-size` (rows per block for states + GEMM)
- **Precision**:
  - `--dtype {float32,float64}` compute precision (and device `c_dtype`)
  - `--return-dtype {float32,float64}` final \(K\) dtype
- **Circuit**:
  - `--layers` (BasicEntanglerLayers depth)

---

## Benchmarks (snapshot)

**Host: “Papavero”** — Intel Xeon (32C/64T), CUDA 12.x GPU  
*Measured Mpairs/s (higher = faster). Your numbers will vary slightly.*

| Config (N, nq)             | Best CPU (NumPy)                    | Best GPU (Torch/CuPy)              |
|----------------------------|-------------------------------------|------------------------------------|
| **N=4096, nq=12**          | **~1.085 Mpairs/s** (`tile=512`, `workers=8`) | ~0.48 Mpairs/s (`tile=2048`)       |
| **N=8192, nq=12**          | —                                   | **~0.94 Mpairs/s** (`tile=2048`)   |

**Key takeaways**
- CPU excels on moderate \(N\) with **8–12 processes** and `tile≈512`.
- GPU needs **large tiles** (`≥1024`) and benefits most at **large N**.
- The **Torch streaming path** avoids CPU↔GPU ping-pong and is the recommended GPU mode.

---

## Datasets → angles

`benchmark_pl_kernel.py` can ingest features or angles:

- `--data-x`, `--data-y`: `.npy/.npz/.csv/.tsv`
- `--data-angles`: skip encoding (already angles \([-π, π]\))
- `--encoder {minmax,zscore,identity}` + `--enc-alpha`
- Auto **pad/truncate** to match `--qubits`.
- If `--data-y` is provided, the run switches to **non symmetric** \(K(X,Y)\).

---

## 📦 Logging & artifacts

- **Weights & Biases** (optional): metrics, confusion matrices, per-epoch timings.  
  Set `WANDB_PROJECT` in your environment or config.
- **Exports**: CSV and (optionally) Excel (`openpyxl`). If `openpyxl` is missing, we gracefully fall back to CSV (see training script note).

---

## Troubleshooting

- **`ModuleNotFoundError: openpyxl`**  
  → `pip/conda install openpyxl`, or add a fallback to CSV in the writer section.
- **CuPy error: `cuda_fp16.h` missing`**  
  → Install **CUDA toolkit** headers (e.g. `mamba install -c conda-forge cuda-toolkit=12.1 cupy`) or switch to `--gram-backend torch`.
- **GPU slower than CPU**  
  - Increase `--tile-size` (e.g., 1024–3072 for 12–14 qubits).  
  - Use `--gram-backend torch` (streaming).  
  - Keep `--dtype float32` unless precision demands otherwise.
- **High CPU contention**  
  - Cap threads: `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.  
  - Try `--workers 8` on 32C/64T hosts.

---

## License & citation

- Code released for academic use within the Parallel Programming course.  
- Please cite the repo and the upstream frameworks (PennyLane, PyTorch) if you build on it.

---

**Author**: Dylan Fouepe — Master’s in AI, University of Florence  
GitHub: [@DylanUnifi](https://github.com/DylanUnifi)
