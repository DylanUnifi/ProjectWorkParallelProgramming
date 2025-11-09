# Parallel Programming Project

[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20PennyLane%20%7C%20CUDA%20%7C%20OpenMP-green?logo=pytorch)](#)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-PL%20Kernel%20%7C%20CPU%20vs%20GPU-blue)](#)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-logging-orange?logo=weightsandbiases)](#-logging--artifacts)

SVM **quantum and classical training** for binary image classification with a focus on **parallel** and **heterogeneous** computation of **quantum kernel matrices**.  
Project developed in the *Parallel Programming* course (University of Florence).

> Core idea: the **Gram matrix** \(K\) (state fidelity kernel) dominates training time. We accelerate it with tiling, multiprocessing and GPU backends (NumPy, CuPy, Torch/CUDA).

---

## Table of Contents
- [Overview](#overview)
- [What’s new](#whats-new)
- [Repo layout](#repo-layout)
- [Setup](#setup)
- [Backends & knobs](#backends--knobs)
- [Benchmarks (snapshot)](#benchmarks-snapshot)
- [Configs (datasets & debug)](#configs-datasets--debug)
- [Experiments Roadmap](#-experiments-roadmap)
- [Logging & artifacts](#-logging--artifacts)
- [Troubleshooting](#troubleshooting)
- [License & citation](#license--citation)

---

## Overview

We provide SVM binary classification of classical images dataset with rbf kernel and computed quantum kernel:

- `scripts/train_svm_qkernel.py` — **script** for SVM Quantum kernels training.
- `scripts/train_svm.py` — **script** for SVM RBF kernel training.
- `quantumkernels/compute_kernel.py` — unified API to compute \(K\) with:
  - **PennyLane / lightning.qubit** (CPU) and **lightning.gpu** (GPU)
  - **Host matmul**: NumPy (CPU), **CuPy** (GPU), **Torch/CUDA** (GPU, *streaming path: states+GEMM entirely on GPU*)
  - **Multiprocessing** (spawn-safe) and **tiling** to bound memory
- `benchmark_pl_kernel.py` — sweeps configurations and reports **throughput** (Mpairs/s).

---

## Repo layout

```
train_svm.py
train_svm_qkernel.py.py
quantumkernels/
  └─ compute_kernel.py               # unified kernel API (NumPy/CuPy/Torch)
benchmark_pl_kernel.py             # grid search for kernel throughput
run_experiments_all.sh
configs/
  ├─ fashion_debug.yaml
  ├─ svhn_debug.yaml
  ├─ cifar10_debug.yaml
  └─ fashion/svhn/cifar10*.yaml
results/                                # plots
```

---

## Setup

### Environment

- **OS**: Linux (tested on Ubuntu 24.04)
- **CUDA**: 12.x recommended (for GPU paths)

```bash
# Clone
git clone https://github.com/DylanUnifi/ProjectWorkParallelProgramming.git
cd ProjectWorkParallelProgramming

# Conda env
conda create -n ProjectWorkParallelProgramming python=3.12 -y
conda activate ProjectWorkParallelProgramming

# Base deps
pip install -r requirements.txt

# GPU backends
pip install "cupy-cuda12x"                # CuPy (CUDA 12.x)
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
```

---

## Backends & knobs

- **Devices** (`--pl-device`):  
  - `lightning.qubit` (CPU), `lightning.gpu` (GPU)
- **Matmul backends** (`--gram-backend`):  
  - `numpy`, `torch`, `cuda_ry`
- **Parallelism**:  
  - `--workers` (processes; GPU forces 1)
  - `--tile-size` (rows per block for states + GEMM)
- **Precision**:  
  - `--dtype {float32,float64}` (compute)
  - `--return-dtype {float32,float64}` (final kernel)

---


## Experiments Roadmap

### Troubleshooting

- **`ModuleNotFoundError: openpyxl`** → `pip install openpyxl`  
- **CuPy error: `cuda_fp16.h` missing** → install CUDA toolkit headers (`conda install -c conda-forge cuda-toolkit=12.1 cupy`)  
- **GPU slower than CPU** → increase `--tile-size` (≥1024) and use `--gram-backend torch`  
- **CPU contention** → `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` 

---

### Benchmarking & HPC scaling
- **Objective**: understand how the kernel backends behave (independent of dataset difficulty).  
- Run `tools/benchmark_pl_kernel.py` to compare:
  - **CPU (NumPy + multiprocessing)** vs **GPU (Torch streaming, CuPy)**.  
  - Vary `tile_size`, `workers`, precision (`float32` vs `float64`).  
- Collect **Mpairs/s throughput** and identify optimal settings per device. 
---

### Snapschot Results

**Host: “Papavero”** — Intel Xeon (32C/64T), CUDA 12.x GPU  

| Config (N, nq)             | CPU (NumPy)             | GPU (Torch/CuPy)         |
|-----------------------------|-------------------------|--------------------------|
| **N=4096, nq=12**           | ~1.08 Mpairs/s          | ~0.48 Mpairs/s           |
| **N=8192, nq=12**           | —                       | ~0.94 Mpairs/s           | 


---

---

### Debug mode (fast checks)
```bash
python scripts/train_svm_qkernel.py \
  --config configs/fashion_debug.yaml
```

Debug versions (fast sanity checks):
- `fashion_debug.yaml` (5 epochs, batch 64)  
- `svhn_debug.yaml` (8 epochs, batch 64)  
- `cifar10_debug.yaml` (10 epochs, batch 64)  

---

### Train (Quantum kernel)
```bash
python scripts/train_svm_qkernel.py \
  --config configs/fashion.yaml \
  --kernel qkernel
  --pl-device lightning.gpu
  --kernel-centering
  --gram-backend cuda_ry
```

👉 Configs: `fashion.yaml`, `svhn.yaml`, `cifar10.yaml`.
---

### Train (RBF kernel)
```bash
python scripts/train_svm.py \
  --config configs/fashion.yaml
```

👉 Configs: `fashion.yaml`, `svhn.yaml`, `cifar10.yaml`.
---

### Reporting
- Aggregate metrics: **F1, AUC, Balanced Accuracy, speed**.  
- Export tables (CSV/Excel).  
- Generate comparative plots: accuracy vs computation time.  
- Draft final **academic-style report** with discussion.

---

## 📦 Logging & artifacts

- **Weights & Biases**: per-epoch metrics, configs, thresholds.  
- **Exports**: CSV and Excel (`openpyxl`).

---

## License & citation

- Code released for academic use within the Parallel Programming course.  
- Please cite the repo and upstream frameworks (PennyLane, PyTorch) if you build on it.

---

✍️ **Author**: Dylan Fouepe — Master’s in AI, University of Florence  
GitHub: [@DylanUnifi](https://github.com/DylanUnifi)
