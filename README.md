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
- [Configs (datasets & debug)](#configs-datasets--debug)
- [🧪 Experiments Roadmap](#-experiments-roadmap)
- [Logging & artifacts](#-logging--artifacts)
- [Troubleshooting](#troubleshooting)
- [License & citation](#license--citation)

---

## Overview

We provide unified training and kernel computation scripts:

- `scripts/train_hybrid_qcnn_svm_unified.py` — **one script** for both RBF and Quantum kernels (QCNN + SVM).
- `scripts/pipeline_backends.py` — unified API to compute \(K\) with:
  - **PennyLane / lightning.qubit** (CPU) and **lightning.gpu** (GPU)
  - **Host matmul**: NumPy (CPU), **CuPy** (GPU), **Torch/CUDA** (GPU, *streaming path: states+GEMM entirely on GPU*)
  - **Multiprocessing** (spawn-safe) and **tiling** to bound memory
- `tools/benchmark_pl_kernel.py` — sweeps configurations and reports **throughput** (Mpairs/s).

---

## What’s new

- **Unified training**: choose `--kernel rbf` or `--kernel qkernel` (quantum precomputed).
- **Torch “streaming” GPU path**: state preparation **and** GEMM on GPU (no host copies).
- **New YAML configs** for multiple datasets (Fashion-MNIST, SVHN, CIFAR-10).
- **Debug configs**: reduced epochs/samples for quick testing.
- **Typed dtypes** end-to-end: `--dtype {float32,float64}` for compute & `--return-dtype` for \(K\).
- **Robust multiprocessing** with explicit thread caps (`OMP_NUM_THREADS=1`).

---

## Repo layout

```
scripts/
  ├─ train_hybrid_qcnn_svm_unified.py   # one script for RBF & quantum kernels
  └─ pipeline_backends.py               # unified kernel API (NumPy/CuPy/Torch, MP-safe)
tools/
  ├─ benchmark_pl_kernel.py             # grid search for kernel throughput
  └─ run_experiments_all.sh
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

# Optional: GPU backends
pip install "cupy-cuda12x"                # CuPy (CUDA 12.x)
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1 -y
pip install openpyxl                      # Excel export
```

---

## Quick start

---

## Configs (datasets & debug)

- `fashion.yaml` — full run, Fashion-MNIST  
- `svhn.yaml` — full run, SVHN  
- `cifar10.yaml` — full run, CIFAR-10  

Debug versions (fast sanity checks):
- `fashion_debug.yaml` (5 epochs, batch 64)  
- `svhn_debug.yaml` (8 epochs, batch 64)  
- `cifar10_debug.yaml` (10 epochs, batch 64)  

---

### Train (RBF kernel)
```bash
python scripts/train_hybrid_qcnn_svm_unified.py \
  --config configs/fashion.yaml
  --kernel rbf
```

### Train (Quantum kernel)
```bash
python scripts/train_hybrid_qcnn_svm_unified.py \
  --config configs/fashion.yaml \
  --kernel qkernel
  --pl-device lightning.gpu
  --kernel-centering
  --gram-backend auto
```

### Debug mode (fast checks)
```bash
python scripts/train_hybrid_qcnn_svm_unified.py \
  --config configs/fashion_debug.yaml --kernel qkernel
```

---

## Backends & knobs

- **Devices** (`--pl-device`):  
  - `lightning.qubit` (CPU), `lightning.gpu` (GPU)
- **Matmul backends** (`--gram-backend`):  
  - `numpy`, `torch`, `auto`, `cuda_ry`
- **Parallelism**:  
  - `--workers` (processes; GPU forces 1)
  - `--tile-size` (rows per block for states + GEMM)
- **Precision**:  
  - `--dtype {float32,float64}` (compute)
  - `--return-dtype {float32,float64}` (final kernel)

---


## 🧪 Experiments Roadmap

### Phase 1 — Benchmarking & HPC scaling
- **Objective**: understand how the kernel backends behave (independent of dataset difficulty).  
- Run `tools/benchmark_pl_kernel.py` to compare:
  - **CPU (NumPy + multiprocessing)** vs **GPU (Torch streaming, CuPy)**.  
  - Vary `tile_size`, `workers`, precision (`float32` vs `float64`).  
- Collect **Mpairs/s throughput** and identify optimal settings per device. 
---

### Phase 2 — Sanity checks (debug datasets)
- **Fashion-MNIST (3 vs 8)** → quick runs (few epochs) to check end-to-end training.  
- **SVHN (3 vs 8)** → natural images, slightly harder.  
- **CIFAR-10 (3 vs 8)** → stress test with augmentations.  

👉 Configs: `fashion_debug.yaml`, `svhn_debug.yaml`, `cifar10_debug.yaml`.

---

### Phase 3 — Full binary experiments
- **Fashion-MNIST (3 vs 8)**  
  - Baseline: RBF kernel.  
  - Quantum kernel with chosen HPC backend.  
- **SVHN (4 vs 9)**  
  - Harder digit classes.  
- **CIFAR-10 (Cat vs Ship)**  
  - Full-size QCNN (8 qubits, 4 layers).  

👉 Configs: `fashion.yaml`, `svhn.yaml`, `cifar10.yaml`.

---

### Phase 4 — Reporting
- Aggregate metrics: **F1, AUC, Balanced Accuracy, speed**.  
- Export tables (CSV/Excel).  
- Generate comparative plots: accuracy vs computation time.  
- Draft final **academic-style report** with discussion.

---

## 📦 Logging & artifacts

- **Weights & Biases**: per-epoch metrics, configs, thresholds.  
- **Exports**: CSV and Excel (`openpyxl`).

---

## Benchmarks and Results

**Host: “Papavero”** — Intel Xeon (32C/64T), CUDA 12.x GPU  

| Config (N, nq)             | CPU (NumPy)             | GPU (Torch/CuPy)         |
|-----------------------------|-------------------------|--------------------------|
| **N=4096, nq=12**           | ~1.08 Mpairs/s          | ~0.48 Mpairs/s           |
| **N=8192, nq=12**           | —                       | ~0.94 Mpairs/s           |


## Troubleshooting

- **`ModuleNotFoundError: openpyxl`** → `pip install openpyxl`  
- **CuPy error: `cuda_fp16.h` missing** → install CUDA toolkit headers (`mamba install -c conda-forge cuda-toolkit=12.1 cupy`)  
- **GPU slower than CPU** → increase `--tile-size` (≥1024) and use `--gram-backend torch`  
- **CPU contention** → `export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`  

---

## License & citation

- Code released for academic use within the Parallel Programming course.  
- Please cite the repo and upstream frameworks (PennyLane, PyTorch) if you build on it.

---

✍️ **Author**: Dylan Fouepe — Master’s in AI, University of Florence  
GitHub: [@DylanUnifi](https://github.com/DylanUnifi)
