# Parallel Programming Project

[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20PennyLane%20%7C%20CuPy%20%7C%20CUDA-green?logo=pytorch)](#)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%206000%20(96GB)%20%7C%20AMD%20EPYC%2074F3-red)](#-hardware-specs)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-logging-orange?logo=weightsandbiases)](#-logging--artifacts)

SVM **quantum and classical training** for binary image classification with a focus on **parallel** and **heterogeneous** computation of **quantum kernel matrices**.  
Project developed in the *Parallel Programming* course (University of Florence).

> **Core Innovation:** We implemented a custom **High-Performance Backend** (`cuda_states`) using Raw CUDA Kernels via CuPy and Zero-Copy memory transfer (DLPack) to bypass standard library overheads.

---

## Table of Contents
- [Overview](#overview)
- [Repo layout](#repo-layout)
- [High-Performance Architecture](#-high-performance-architecture)
- [Hardware Specs](#-hardware-specs)
- [Setup](#setup)
- [Backends & knobs](#backends--knobs)
- [Training & Usage](#-training--usage)
- [Benchmarking](#benchmarking)
- [Logging & artifacts](#-logging--artifacts)
- [License & citation](#license--citation)

---

## Overview

We provide SVM binary classification of classical images dataset (CIFAR-10, Fashion-MNIST, SVHN) using Quantum Kernels computed on GPU:

- `scripts/train_svm_qkernel.py` — **Main script** for SVM Quantum kernels training (CV, Optuna, Cache).
- `scripts/pipeline_backends.py` — **The Engine**: Unified API to compute $K$ with:
  - **`cuda_states`**: *Flagship backend*. Custom C++ CUDA kernels compiled via NVRTC + CuPy. Supports massive tiling.
  - **`torch`**: Streaming GPU implementation using native PyTorch operations.
  - **`numpy`**: Fallback CPU implementation with multiprocessing.
- `tools/benchmark_pl_kernel.py` — Tool to measure throughput (Mpairs/s) and VRAM usage.

---

## Repo layout

```text
train_svm_qkernel.py               # Main Entry Point
scripts/
 └─ pipeline_backends.py           # Unified kernel API (The "Engine")
tools/
 ├─ benchmark_pl_kernel.py         # Throughput benchmark
 └─ check_nan.py                   # Numerical stability check
configs/
 ├─ fashion_med.yaml                 
 ├─ fashion_easy.yaml
 └─ fashion_hard.yaml
models/
 └─ svm_extension.py               # Custom SVM wrapper (Save/Load/Thresholds)
kernel_cache/                      # Stores computed .npy matrices

```

---

## 🚀 High-Performance Architecture

This project implements a custom **GPU-accelerated pipeline** (`cuda_states`) designed for massive quantum kernel computations on NVIDIA GPUs.

### Key Features

1. **Zero-Copy Memory Management:** Uses `DLPack` to transfer state vectors from PennyLane/PyTorch to CuPy/CUDA without CPU round-trips.
2. **Custom CUDA Kernels:** Implements raw C++ CUDA kernels (`cgemm_abs2_tiled`) to fuse dot-product and magnitude-squared operations, minimizing VRAM bandwidth.
3. **Synchronization:** Explicit CUDA stream synchronization to prevent race conditions between PyTorch (State Generation) and CuPy (Kernel Calculation).
4. **Float64 Support:** Full support for double precision to ensure numerical stability in SVM solvers.

---

## 💻 Hardware Specs

Benchmarks and training were performed on a high-end HPC node:

* **GPU:** 2x **NVIDIA RTX 6000 Ada Generation** (96 GB VRAM each)
* **CPU:** Dual **AMD EPYC 74F3** 24-Core Processor (96 threads total)
* **RAM:** 512 GB DDR4
* **CUDA:** Version 13.0

---

## Setup

### Environment

* **OS**: Linux (tested on Ubuntu 22.04/24.04)
* **CUDA**: 12.x or 13.x

```bash
# Clone
git clone [https://github.com/DylanUnifi/ProjectWorkParallelProgramming.git](https://github.com/DylanUnifi/ProjectWorkParallelProgramming.git)
cd ProjectWorkParallelProgramming

# Install dependencies
pip install -r requirements.txt

```

### Docker (Recommended)

We provide a Docker image optimized for CUDA 13.0 and Blackwell/Ada architectures.

```bash
# Build GPU image
docker build -t parallel-programming:gpu -f Dockerfile.gpu25 .

# Run container (mounting current dir)
docker run --rm -it --gpus all --shm-size=16g -v $(pwd):/app parallel-programming:gpu

```

---

## Backends & knobs

The script `train_svm_qkernel.py` exposes several knobs to tune performance:

* **`--gram-backend`**:
* `cuda_states`: Custom CUDA kernels. Requires CuPy.
* `torch`: Uses PyTorch streams.
* `numpy`: CPU only.


* **`--tile-size`**: Number of rows computed at once.
* **`--dtype`**: `float32` (speed) or `float64` (precision). **Float64 is recommended** for stability.
* **`--cache-kernels`**: Saves computed matrices to disk to skip re-computation during hyperparameter tuning.

---

## 🔥 Training & Usage

### 1. Ultra-High Performance (Recommended)

To unleash the full performance on high-end GPUs, use `cuda_states` with huge tiles.

```bash
python train_svm_qkernel.py \
  --config configs/fashion_easy.yaml \
  --gram-backend cuda_states \
  --dtype float64 \
  --tile-size -1 \
  --cache-kernels \
  --pca-components 16 \
  --embed-mode ryrz \
  --kernel-centering \
  --normalize-kernel \
  --angle-scale 0.1

```

### 2. Multi-GPU Training

If your system has multiple GPUs, you can train multiple configurations simultaneously:

```bash
# Terminal 1: GPU 0 -> Fashion-MNIST EASY
CUDA_VISIBLE_DEVICES=0 python scripts/train_svm_qkernel.py --config configs/fashion_easy.yaml ...

# Terminal 2: GPU 1 -> Fashion-MNIST HARD
CUDA_VISIBLE_DEVICES=1 python scripts/train_svm_qkernel.py --config configs/fashion_hard.yaml ...

```

---

## Benchmarking

To measure raw throughput (Million pairs/second) without training the SVM:

```bash
python tools/benchmark_pl_kernel.py \
  --samples 20000 \
  --qubits 10 \
  --gram-backend cuda_states \
  --dtype float64 \
  --state-tile 10000 \
  --device lightning.gpu

```

---

## 📦 Logging & artifacts

* **Weights & Biases**: Tracks F1-score, AUC, Accuracy, and **Confusion Matrices**.
* **Kernel Cache**: Computed Gram matrices are stored in `./kernel_cache/` (md5 hashed based on data & params).

---

## License & citation

* Code released for academic use within the Parallel Programming course.
* Please cite the repo and upstream frameworks (PennyLane, PyTorch, CuPy) if you build on it.

---

✍️ **Author**: Dylan Fouepe — Master’s in AI, University of Florence

GitHub: [@DylanUnifi](https://github.com/DylanUnifi)
