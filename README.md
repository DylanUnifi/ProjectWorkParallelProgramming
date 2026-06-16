# Acceleration of Quantum Kernel Methods for Image Classification

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-13.1-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-GPU-ee4c2c.svg)
![CuPy](https://img.shields.io/badge/CuPy-Custom_Kernels-blue)

This repository contains the codebase and reports for a parallel programming 
project studying the acceleration of quantum-kernel Support Vector Machine (SVM) pipelines.

To overcome the $\mathcal{O}(N^2 \cdot 2^Q)$ Gram matrix bottleneck in 
high-dimensional Hilbert spaces, we evaluate three execution paths:

- **CPU base:** Classical CPU baseline and reference path (NumPy).
- **GPU base:** Native PyTorch GPU backend with pinned memory and streams.
- **GPU custom:** Advanced backend utilizing CuPy, DLPack (zero-copy), and 
a custom output-stationary CUDA kernel in FP64.

Datasets evaluated: **Fashion-MNIST**, **CIFAR-10**, and **SVHN**.

---

## Hardware & Environment Setup

All benchmarks and training pipelines reported in this repository were 
executed on the following HPC configuration:

- **GPU:** NVIDIA RTX 6000 Ada Generation (48 GB VRAM)
- **CPU:** AMD EPYC 7343 16-Core Processor (64 threads)
- **Environment:** CUDA 13.1

---

## Quick Start

### Prerequisites

- Docker and Docker Compose
- **NVIDIA Container Toolkit** (required for GPU passthrough to the containers)

### 1. Build runtime containers

```bash
docker compose build trainer-classical trainer-quantum
```

### 2. Run full benchmark sweeps

```bash
bash run_all_classical.sh
bash run_all_quantum.sh
```

Alternatively, run individual evaluations locally using Python directly:

```bash
python train_svm_qkernel.py --config configs/fashion_easy.yaml --backend 
gpu_custom
```

---

## Key Results

### Global Benchmark Profile

*Throughput profile across backends. Notice how the custom CUDA kernel 
dominates at smaller sample sizes, while the PyTorch backend excels at 
larger batches due to dense matrix optimizations.*

| Metric | Value | Context |
|---|---:|---|
| Best quantum F1 | 0.9653 | Fashion-MNIST, GPU base, size 1000 |
| Best quantum AUC | 0.9956 | Fashion-MNIST, GPU base, size 1000 |
| Classical wins | 8/9 | Paired comparisons at size 500 and 1000 |

### Predictive Highlights (Size 1000)

| Comparison | Delta F1 | Delta AUC | Delta Time (s) |
|---|---:|---:|---:|
| Classical - GPU base | +0.1721 | +0.1600 | -274.52 |
| Classical - GPU custom | +0.2676 | +0.1990 | -475.66 |

### Collapse Indicators (Hard CIFAR-10, 6 Layers)

Deeper quantum circuits on complex datasets lead to state concentration 
(kernel collapse).

| Indicator | Value |
|---|---:|
| Kernel std | 0.0005 |
| Support-vector fraction | 1.0000 |
| Test F1 | 0.0000 |
| Test AUC | 0.4757 |

---

## Repository Structure

```
.
├── scripts/                    # HPC pipeline scripts and GPU memory optimization utilities
├── configs/                    # YAML configuration files per dataset and backend
├── benchmark_results/          # Generated throughput plots
├── train_svm_classical.py      # Classical SVM training loop
├── train_svm_qkernel.py        # Quantum kernel SVM training loop
├── benchmark.py                # Benchmark suite for timing isolated Gram matrix builds
├── report_pw_pp.pdf            # Full academic report detailing kernel logic and analysis
└── presentation pw-pp.pdf      # Slide deck summarizing the project
```

---

## Important Figures

| Figure | Description | Path |
|---|---|---|
| Global benchmark | Overall throughput profile across backends | `benchmark_results/benchmark.png` |
| Fashion benchmark | Throughput and backend profile on Fashion-MNIST | `benchmark_results/fashion/benchmark.png` |
| CIFAR-10 benchmark | Throughput and backend profile on CIFAR-10 | `benchmark_results/cifar10/benchmark.png` |
| SVHN benchmark | Throughput and backend profile on SVHN | `benchmark_results/svhn/benchmark.png` |

Full architecture diagrams and result figures are available in `report_pw_pp.pdf` and `presentation pw-pp.pdf`.


## Author

Dylan Fouepe — University of Florence  
*Project Work, Parallel Programming — 2025*
