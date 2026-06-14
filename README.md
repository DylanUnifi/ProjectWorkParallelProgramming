# Parallel Programming Project

[![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%20%7C%20PennyLane%20%7C%20CuPy%20%7C%20CUDA-green?logo=pytorch)](#overview)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20RTX%206000%20(96GB)%20%7C%20AMD%20EPYC%2074F3-red)](#-hardware-specs)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-logging-orange?logo=weightsandbiases)](#-logging--artifacts)

SVM **quantum and classical training** for binary image classification on CIFAR-10, Fashion-MNIST, and SVHN, with a focus on **parallel** and **heterogeneous** computation of **quantum kernel matrices**.  
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

We provide binary SVM classification on classical image datasets (CIFAR-10, Fashion-MNIST, SVHN) using quantum kernels computed on GPU:

- `train_svm_qkernel.py` — Main script for quantum-kernel SVM training (CV, Optuna, caching).
- `train_svm_classical.py` — Classical SVM baseline for direct CPU comparisons.
- `scripts/pipeline_backends.py` — Unified API to compute $K$ with:
  - **`cuda_states`**: *Flagship backend*. Custom C++ CUDA kernels compiled via NVRTC + CuPy. Supports massive tiling.
  - **`torch`**: Streaming GPU implementation using native PyTorch operations.
  - **`numpy`**: Fallback CPU implementation with multiprocessing.
- `tools/benchmark_pl_kernel.py` — Tool to measure throughput (Mpairs/s) and VRAM usage.
- `tools/profile_kernel.py` — Standalone profiler for kernel execution and memory usage.
- `benchmark.py` — End-to-end benchmark suite that writes CSV, JSON, and plot artifacts.

---

## Repo layout

```text
train_svm_qkernel.py               # Main Entry Point
train_svm_classical.py             # Classical baseline
benchmark.py                       # Full benchmark suite
run_all_cpu.sh                     # CPU batch launcher
run_all_gpu.sh                     # GPU batch launcher
scripts/
 └─ pipeline_backends.py           # Unified kernel API (The "Engine")
tools/
 ├─ benchmark_pl_kernel.py         # Throughput benchmark
 └─ profile_kernel.py              # Kernel profiler
tests/
 └─ check_nan.py                   # Numerical stability check
configs/
 ├─ fashion_med.yaml
 ├─ fashion_easy.yaml
 └─ fashion_hard.yaml
models/
 └─ svm_extension.py               # Custom SVM wrapper (Save/Load/Thresholds)
benchmark_results/                 # Default benchmark outputs
kernel_cache/                      # Runtime kernel cache

```

---

## 🚀 High-Performance Architecture

This project implements a custom **GPU-accelerated pipeline** (`cuda_states`) designed for massive quantum kernel computations on NVIDIA GPUs.

### Key Features

1. **Zero-Copy Memory Management:** Uses `DLPack` to transfer state vectors from PennyLane/PyTorch to CuPy/CUDA without CPU round-trips.
2. **Custom CUDA Kernels:** Implements output-stationary raw CUDA kernels (`cgemm_abs2_os_full`) to fuse dot-product and magnitude-squared operations while keeping numerical accumulation in double precision.
3. **Synchronization:** Explicit CUDA stream synchronization to prevent race conditions between PyTorch (State Generation) and CuPy (Kernel Calculation).
4. **Float64 Support:** Full support for double precision to ensure numerical stability in SVM solvers.

---

## 💻 Hardware Specs

Benchmarks and training were performed on a high-end HPC node:

- **GPU:** 2x **NVIDIA RTX 6000 Ada Generation** (96 GB VRAM each)
- **CPU:** Dual **AMD EPYC 74F3** 24-Core Processor (96 threads total)
- **RAM:** 512 GB DDR4
- **CUDA:** Version 13.0

---

## Setup

### Environment

- **OS**: Linux (tested on Ubuntu 22.04/24.04)
- **CUDA**: 12.x or 13.x

```bash
# Clone
git clone https://github.com/DylanUnifi/ProjectWorkParallelProgramming.git
cd ProjectWorkParallelProgramming

# Install dependencies
pip install -r requirements.txt

```

### Docker (Recommended)

We provide two Docker images: a CPU baseline and a slimmer GPU image built on Python 3.10 with CUDA 13.0 wheels.

```bash
# Build CPU image
docker build -t parallel-programming:cpu -f Dockerfile .

# Build GPU image
docker build -t parallel-programming:gpu -f Dockerfile.gpu130 .

# Run CPU container
docker run --rm -it -v $(pwd):/app parallel-programming:cpu

# Run GPU container (mounting current dir)
docker run --rm -it --gpus all --shm-size=16g -v $(pwd):/app parallel-programming:gpu

```

With docker compose, `trainer-cpu` builds the CPU image and `trainer-gpu130` / `run-all-gpu130` build the CUDA 13.0 image from `Dockerfile.gpu130`. On the server, run scripts through Docker only: `run_all_cpu.sh` and `run_all_gpu.sh` now re-execute themselves inside the right container, and direct commands should use `docker compose run`. The default GPU image keeps the optional `lightning.gpu` extra out to stay within tighter Docker storage budgets.

---

## Backends & knobs

The script `train_svm_qkernel.py` exposes several knobs to tune performance:

- **`--gram-backend`**: `cuda_states`, `torch`, or `numpy`.
- **`--state-tile`**: Number of quantum states processed per GPU batch for `cuda_states`.
- **`--tile-size`**: Tile size used by the non-`cuda_states` backends.
- **`--dtype`**: `float32` (speed) or `float64` (precision). `float64` is recommended for numerical stability.
- **`--cache-kernels`**: Save computed matrices to disk to skip re-computation during hyperparameter tuning.

---

## 🔥 Training & Usage

### Optimal Configuration (Benchmark-Validated)

```bash
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    python3 train_svm_qkernel.py \
        --config configs/cifar10_med.yaml \
        --gram-backend cuda_states \
        --pl-device lightning.gpu \
        --state-tile -1 \
        --vram-fraction 0.95 \
        --num-streams 2 \
        --precompute-all-states \
        --no-dynamic-batch \
        --no-cuda-graphs \
        --dtype float64
```

**Key findings:**

- `precompute_all_states=True` provides **74% speedup** (most critical)
- `state_tile=-1` (auto) is **78% faster** than fixed sizes
- `vram_fraction=0.95` maximizes GPU utilization
- `dynamic_batch` and `cuda_graphs` show no benefit, slight overhead

### 1. Ultra-High Performance (Recommended)

To unleash the full performance on high-end GPUs, use `cuda_states` with huge tiles.

```bash
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    python3 train_svm_qkernel.py \
        --config configs/fashion_easy.yaml \
        --gram-backend cuda_states \
        --dtype float64 \
        --state-tile -1 \
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
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    bash -lc 'CUDA_VISIBLE_DEVICES=0 python3 train_svm_qkernel.py --config configs/fashion_easy.yaml ...'

# Terminal 2: GPU 1 -> Fashion-MNIST HARD
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    bash -lc 'CUDA_VISIBLE_DEVICES=1 python3 train_svm_qkernel.py --config configs/fashion_hard.yaml ...'
```

Batch helpers are also available:

```bash
bash run_all_gpu.sh   # quantum runs
bash run_all_cpu.sh   # classical baseline runs

```

---

## Benchmarking

### Example 1: Quick Backend Comparison

Test the three backends with a smaller workload:

```bash
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    python3 benchmark.py --backend-comparison --n-samples 4000 --n-qubits 16
```

### Example 2: Full cuda_states Optimization Study

Comprehensive study of all cuda_states optimizations:

```bash
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    python3 benchmark.py \
        --cuda-states-ablation \
        --cuda-states-state-tile \
        --cuda-states-vram \
        --cuda-states-streams \
        --n-samples 8000 \
        --n-qubits 16 \
        --verbose
```

### Example 3: Focused torch Comparison

Compare `torch` directly against `cuda_states` and `numpy`:

```bash
docker compose -f docker-compose.yml run --rm trainer-gpu130 \
    python3 benchmark.py \
        --backend-comparison \
        --backends torch cuda_states numpy \
        --n-samples 8000 \
        --verbose
```

### Example 4: Memory Profiling

Run with detailed memory profiling:

```bash
python benchmark.py \
    --memory-profiling \
    --backend-comparison \
    --n-samples 4000 \
    --verbose
```

### Example 5: Scaling Analysis

Analyze how performance scales:

```bash
python benchmark.py \
    --qubit-scaling \
    --sample-scaling \
    --verbose
```

### Example 6: Production-Ready Full Suite

Run the complete benchmark suite with production settings:

```bash
python benchmark.py \
    --all \
    --n-samples 1000 \
    --n-qubits 16 \
    --warmup-runs 2 \
    --benchmark-runs 5 \
    --output-dir benchmark_results \
    --verbose
```

### Example 7: Dataset Profiles (Fashion, CIFAR10, SVHN)

Use dataset-specific benchmark presets for fair backend comparison across your three datasets.

```bash
# Fashion-MNIST profile
docker compose run --rm trainer-gpu130 python3 benchmark.py \
    --all \
    --parallel-gpus 5 \
    --dataset-profile fashion \
    --warmup-runs 2 \
    --benchmark-runs 2 \
    --output-dir benchmark_results/fashion

# CIFAR10 profile
docker compose run --rm trainer-gpu130 python3 benchmark.py \
    --all \
    --parallel-gpus 5 \
    --dataset-profile cifar10 \
    --warmup-runs 2 \
    --benchmark-runs 2 \
    --output-dir benchmark_results/cifar10

# SVHN profile
docker compose run --rm trainer-gpu130 python3 benchmark.py \
    --all \
    --parallel-gpus 5 \
    --dataset-profile svhn \
    --warmup-runs 2 \
    --benchmark-runs 2 \
    --output-dir benchmark_results/svhn
```

Notes:

- `--dataset-profile` configures `QUBITS_RANGE`, `SAMPLE_SIZES`, and default benchmark scales.
- `--warmup-runs` controls GPU warmup passes before timing.
- `--benchmark-runs` controls timed repetitions and averages per test point.

## Output Files

The benchmark generates comprehensive output in the selected output directory. By default this is `benchmark_results/`.

### Main Artifacts

- `benchmark.csv` - Combined row-wise results from every executed benchmark test.
- `benchmark_summary.json` - Aggregated summary statistics.
- `benchmark.png` - Comprehensive visualization with 6 subplots:
  - Throughput vs Qubits
  - Time vs Qubits (Log Scale)
  - GPU Memory Usage
  - Sample Scaling (O(N²) verification)
  - Tile Size Optimization
  - GPU Speedup vs CPU

## Interpreting Results

### Backend Comparison

Look for:

- **Best throughput** (Mpairs/s): Higher is better
- **GPU memory usage**: Ensure it fits in available VRAM
- **Speedup vs CPU**: How much faster GPU backends are

### Optimization Ablation

- **Baseline**: Performance with all optimizations OFF
- **Individual optimizations**: Contribution of each optimization
- **Full optimizations**: Performance with all optimizations ON
- **Speedup factor**: Full optimizations vs Baseline

### Tile Size Optimization

- **Optimal tile size**: Balance between memory usage and performance
- **Performance curve**: How throughput varies with tile size
- **Memory impact**: Larger tiles may use more memory

### VRAM Fraction Impact

- **Sweet spot**: Best vram_fraction for your workload
- **Stability**: Higher fractions may be less stable
- **Throughput**: Impact on performance

### Stream Pool Impact

- **Optimal stream count**: More isn't always better
- **Diminishing returns**: Look for performance plateau
- **Overhead**: Too many streams can add overhead

### Scaling Analysis

- **Qubit scaling**: Should show exponential growth (2^n)
- **Sample scaling**: Should show O(N²) behavior
- **Backend comparison**: How different backends scale

## Performance Tips

1. **Start small**: Test with smaller workloads first (e.g., --n-samples 2000)
2. **Use warmup**: Always use at least 1 warmup run to prime the GPU
3. **Multiple runs**: Use 3-5 benchmark runs for accurate timing
4. **Monitor memory**: Watch GPU memory usage, especially for large qubit counts
5. **Optimize incrementally**: Test one optimization at a time before combining

## Troubleshooting

### Out of Memory Errors

- Reduce `--n-samples` or `--n-qubits`
- Lower `vram_fraction` (e.g., 0.7 instead of 0.95)
- Use smaller state_tile values
- Close other GPU applications

### Slow Performance

- Ensure GPU isn't being used by other processes
- Check that CUDA is properly installed
- Verify that optimizations are enabled
- Try different tile sizes

### Import Errors

Ensure all dependencies are installed:

```bash
pip install numpy pandas matplotlib seaborn torch cupy
```

## Advanced Usage

### Custom Configuration

You can modify `BACKEND_CONFIGS` in the script to test custom configurations:

```python
BACKEND_CONFIGS = {
    "cuda_states": {
        "state_tile": 4096,  # Custom value
        "vram_fraction": 0.9,  # Custom value
        "num_streams": 8,  # Custom value
        # ... other parameters
    },
}
```

### Batch Testing

Create a shell script to run multiple configurations:

```bash
#!/bin/bash
for samples in 2000 4000 8000 16000; do
    for qubits in 8 10 12 14; do
        python benchmark.py \
            --backend-comparison \
            --n-samples $samples \
            --n-qubits $qubits \
            --output-dir results_${samples}_${qubits}
    done
done
```

## Hardware Requirements

- **Minimum**: NVIDIA GPU with 8GB VRAM, CUDA 11.0+
- **Recommended**: NVIDIA GPU with 16GB+ VRAM, CUDA 12.0+
- **Optimal**: NVIDIA RTX 6000 Ada (96GB VRAM), CUDA 13.0

---

## 📦 Logging & artifacts

- **Weights & Biases**: Tracks F1-score, AUC, Accuracy, and **Confusion Matrices**.
- **Kernel Cache**: Computed Gram matrices are stored in `./kernel_cache/` (md5 hashed based on data and parameters).
- **Batch Logs**: `run_all_cpu.sh` writes `log_*_classical_*.txt` and `run_all_gpu.sh` writes `log_*_torch_*.txt`.

---

## License & citation

- Code released for academic use within the Parallel Programming course.
- Please cite the repo and upstream frameworks (PennyLane, PyTorch, CuPy) if you build on it.

---

✍️ **Author**: Dylan Fouepe — Master’s in AI, University of Florence

GitHub: [@DylanUnifi](https://github.com/DylanUnifi)
