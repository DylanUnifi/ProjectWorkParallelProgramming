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

### Optimal Configuration (Benchmark-Validated)

Based on extensive benchmarks on RTX 6000 Ada (102GB):

```bash
python train_svm_qkernel.py \
    --config configs/cifar10.yaml \
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

### Example 1: Quick Backend Comparison

Test the three backends with a smaller workload:

```bash
python benchmark_production.py --backend-comparison --n-samples 4000 --n-qubits 8
```

### Example 2: Full cuda_states Optimization Study

Comprehensive study of all cuda_states optimizations:

```bash
python benchmark_production.py \
    --cuda-states-ablation \
    --cuda-states-state-tile \
    --cuda-states-vram \
    --cuda-states-streams \
    --n-samples 8000 \
    --n-qubits 10 \
    --verbose
```

### Example 3: torch Backend Optimization

Find optimal torch settings:

```bash
python benchmark_production.py \
    --torch-ablation \
    --torch-tiles \
    --n-samples 8000 \
    --verbose
```

### Example 4: Memory Profiling

Run with detailed memory profiling:

```bash
python benchmark_production.py \
    --memory-profiling \
    --backend-comparison \
    --n-samples 4000 \
    --verbose
```

### Example 5: Scaling Analysis

Analyze how performance scales:

```bash
python benchmark_production.py \
    --qubit-scaling \
    --sample-scaling \
    --verbose
```

### Example 6: Production-Ready Full Suite

Run the complete benchmark suite with production settings:

```bash
python benchmark_production.py \
    --all \
    --n-samples 8000 \
    --n-qubits 10 \
    --warmup-runs 2 \
    --benchmark-runs 5 \
    --output-dir production_benchmark_results \
    --verbose
```

## Output Files

The benchmark generates comprehensive output in the specified output directory:

### CSV Files

- `backend_comparison.csv` - Backend comparison results
- `cuda_states_ablation.csv` - cuda_states ablation study results
- `cuda_states_state_tile.csv` - State tile optimization results
- `cuda_states_vram_fraction.csv` - VRAM fraction impact results
- `cuda_states_streams.csv` - Stream pool impact results
- `torch_ablation.csv` - torch ablation study results
- `torch_tile_sizes.csv` - torch tile optimization results
- `memory_profiling.csv` - Memory profiling results
- `qubit_scaling.csv` - Qubit scaling results
- `sample_scaling.csv` - Sample scaling results
- `production_benchmark.csv` - Combined results from all tests
- `production_benchmark_summary.json` - Summary statistics

### Plots

- `production_benchmark.png` - Comprehensive visualization with 6 subplots:
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
        python benchmark_production.py \
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

* **Weights & Biases**: Tracks F1-score, AUC, Accuracy, and **Confusion Matrices**.
* **Kernel Cache**: Computed Gram matrices are stored in `./kernel_cache/` (md5 hashed based on data & params).

---

## License & citation

* Code released for academic use within the Parallel Programming course.
* Please cite the repo and upstream frameworks (PennyLane, PyTorch, CuPy) if you build on it.

---

✍️ **Author**: Dylan Fouepe — Master’s in AI, University of Florence

GitHub: [@DylanUnifi](https://github.com/DylanUnifi)
