# Parallel Programming Project

This repository studies acceleration of quantum-kernel SVM pipelines for image classification with three execution paths:

- CPU base: classical CPU baseline and reference path
- GPU base: PyTorch GPU backend
- GPU custom: CuPy + custom CUDA kernel backend

Datasets used in experiments:

- Fashion-MNIST
- CIFAR-10
- SVHN

## Repository Contents

- Training scripts:
  - `train_svm_classical.py`
  - `train_svm_qkernel.py`
- Batch launchers:
  - `run_all_classical.sh`
  - `run_all_quantum.sh`
- Benchmark suite:
  - `benchmark.py`
- Result extraction:
  - `extract_results.py`
- Main report source:
  - `report_pw_pp.tex`
- Main presentation source:
  - `presentation pw-pp.tex`

## Quick Start

### 1. Build runtime containers

```bash
docker compose build trainer-classical trainer-quantum
```

### 2. Run full sweeps

```bash
bash run_all_classical.sh
bash run_all_quantum.sh
```

## Key Results

### Highlights

| Metric | Value | Context |
|---|---:|---|
| Best quantum F1 | 0.9653 | Fashion-MNIST, GPU base, size 1000 |
| Best quantum AUC | 0.9956 | Fashion-MNIST, GPU base, size 1000 |
| Classical wins | 8/9 | Paired comparisons at size 500 and 1000 |

### Classical vs Quantum (Size 1000)

| Comparison | Delta F1 | Delta AUC | Delta Time (s) |
|---|---:|---:|---:|
| Classical - GPU base | +0.1721 | +0.1600 | -274.52 |
| Classical - GPU custom | +0.2676 | +0.1990 | -475.66 |

### Collapse Indicators (Hard CIFAR-10, 6 Layers)

| Indicator | Value |
|---|---:|
| Kernel std | 0.0005 |
| Support-vector fraction | 1.0000 |
| Test F1 | 0.0000 |
| Test AUC | 0.4757 |

These values are reported in `report_pw_pp.tex`.

## Important Figures

| Figure | Description | Path |
|---|---|---|
| Fashion benchmark | Throughput and backend profile on Fashion-MNIST | `benchmark_results/fashion/benchmark.png` |
| CIFAR-10 benchmark | Throughput and backend profile on CIFAR-10 | `benchmark_results/cifar10/benchmark.png` |
| SVHN benchmark | Throughput and backend profile on SVHN | `benchmark_results/svhn/benchmark.png` |
| HPC architecture diagrams | Backend API and kernel-construction figures | `report_pw_pp.tex` |
| Presentation visuals | Condensed storyline and result figures | `presentation pw-pp.tex` |


## Author

Dylan Fouepe
