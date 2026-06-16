# Parallel Programming Exam Project

This repository contains the exam work on parallel acceleration of SVM pipelines with:

- classical baseline (`classical`)
- quantum-kernel backend with PyTorch (`torch`)
- custom CUDA backend (`cuda_states`)

Target datasets:

- Fashion-MNIST
- CIFAR-10
- SVHN

## What Is Included

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

## Quick Start (Docker Compose)

### 1. Build containers

```bash
docker compose build trainer-classical trainer-quantum
```

### 2. Run full exam sweeps

```bash
bash run_all_classical.sh
bash run_all_quantum.sh
```

### 3. Extract results (latest run only)

```bash
docker compose run --rm extract-results \
  python3 extract_results.py --latest-run --csv summary_results_latest.csv
```

### 4. Extract a specific run

```bash
docker compose run --rm extract-results \
  python3 extract_results.py --run 20260615_211954 --csv summary_results_run_20260615_211954.csv
```

## Main Files Used for the Report

- Full historical extraction:
  - `summary_results_v2.csv`
- Latest quantum run:
  - `summary_results_latest_alias.csv`
- Classical run used for paired comparison:
  - `summary_results_classical_20260615_182830.csv`
- Final paired table (dataset/difficulty):
  - `summary_comparison_by_dataset_difficulty.csv`
- Report source:
  - `parallel_qkernel_report_final.tex`

## Benchmark Figures

- Global: `benchmark_results/benchmark.png`
- Fashion: `benchmark_results/fashion/benchmark.png`
- CIFAR-10: `benchmark_results/cifar10/benchmark.png`
- SVHN: `benchmark_results/svhn/benchmark.png`

## Notes for This Exam Repository

- `.gitignore` is configured to avoid committing generated logs, local caches, benchmark outputs, summary CSVs, and LaTeX build artifacts.
- Re-run experiments locally when needed; generated artifacts are intentionally ignored to keep the repository clean.

## Author

Dylan Fouepe
