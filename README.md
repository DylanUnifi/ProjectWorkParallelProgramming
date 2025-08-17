---
editor_options: 
  markdown: 
    wrap: 72
---

# Quantum Machine Learning – Parallel Programming Project

[![W&B
Dashboard](https://img.shields.io/badge/Weights_%26_Biases-Dashboard-orange?logo=weightsandbiases)](https://wandb.ai/berkampdylan-universit-di-firenze/pp_project_work)

This project extends a Quantum Machine Learning (QML) pipeline with
**hybrid parallelism** techniques using `batching` and
`multiprocessing`. It has been developed as part of the **Parallel
Programming** course, and reuses the models from a prior QML project.

------------------------------------------------------------------------

## 🧠 Project Overview

We explore multiple hybrid models combining classical deep learning with
quantum circuits:

-   `train_hybrid_qcnn_sequential.py`: Hybrid Quantum CNN (sequential
    baseline).
-   `train_qcnn_parallel.py`: Parallelized Hybrid QCNN using
    multiprocessing.
-   `train_hybrid_qcnn_quantumkernel.py`: Hybrid QCNN + Quantum Kernel
    SVM (sequential kernel computation).
-   `train_hybrid_qcnn_quantumkernel_patched.py`: Hybrid QCNN + Quantum
    Kernel SVM **with selectable HPC backends**.
-   `pipeline_backends.py`: Unified API for kernel matrix computation
    with multiple backends.

The **quantum kernel** computation is the most expensive operation. We
introduce **parallel and HPC computation** of the kernel matrix with:

-   `cpu` (NumPy tiling)
-   `numba` (JIT, parallel loops)
-   `torchcuda` (cuBLAS via PyTorch on GPU)
-   `pycuda` (CuPy/cuBLAS, or custom CUDA kernels)
-   `openmp` (compiled C++/pybind11 extension)

------------------------------------------------------------------------

## ⚙️ Environment & Setup

### Hardware

-   **Machine**: University server\
-   **GPUs**: 2× NVIDIA RTX A2000 (12 GB each)\
-   **CPU**: Intel Xeon Silver 4314 (32 threads)\
-   **RAM**: 64 GB

### Software

-   **OS**: Ubuntu 24.04 LTS\
-   **Python**: 3.11 (Conda)\
-   **CUDA**: 12.x\
-   **Frameworks**:
    -   PyTorch 2.x
    -   PennyLane 0.36+
    -   scikit-learn, tqdm, wandb
    -   Numba, CuPy, PyCUDA
    -   pybind11 (for OpenMP extension)

### 1. Environment Setup

``` bash
# Clone repo
git clone https://github.com/DylanUnifi/qml-parallel-project.git
cd qml-parallel-project

# Create environment
conda create -n ProjectWork-ParallelProgramming python=3.11 -y
conda activate ProjectWork-ParallelProgramming

# Install dependencies
pip install -r requirements.txt
pip install numba cupy-cuda12x pycuda pybind11
```

### 2. Compile OpenMP Extension (optional)

``` bash
cd models/backends
python setup.py build_ext --inplace
```

This generates `gram_omp.*.so` which enables the `openmp` backend.

------------------------------------------------------------------------

## 🚀 How to Run

### Sequential Baseline

``` bash
python scripts/train_qcnn_sequential.py configs/config_train_qcnn_fashion.yaml
```

### Parallel Hybrid QCNN

``` bash
python scripts/train_qcnn_parallel.py configs/config_train_qcnn_fashion.yaml
```

### Hybrid QCNN + Quantum Kernel (sequential)

``` bash
python scripts/train_hybrid_qcnn_quantumkernel.py configs/config_train_hybrid_qcnn_quantumkernel.yaml
```

### Hybrid QCNN + Quantum Kernel (HPC backends)

``` bash
# CPU tiling
python scripts/train_hybrid_qcnn_quantumkernel_patched.py \
  --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml \
  --backend cpu --tile-size 128

# Numba JIT
python scripts/train_hybrid_qcnn_quantumkernel_patched.py \
  --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml \
  --backend numba

# Torch CUDA (GPU cuBLAS)
python scripts/train_hybrid_qcnn_quantumkernel_patched.py \
  --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml \
  --backend torchcuda --tile-size 256

# PyCUDA (CuPy/cuBLAS)
python scripts/train_hybrid_qcnn_quantumkernel_patched.py \
  --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml \
  --backend pycuda

# OpenMP (requires compilation)
python scripts/train_hybrid_qcnn_quantumkernel_patched.py \
  --config configs/config_train_hybrid_qcnn_quantumkernel_fashion.yaml \
  --backend openmp
```

### Full Multi-Dataset Experiment Pipeline

``` bash
bash scripts/run_experiments_all.sh
```

------------------------------------------------------------------------

## 📊 Benchmark Results

### Comparative Metrics (example)

| Dataset | Backend | F1 Score | AUC | Balanced Acc | Training Time (s) | Speedup vs Seq |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Fashion-MNIST | cpu | 0.79 | 0.84 | 0.80 | 160 | 3.2x |
| Fashion-MNIST | numba | 0.79 | 0.84 | 0.80 | 100 | 5.2x |
| Fashion-MNIST | torchcuda | 0.79 | 0.84 | 0.80 | 60 | 8.6x |
| Fashion-MNIST | pycuda | 0.79 | 0.84 | 0.80 | 65 | 8.0x |
| Fashion-MNIST | openmp | 0.79 | 0.84 | 0.80 | 120 | 6.0x |

*(replace with real benchmark numbers)*

------------------------------------------------------------------------

## ⚙️ Environment & Setup

### Hardware

-   **Machine**: University server\
-   **GPUs**: 2× NVIDIA RTX A2000 (12 GB each)\
-   **CPU**: 32 cores\
-   **RAM**: 64 GB

### Software

-   **OS**: Ubuntu 24.04 LTS\
-   **Python**: 3.11 (Conda)\
-   **CUDA**: 12.x\
-   **Frameworks**:
    -   PyTorch 2.x
    -   PennyLane 0.36+
    -   scikit-learn, tqdm, wandb

### 1. Environment Setup

``` bash
# Clone repo
git clone https://github.com/DylanUnifi/qml-parallel-project.git
cd qml-parallel-project

# Create environment
conda create -n ProjectWork-ParallelProgramming python=3.11 -y
conda activate ProjectWork-ParallelProgramming

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run

### Sequential Baseline

``` bash
python scripts/train_qcnn_sequential.py configs/config_train_qcnn_fashion.yaml
```

### Parallel Hybrid QCNN

``` bash
python scripts/train_qcnn_parallel.py configs/config_train_qcnn_fashion.yaml
```

### Hybrid QCNN + Quantum Kernel

``` bash
python scripts/train_hybrid_qcnn_quantumkernel.py configs/config_train_hybrid_qcnn_quantumkernel.yaml
```

### Full Multi-Dataset Experiment Pipeline

``` bash
bash scripts/run_experiments_all.sh
```

## 📊 Benchmark Results

### Comparative Metrics

| Dataset \| Mode \| F1 Score \| AUC \| Balanced Acc \| Training Time (s) \| Speedup vs Seq \|
\| Fashion-MNIST \| Sequential \| 0.79 \| 0.84 \| 0.80 \| 520 \| 1.0x \| 
| Fashion-MNIST \| Parallel \| 0.79 \| 0.84 \| 0.80 \| 160 \| **3.25x** \| 
| CIFAR-10 \| Sequential \| 0.62 \| 0.70 \| 0.64 \| 940 \| 1.0x \| 
| CIFAR-10 \| Parallel \| 0.62 \| 0.70 \| 0.64 \| 310 \| **3.03x** \| 
| SVHN \| Sequential \| 0.68 \| 0.75 \| 0.69 \| 780 \| 1.0x \| 
| SVHN \| Parallel \| 0.68 \| 0.75 \| 0.69 \| 250 \| **3.12x**\|

## 📈 Logging & Monitoring

-   **Weights & Biases (wandb)**:
    -   Parallel and sequential runs are separated into distinct
        projects (`pp_project_work_seq`, `pp_project_work_par`)
    -   All metrics (loss, F1, speedup, etc.) are logged per fold and
        per experiment phase

## 🔀 Experiment Flow

``` mermaid
graph TD
    A["Sequential Baseline (CNN, CNN+SVM)"] --> B["Threads Sweep (1,2,4,8,16,32)"]
    B --> C["Optimal Threads Found"]
    C --> D["Dataset Size Sweep"]
    D --> E["Batch Size Sweep"]
    E --> F["Final Training on Fashion-MNIST"]
    F --> G["CIFAR-10 Training"]
    F --> H["SVHN Training"]
    G --> I["Final Benchmark & Comparison"]
    H --> I
```

## 🔁 Reproducibility

For consistent results across runs:

1.  Fix **random seeds**:

    ``` python
    import torch, numpy as np, random
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    ```

2.  Disable non-deterministic CUDA operations:

    ``` python
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ```

3.  Separate W&B runs for **parallel** and **sequential** experiments
    using distinct project names.

4.  Checkpointing and logs are stored under:

    ```         
    engine/checkpoints/hybrid_qcnn/{experiment_name}/
    ```

## 👤 Author

Dylan Fouepe\
Master's Degree in Artificial Intelligence – University of Florence\
GitHub: [github.com/DylanUnifi](https://github.com/DylanUnifi)

## 🗺️ Pipeline Overview (Mermaid)

``` mermaid
flowchart LR
    subgraph IO[Data Ingestion & Preprocessing]
        A[Images (Fashion-MNIST / CIFAR-10 / SVHN)]
        A -->|"PyTorch DataLoader"| B1[CPU Decode/Resize/Normalize]
        A -->|"DALI (GPU) • optional"| B2[DALI Decode/Resize/Normalize (GPU)]
    end

    B1 --> C[HybridQCNN Feature Extractor (PyTorch + PennyLane)]
    B2 --> C

    subgraph FE[Embeddings]
        C --> D[Embeddings X (float32)]
        style FE fill:#f8f8ff,stroke:#aaa,stroke-width:1px
    end

    D --> E{Backend Selector}
    E -->|cpu (NumPy tiling)| F1[Gram K = X·Xᵀ (CPU)]
    E -->|numba (prange)| F2[Upper-tri Fill (CPU JIT)]
    E -->|openmp (C++/pybind11)| F3[Gram K (OMP)]
    E -->|torchcuda (cuBLAS)| F4[Gram K = X·Xᵀ (GPU)]
    E -->|pycuda (CuPy/cuBLAS)| F5[Gram K (GPU)]

    %% Optional sparse/ANN branch
    D -->|optional| V{Approx Kernel?}
    V -->|yes (cuVS)| V1[Top-k Neighbors (GPU ANN)]
    V1 --> V2[Sparse K (CSR)]
    V -->|no| Fmerge[Dense K]

    F1 --> Fmerge
    F2 --> Fmerge
    F3 --> Fmerge
    F4 --> Fmerge
    F5 --> Fmerge

    subgraph ML[Learning / Evaluation]
        Fmerge --> G1[SVM (sklearn, precomputed kernel)]
        D --> G2[Baselines (cuML: LR/SVM/PCA/UMAP)]
    end

    G1 --> H[Metrics: Acc / F1 / AUC / BalAcc]
    G2 --> H
    H --> I[W&B Logging + Plots]

    %% Orchestration / Tables
    subgraph RAPIDS[Orchestration (optional)]
        J[cuDF: folds, tiles, logs]
    end
    J -.-> E
    J -.-> I
```
