# Quantum Machine Learning – Parallel Programming Project

This project extends a Quantum Machine Learning (QML) pipeline with **hybrid parallelism** techniques using `batching` and `multiprocessing`. It has been developed as part of the **Parallel Programming** course, and reuses the models from a prior QML project.

## 🧠 Project Structure

We explore multiple hybrid models combining classical deep learning with quantum circuits:

- `train_hybrid_qcnn.py`: Hybrid Quantum CNN trained classically (sequential).
- `train_hybrid_qcnn_svm.py`: Hybrid QCNN with feature extraction + classical SVM.
- `train_quantum_mlp.py`: Quantum residual MLP with classical preprocessing.
- `train_hybrid_qcnn_quantumkernel.py`: Hybrid QCNN + **Quantum Kernel SVM**, where the kernel matrix is computed with a quantum embedding.

The **quantum kernel** computation is the most expensive operation. We introduce **parallel computation** of the kernel matrix using Python's `multiprocessing` module.

---

## ⚙️ Parallelization Techniques

- **Batching**: Used in all models via DataLoader and circuit batching (when supported).
- **Multiprocessing**:
  - Parallel kernel matrix computation (`train_hybrid_qcnn_quantumkernel.py`)
  - Designed to reduce \( O(N^2) \) bottleneck from pairwise quantum kernel estimation.
  - Easily configurable from YAML or CLI (e.g. `--n_processes 4`).

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Hybrid QCNN + Quantum Kernel (with multiprocessing)
```bash
python train_hybrid_qcnn_quantumkernel.py --config configs/config_train_hybrid_qcnn_quantumkernel.yaml
```

### 3. Train Quantum MLP
```bash
python train_quantum_mlp.py
```

### 4. Train Hybrid QCNN (Sequential)
```bash
python train_hybrid_qcnn.py
```

---

## 📂 Project Layout

```
.
├── models/
├── engine/
│   └── checkpoints/
├── configs/
├── utils/
├── train_hybrid_qcnn.py
├── train_hybrid_qcnn_svm.py
├── train_quantum_mlp.py
└── train_hybrid_qcnn_quantumkernel.py
```

---

## 📈 Logging & Monitoring

- **TensorBoard**: Local logs stored per fold in `engine/checkpoints/...`
- **Weights & Biases (wandb)**: Automatically logs metrics and plots if enabled.

---

## 👤 Author

Dylan Fouepe  
Master's Degree in Artificial Intelligence – University of Florence  
GitHub: [github.com/DylanUnifi](https://github.com/DylanUnifi)

