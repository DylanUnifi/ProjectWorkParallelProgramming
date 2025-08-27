# --- project root on sys.path (CLion/PyCharm safe) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import optuna
import pennylane as qml

from data_loader.utils import load_dataset_by_name
from models.hybrid_qcnn import HybridQCNNFeatures
from utils.logger import init_logger, write_log
from models.svm_extension import EnhancedSVM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Extracting features"):
            batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
            feats = model(batch_X).cpu().numpy()
            features.append(feats)
            labels.append(batch_y.cpu().numpy())
    return np.vstack(features), np.concatenate(labels)

def run_train_hybrid_qcnn_quantumkernel(config, config_path=None):
    dataset_name = config["dataset"]["name"]
    EXPERIMENT_NAME = f"{dataset_name}_hybrid_qcnn_quantumkernel"
    SAVE_DIR = os.path.join("engine/checkpoints", "hybrid_qcnn_quantumkernel", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(
        project="qml_project",
        name=EXPERIMENT_NAME,
        config=config,
        group=dataset_name
    )

    # Adds the path to the config file to the wandb logs
    if config_path:
        wandb.config.update({"config_path": os.path.abspath(config_path)})

    # Loading datasets
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        binary_classes=config.get("binary_classes", [3, 8])
    )

    # Random selection of 500 examples for the train (modify if you want to use more/all of them)
    indices = torch.randperm(len(train_dataset))[:500]
    train_dataset = Subset(train_dataset, indices)
    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")

    # KFold cross-validation
    kfold = KFold(n_splits=config["training"]["kfold"], shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Starting Hybrid QCNN + Quantum Kernel SVM training...")

        sample_x, _ = train_dataset[0]
        input_size = sample_x.numel()

        feature_extractor = HybridQCNNFeatures(input_size=input_size).to(DEVICE)
        print(f"[Fold {fold}] Extraction des features train/val/test...")

        trainval_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"])
        X_trainval, y_trainval = extract_features(feature_extractor, trainval_loader)

        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
        X_test, y_test = extract_features(feature_extractor, test_loader)

        scaler = StandardScaler()
        X_trainval_scaled = scaler.fit_transform(X_trainval)
        X_test_scaled = scaler.transform(X_test)

        # Separate part of the trainval as internal value for Optuna
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval_scaled, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
        )

        n_qubits = X_train.shape[1]
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def quantum_feature_map(x):
            for i in range(len(x)):
                qml.RY(x[i], wires=i)
            return qml.state()

        def quantum_kernel(x1, x2):
            psi1 = quantum_feature_map(x1)
            psi2 = quantum_feature_map(x2)
            return np.abs(np.dot(np.conj(psi1), psi2))**2

        print("🔹 Building the kernel train matrix for Optuna...")
        K_train = np.zeros((len(X_train), len(X_train)))
        for i in tqdm(range(len(X_train)), desc="Kernel train"):
            for j in range(i, len(X_train)):
                k = quantum_kernel(X_train[i], X_train[j])
                K_train[i, j] = K_train[j, i] = k

        print("🔹 Building the kernel matrix val...")
        K_val = np.zeros((len(X_val), len(X_train)))
        for i in tqdm(range(len(X_val)), desc="Kernel val"):
            for j in range(len(X_train)):
                K_val[i, j] = quantum_kernel(X_val[i], X_train[j])

        print("🔹 Optimize C with Optuna...")

        def objective(trial):
            C = trial.suggest_loguniform("C", 1e-3, 1e3)
            svm = EnhancedSVM(C=C, kernel='precomputed', probability=True)
            svm.fit(K_train, y_train)
            y_pred = svm.predict(K_val)
            return 1.0 - f1_score(y_val, y_pred, average="weighted")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        best_C = study.best_params["C"]
        print(f"Best C found by Optuna: {best_C}")

        # Full kernel calculation for trainval and test
        print("🔹 Build the final trainval kernel matrix...")
        K_trainval = np.zeros((len(X_trainval_scaled), len(X_trainval_scaled)))
        for i in tqdm(range(len(X_trainval_scaled)), desc="Kernel trainval"):
            for j in range(i, len(X_trainval_scaled)):
                k = quantum_kernel(X_trainval_scaled[i], X_trainval_scaled[j])
                K_trainval[i, j] = K_trainval[j, i] = k

        print("🔹 Build the final test kernel matrix...")
        K_test = np.zeros((len(X_test_scaled), len(X_trainval_scaled)))
        for i in tqdm(range(len(X_test_scaled)), desc="Kernel test"):
            for j in range(len(X_trainval_scaled)):
                K_test[i, j] = quantum_kernel(X_test_scaled[i], X_trainval_scaled[j])

        # Heatmap
        plt.figure(figsize=(8,6))
        plt.imshow(K_trainval, cmap="viridis")
        plt.title("Quantum Kernel Matrix - Trainval")
        plt.colorbar()
        heatmap_path = os.path.join(SAVE_DIR, f"kernel_heatmap_fold_{fold}.png")
        plt.savefig(heatmap_path)
        wandb.log({"kernel_heatmap": wandb.Image(heatmap_path)})

        print("🔹 Train the SVM with best C...")
        best_svm = EnhancedSVM(C=best_C, kernel='precomputed', probability=True)
        best_svm.fit(K_trainval, y_trainval)

        print("🔹 Predict on the test set...")
        y_pred = best_svm.predict(K_test)
        acc, f1, precision, recall = (
            accuracy_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred)
        )
        try:
            y_probs = best_svm.model.decision_function(K_test)
            auc = roc_auc_score(y_test, y_probs)
        except Exception:
            auc = 0.0
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        print(f"\n[Test Set] Test Acc: {acc:.4f} | F1: {f1:.4f} | BalAcc: {bal_acc:.4f} | AUC: {auc:.4f}")
        wandb.log({
            "test/qsvm_f1": f1,
            "test/qsvm_accuracy": acc,
            "test/qsvm_precision": precision,
            "test/qsvm_recall": recall,
            "test/qsvm_balanced_accuracy": bal_acc,
            "test/qsvm_auc": auc,
        })

    wandb.finish()
    print("Hybrid QCNN + Quantum Kernel SVM training complete.")

import argparse
import os
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start training the Hybrid QCNN + Quantum Kernel SVM model.")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load the YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_train_hybrid_qcnn_quantumkernel(config, config_path=args.config)

