import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score
)
from tqdm import tqdm
import wandb
import optuna
import matplotlib.pyplot as plt
import yaml

from data_loader.utils import load_dataset_by_name
from models.hybrid_qcnn import HybridQCNNFeatures
from utils.logger import init_logger
from models.svm_extension import EnhancedSVM

# PennyLane-only backend
try:
    from pipeline_backends import compute_kernel_matrix
except Exception as e:
    raise ImportError("pipeline_backends.py not found. Place it in PYTHONPATH or project root.") from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Reproductibilité --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -------------------- Utils --------------------
def extract_angles(model, loader):
    model.eval()
    angles, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Extracting angles"):
            batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
            A = model.compute_angles(batch_X).detach().cpu().numpy()  # [B, n_qubits]
            angles.append(A)
            labels.append(batch_y.detach().cpu().numpy())
    return np.vstack(angles), np.concatenate(labels)

def _center_kernel_train(K: np.ndarray) -> np.ndarray:
    """Double-centre un kernel carré K (Schölkopf)."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _center_kernel_test(K_test: np.ndarray, K_train: np.ndarray) -> np.ndarray:
    """
    Centre K_test (m×n) par rapport à K_train (n×n) déjà centré.
    K_test_c = K_test - 1_m K̄cols - K̄rows^T 1_n + mean(K_train)
    """
    n = K_train.shape[0]
    mean_cols = K_train.mean(axis=0, keepdims=True)   # (1, n)
    mean_rows = K_train.mean(axis=1, keepdims=True)   # (n, 1)
    mean_all  = K_train.mean()                        # scalaire
    m = K_test.shape[0]
    return K_test - np.ones((m, 1)) @ mean_cols - mean_rows.T @ np.ones((1, n)) + mean_all

# -------------------- Main --------------------
def run_train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    EXPERIMENT_NAME = f"{dataset_name}_qcnn_fidelity_kernel_pennylane"
    SAVE_DIR = os.path.join("engine/checkpoints", "qcnn_fidelity_kernel", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(project="qml_project", name=EXPERIMENT_NAME, config={**config, **vars(args)}, group=dataset_name)

    # Dataset (binaire)
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", config.get("binary_classes", [3, 8]))
    )

    # Subset optionnel
    if args.train_subset is not None:
        indices = torch.randperm(len(train_dataset))[:args.train_subset]
        train_dataset = Subset(train_dataset, indices)

    # KFold
    kfold = KFold(n_splits=config["training"]["kfold"], shuffle=True, random_state=SEED)

    # Modèle (extract angles)
    sample_X, _ = train_dataset[0]
    input_size = sample_X.numel()
    feature_extractor = HybridQCNNFeatures(input_size=input_size).to(DEVICE)

    # Poids d’entrelacement (fixes)
    W = next(feature_extractor.quantum_layer.parameters()).detach().cpu().numpy()  # [n_layers, n_qubits]

    # Paramètres PennyLane (priorité CLI > YAML)
    pl_device  = args.pl_device  or config.get("pennylane", {}).get("device", "lightning.qubit")
    pl_workers = args.pl_workers if args.pl_workers is not None else config.get("pennylane", {}).get("workers", 0)
    tile_size  = args.tile_size  or config.get("pennylane", {}).get("tile_size", 128)
    do_center  = args.kernel_centering or bool(config.get("pennylane", {}).get("kernel_centering", False))

    # Pré-extraction angles test
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
    A_test, y_test = extract_angles(feature_extractor, test_loader)
    y_test = y_test.astype(np.int64)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        init_logger(os.path.join(SAVE_DIR, "logs"), fold)

        train_subset = Subset(train_dataset, train_idx)
        val_subset   = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=config["training"]["batch_size"])

        A_train, y_train = extract_angles(feature_extractor, train_loader)
        A_val,   y_val   = extract_angles(feature_extractor, val_loader)
        y_train = y_train.astype(np.int64)
        y_val   = y_val.astype(np.int64)

        # ---------------- Kernels (Optuna) ----------------
        print("🔹 Kernel train (PennyLane, mp)...")
        K_train = compute_kernel_matrix(
            A_train, weights=W, device_name=pl_device,
            tile_size=tile_size, symmetric=True, n_workers=pl_workers
        )
        print("🔹 Kernel val (PennyLane, mp)...")
        K_val = compute_kernel_matrix(
            A_val, Y=A_train, weights=W, device_name=pl_device,
            tile_size=tile_size, symmetric=False, n_workers=pl_workers
        )

        if do_center:
            K_train = _center_kernel_train(K_train)
            K_val   = _center_kernel_test(K_val, K_train)

        # ---------------- Optuna (binaire) ----------------
        def objective(trial):
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            svm = EnhancedSVM(C=C, kernel='precomputed', probability=True)
            svm.fit(K_train, y_train)
            y_pred = svm.predict(K_val)
            # si labels binaires {0,1}, utilise average="binary"
            avg = "binary" if np.unique(y_val).size == 2 else "weighted"
            return 1.0 - f1_score(y_val, y_pred, average=avg)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.get("svm", {}).get("n_trials", 20))
        best_C = study.best_params["C"]

        # ---------------- Final: trainval/test ----------------
        A_trainval = np.vstack([A_train, A_val])
        y_trainval = np.concatenate([y_train, y_val]).astype(np.int64)

        print("🔹 Kernel trainval (PennyLane, mp)...")
        K_trainval = compute_kernel_matrix(
            A_trainval, weights=W, device_name=pl_device,
            tile_size=tile_size, symmetric=True, n_workers=pl_workers
        )
        print("🔹 Kernel test (PennyLane, mp)...")
        K_test = compute_kernel_matrix(
            A_test, Y=A_trainval, weights=W, device_name=pl_device,
            tile_size=tile_size, symmetric=False, n_workers=pl_workers
        )

        if do_center:
            K_trainval = _center_kernel_train(K_trainval)
            K_test     = _center_kernel_test(K_test, K_trainval)
            try:
                wandb.config.update({"kernel_centering": True}, allow_val_change=True)
            except Exception:
                pass

        # Heatmap
        plt.figure(figsize=(7, 5))
        plt.imshow(K_trainval, cmap="viridis")
        plt.title(f"Fidelity Kernel (trainval) - Fold {fold}")
        plt.colorbar()
        heatmap_path = os.path.join(SAVE_DIR, f"kernel_heatmap_fold_{fold}.png")
        plt.savefig(heatmap_path, bbox_inches="tight")
        wandb.log({f"kernel_heatmap_fold_{fold}": wandb.Image(heatmap_path)})

        # SVM + métriques binaires
        clf = EnhancedSVM(C=best_C, kernel='precomputed', probability=True)
        clf.fit(K_trainval, y_trainval)
        y_pred = clf.predict(K_test)

        is_binary = (np.unique(y_test).size == 2)
        avg = "binary" if is_binary else "weighted"
        acc      = accuracy_score(y_test, y_pred)
        f1       = f1_score(y_test, y_pred, average=avg)
        precision= precision_score(y_test, y_pred, average=avg)
        recall   = recall_score(y_test, y_pred, average=avg)
        bal_acc  = balanced_accuracy_score(y_test, y_pred)

        # AUC (binaire)
        try:
            y_scores = clf.model.decision_function(K_test)
            if y_scores.ndim > 1 and y_scores.shape[1] == 2:
                y_scores = y_scores[:, 1]
            auc = roc_auc_score(y_test, y_scores) if is_binary else 0.0
        except Exception:
            auc = 0.0

        wandb.log({
            f"fold_{fold}/qsvm_accuracy": acc,
            f"fold_{fold}/qsvm_f1": f1,
            f"fold_{fold}/qsvm_precision": precision,
            f"fold_{fold}/qsvm_recall": recall,
            f"fold_{fold}/qsvm_balanced_accuracy": bal_acc,
            f"fold_{fold}/qsvm_auc": auc,
        })

    wandb.finish()

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    # PennyLane-only options
    p.add_argument("--pl-device", type=str, default=None, help="PennyLane device (lightning.qubit, lightning.gpu)")
    p.add_argument("--pl-workers", type=int, default=None, help="#processes (0 => cpu_count-1, 1 recommended on GPU)")
    p.add_argument("--tile-size", type=int, default=None, help="Tile size for states/Gram (e.g., 128 CPU, 256 GPU)")
    p.add_argument("--kernel-centering", action="store_true", help="Apply kernel centering before SVM")
    p.add_argument("--train-subset", type=int, default=None, help="Subsample train for faster runs")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_train(args)
