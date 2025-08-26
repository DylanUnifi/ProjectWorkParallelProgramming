# scripts/train_hybrid_qcnn_quantumkernel_patched.py

# --- project root on sys.path (CLion/PyCharm safe) ---
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------

import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, precision_recall_curve
)
import wandb
import optuna
import yaml

from data_loader.utils import load_dataset_by_name
from models.hybrid_qcnn import HybridQCNNFeatures
from models.svm_extension import EnhancedSVM  # OK without threshold methods (we manage the threshold here)
from utils.logger import init_logger

# PennyLane-only backend
try:
    from pipeline_backends import compute_kernel_matrix
except Exception as e:
    raise ImportError("pipeline_backends.py not found. Place it in PYTHONPATH or project root.") from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------- Reproducibility --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -------------------- Helpers --------------------
def extract_angles(model, loader, dtype=np.float32):
    model.eval()
    angles, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
            A = model.compute_angles(batch_X).detach().cpu().numpy().astype(dtype, copy=False)
            angles.append(A)
            labels.append(batch_y.detach().cpu().numpy())
    return np.vstack(angles), np.concatenate(labels)


def _center_kernel_train(K: np.ndarray) -> np.ndarray:
    """Double-center a square kernel K (Schölkopf)."""
    K = np.asarray(K, dtype=np.float64, order="C")
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _center_kernel_test(K_test: np.ndarray, K_train_uncentered: np.ndarray) -> np.ndarray:
    """Center K_test relative to K_train not centered (Schölkopf)."""
    Kt = np.asarray(K_test, dtype=np.float64, order="C")
    Ku = np.asarray(K_train_uncentered, dtype=np.float64, order="C")
    mean_cols = Ku.mean(axis=0, keepdims=True)
    mean_rows = Ku.mean(axis=1, keepdims=True)
    mean_all = Ku.mean()
    Kt_c = Kt - np.ones((Kt.shape[0], 1)) @ mean_cols - np.ones((Kt.shape[0], 1)) @ mean_rows.T + mean_all
    return Kt_c


def dataset_labels_numpy(dataset) -> np.ndarray:
    """Retrieves all labels from a Dataset (int64)."""
    ys = np.empty(len(dataset), dtype=np.int64)
    for i in range(len(dataset)):
        _, y = dataset[i]
        ys[i] = int(y)
    return ys


def stratified_subset_dataset(dataset, n_samples, seed=42):
    ys = dataset_labels_numpy(dataset)
    if n_samples is None or n_samples >= len(ys):
        return dataset
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=seed)
    idx, _ = next(sss.split(np.zeros_like(ys), ys))
    return Subset(dataset, idx)


def find_best_thr_decision(scores: np.ndarray, y_true: np.ndarray, min_recall: float = 0.10):
    """
    Find the best threshold on `decision_function` that maximizes F1
    while imposing a minimum recall.
    """
    p, r, thr = precision_recall_curve(y_true, scores)
    # `thr` a len = len(p) - 1 ; on étend pour itérer proprement
    thr_ext = np.r_[thr, thr[-1] if thr.size else 0.0]
    best_f1, best_thr = -1.0, 0.0
    for pi, ri, ti in zip(p, r, thr_ext):
        if ri >= min_recall:
            f1 = 0.0 if (pi + ri) == 0 else 2 * pi * ri / (pi + ri)
            if f1 > best_f1:
                best_f1, best_thr = f1, ti
    return float(best_thr)


def binarize_by_threshold(scores: np.ndarray, thr: float) -> np.ndarray:
    return (scores >= thr).astype(np.int64)


# -------------------- Main --------------------
def run_train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    EXPERIMENT_NAME = f"{dataset_name}_qcnn_fidelity_kernel_pennylane"
    SAVE_DIR = os.path.join("engine/checkpoints", "qcnn_fidelity_kernel", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(project="qml_project", name=EXPERIMENT_NAME,
               config={**config, **vars(args)}, group=dataset_name)

    # dtypes
    feat_dtype = np.float32 if args.dtype == "float32" else np.float64
    ret_dtype = np.float32 if args.return_dtype == "float32" else np.float64

    # Binary dataset
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", config.get("binary_classes", [3, 8])),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )

    # Optional stratified subset
    if args.train_subset is not None:
        train_dataset = stratified_subset_dataset(train_dataset, args.train_subset, seed=SEED)

    # Prepare StratifiedKFold (binary)
    y_all = dataset_labels_numpy(train_dataset)
    skf = StratifiedKFold(n_splits=config["training"]["kfold"], shuffle=True, random_state=SEED)

    # Angle extraction model
    sample_X, _ = train_dataset[0]
    input_size = sample_X.numel()
    feature_extractor = HybridQCNNFeatures(input_size=input_size).to(DEVICE)

    # Interlacing weights (fixed)
    W = next(feature_extractor.quantum_layer.parameters()).detach().cpu().numpy()

    # PennyLane settings
    pl_device = args.pl_device or config.get("pennylane", {}).get("device", "lightning.qubit")
    pl_workers = args.pl_workers if args.pl_workers is not None else config.get("pennylane", {}).get("workers", 0)
    tile_size = args.tile_size or config.get("pennylane", {}).get("tile_size", 128)
    do_center = bool(args.kernel_centering or config.get("pennylane", {}).get("kernel_centering", False))

    # Pre-extraction test angles
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
    A_test, y_test = extract_angles(feature_extractor, test_loader, dtype=feat_dtype)
    y_test = y_test.astype(np.int64)

    # thresholds per fold (to calculate an overall median)
    per_fold_thresholds = []

    # KFold loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros_like(y_all), y_all)):
        init_logger(os.path.join(SAVE_DIR, "logs"), fold)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config["training"]["batch_size"])

        A_train, y_train = extract_angles(feature_extractor, train_loader, dtype=feat_dtype)
        A_val, y_val = extract_angles(feature_extractor, val_loader, dtype=feat_dtype)
        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)

        # ---------------- Kernels ----------------
        print(f"🔹 Kernel train | fold {fold}")
        K_train = compute_kernel_matrix(A_train, weights=W, device_name=pl_device,
                                        tile_size=tile_size, symmetric=True, n_workers=pl_workers,
                                        progress=True, desc=f"Fold {fold} | train").astype(ret_dtype, copy=False)

        print(f"🔹 Kernel val | fold {fold}")
        K_val = compute_kernel_matrix(A_val, Y=A_train, weights=W, device_name=pl_device,
                                      tile_size=tile_size, symmetric=False, n_workers=pl_workers,
                                      progress=True, desc=f"Fold {fold} | val").astype(ret_dtype, copy=False)

        K_train_u = np.asarray(K_train, dtype=ret_dtype, order="C")

        if do_center:
            # centre train et val
            K_train = _center_kernel_train(K_train_u.copy())
            K_val = _center_kernel_test(K_val, K_train_u)
            wandb.config.update({"kernel_centering": True}, allow_val_change=True)

        # ---------------- Optuna (binary): minimizes 1 - F1 ----------------
        def objective(trial):
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            svm = EnhancedSVM(C=C, kernel="precomputed", probability=True)
            svm.fit(K_train, y_train)
            y_val_pred = svm.predict(K_val)
            return 1.0 - f1_score(y_val, y_val_pred, average="binary")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.get("svm", {}).get("n_trials", 50))
        best_C = float(study.best_params["C"])
        wandb.log({f"fold_{fold}/best_C": best_C})

        # --------- Optimal threshold on the val (decision_function) ----------
        svm_val = EnhancedSVM(C=best_C, kernel="precomputed", probability=True)
        svm_val.fit(K_train, y_train)

        # decision scores (plus stables que proba pour le seuil)
        val_scores = svm_val.model.decision_function(K_val)
        if val_scores.ndim > 1 and val_scores.shape[1] == 2:
            val_scores = val_scores[:, 1]

        best_thr = find_best_thr_decision(val_scores, y_val, min_recall=0.10)
        wandb.log({f"fold_{fold}/svm_threshold": best_thr})
        per_fold_thresholds.append(best_thr)

        # ---------------- Final: trainval/test ----------------
        A_trainval = np.vstack([A_train, A_val])
        y_trainval = np.concatenate([y_train, y_val]).astype(np.int64)

        print(f"🔹 Kernel trainval | fold {fold}")
        K_trainval = compute_kernel_matrix(A_trainval, weights=W, device_name=pl_device,
                                           tile_size=tile_size, symmetric=True, n_workers=pl_workers,
                                           progress=True, desc=f"Fold {fold} | trainval").astype(ret_dtype, copy=False)
        print(f"🔹 Kernel test | fold {fold}")
        K_test = compute_kernel_matrix(A_test, Y=A_trainval, weights=W, device_name=pl_device,
                                       tile_size=tile_size, symmetric=False, n_workers=pl_workers,
                                       progress=True, desc=f"Fold {fold} | test").astype(ret_dtype, copy=False)

        if do_center:
            K_trainval_u = np.asarray(K_trainval, dtype=ret_dtype, order="C")
            K_trainval = _center_kernel_train(K_trainval_u.copy())
            K_test = _center_kernel_test(K_test, K_trainval_u)

        # Final SVM (fit on trainval)
        clf = EnhancedSVM(C=best_C, kernel="precomputed", probability=True)
        clf.fit(K_trainval, y_trainval)

        # --- metrics without custom thresholds (predict = 0.0) ---
        y_pred = clf.predict(K_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")
        precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
        recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
        bal_acc = balanced_accuracy_score(y_test, y_pred)

        try:
            test_scores = clf.model.decision_function(K_test)
            if test_scores.ndim > 1 and test_scores.shape[1] == 2:
                test_scores = test_scores[:, 1]
            auc = roc_auc_score(y_test, test_scores)
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

        # --- metrics WITH optimal fold threshold (globalthr will come later) ---
        y_pred_thr = binarize_by_threshold(test_scores, best_thr)
        acc_g = accuracy_score(y_test, y_pred_thr)
        f1_g = f1_score(y_test, y_pred_thr, average="binary")
        precision_g = precision_score(y_test, y_pred_thr, average="binary", zero_division=0)
        recall_g = recall_score(y_test, y_pred_thr, average="binary", zero_division=0)
        bal_acc_g = balanced_accuracy_score(y_test, y_pred_thr)
        # not identical because independent of the threshold
        wandb.log({
            f"fold_{fold}/qsvm_accuracy_globalthr": acc_g,
            f"fold_{fold}/qsvm_f1_globalthr": f1_g,
            f"fold_{fold}/qsvm_precision_globalthr": precision_g,
            f"fold_{fold}/qsvm_recall_globalthr": recall_g,
            f"fold_{fold}/qsvm_balanced_accuracy_globalthr": bal_acc_g,
            f"fold_{fold}/qsvm_auc_globalthr": auc,
        })

    # ----- Overall median threshold (useful for subsequent runs) -----
    if len(per_fold_thresholds) > 0:
        thr_global = float(np.median(per_fold_thresholds))
        wandb.log({f"global/svm_threshold_median": thr_global})

    wandb.finish()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    p.add_argument("--pl-device", type=str, default=None, help="PennyLane device (lightning.qubit, lightning.gpu)")
    p.add_argument("--pl-workers", type=int, default=None, help="#processes (0 => cpu_count-1, 1 recommended on GPU)")
    p.add_argument("--tile-size", type=int, default=None, help="Tile size for states/Gram")
    p.add_argument("--kernel-centering", action="store_true", help="Apply kernel centering before SVM")
    p.add_argument("--train-subset", type=int, default=None, help="Subsample train (stratified) for faster runs")

    # dtypes
    p.add_argument("--dtype", type=str, default="float32",
                   choices=["float32", "float64"], help="dtype utilisé pour les features (angles)")
    p.add_argument("--return-dtype", type=str, default="float64",
                   choices=["float32", "float64"], help="dtype final des matrices kernel")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_train(args)
