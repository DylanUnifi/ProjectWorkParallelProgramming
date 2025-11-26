import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, precision_recall_curve
)
from tqdm import tqdm
import yaml
import wandb
import optuna

from data_loader.utils import load_dataset_by_name
from models.svm_extension import EnhancedSVM
from utils.scheduler import get_scheduler
from utils.logger import init_logger
from scripts.pipeline_backends import compute_kernel_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def _center_kernel_train(K):
    K = np.asarray(K, dtype=np.float64, order="C")
    n = K.shape[0]; H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _center_kernel_test(K_test, K_train_unc):
    Kt = np.asarray(K_test, dtype=np.float64, order="C")
    Ku = np.asarray(K_train_unc, dtype=np.float64, order="C")
    mc = Ku.mean(axis=0, keepdims=True); mr = Ku.mean(axis=1, keepdims=True); ma = Ku.mean()
    return Kt - np.ones((Kt.shape[0],1)) @ mc - np.ones((Kt.shape[0],1)) @ mr.T + ma

def find_best_thr_decision(scores, y_true, min_recall=None):
    p, r, thr = precision_recall_curve(y_true, scores)
    thr_ext = np.r_[thr, thr[-1] if thr.size else 0.0]
    best_f1, best_thr = -1.0, 0.0
    for pi, ri, ti in zip(p, r, thr_ext):
        if (min_recall is None) or (ri >= min_recall):
            f1 = 0.0 if (pi + ri) == 0 else 2 * pi * ri / (pi + ri)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(ti)
    return best_thr

def binarize_by_threshold(scores, thr):
    return (scores >= thr).astype(np.int64)

# ---------- main ----------
def run_train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # W&B en ligne uniquement
    os.environ["WANDB_MODE"] = "online"

    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}"

    SAVE_DIR = os.path.join("engine/checkpoints", "fidelity_kernel", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(project="pp_project", name=EXPERIMENT_NAME, config={**config, **vars(args)}, group=dataset_name)
    wandb.config.update({"config_path": os.path.abspath(args.config)})

    feat_dtype = np.float32 if args.dtype == "float32" else np.float64
    ret_dtype  = np.float32 if args.return_dtype == "float32" else np.float64

    # Pennylane backends / kernel opts
    pl_device   = args.pl_device or config.get("pennylane", {}).get("device", "lightning.qubit")
    pl_workers  = args.pl_workers if args.pl_workers is not None else config.get("pennylane", {}).get("workers", 0)
    tile_size   = args.tile_size or config.get("pennylane", {}).get("tile_size", 128)
    do_center   = bool(args.kernel_centering or config.get("pennylane", {}).get("kernel_centering", False))
    gram_backend = args.gram_backend or config.get("pennylane", {}).get("gram_backend", "auto")
    wandb.config.update({"pennylane/gram_backend": gram_backend}, allow_val_change=True)

    # Dataset
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", config.get("binary_classes", [3, 8])),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )

    indices = torch.randperm(len(train_dataset))[:500].tolist()
    train_dataset = Subset(train_dataset, indices)

    print(f"Nombre d'exemples chargés dans train_dataset : {len(train_dataset)}")


    # kernels
    print(f"🔹 Kernel train")
    K_train = compute_kernel_matrix(
        A_train, weights=W, device_name=pl_device, tile_size=tile_size,
        symmetric=True, n_workers=pl_workers, dtype=args.dtype, return_dtype=args.return_dtype,
        gram_backend=gram_backend, progress=True, desc=f"Fold {fold} | train"
    ).astype(ret_dtype, copy=False)

    print(f"🔹 Kernel val")
    K_val = compute_kernel_matrix(
        A_val, Y=A_train, weights=W, device_name=pl_device, tile_size=tile_size,
        symmetric=False, n_workers=pl_workers, dtype=args.dtype, return_dtype=args.return_dtype,
        gram_backend=gram_backend, progress=True, desc=f"Fold {fold} | val"
    ).astype(ret_dtype, copy=False)

    K_train_u = np.asarray(K_train, dtype=ret_dtype, order="C")
    if do_center:
        K_train = _center_kernel_train(K_train_u.copy())
        K_val   = _center_kernel_test(K_val, K_train_u)
        wandb.config.update({"kernel_centering": True}, allow_val_change=True)

    # Optuna sur C (minimise 1 - F1)
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        svm = EnhancedSVM(C=C, kernel="precomputed", probability=True)
        svm.fit(K_train, y_train)
        y_pred = svm.predict(K_val)
        return 1.0 - f1_score(y_val, y_pred, average="binary")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=config.get("svm", {}).get("n_trials", 50))
    best_C = float(study.best_params["C"])
    wandb.log({f"fold_{fold}/best_C": best_C})

    # Seuil sur decision_function
    svm_val = EnhancedSVM(C=best_C, kernel="precomputed", probability=True)
    svm_val.fit(K_train, y_train)
    val_scores = svm_val.model.decision_function(K_val)

    # Hinge loss (train et val) pour monitoring
    train_scores_tmp = svm_val.model.decision_function(K_train)
    hinge_train = hinge_loss_from_scores(train_scores_tmp, y_train)
    hinge_val = hinge_loss_from_scores(val_scores, y_val)
    wandb.log({
        f"fold_{fold}/svm_hinge_train": hinge_train,
        f"fold_{fold}/svm_hinge_val": hinge_val,
    })

    if val_scores.ndim > 1 and val_scores.shape[1] == 2:
        val_scores = val_scores[:, 1]
    best_thr = find_best_thr_decision(val_scores, y_val, min_recall=0.10)
    wandb.log({f"fold_{fold}/svm_threshold": best_thr})
    per_fold_thresholds.append(best_thr)

    # Trainval/test
    A_trainval = np.vstack([A_train, A_val])
    y_trainval = np.concatenate([y_train, y_val]).astype(np.int64)

    print(f"🔹 Kernel trainval | fold {fold}")
    K_trainval = compute_kernel_matrix(
        A_trainval, weights=W, device_name=pl_device, tile_size=tile_size,
        symmetric=True, n_workers=pl_workers, dtype=args.dtype, return_dtype=args.return_dtype,
        gram_backend=gram_backend, progress=True, desc=f"Fold {fold} | trainval"
    ).astype(ret_dtype, copy=False)

    print(f"🔹 Kernel test | fold {fold}")
    K_test = compute_kernel_matrix(
        A_test, Y=A_trainval, weights=W, device_name=pl_device, tile_size=tile_size,
        symmetric=False, n_workers=pl_workers, dtype=args.dtype, return_dtype=args.return_dtype,
        gram_backend=gram_backend, progress=True, desc=f"Fold {fold} | test"
    ).astype(ret_dtype, copy=False)

    if do_center:
        K_trainval_u = np.asarray(K_trainval, dtype=ret_dtype, order="C")
        K_trainval   = _center_kernel_train(K_trainval_u.copy())
        K_test       = _center_kernel_test(K_test, K_trainval_u)

    clf = EnhancedSVM(C=best_C, kernel="precomputed", probability=True)
    clf.fit(K_trainval, y_trainval)

    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary")
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="binary", zero_division=0)
    bal  = balanced_accuracy_score(y_test, y_pred)
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
        f"fold_{fold}/qsvm_precision": prec,
        f"fold_{fold}/qsvm_recall": rec,
        f"fold_{fold}/qsvm_balanced_accuracy": bal,
        f"fold_{fold}/qsvm_auc": auc,
    })

    # Hinge loss sur trainval et test (indépendante du threshold)
    trainval_scores = clf.model.decision_function(K_trainval)
    hinge_trainval = hinge_loss_from_scores(trainval_scores, y_trainval)
    hinge_test = hinge_loss_from_scores(test_scores, y_test)
    wandb.log({
        f"fold_{fold}/svm_hinge_trainval": hinge_trainval,
        f"fold_{fold}/svm_hinge_test": hinge_test,
    })

    # Avec seuil optimal du fold
    y_pred_thr = binarize_by_threshold(test_scores, best_thr)
    acc_g = accuracy_score(y_test, y_pred_thr)
    f1_g  = f1_score(y_test, y_pred_thr, average="binary")
    prec_g= precision_score(y_test, y_pred_thr, average="binary", zero_division=0)
    rec_g = recall_score(y_test, y_pred_thr, average="binary", zero_division=0)
    bal_g = balanced_accuracy_score(y_test, y_pred_thr)

    wandb.log({
        f"fold_{fold}/qsvm_accuracy_globalthr": acc_g,
        f"fold_{fold}/qsvm_f1_globalthr": f1_g,
        f"fold_{fold}/qsvm_precision_globalthr": prec_g,
        f"fold_{fold}/qsvm_recall_globalthr": rec_g,
        f"fold_{fold}/qsvm_balanced_accuracy_globalthr": bal_g,
        f"fold_{fold}/qsvm_auc_globalthr": auc,
    })

if len(per_fold_thresholds) > 0:
    thr_global = float(np.median(per_fold_thresholds))
    wandb.log({"global/svm_threshold_median": thr_global})

wandb.finish()

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--pl-device", type=str, default=None)
    p.add_argument("--pl-workers", type=int, default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--kernel-centering", action="store_true")
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    p.add_argument("--return-dtype", type=str, default="float64", choices=["float32","float64"])
    p.add_argument("--gram-backend", type=str, default=None,
        choices=["auto", "torch", "cupy", "numpy", "cuda_states", "cuda_ry"],
        help="Backend pour compute_kernel_matrix (priorité sur le YAML s'il est fourni)."
    )
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_train(args)
