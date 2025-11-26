# train_qsvm.py - CORRECTED & OPTIMIZED
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
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
from scripts.pipeline_backends import compute_kernel_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ═══════════════════════════════════════════════════════════
# Kernel Centering
# ═══════════════════════════════════════════════════════════
def _center_kernel_train(K):
    """Centre le kernel d'entraînement."""
    K = np.asarray(K, dtype=np.float64, order="C")
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def _center_kernel_test(K_test, K_train_uncentered):
    """
    Centre le kernel de test selon les statistiques du train.
    
    Args:
        K_test: (m, n) - kernel test-train
        K_train_uncentered: (n, n) - kernel train-train non centré
    
    Returns:
        (m, n) - kernel test centré
    """
    Kt = np.asarray(K_test, dtype=np.float64, order="C")
    Ku = np.asarray(K_train_uncentered, dtype=np.float64, order="C")
    
    n = Ku.shape[0]
    m = Kt.shape[0]
    
    # Moyennes du kernel train
    col_mean = Ku.mean(axis=0)  # (n,) moyenne par colonne
    row_mean = Ku.mean(axis=1)  # (n,) moyenne par ligne  
    total_mean = Ku.mean()      # scalaire
    
    # Centrage de K_test
    # Pour chaque élément K_test[i,j]:
    # K_centered[i,j] = K_test[i,j] - col_mean[j] - row_mean.mean() + total_mean
    
    K_centered = Kt.copy()
    
    # Soustraire moyenne par colonne (broadcast sur axis=0)
    K_centered -= col_mean[np.newaxis, :]
    
    # Soustraire moyenne globale des lignes
    K_centered -= row_mean.mean()
    
    # Ajouter moyenne totale
    K_centered += total_mean
    
    return K_centered


# ═══════════════════════════════════════════════════════════
# SVM Utilities
# ═══════════════════════════════════════════════════════════
def hinge_loss_from_scores(scores, y_true):
    """Calcule hinge loss: max(0, 1 - y*score)."""
    y_signed = 2 * y_true - 1  # {0,1} → {-1,+1}
    losses = np.maximum(0, 1 - y_signed * scores)
    return float(np.mean(losses))

def find_best_threshold(scores, y_true, min_recall=0.10):
    """Trouve le meilleur seuil de décision (maximise F1)."""
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
    """Applique seuil de décision."""
    return (scores >= thr).astype(np.int64)

# ═══════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════
def extract_features(dataset, use_pca=True, n_components=None, pca_model=None):
    """
    Extrait features depuis un dataset PyTorch.
    Returns: (features, labels, pca_model)
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    
    all_features = []
    all_labels = []
    
    print("📦 Extraction des features...")
    for images, labels in tqdm(loader, desc="Feature extraction"):
        # Flatten images
        batch_size = images.shape[0]
        features = images.view(batch_size, -1).numpy()  # [B, C×H×W]
        
        all_features.append(features)
        all_labels.append(labels.numpy())
    
    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)
    
    # PCA optionnel
    if use_pca and n_components:
        if pca_model is None:
            print(f"🔬 Applying PCA: {X.shape[1]} → {n_components} dims")
            pca_model = PCA(n_components=n_components, random_state=SEED)
            X = pca_model.fit_transform(X)
        else:
            X = pca_model.transform(X)
        
        print(f"   Variance explained: {pca_model.explained_variance_ratio_.sum():.2%}")
    
    return X.astype(np.float32), y, pca_model

# ═══════════════════════════════════════════════════════════
# Quantum Kernel + SVM Training
# ═══════════════════════════════════════════════════════════
def train_fold(
    fold_idx,
    train_idx, val_idx, test_idx,
    X_full, y_full,
    weights,
    config, args
):
    """Entraîne un fold de cross-validation."""
    
    # Split data
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    X_test, y_test = X_full[test_idx], y_full[test_idx]
    
    # Config kernel
    pl_device = args.pl_device or config.get("pennylane", {}).get("device", "lightning.qubit")
    pl_workers = args.pl_workers if args.pl_workers is not None else config.get("pennylane", {}).get("workers", 0)
    tile_size = args.tile_size or config.get("pennylane", {}).get("tile_size", 128)
    do_center = bool(args.kernel_centering or config.get("pennylane", {}).get("kernel_centering", False))
    gram_backend = args.gram_backend or config.get("pennylane", {}).get("gram_backend", "auto")
    
    # ── Compute Kernels ──
    print(f"\n{'═'*60}")
    print(f"FOLD {fold_idx + 1}: Computing Quantum Kernels")
    print(f"{'═'*60}")
    
    # Train kernel
    print(f"🔹 K_train ({X_train.shape[0]}×{X_train.shape[0]})")
    K_train = compute_kernel_matrix(
        X_train,
        weights=weights,
        device_name=pl_device,
        tile_size=tile_size,
        symmetric=True,
        n_workers=pl_workers,
        dtype=args.dtype,
        return_dtype=args.return_dtype,
        gram_backend=gram_backend,
        progress=True,
        desc=f"Fold {fold_idx+1} | train"
    )
    
    # Val kernel
    print(f"🔹 K_val ({X_val.shape[0]}×{X_train.shape[0]})")
    K_val = compute_kernel_matrix(
        X_val, Y=X_train,
        weights=weights,
        device_name=pl_device,
        tile_size=tile_size,
        symmetric=False,
        n_workers=pl_workers,
        dtype=args.dtype,
        return_dtype=args.return_dtype,
        gram_backend=gram_backend,
        progress=True,
        desc=f"Fold {fold_idx+1} | val"
    )
    
    # Centering
    if do_center:
        print("🔄 Centering kernels...")
        K_train_unc = K_train.copy()
        K_train = _center_kernel_train(K_train)
        K_val = _center_kernel_test(K_val, K_train_unc)
    
    # ── Optimize C with Optuna ──
    print(f"\n🔍 Optimizing SVM hyperparameter C (Optuna)...")
    
    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e3, log=True)
        svm = EnhancedSVM(C=C, kernel="precomputed", probability=True)
        svm.fit(K_train, y_train)
        y_pred = svm.predict(K_val)
        f1 = f1_score(y_val, y_pred, average="binary")
        return 1.0 - f1  # Minimize (1 - F1)
    
    n_trials = config.get("svm", {}).get("n_trials", 30)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_C = float(study.best_params["C"])
    print(f"✅ Best C: {best_C:.4e} (F1_val: {1 - study.best_value:.4f})")
    
    wandb.log({f"fold_{fold_idx}/best_C": best_C})
    
    # ── Train with best C & find threshold ──
    svm_val = EnhancedSVM(C=best_C, kernel="precomputed", probability=True)
    svm_val.fit(K_train, y_train)
    
    # Decision scores for threshold tuning
    val_scores = svm_val.model.decision_function(K_val)
    if val_scores.ndim > 1:
        val_scores = val_scores[:, 1]
    
    best_thr = find_best_threshold(val_scores, y_val, min_recall=0.10)
    print(f"🎯 Best threshold: {best_thr:.4f}")
    
    wandb.log({f"fold_{fold_idx}/best_threshold": best_thr})
    
    # Hinge loss (monitoring)
    train_scores = svm_val.model.decision_function(K_train)
    hinge_train = hinge_loss_from_scores(train_scores, y_train)
    hinge_val = hinge_loss_from_scores(val_scores, y_val)
    
    wandb.log({
        f"fold_{fold_idx}/hinge_train": hinge_train,
        f"fold_{fold_idx}/hinge_val": hinge_val,
    })
    
    # ── Final Evaluation on Test ──
    print(f"\n🧪 Final evaluation on test set...")
    
    # Trainval merge
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    print(f"🔹 K_trainval ({X_trainval.shape[0]}×{X_trainval.shape[0]})")
    K_trainval = compute_kernel_matrix(
        X_trainval,
        weights=weights,
        device_name=pl_device,
        tile_size=tile_size,
        symmetric=True,
        n_workers=pl_workers,
        dtype=args.dtype,
        return_dtype=args.return_dtype,
        gram_backend=gram_backend,
        progress=True,
        desc=f"Fold {fold_idx+1} | trainval"
    )
    
    print(f"🔹 K_test ({X_test.shape[0]}×{X_trainval.shape[0]})")
    K_test = compute_kernel_matrix(
        X_test, Y=X_trainval,
        weights=weights,
        device_name=pl_device,
        tile_size=tile_size,
        symmetric=False,
        n_workers=pl_workers,
        dtype=args.dtype,
        return_dtype=args.return_dtype,
        gram_backend=gram_backend,
        progress=True,
        desc=f"Fold {fold_idx+1} | test"
    )
    
    if do_center:
        K_trainval_unc = K_trainval.copy()
        K_trainval = _center_kernel_train(K_trainval)
        K_test = _center_kernel_test(K_test, K_trainval_unc)
    
    # Final SVM
    clf = EnhancedSVM(C=best_C, kernel="precomputed", probability=True)
    clf.fit(K_trainval, y_trainval)
    
    # Standard predict
    y_pred = clf.predict(K_test)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # AUC avec scores
    test_scores = clf.model.decision_function(K_test)
    if test_scores.ndim > 1:
        test_scores = test_scores[:, 1]
    
    try:
        auc = roc_auc_score(y_test, test_scores)
    except:
        auc = 0.0
    
    # Hinge loss test
    hinge_test = hinge_loss_from_scores(test_scores, y_test)
    
    # Metrics avec threshold
    y_pred_thr = binarize_by_threshold(test_scores, best_thr)
    acc_thr = accuracy_score(y_test, y_pred_thr)
    f1_thr = f1_score(y_test, y_pred_thr, average="binary")
    prec_thr = precision_score(y_test, y_pred_thr, average="binary", zero_division=0)
    rec_thr = recall_score(y_test, y_pred_thr, average="binary", zero_division=0)
    bal_thr = balanced_accuracy_score(y_test, y_pred_thr)
    
    # Log all
    wandb.log({
        # Standard
        f"fold_{fold_idx}/test_accuracy": acc,
        f"fold_{fold_idx}/test_f1": f1,
        f"fold_{fold_idx}/test_precision": prec,
        f"fold_{fold_idx}/test_recall": rec,
        f"fold_{fold_idx}/test_balanced_accuracy": bal_acc,
        f"fold_{fold_idx}/test_auc": auc,
        f"fold_{fold_idx}/hinge_test": hinge_test,
        
        # With threshold
        f"fold_{fold_idx}/test_accuracy_thr": acc_thr,
        f"fold_{fold_idx}/test_f1_thr": f1_thr,
        f"fold_{fold_idx}/test_precision_thr": prec_thr,
        f"fold_{fold_idx}/test_recall_thr": rec_thr,
        f"fold_{fold_idx}/test_balanced_accuracy_thr": bal_thr,
    })
    
    print(f"\n📊 Test Results (Fold {fold_idx+1}):")
    print(f"   Standard predict: F1={f1:.4f}, Acc={acc:.4f}, AUC={auc:.4f}")
    print(f"   With threshold:   F1={f1_thr:.4f}, Acc={acc_thr:.4f}")
    
    return {
        "fold": fold_idx,
        "best_C": best_C,
        "best_threshold": best_thr,
        "test_f1": f1,
        "test_f1_thr": f1_thr,
        "test_auc": auc,
    }

# ═══════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════
def run_train(args):
    """Main training pipeline."""
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup W&B
    os.environ["WANDB_MODE"] = "online"
    
    dataset_name = config["dataset"]["name"]
    exp_name = config.get("experiment_name", "default_exp")
    EXPERIMENT_NAME = f"{dataset_name}_{exp_name}"
    
    wandb.init(
        project="pp_project",
        name=EXPERIMENT_NAME,
        config={**config, **vars(args)},
        group=dataset_name
    )
    
    # Load dataset
    print(f"\n{'═'*60}")
    print(f"Loading Dataset: {dataset_name}")
    print(f"{'═'*60}")
    
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", [3, 8]),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )
    
    # Subset for fast testing
    if args.train_subset:
        indices = torch.randperm(len(train_dataset))[:args.train_subset].tolist()
        train_dataset = Subset(train_dataset, indices)
        print(f"⚠️  Using subset: {args.train_subset} samples")
    
    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Test:  {len(test_dataset)} samples")
    
    # Extract features
    use_pca = config.get("svm", {}).get("use_pca", False)
    n_components = config.get("svm", {}).get("pca_components", 50)
    
    X_train, y_train, pca_model = extract_features(
        train_dataset,
        use_pca=use_pca,
        n_components=n_components
    )
    
    X_test, y_test, _ = extract_features(
        test_dataset,
        use_pca=use_pca,
        n_components=n_components,
        pca_model=pca_model
    )
    
    # Combine for cross-validation
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    
    n_features = X_full.shape[1]
    print(f"\n📊 Feature dimensions: {n_features}")
    
    # Initialize quantum weights
    n_qubits = min(n_features, 10)  # Max 10 qubits pour éviter explosion
    n_layers = config.get("pennylane", {}).get("layers", 2)
    
    print(f"🔬 Quantum circuit: {n_qubits} qubits, {n_layers} layers")
    
    rng = np.random.default_rng(SEED)
    weights = rng.normal(0, 0.1, (n_layers, n_qubits)).astype(np.float32)
    
    wandb.config.update({
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_features": n_features,
    })
    
    # Truncate features to n_qubits
    if n_features > n_qubits:
        print(f"⚠️  Truncating features: {n_features} → {n_qubits}")
        X_full = X_full[:, :n_qubits]
    
    # Cross-validation
    n_folds = config.get("cv_folds", 3)
    test_size = len(X_test)
    
    # Keep test set fixed, CV on train
    test_indices = list(range(len(X_full) - test_size, len(X_full)))
    train_indices = list(range(len(X_full) - test_size))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_indices)):
        # Map indices
        train_idx_global = [train_indices[i] for i in train_idx]
        val_idx_global = [train_indices[i] for i in val_idx]
        
        result = train_fold(
            fold_idx=fold_idx,
            train_idx=train_idx_global,
            val_idx=val_idx_global,
            test_idx=test_indices,
            X_full=X_full,
            y_full=y_full,
            weights=weights,
            config=config,
            args=args
        )
        
        fold_results.append(result)
    
    # Aggregate results
    avg_f1 = np.mean([r["test_f1"] for r in fold_results])
    avg_f1_thr = np.mean([r["test_f1_thr"] for r in fold_results])
    avg_auc = np.mean([r["test_auc"] for r in fold_results])
    
    wandb.log({
        "avg/test_f1": avg_f1,
        "avg/test_f1_thr": avg_f1_thr,
        "avg/test_auc": avg_auc,
    })
    
    print(f"\n{'═'*60}")
    print(f"FINAL RESULTS ({n_folds} folds)")
    print(f"{'═'*60}")
    print(f"Average F1 (standard):    {avg_f1:.4f}")
    print(f"Average F1 (w/ threshold): {avg_f1_thr:.4f}")
    print(f"Average AUC:              {avg_auc:.4f}")
    
    wandb.finish()

# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
def build_argparser():
    p = argparse.ArgumentParser(description="Train Quantum SVM")
    
    p.add_argument("--config", type=str, required=True, help="Path to YAML config")
    
    # PennyLane
    p.add_argument("--pl-device", type=str, default=None, help="PennyLane device (lightning.qubit|lightning.gpu)")
    p.add_argument("--pl-workers", type=int, default=None, help="Number of workers (CPU only)")
    p.add_argument("--tile-size", type=int, default=None, help="Tile size for kernel computation")
    p.add_argument("--kernel-centering", action="store_true", help="Enable kernel centering")
    
    # Backend
    p.add_argument("--gram-backend", type=str, default=None,
                   choices=["auto", "torch", "cupy", "numpy", "cuda_states", "tensorcore"],
                   help="Backend for kernel matrix computation")
    
    # Precision
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"],
                   help="Dtype for kernel computation")
    p.add_argument("--return-dtype", type=str, default="float32", choices=["float32", "float64"],
                   help="Dtype for output kernel matrix")
    
    # Data
    p.add_argument("--train-subset", type=int, default=None, help="Use subset of training data")
    
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_train(args)
