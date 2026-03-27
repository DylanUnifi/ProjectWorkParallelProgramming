import os
import argparse
import random
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KernelCenterer
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score
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


class KernelCacheManager:
    """Manage caching for expensive kernel matrices."""
    def __init__(self, cache_dir="./cache_kernels"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_hash(self, X, Y_shape, params):
        """Build a stable hash from data statistics and parameters."""
        data_summary = f"{X.shape}_{X.mean():.4f}_{X.std():.4f}_{Y_shape}"
        param_str = json.dumps(params, sort_keys=True)
        content = f"{data_summary}_{param_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def load(self, X, Y_shape, params, desc=""):
        h = self._compute_hash(X, Y_shape, params)
        path = os.path.join(self.cache_dir, f"kernel_{desc}_{h}.npy")
        if os.path.exists(path):
            print(f"Cache hit for {desc}: {path}")
            return np.load(path)
        return None

    def save(self, K, X, Y_shape, params, desc=""):
        h = self._compute_hash(X, Y_shape, params)
        path = os.path.join(self.cache_dir, f"kernel_{desc}_{h}.npy")
        np.save(path, K)
        print(f"Saved kernel: {path}")

def _center_kernel_train(K):
    """Center the training kernel matrix."""
    K = np.asarray(K, dtype=np.float64, order="C")
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _center_kernel_test(K_test, K_train_uncentered):
    """Center the test kernel matrix using training statistics."""
    Kt = np.asarray(K_test, dtype=np.float64, order="C")
    Ku = np.asarray(K_train_uncentered, dtype=np.float64, order="C")
    n = Ku.shape[0]
    col_mean = Ku.mean(axis=0)
    row_mean = Ku.mean(axis=1)
    total_mean = Ku.mean()
    
    K_centered = Kt - col_mean[np.newaxis, :] - row_mean.mean() + total_mean
    return K_centered

def preprocess_features(X_train, X_test, scaler_type="minmax", feature_range=(0, np.pi)):
    """Scale features for quantum embedding."""
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        return X_train, X_test

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def extract_features(dataset):
    """Extract flattened features from a dataset."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    
    all_features, all_labels = [], []
    
    for images, labels in tqdm(loader, desc="Extracting", leave=False):
        batch_size = images.shape[0]
        features = images.view(batch_size, -1).numpy()
        all_features.append(features)
        all_labels.append(labels.numpy())
    
    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)
    
    return X, y

def _compute_binary_metrics(y_true, y_pred, y_score):
    """Compute a compact set of binary classification metrics."""
    metrics = {
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
    }

    if np.unique(y_true).size < 2:
        metrics["auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
        return metrics

    metrics["auc"] = roc_auc_score(y_true, y_score)
    metrics["pr_auc"] = average_precision_score(y_true, y_score)
    return metrics

def train_fold(
    fold_idx, train_idx, val_idx, test_idx,
    X_full, y_full, weights, config, args, cache_mgr
):
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    X_test, y_test = X_full[test_idx], y_full[test_idx]
    
    n_components = args.pca_components or config.get("svm", {}).get("pca_components", 16)
    if n_components:
        pca = PCA(n_components=n_components, random_state=SEED)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)
        
    scale_range = (0, np.pi) if args.embed_mode == "angle" else (0, 1)
    scaler = MinMaxScaler(feature_range=scale_range)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    pl_cfg = config.get("pennylane", {})
    
    backend_params = {
        "device_name": args.pl_device or pl_cfg.get("device", "lightning.qubit"),
        "tile_size": args.tile_size or (args.torch_tile_size if args.gram_backend == "torch" else pl_cfg.get("tile_size", 128)),
        "n_workers": args.pl_workers if args.pl_workers is not None else pl_cfg.get("workers", 0),
        "gram_backend": args.gram_backend or pl_cfg.get("gram_backend", "auto"),
        "dtype": args.dtype,
        "return_dtype": args.return_dtype,
        "angle_scale": args.angle_scale,
        "embed_mode": args.embed_mode,
        "normalize": args.normalize_kernel,
        "state_tile": args.state_tile,
        "vram_fraction": args.vram_fraction,
        "autotune": args.autotune,
        "precompute_all_states": args.precompute_all_states,
        "dynamic_batch": args.dynamic_batch,
        "num_streams": args.num_streams,
        "learn_tiles": args.learn_tiles,
        "use_cuda_graphs": args.use_cuda_graphs,
        "profile_memory": args.profile_memory,
        "verbose_profile": args.verbose_profile,
    }
    
    if args.gram_backend == "torch":
        backend_params.update({
            "use_pinned_memory": args.torch_pinned_memory,
            "use_cuda_streams": args.torch_cuda_streams,
            "use_amp": args.torch_amp,
            "use_compile": args.torch_compile,
        })
    
    cache_params = {**backend_params, "weights_hash": hashlib.md5(weights.tobytes()).hexdigest()[:8]}

    print(f"\n🔹 Fold {fold_idx+1}: Kernel Train ({len(X_train)}x{len(X_train)})")
    K_train = cache_mgr.load(X_train, X_train.shape, cache_params, desc="train")
    if K_train is None:
        K_train = compute_kernel_matrix(
            X_train, weights=weights, symmetric=True,
            progress=True, desc=f"Fold {fold_idx+1} train",
            **backend_params
        )
        if args.cache_kernels:
            cache_mgr.save(K_train, X_train, X_train.shape, cache_params, desc="train")

    print(f"🔹 Fold {fold_idx+1}: Kernel Val ({len(X_val)}x{len(X_train)})")
    K_val = cache_mgr.load(X_val, X_train.shape, cache_params, desc="val")
    if K_val is None:
        K_val = compute_kernel_matrix(
            X_val, Y=X_train, weights=weights, symmetric=False,
            progress=True, desc=f"Fold {fold_idx+1} val",
            **backend_params
        )
        if args.cache_kernels:
            cache_mgr.save(K_val, X_val, X_train.shape, cache_params, desc="val")

    if args.kernel_centering:
        centerer = KernelCenterer()
        K_train = centerer.fit_transform(K_train)
        K_val = centerer.transform(K_val)

    optuna_best_value = float("nan")
    if args.svm_c is not None:
        best_C = float(args.svm_c)
        print(f"Using user-provided C: {best_C}")
    else:
        print(f"🔍 Optimizing C (Optuna)...")

        def objective(trial):
            C = trial.suggest_float('C', 0.1, 10.0, log=True)
            svm = EnhancedSVM(C=C, kernel="precomputed", probability=True, class_weight="balanced", max_iter=50000, tol=1e-4)
            svm.fit(K_train, y_train)
            y_pred = svm.predict(K_val)
            return 1.0 - f1_score(y_val, y_pred, average="binary")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config.get("svm", {}).get("n_trials", 10))
        best_C = study.best_params["C"]
        optuna_best_value = study.best_value
        print(f"Best C found by Optuna: {best_C}")

    selected_val_svm = EnhancedSVM(
        C=best_C,
        kernel="precomputed",
        probability=True,
        class_weight="balanced",
        max_iter=50000,
        tol=1e-4
    )
    selected_val_svm.fit(K_train, y_train)
    val_pred = selected_val_svm.predict(K_val)
    val_proba = selected_val_svm.predict_proba(K_val)[:, 1]
    val_metrics = _compute_binary_metrics(y_val, val_pred, val_proba)
    
    wandb.log({
        f"fold_{fold_idx}/best_C": best_C,
        f"fold_{fold_idx}/optuna_objective": optuna_best_value,
        f"fold_{fold_idx}/val_f1": val_metrics["f1"],
        f"fold_{fold_idx}/val_acc": val_metrics["acc"],
        f"fold_{fold_idx}/val_precision": val_metrics["precision"],
        f"fold_{fold_idx}/val_recall": val_metrics["recall"],
        f"fold_{fold_idx}/val_balanced_acc": val_metrics["balanced_acc"],
        f"fold_{fold_idx}/val_auc": val_metrics["auc"],
        f"fold_{fold_idx}/val_pr_auc": val_metrics["pr_auc"],
        f"fold_{fold_idx}/config/state_tile": args.state_tile,
        f"fold_{fold_idx}/config/vram_fraction": args.vram_fraction,
        f"fold_{fold_idx}/config/autotune": args.autotune,
        f"fold_{fold_idx}/config/num_streams": args.num_streams,
        f"fold_{fold_idx}/config/dynamic_batch": args.dynamic_batch,
        f"fold_{fold_idx}/config/use_cuda_graphs": args.use_cuda_graphs,
        f"fold_{fold_idx}/config/precompute_all_states": args.precompute_all_states,
        f"fold_{fold_idx}/config/learn_tiles": args.learn_tiles,
    })

    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    print(f"🔹 Fold {fold_idx+1}: Kernel TrainVal ({len(X_trainval)}x{len(X_trainval)})")
    K_trainval = cache_mgr.load(X_trainval, X_trainval.shape, cache_params, desc="trainval")
    if K_trainval is None:
        K_trainval = compute_kernel_matrix(
            X_trainval, weights=weights, symmetric=True,
            progress=True, desc="TrainVal", **backend_params
        )
        if args.cache_kernels:
             cache_mgr.save(K_trainval, X_trainval, X_trainval.shape, cache_params, desc="trainval")

    print(f"🔹 Fold {fold_idx+1}: Kernel Test ({len(X_test)}x{len(X_trainval)})")
    K_test = cache_mgr.load(X_test, X_trainval.shape, cache_params, desc="test")
    if K_test is None:
        K_test = compute_kernel_matrix(
            X_test, Y=X_trainval, weights=weights, symmetric=False,
            progress=True, desc="Test", **backend_params
        )
        if args.cache_kernels:
             cache_mgr.save(K_test, X_test, X_trainval.shape, cache_params, desc="test")

    if args.kernel_centering:
        centerer_final = KernelCenterer()
        K_trainval = centerer_final.fit_transform(K_trainval)
        K_test = centerer_final.transform(K_test)

    print(f"🚀 Retraining final SVM with best C: {best_C}...")
    final_svm = EnhancedSVM(
        C=best_C, 
        kernel="precomputed", 
        probability=True, 
        class_weight="balanced",
        max_iter=50000,
        tol=1e-4
    )
    final_svm.fit(K_trainval, y_trainval)
    
    y_pred = final_svm.predict(K_test)
    y_proba = final_svm.predict_proba(K_test)[:, 1]
    test_metrics = _compute_binary_metrics(y_test, y_pred, y_proba)
    
    wandb.log({
        f"fold_{fold_idx}/test_f1": test_metrics["f1"],
        f"fold_{fold_idx}/test_acc": test_metrics["acc"],
        f"fold_{fold_idx}/test_precision": test_metrics["precision"],
        f"fold_{fold_idx}/test_recall": test_metrics["recall"],
        f"fold_{fold_idx}/test_balanced_acc": test_metrics["balanced_acc"],
        f"fold_{fold_idx}/test_auc": test_metrics["auc"],
        f"fold_{fold_idx}/test_pr_auc": test_metrics["pr_auc"],
        f"fold_{fold_idx}/test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test, preds=y_pred,
            class_names=["Class 0", "Class 1"]
        )
    })
    
    print(f"✅ Result Fold {fold_idx+1}: F1={test_metrics['f1']:.4f} AUC={test_metrics['auc']:.4f}")
    return {
        "val_f1": val_metrics["f1"],
        "val_acc": val_metrics["acc"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_balanced_acc": val_metrics["balanced_acc"],
        "val_auc": val_metrics["auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "test_f1": test_metrics["f1"],
        "test_acc": test_metrics["acc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_balanced_acc": test_metrics["balanced_acc"],
        "test_auc": test_metrics["auc"],
        "test_pr_auc": test_metrics["pr_auc"],
    }


def run_train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    ds_name = config["dataset"]["name"]
    backend_short = args.gram_backend if args.gram_backend else "auto"
    exp_name = f"{ds_name}_qsvm_{backend_short}_{args.embed_mode}"
    if args.angle_scale != 1.0:
        exp_name += f"_scale{args.angle_scale}"
    
    wandb.init(project="pp_project", name=exp_name, config={**config, **vars(args)}, group="qsvm")
    
    print(f"📂 Loading {ds_name}...")
    train_dataset, test_dataset = load_dataset_by_name(
        name=ds_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", [3, 8]),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )
    
    if args.train_subset:
        train_dataset = Subset(train_dataset, range(min(len(train_dataset), args.train_subset)))
        print(f"Using train subset: {len(train_dataset)}")

    n_components = args.pca_components or config.get("svm", {}).get("pca_components", 16)
    X_train_raw, y_train = extract_features(train_dataset)
    X_test_raw, y_test = extract_features(test_dataset)
    
    X_full = np.vstack([X_train_raw, X_test_raw])
    y_full = np.concatenate([y_train, y_test])
    
    n_qubits = args.pca_components or config.get("svm", {}).get("pca_components", X_full.shape[1])
    n_layers = config.get("pennylane", {}).get("layers", 2)
    rng = np.random.default_rng(SEED)
    weights = rng.normal(0, 0.1, (n_layers, n_qubits)).astype(np.float32)
    
    cache_mgr = KernelCacheManager(args.cache_dir)
    
    n_folds = config.get("cv_folds", 3)
    test_size = len(X_test_raw)
    train_indices = list(range(len(X_full) - test_size))
    test_indices = list(range(len(X_full) - test_size, len(X_full)))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    results = []
    
    print(f"🚀 Starting {n_folds}-Fold CV on {args.pl_device} via {args.gram_backend}")
    print(f"⚙️  Params: tile={args.tile_size}, angle_scale={args.angle_scale}, embed={args.embed_mode}")
    
    for fold_idx, (t_idx, v_idx) in enumerate(kf.split(train_indices)):
        train_idx_global = [train_indices[i] for i in t_idx]
        val_idx_global = [train_indices[i] for i in v_idx]
        
        res = train_fold(
            fold_idx, train_idx_global, val_idx_global, test_indices,
            X_full, y_full, weights, config, args, cache_mgr
        )
        results.append(res)
        
    summary = {}
    for metric_name in results[0]:
        values = np.asarray([r[metric_name] for r in results], dtype=np.float64)
        summary[f"mean/{metric_name}"] = float(np.nanmean(values))
        summary[f"std/{metric_name}"] = float(np.nanstd(values))
    
    wandb.log(summary)
    print(
        f"\n🏆 Final Average: "
        f"val_F1={summary['mean/val_f1']:.4f} "
        f"test_F1={summary['mean/test_f1']:.4f} "
        f"test_AUC={summary['mean/test_auc']:.4f}"
    )
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--pl-device", type=str, default=None)
    p.add_argument("--pl-workers", type=int, default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--gram-backend", type=str, default="auto", 
                   choices=["auto", "numpy", "torch", "cuda_states"])
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--return-dtype", type=str, default="float32")
    
    p.add_argument("--angle-scale", type=float, default=0.1, help="Scaling factor for AngleEmbedding (gamma)")
    p.add_argument("--embed-mode", type=str, default="ryrz", choices=["angle", "ry", "ryrz"])
    p.add_argument("--normalize-kernel", action="store_true", help="Force diag(K)=1")
    p.add_argument("--kernel-centering", action="store_true")
    
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--pca-components", type=int, default=None)
    p.add_argument("--cache-kernels", action="store_true", help="Enable disk caching for kernels")
    p.add_argument("--cache-dir", type=str, default="./kernel_cache")
    p.add_argument("--svm-c", type=float, default=None, help="Force SVM C parameter (bypasses Optuna search)")

    p.add_argument("--state-tile", type=int, default=-1,
                   help="State tile size (-1 for auto VRAM-aware)")
    p.add_argument("--vram-fraction", type=float, default=0.95,
                   help="Maximum VRAM fraction to use (0-1)")
    p.add_argument("--autotune", action="store_true", default=True,
                   help="Enable kernel tile autotuning")
    p.add_argument("--no-autotune", action="store_false", dest="autotune")
    p.add_argument("--precompute-all-states", action="store_true", default=True,
                   help="Bulk precompute all quantum states")
    p.add_argument("--no-precompute", action="store_false", dest="precompute_all_states")
    p.add_argument("--dynamic-batch", action="store_true", default=False,
                   help="Enable dynamic batch sizing")
    p.add_argument("--no-dynamic-batch", action="store_false", dest="dynamic_batch")
    p.add_argument("--num-streams", type=int, default=2,
                   help="Number of CUDA streams for parallelism")
    p.add_argument("--learn-tiles", action="store_true", default=True,
                   help="Learn optimal tiles from run history")
    p.add_argument("--no-learn-tiles", action="store_false", dest="learn_tiles")
    p.add_argument("--use-cuda-graphs", action="store_true", default=False,
                   help="Enable CUDA graph optimization")
    p.add_argument("--no-cuda-graphs", action="store_false", dest="use_cuda_graphs")
    p.add_argument("--profile-memory", action="store_true", default=False,
                   help="Enable GPU memory profiling")
    p.add_argument("--verbose-profile", action="store_true", default=False,
                   help="Show detailed profiling output")
    
    p.add_argument("--torch-tile-size", type=int, default=512,
                   help="Tile size for torch backend")
    p.add_argument("--torch-pinned-memory", action="store_true", default=False,
                   help="Use pinned memory for torch transfers")
    p.add_argument("--torch-cuda-streams", action="store_true", default=False,
                   help="Use CUDA streams for torch overlap")
    p.add_argument("--torch-amp", action="store_true", default=False,
                   help="Use automatic mixed precision (experimental)")
    p.add_argument("--torch-compile", action="store_true", default=False,
                   help="Use torch.compile (PyTorch 2.0+)")

    args = p.parse_args()
    run_train(args)
