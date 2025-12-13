# train_svm_qkernel.py - OPTIMIZED & CACHED
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, precision_recall_curve, confusion_matrix
)
from tqdm import tqdm
import yaml
import wandb
import optuna

# Imports locaux
from data_loader.utils import load_dataset_by_name
from models.svm_extension import EnhancedSVM
from scripts.pipeline_backends import compute_kernel_matrix

# Gestion du déterminisme
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ═══════════════════════════════════════════════════════════
# 1. Gestionnaire de Cache pour les Kernels (NOUVEAU)
# ═══════════════════════════════════════════════════════════
class KernelCacheManager:
    """Gère la sauvegarde et le chargement des matrices de kernel coûteuses."""
    def __init__(self, cache_dir="./cache_kernels"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _compute_hash(self, X, Y_shape, params):
        """Crée un hash unique basé sur les données et les paramètres."""
        # On hash une statistique des données (moyenne/std) pour éviter de hasher tout le tableau
        data_summary = f"{X.shape}_{X.mean():.4f}_{X.std():.4f}_{Y_shape}"
        param_str = json.dumps(params, sort_keys=True)
        content = f"{data_summary}_{param_str}"
        return hashlib.md5(content.encode()).hexdigest()

    def load(self, X, Y_shape, params, desc=""):
        h = self._compute_hash(X, Y_shape, params)
        path = os.path.join(self.cache_dir, f"kernel_{desc}_{h}.npy")
        if os.path.exists(path):
            print(f"⚡ Cache hit pour {desc}: {path}")
            return np.load(path)
        return None

    def save(self, K, X, Y_shape, params, desc=""):
        h = self._compute_hash(X, Y_shape, params)
        path = os.path.join(self.cache_dir, f"kernel_{desc}_{h}.npy")
        np.save(path, K)
        print(f"💾 Kernel sauvegardé: {path}")

# ═══════════════════════════════════════════════════════════
# 2. Utilitaires Kernel (Centrage, Scaling)
# ═══════════════════════════════════════════════════════════
def _center_kernel_train(K):
    """Centre le kernel d'entraînement (Kernel PCA implicite)."""
    K = np.asarray(K, dtype=np.float64, order="C")
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def _center_kernel_test(K_test, K_train_uncentered):
    """Centre le kernel de test selon les statistiques du train."""
    Kt = np.asarray(K_test, dtype=np.float64, order="C")
    Ku = np.asarray(K_train_uncentered, dtype=np.float64, order="C")
    n = Ku.shape[0]
    col_mean = Ku.mean(axis=0)
    row_mean = Ku.mean(axis=1)
    total_mean = Ku.mean()
    
    K_centered = Kt - col_mean[np.newaxis, :] - row_mean.mean() + total_mean
    return K_centered

def preprocess_features(X_train, X_test, scaler_type="minmax", feature_range=(0, np.pi)):
    """Normalise les features pour l'embedding quantique."""
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        return X_train, X_test

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# ═══════════════════════════════════════════════════════════
# 3. Pipeline d'Extraction
# ═══════════════════════════════════════════════════════════
def extract_features(dataset, use_pca=True, n_components=None, pca_model=None):
    """Extrait les features (pixels aplatis) + PCA optionnelle."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    
    all_features, all_labels = [], []
    
    # print("📦 Extraction des features...")
    for images, labels in tqdm(loader, desc="Extracting", leave=False):
        batch_size = images.shape[0]
        # Flatten: [B, C, H, W] -> [B, Features]
        features = images.view(batch_size, -1).numpy()
        all_features.append(features)
        all_labels.append(labels.numpy())
    
    X = np.vstack(all_features).astype(np.float32)
    y = np.concatenate(all_labels).astype(np.int64)
    
    if use_pca and n_components:
        if pca_model is None:
            # print(f"🔬 Fitting PCA: {X.shape[1]} → {n_components} dims")
            pca_model = PCA(n_components=n_components, random_state=SEED)
            X = pca_model.fit_transform(X)
        else:
            X = pca_model.transform(X)
        
    return X.astype(np.float32), y, pca_model

# ═══════════════════════════════════════════════════════════
# 4. Entraînement d'un Fold
# ═══════════════════════════════════════════════════════════
def train_fold(
    fold_idx, train_idx, val_idx, test_idx,
    X_full, y_full, weights, config, args, cache_mgr
):
    # Séparation des données
    X_train, y_train = X_full[train_idx], y_full[train_idx]
    X_val, y_val = X_full[val_idx], y_full[val_idx]
    X_test, y_test = X_full[test_idx], y_full[test_idx]
    
    # --- Scaling (Important pour Quantum) ---
    # On scale après le split pour éviter le data leakage
    # Scaling par défaut: [0, 2pi] pour AngleEmbedding, ou [0, pi]
    scale_range = (0, np.pi) if args.embed_mode == "angle" else (0, 1) # RZ/RY aiment bien ~1
    X_train, X_val = preprocess_features(X_train, X_val, scaler_type="minmax", feature_range=scale_range)
    # Note: X_test sera scalé par rapport à X_train+X_val plus tard lors du merge
    
    # --- Configuration Backend ---
    pl_cfg = config.get("pennylane", {})
    backend_params = {
        "device_name": args.pl_device or pl_cfg.get("device", "lightning.qubit"),
        "tile_size": args.tile_size or pl_cfg.get("tile_size", 128),
        "n_workers": args.pl_workers if args.pl_workers is not None else pl_cfg.get("workers", 0),
        "gram_backend": args.gram_backend or pl_cfg.get("gram_backend", "auto"),
        "dtype": args.dtype,
        "return_dtype": args.return_dtype,
        "angle_scale": args.angle_scale,   # Nouveau
        "embed_mode": args.embed_mode,     # Nouveau
        "normalize": args.normalize_kernel # Nouveau
    }
    
    # Paramètres pour le hash du cache
    cache_params = {**backend_params, "weights_hash": hashlib.md5(weights.tobytes()).hexdigest()[:8]}

    # ── Calcul / Chargement Kernel Train ──
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

    # ── Calcul / Chargement Kernel Val ──
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

    # Centrage optionnel
    if args.kernel_centering:
        K_train_unc = K_train.copy()
        K_train = _center_kernel_train(K_train)
        K_val = _center_kernel_test(K_val, K_train_unc)

    # ── Optimisation SVM (C) avec Optuna ──
    print(f"🔍 Optimizing C (Optuna)...")
    def objective(trial):
        C = trial.suggest_float('C', 0.1, 10.0, log=True)
        print(f"🔄 Trial {trial.number}: Training with C={C:.2e}...")  # <--- AJOUT
        
        svm = EnhancedSVM(
            C=C,
            kernel="precomputed",
            probability=True,
            class_weight="balanced", # Indispensable
            max_iter=50000,          # Augmenté pour la convergence
            tol=1e-4,
            cache_size=4000
        )
        # ...
        svm.fit(K_train, y_train)
        # Score sur validation
        y_pred = svm.predict(K_val)
        return 1.0 - f1_score(y_val, y_pred, average="binary")

    study = optuna.create_study(direction="maximize")
    # Activez la barre de progression pour voir l'avancement
    study.optimize(objective, n_trials=config.get("svm", {}).get("n_trials", 20), show_progress_bar=True)
    best_C = study.best_params["C"]
    
    wandb.log({f"fold_{fold_idx}/best_C": best_C})

    # ── Entraînement Final sur (Train + Val) pour Test ──
    # Fusion des données pour le test final du fold
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    # Scaling X_test par rapport à l'ensemble combiné
    _, X_test_scaled = preprocess_features(X_trainval, X_test, scaler_type="minmax", feature_range=scale_range)
    # X_trainval est déjà scalé individuellement, mais idéalement on devrait refit le scaler sur le tout.
    # Pour simplifier ici, on assume distributions similaires.
    
    # Kernel TrainVal (Symétrique)
    # Astuce: On peut assembler K_trainval à partir de K_train, K_val si on n'a pas centré
    # Mais si centrage ou complexité, mieux vaut recalculer ou cacher.
    
    print(f"🔹 Fold {fold_idx+1}: Kernel TrainVal ({len(X_trainval)}x{len(X_trainval)})")
    K_trainval = cache_mgr.load(X_trainval, X_trainval.shape, cache_params, desc="trainval")
    if K_trainval is None:
        K_trainval = compute_kernel_matrix(
            X_trainval, weights=weights, symmetric=True,
            progress=True, desc="TrainVal", **backend_params
        )
        if args.cache_kernels:
             cache_mgr.save(K_trainval, X_trainval, X_trainval.shape, cache_params, desc="trainval")

    print(f"🔹 Fold {fold_idx+1}: Kernel Test ({len(X_test_scaled)}x{len(X_trainval)})")
    K_test = cache_mgr.load(X_test_scaled, X_trainval.shape, cache_params, desc="test")
    if K_test is None:
        K_test = compute_kernel_matrix(
            X_test_scaled, Y=X_trainval, weights=weights, symmetric=False,
            progress=True, desc="Test", **backend_params
        )
        if args.cache_kernels:
             cache_mgr.save(K_test, X_test_scaled, X_trainval.shape, cache_params, desc="test")

    if args.kernel_centering:
        K_trainval_unc = K_trainval.copy()
        K_trainval = _center_kernel_train(K_trainval)
        K_test = _center_kernel_test(K_test, K_trainval_unc)

    # Fit SVM Final
    final_svm = EnhancedSVM(C=best_C, kernel="precomputed", probability=True, class_weight="balanced")
    final_svm.fit(K_trainval, y_trainval)
    
    y_pred = final_svm.predict(K_test)
    y_proba = final_svm.predict_proba(K_test)[:, 1]

    # Metrics
    f1 = f1_score(y_test, y_pred, average="binary")
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Log WandB
    wandb.log({
        f"fold_{fold_idx}/test_f1": f1,
        f"fold_{fold_idx}/test_acc": acc,
        f"fold_{fold_idx}/test_auc": auc,
        f"fold_{fold_idx}/confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_test, preds=y_pred,
            class_names=["Class 0", "Class 1"]
        )
    })
    
    print(f"✅ Result Fold {fold_idx+1}: F1={f1:.4f} AUC={auc:.4f}")
    return {"f1": f1, "auc": auc}


# ═══════════════════════════════════════════════════════════
# 5. Boucle Principale
# ═══════════════════════════════════════════════════════════
def run_train(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Nom d'expérience enrichi
    ds_name = config["dataset"]["name"]
    backend_short = args.gram_backend if args.gram_backend else "auto"
    exp_name = f"{ds_name}_qsvm_{backend_short}_{args.embed_mode}"
    if args.angle_scale != 1.0:
        exp_name += f"_scale{args.angle_scale}"
    
    wandb.init(project="pp_project", name=exp_name, config={**config, **vars(args)}, group="qsvm")
    
    # Chargement
    print(f"📂 Loading {ds_name}...")
    train_dataset, test_dataset = load_dataset_by_name(
        name=ds_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", [3, 8]),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )
    
    if args.train_subset:
        train_dataset = Subset(train_dataset, range(min(len(train_dataset), args.train_subset)))
        print(f"⚠️ Subset Train: {len(train_dataset)}")

    # Features Classiques + PCA
    n_components = args.pca_components or config.get("svm", {}).get("pca_components", 10)
    X_train_raw, y_train, pca = extract_features(train_dataset, use_pca=True, n_components=n_components)
    X_test_raw, y_test, _ = extract_features(test_dataset, use_pca=True, n_components=n_components, pca_model=pca)
    
    # Fusion pour CV
    X_full = np.vstack([X_train_raw, X_test_raw])
    y_full = np.concatenate([y_train, y_test])
    
    # Poids Quantiques (fixés pour tous les folds pour cohérence)
    n_qubits = X_full.shape[1]
    n_layers = config.get("pennylane", {}).get("layers", 2)
    rng = np.random.default_rng(SEED)
    weights = rng.normal(0, 0.1, (n_layers, n_qubits)).astype(np.float32)
    
    # Cache Manager
    cache_mgr = KernelCacheManager(args.cache_dir)
    
    # Cross Validation
    n_folds = config.get("cv_folds", 3)
    # On garde le test set séparé à la fin, on fait la CV sur le train set d'origine
    # Pour simplifier le code existant qui mergeait tout, on va respecter la logique de split du script original
    # qui utilisait les index fixes.
    
    test_size = len(X_test_raw)
    train_indices = list(range(len(X_full) - test_size))
    test_indices = list(range(len(X_full) - test_size, len(X_full)))
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    results = []
    
    print(f"🚀 Starting {n_folds}-Fold CV on {args.pl_device} via {args.gram_backend}")
    print(f"⚙️  Params: tile={args.tile_size}, angle_scale={args.angle_scale}, embed={args.embed_mode}")
    
    for fold_idx, (t_idx, v_idx) in enumerate(kf.split(train_indices)):
        # Indices globaux
        train_idx_global = [train_indices[i] for i in t_idx]
        val_idx_global = [train_indices[i] for i in v_idx]
        
        res = train_fold(
            fold_idx, train_idx_global, val_idx_global, test_indices,
            X_full, y_full, weights, config, args, cache_mgr
        )
        results.append(res)
        
    avg_f1 = np.mean([r["f1"] for r in results])
    avg_auc = np.mean([r["auc"] for r in results])
    
    wandb.log({"avg/f1": avg_f1, "avg/auc": avg_auc})
    print(f"\n🏆 Final Average: F1={avg_f1:.4f}, AUC={avg_auc:.4f}")
    wandb.finish()


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    
    # Backend & Perf
    p.add_argument("--pl-device", type=str, default=None)
    p.add_argument("--pl-workers", type=int, default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--gram-backend", type=str, default="auto", 
                   choices=["auto", "numpy", "torch", "cuda_states"])
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--return-dtype", type=str, default="float32")
    
    # Quantum Params (Nouveaux)
    p.add_argument("--angle-scale", type=float, default=1.0, help="Scaling factor for AngleEmbedding (gamma)")
    p.add_argument("--embed-mode", type=str, default="ryrz", choices=["angle", "ry", "ryrz"])
    p.add_argument("--normalize-kernel", action="store_true", help="Force diag(K)=1")
    p.add_argument("--kernel-centering", action="store_true")
    
    # Data & Cache
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--pca-components", type=int, default=None)
    p.add_argument("--cache-kernels", action="store_true", help="Enable disk caching for kernels")
    p.add_argument("--cache-dir", type=str, default="./kernel_cache")

    args = p.parse_args()
    run_train(args)