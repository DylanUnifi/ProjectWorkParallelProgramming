import argparse, yaml, optuna
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

from data_loader.utils import load_dataset_by_name

optuna.logging.set_verbosity(optuna.logging.WARNING)

def extract_raw_features(dataset):
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
    all_features, all_labels = [], []
    for images, labels in tqdm(loader, desc="Extracting", leave=False):
        all_features.append(images.view(images.shape[0], -1).numpy())
        all_labels.append(labels.numpy())
    return np.vstack(all_features).astype(np.float32), np.concatenate(all_labels).astype(np.int64)

def run_train(args):
    with open(args.config, "r") as f: config = yaml.safe_load(f)
    ds_name = config["dataset"]["name"]
    
    print(f"📂 Loading {ds_name} (Classical SVM - {args.kernel.upper()} Kernel)...")
    train_dataset, test_dataset = load_dataset_by_name(
        name=ds_name,
        binary_classes=config.get("dataset", {}).get("binary_classes", [3, 8]),
        grayscale=config.get("dataset", {}).get("grayscale", True),
        root=config.get("dataset", {}).get("root", "./data"),
    )
    
    if args.train_subset:
        train_dataset = Subset(train_dataset, range(min(len(train_dataset), args.train_subset)))
        print(f"⚠️ Subset Train: {len(train_dataset)}")
        
    X_train_raw, y_train = extract_raw_features(train_dataset)
    X_test_raw, y_test = extract_raw_features(test_dataset)
    
    X_full = np.vstack([X_train_raw, X_test_raw])
    y_full = np.concatenate([y_train, y_test])
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    test_size = len(X_test_raw)
    train_indices = list(range(len(X_full) - test_size))
    test_indices = list(range(len(X_full) - test_size, len(X_full)))
    
    results = []
    print(f"🚀 Starting 3-Fold CV | PCA: {args.pca_components}")
    
    for fold_idx, (t_idx, v_idx) in enumerate(kf.split(train_indices)):
        train_idx_global = [train_indices[i] for i in t_idx]
        val_idx_global = [train_indices[i] for i in v_idx]
        
        X_tr, y_tr = X_full[train_idx_global], y_full[train_idx_global]
        X_va, y_va = X_full[val_idx_global], y_full[val_idx_global]
        X_te, y_te = X_full[test_indices], y_full[test_indices]
        
        if args.pca_components:
            pca = PCA(n_components=args.pca_components, random_state=42)
            X_tr = pca.fit_transform(X_tr)
            X_va = pca.transform(X_va)
            X_te = pca.transform(X_te)
            
        scaler = MinMaxScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te = scaler.transform(X_te)
        
        def objective(trial):
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
            gamma = trial.suggest_float('gamma', 1e-4, 1.0, log=True) if args.kernel != 'linear' else 'scale'
            svm = SVC(C=C, gamma=gamma, kernel=args.kernel, class_weight="balanced")
            svm.fit(X_tr, y_tr)
            return 1.0 - f1_score(y_va, svm.predict(X_va), average="binary")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)
        best_p = study.best_params
        
        X_trva = np.vstack([X_tr, X_va])
        y_trva = np.concatenate([y_tr, y_va])
        
        gamma_val = best_p.get('gamma', 'scale')
        final_svm = SVC(C=best_p['C'], gamma=gamma_val, kernel=args.kernel, probability=True, class_weight="balanced")
        final_svm.fit(X_trva, y_trva)
        
        y_pred = final_svm.predict(X_te)
        y_proba = final_svm.predict_proba(X_te)[:, 1]
        
        f1 = f1_score(y_te, y_pred, average="binary")
        auc = roc_auc_score(y_te, y_proba)
        
        print(f"🔹 Fold {fold_idx+1}: F1={f1:.4f}, AUC={auc:.4f} (C={best_p['C']:.2f})")
        results.append({"f1": f1, "auc": auc})
        
    print(f"\n🏆 AVERAGE {args.kernel.upper()} SVM | F1: {np.mean([r['f1'] for r in results]):.4f} | AUC: {np.mean([r['auc'] for r in results]):.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--train-subset", type=int, default=None)
    p.add_argument("--pca-components", type=int, default=16)
    p.add_argument("--kernel", type=str, default="rbf", choices=["rbf", "linear", "poly"])
    args = p.parse_args()
    run_train(args)
