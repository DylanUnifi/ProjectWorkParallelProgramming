
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score
from tqdm import tqdm
import wandb
import optuna

from data_loader.utils import load_dataset_by_name
from models.hybrid_qcnn import HybridQCNNFeatures
from utils.logger import init_logger
from models.svm_extension import EnhancedSVM

# unified backend API
try:
    from models.quantum_kernel import compute_kernel_matrix
except Exception as e:
    raise ImportError("pipeline_backends.py not found. Place it in PYTHONPATH or project root.") from e

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in tqdm(loader, desc="Extracting features"):
            batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
            feats = model(batch_X).detach().cpu().numpy()
            features.append(feats)
            labels.append(batch_y.detach().cpu().numpy())
    return np.vstack(features), np.concatenate(labels)

def run_train(args):
    # Load config
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset"]["name"]
    EXPERIMENT_NAME = f"{dataset_name}_hybrid_qcnn_qkernel_{args.backend}"
    SAVE_DIR = os.path.join("engine/checkpoints", "hybrid_qcnn_quantumkernel", EXPERIMENT_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)

    wandb.init(project="qml_project", name=EXPERIMENT_NAME, config={**config, **vars(args)}, group=dataset_name)

    # Dataset
    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=config["training"]["batch_size"],
        binary_classes=config.get("binary_classes", [3, 8])
    )

    # (Option) Subsample for speed during dev
    if args.train_subset is not None:
        indices = torch.randperm(len(train_dataset))[:args.train_subset]
        train_dataset = Subset(train_dataset, indices)

    # KFold
    kfold = KFold(n_splits=config["training"]["kfold"], shuffle=True, random_state=42)

    # Feature extractor
    sample_X, _ = train_dataset[0]
    input_size = sample_X.numel()
    feature_extractor = HybridQCNNFeatures(input_size=input_size).to(DEVICE)

    # Pre-extract test embeddings once
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"])
    X_test, y_test = extract_features(feature_extractor, test_loader)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"[Fold {fold}] Running with backend={args.backend}, tile_size={args.tile_size}")
        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_subset,   batch_size=config["training"]["batch_size"])

        # Extract embeddings
        X_train, y_train = extract_features(feature_extractor, train_loader)
        X_val,   y_val   = extract_features(feature_extractor, val_loader)

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        X_test_s  = scaler.transform(X_test)

        # (Optional) move to GPU for torchcuda backend
        to_gpu = (args.backend == "torchcuda")
        if to_gpu:
            X_train_s_t = torch.from_numpy(X_train_s).cuda()
            X_val_s_t   = torch.from_numpy(X_val_s).cuda()
            X_test_s_t  = torch.from_numpy(X_test_s).cuda()
        else:
            X_train_s_t, X_val_s_t, X_test_s_t = X_train_s, X_val_s, X_test_s

        # Build kernels
        print("🔹 Building K_train ...")
        K_train = compute_kernel_matrix(X_train_s_t, backend=args.backend, tile_size=args.tile_size, symmetric=True)
        print("🔹 Building K_val ...")
        K_val   = compute_kernel_matrix(X_val_s_t,   Y=X_train_s_t, backend=args.backend, tile_size=args.tile_size, symmetric=False)

        # bring back to CPU numpy for sklearn
        if args.backend == "torchcuda":
            K_train = K_train.detach().cpu().numpy()
            K_val   = K_val.detach().cpu().numpy()

        # Optuna over C
        def objective(trial):
            C = trial.suggest_float("C", 1e-3, 1e3, log=True)
            svm = EnhancedSVM(C=C, kernel='precomputed', probability=True)
            svm.fit(K_train, y_train)
            y_pred = svm.predict(K_val)
            return 1.0 - f1_score(y_val, y_pred, average="weighted")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=config["svm"].get("n_trials", 20))
        best_C = study.best_params["C"]
        print(f"✅ Best C: {best_C}")

        # Final kernels
        X_trainval_s = np.vstack([X_train_s, X_val_s])
        y_trainval   = np.concatenate([y_train, y_val])
        if to_gpu:
            X_trainval_s_t = torch.from_numpy(X_trainval_s).cuda()
        else:
            X_trainval_s_t = X_trainval_s

        print("🔹 Building K_trainval ...")
        K_trainval = compute_kernel_matrix(X_trainval_s_t, backend=args.backend, tile_size=args.tile_size, symmetric=True)
        print("🔹 Building K_test ...")
        K_test     = compute_kernel_matrix(X_test_s_t, Y=X_trainval_s_t, backend=args.backend, tile_size=args.tile_size, symmetric=False)

        if args.backend == "torchcuda":
            K_trainval = K_trainval.detach().cpu().numpy()
            K_test     = K_test.detach().cpu().numpy()

        # Heatmap
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7,5))
        plt.imshow(K_trainval, cmap="viridis")
        plt.title(f"Quantum Kernel Matrix - Fold {fold}")
        plt.colorbar()
        heatmap_path = os.path.join(SAVE_DIR, f"kernel_heatmap_fold_{fold}.png")
        plt.savefig(heatmap_path)
        wandb.log({f"kernel_heatmap_fold_{fold}": wandb.Image(heatmap_path)})

        # Train SVM
        clf = EnhancedSVM(C=best_C, kernel='precomputed', probability=True)
        clf.fit(K_trainval, y_trainval)
        y_pred = clf.predict(K_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall    = recall_score(y_test, y_pred)
        bal_acc   = balanced_accuracy_score(y_test, y_pred)
        try:
            y_probs = clf.model.decision_function(K_test)
            auc = roc_auc_score(y_test, y_probs)
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
    p.add_argument("--backend", type=str, default="cpu",
                   choices=["cpu","numba","torchcuda","pycuda","openmp"])
    p.add_argument("--tile-size", type=int, default=128)
    p.add_argument("--train-subset", type=int, default=None, help="Subsample train for faster runs")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_train(args)
