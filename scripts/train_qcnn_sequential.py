# train_qcnn_sequential.py
# Entraînement Hybrid QCNN – Séquentiel (single process, baseline)

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm, trange
from benchmark.benchmark_logger import timeit, log_benchmark
import wandb

from models.hybrid_qcnn import HybridQCNNBinaryClassifier
from utils.checkpoint import save_checkpoint, safe_load_checkpoint
from utils.early_stopping import EarlyStopping
from utils.metrics import log_metrics
from data_loader.utils import load_dataset_by_name
from utils.scheduler import get_scheduler
from utils.visual import save_plots
from utils.logger import init_logger, write_log

@timeit
def run_train_hybrid_qcnn_sequential(config):
    # --- Variables d'expérimentation ---
    EXP_MODE = os.environ.get("EXP_MODE", None)
    EXP_VALUE = os.environ.get("EXP_VALUE", None)

    # Override taille dataset depuis env si present
    custom_dataset_size = os.environ.get("DATASET_SIZE", None)
    if custom_dataset_size is not None:
        custom_dataset_size = int(custom_dataset_size)
    else:
        custom_dataset_size = 1000
    if EXP_MODE:
        print(f"⚙️ Mode expérimental (SEQ): {EXP_MODE} = {EXP_VALUE}")

    dataset_name = config["dataset"]["name"]
    base_exp_name = config.get("experiment_name", "default_exp_seq")
    EXPERIMENT_NAME = f"{dataset_name}_{base_exp_name}_sequential_{EXP_MODE}_{EXP_VALUE}"
    SAVE_DIR = os.path.join("engine/checkpoints", "hybrid_qcnn", EXPERIMENT_NAME)
    CHECKPOINT_DIR = os.path.join(SAVE_DIR, "folds")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    wandb.init(
        project="pp_project_work_seq",
        name=EXPERIMENT_NAME,
        config=config
    )

    # DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    DEVICE = torch.device("cpu")
    print(f"📌 Device utilisé (SEQ) : {DEVICE}")

    BATCH_SIZE = config["training"]["batch_size"]
    EPOCHS = config["training"]["epochs"]
    LR = config["training"]["learning_rate"]
    WARMUP_EPOCHS = config["training"].get("warmup_epochs", 0)
    SCHEDULER_TYPE = config.get("scheduler", None)
    SCHEDULER_PARAMS = config.get("scheduler_params", {})
    KFOLD = config["training"]["kfold"]
    PATIENCE = config["training"]["early_stopping"]
    SCHEDULER_TYPE = config.get("scheduler", None)

    train_dataset, test_dataset = load_dataset_by_name(
        name=dataset_name,
        batch_size=BATCH_SIZE,
        binary_classes=config.get("binary_classes", [3, 8])
    )

    dataset_size = custom_dataset_size
    if EXP_MODE == "dataset_size" and EXP_VALUE:
        dataset_size = int(EXP_VALUE)

    indices = torch.randperm(len(train_dataset))[:dataset_size]
    train_dataset = Subset(train_dataset, indices)
    print(f"Nombre d'exemples chargés dans train_dataset (SEQ) : {len(train_dataset)}")

    if EXP_MODE == "batch_size" and EXP_VALUE:
        BATCH_SIZE = int(EXP_VALUE)

    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)
    labels = np.array([target for _, target in train_dataset])
    total_training_time = 0
    fold_f1_scores, fold_accuracies, fold_aucs, fold_bal_accs = [], [], [], []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        # Log distribution des classes dans chaque split :
        print(f"Fold {fold} - Train class dist:", np.unique(labels[train_idx], return_counts=True))
        print(f"Fold {fold} - Val class dist:", np.unique(labels[val_idx], return_counts=True))
        print(f"[Fold {fold}] Starting Hybrid QCNN training (Sequential)...")
        early_stopping = EarlyStopping(patience=PATIENCE)
        log_path, log_file = init_logger(os.path.join(SAVE_DIR, "logs"), fold)
        write_log(log_file, f"[Fold {fold}] Hybrid QCNN Training Log (Sequential)\n")
        train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(train_dataset, val_idx), batch_size=BATCH_SIZE)
        input_size = train_dataset[0][0].numel()
        model = HybridQCNNBinaryClassifier(
            input_size=input_size,
            parallel=False,
            device=DEVICE
        ).to(DEVICE)
        print(f"🚀 Mode utilisé : Séquentiel")
        wandb.log({"mode": "sequential"})
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Warmup scheduler
        if WARMUP_EPOCHS > 0:
            def lr_lambda(epoch):
                if epoch < WARMUP_EPOCHS:
                    return float(epoch + 1) / float(WARMUP_EPOCHS)
                else:
                    return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            scheduler_type = SCHEDULER_TYPE  # Sauvegarde pour l'après-warmup
        else:
            scheduler = None
            scheduler_type = SCHEDULER_TYPE

        # Après le warmup, StepLR prendra le relais
        step_scheduler = None
        if scheduler_type == "StepLR":
            step_size = SCHEDULER_PARAMS.get("step_size", 5)
            gamma = SCHEDULER_PARAMS.get("gamma", 0.5)
            step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == "CosineAnnealingLR":
            T_max = SCHEDULER_PARAMS.get("T_max", 20)
            eta_min = SCHEDULER_PARAMS.get("eta_min", 1e-5)
            step_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

        criterion = nn.BCELoss()
        start_epoch = 0
        try:
            model, optimizer, start_epoch = safe_load_checkpoint(model, optimizer, CHECKPOINT_DIR, fold)
            print(f"Resuming from epoch {start_epoch}")
        except FileNotFoundError:
            print("No checkpoint found, starting from scratch")
        loss_history, f1_history = [], []
        best_f1, best_epoch = 0, 0
        acc, auc, bal_acc = 0, 0, 0  # ← PATCH
        stopped_early = False
        start_time = time.time()

        for epoch in trange(start_epoch, EPOCHS, desc=f"[Fold {fold}] Hybrid QCNN Training (SEQ)"):
            model.train()
            total_loss = 0
            grad_norms = []
            for batch_idx, (batch_X, batch_y) in enumerate(tqdm(train_loader, desc=f"[Fold {fold}] Batches")):
                batch_X, batch_y = batch_X.view(batch_X.size(0), -1).to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_X)
                outputs = outputs.view(-1)
                batch_y = batch_y.view(-1)
                loss = criterion(outputs, batch_y.float())
                loss.backward()
                # Log la norme des gradients pour chaque batch
                total_norm = 0.
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                grad_norms.append(total_norm)
                wandb.log({"grad_norm": total_norm}, step=epoch * len(train_loader) + batch_idx)
                optimizer.step()
                total_loss += loss.item()
            # Optionnel : log moyenne par epoch
            mean_grad_norm = sum(grad_norms) / len(grad_norms)
            wandb.log({"mean_grad_norm": mean_grad_norm}, step=epoch)

            model.eval()
            y_true, y_pred, y_probs = [], [], []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.view(batch_X.size(0), -1).to(DEVICE)
                    preds_logits = model(batch_X).squeeze()
                    preds = (preds_logits >= 0.5).float()

                    def tolist_safe(x):
                        if isinstance(x, torch.Tensor):
                            return x.cpu().numpy().flatten().tolist()
                        if isinstance(x, float) or isinstance(x, int):
                            return [x]
                        return list(x)

                    y_true.extend(tolist_safe(batch_y))
                    y_pred.extend(tolist_safe(preds))
                    y_probs.extend(tolist_safe(preds_logits))
            acc, f1, precision, recall = log_metrics(y_true, y_pred)
            try:
                auc = roc_auc_score(y_true, y_probs)
            except ValueError:
                auc = 0.0
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            val_loss = total_loss / len(train_loader)
            wandb.log({
                "val/loss": val_loss, "val/f1": f1, "val/accuracy": acc,
                "val/precision": precision, "val/recall": recall,
                "val/balanced_accuracy": bal_acc, "val/auc": auc
            })
            write_log(log_file, f"[Epoch {epoch}] Loss: {val_loss:.4f} | F1: {f1:.4f} | Acc: {acc:.4f} | "
                                f"BalAcc: {bal_acc:.4f} | AUC: {auc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f}")
            loss_history.append(val_loss)
            f1_history.append(f1)
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                save_checkpoint(model, optimizer, epoch, CHECKPOINT_DIR, fold, best_f1)
                write_log(log_file, f"[Epoch {epoch}] New best F1: {f1:.4f} (Saved model)")
                wandb.run.summary[f"fold_{fold}/best_f1"] = best_f1
                wandb.run.summary[f"fold_{fold}/best_epoch"] = best_epoch
            if early_stopping(f1):
                print("Early stopping triggered.")
                write_log(log_file, f"Early stopping triggered at epoch {epoch}")
                stopped_early = True
                break
            if scheduler is not None and epoch < WARMUP_EPOCHS:
                scheduler.step()
            elif step_scheduler is not None:
                step_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            wandb.log({"learning_rate": current_lr}, step=epoch)

        if hasattr(model, "close_pool"):
            model.close_pool()
        end_time = time.time()
        duration = end_time - start_time
        total_training_time += duration
        fold_f1_scores.append(best_f1)
        fold_accuracies.append(acc)
        fold_aucs.append(auc)
        fold_bal_accs.append(bal_acc)
        save_plots(fold, loss_history, f1_history, os.path.join(SAVE_DIR, "plots"))
        write_log(log_file, f"\n[Fold {fold}] Best F1: {best_f1:.4f} at epoch {best_epoch}")
        log_benchmark(
            experiment="Hybrid QCNN",
            version=f"Sequential_{EXP_MODE}_{EXP_VALUE}",
            dataset=dataset_name,
            num_epochs=best_epoch,
            early_stop=stopped_early,
            training_time=duration,
            f1_score=best_f1,
            accuracy=acc,
            auc=auc,
            balanced_accuracy=bal_acc,
            csv_path="benchmark_results.csv"
        )
    wandb.finish()

if __name__ == "__main__":
    import yaml
    with open("../configs/config_train_qcnn_fashion.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_train_hybrid_qcnn_sequential(config)
