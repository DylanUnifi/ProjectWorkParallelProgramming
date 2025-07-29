import os
import matplotlib.pyplot as plt

def save_plots(fold, loss_history, f1_history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(loss_history, label="Loss")
    plt.plot(f1_history, label="F1")
    plt.legend()
    plt.title(f"Fold {fold} - Loss/F1")
    plt.savefig(os.path.join(save_dir, f"fold_{fold}_loss_f1.png"))
    plt.close()
