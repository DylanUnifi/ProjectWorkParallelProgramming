import os
import torch

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, fold, best_score):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"best_fold_{fold}.pt")
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_score': best_score
    }, path)

def safe_load_checkpoint(model, optimizer, checkpoint_dir, fold):
    path = os.path.join(checkpoint_dir, f"best_fold_{fold}.pt")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    return model, optimizer, start_epoch
