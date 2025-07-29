import torch

def get_scheduler(optimizer, scheduler_type=None):
    if scheduler_type is None:
        return None
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    else:
        return None
