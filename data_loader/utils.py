# data_loader/utils.py
from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
from torchvision.datasets import SVHN


def build_transform(grayscale: bool = True, augment: bool = False):
    """Standardized transformation pipeline."""
    tfms = []
    if grayscale:
        tfms.append(transforms.Grayscale())

    if augment:
        tfms.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(28, padding=4),
        ])

    tfms.append(transforms.ToTensor())
    if grayscale:
        tfms.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # 1 channel
    else:
        tfms.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]))     # 3 channels

    return transforms.Compose(tfms)


def relabel_subset(subset: Subset, targets, binary_classes):
    """
    Map the second class (binary_classes[1]) to 1, the other to 0 in `targets` in-place.
    """
    binary_targets = [1 if int(targets[i]) == int(binary_classes[1]) else 0 for i in subset.indices]
    for idx, label in zip(subset.indices, binary_targets):
        targets[idx] = int(label)
    return subset


def load_dataset_by_name(name, binary_classes=None, grayscale=True, root='./data'):
    """
    Load (train_dataset, test_dataset) filtered on `binary_classes` and binary labels {0,1}.
    - fashion_mnist : Subset of torchvision.FashionMNIST
    - cifar10       : Subset of torchvision.CIFAR10
    - svhn          : TensorDataset (X, y) with y in int64 (torch.long)
    """
    if binary_classes is None:
        binary_classes = [3, 8]

    name = str(name).lower()

    if name == 'fashion_mnist':
        transform = build_transform(grayscale=True)
        train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set  = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

        train_idx = [i for i, t in enumerate(train_set.targets) if int(t) in binary_classes]
        test_idx  = [i for i, t in enumerate(test_set.targets)  if int(t) in binary_classes]

        train_subset = relabel_subset(Subset(train_set, train_idx), train_set.targets, binary_classes)
        test_subset  = relabel_subset(Subset(test_set,  test_idx),  test_set.targets,  binary_classes)
        return train_subset, test_subset

    elif name == 'cifar10':
        transform = build_transform(grayscale=grayscale, augment=True)
        train_set = datasets.CIFAR10(root=root, train=True,  download=True, transform=transform)
        test_set  = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        train_idx = [i for i, t in enumerate(train_set.targets) if int(t) in binary_classes]
        test_idx  = [i for i, t in enumerate(test_set.targets)  if int(t) in binary_classes]

        train_subset = relabel_subset(Subset(train_set, train_idx), train_set.targets, binary_classes)
        test_subset  = relabel_subset(Subset(test_set,  test_idx),  test_set.targets,  binary_classes)
        return train_subset, test_subset

    elif name == 'svhn':
        # SVHN returns labels in {0..9}, here we filter, and return y in torch.long
        transform = build_transform(grayscale=grayscale, augment=False)
        train_set = SVHN(root=root, split='train', download=True, transform=transform)
        test_set  = SVHN(root=root, split='test',  download=True, transform=transform)

        def filter_and_process(dataset):
            X, y = [], []
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                label = int(label)
                if label in binary_classes:
                    X.append(img)
                    y.append(1 if label == int(binary_classes[1]) else 0)
            X = torch.stack(X) if len(X) else torch.empty(0)
            y = torch.tensor(y, dtype=torch.long)  # int64 for sklearn
            return torch.utils.data.TensorDataset(X, y)

        train_dataset = filter_and_process(train_set)
        test_dataset  = filter_and_process(test_set)
        return train_dataset, test_dataset

    else:
        raise ValueError(f"Dataset inconnu: {name}")
