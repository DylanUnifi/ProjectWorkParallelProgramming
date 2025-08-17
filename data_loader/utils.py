
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
from torchvision.datasets import SVHN

def build_transform(grayscale=True, augment=False):
    """Crée une pipeline de transformations standardisée."""
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale())
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomCrop(28, padding=4)
        ])
    if grayscale:
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))  # 🔥 1 canal
    else:
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))  # 🔥 3 canaux

    return transforms.Compose(transform_list)


def relabel_subset(subset, targets, binary_classes):
    binary_targets = [1 if targets[i] == binary_classes[1] else 0 for i in subset.indices]
    for idx, label in zip(subset.indices, binary_targets):
        targets[idx] = label
    return subset

# Interface commune de chargement de datasets

def load_dataset_by_name(name, binary_classes=None, grayscale=True, root='./data'):
    """
    Charge le dataset spécifié et renvoie train_dataset, test_dataset.
    """
    if binary_classes is None:
        binary_classes = [3, 8]

    name = name.lower()

    if name == 'fashion_mnist':
        transform = build_transform(grayscale=True)
        train_set = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

        train_idx = [i for i, t in enumerate(train_set.targets) if t in binary_classes]
        test_idx = [i for i, t in enumerate(test_set.targets) if t in binary_classes]

        train_subset = relabel_subset(Subset(train_set, train_idx), train_set.targets, binary_classes)
        test_subset = relabel_subset(Subset(test_set, test_idx), test_set.targets, binary_classes)

        return train_subset, test_subset

    elif name == 'cifar10':
        transform = build_transform(grayscale=grayscale, augment=True)
        train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        train_idx = [i for i, t in enumerate(train_set.targets) if t in binary_classes]
        test_idx = [i for i, t in enumerate(test_set.targets) if t in binary_classes]

        train_subset = relabel_subset(Subset(train_set, train_idx), train_set.targets, binary_classes)
        test_subset = relabel_subset(Subset(test_set, test_idx), test_set.targets, binary_classes)

        return train_subset, test_subset

    elif name == 'svhn':
        transform = build_transform(grayscale=grayscale)
        train_set = SVHN(root=root, split='train', download=True, transform=transform)
        test_set = SVHN(root=root, split='test', download=True, transform=transform)

        def filter_and_process(dataset):
            X, y = [], []
            for idx in range(len(dataset)):
                img, label = dataset[idx]
                if label in binary_classes:
                    # tensor shape follows transform (1 or 3 channels)
                    X.append(img)
                    y.append(1 if label == binary_classes[1] else 0)
            return torch.utils.data.TensorDataset(torch.stack(X), torch.tensor(y, dtype=torch.float32))

        train_dataset = filter_and_process(train_set)
        test_dataset = filter_and_process(test_set)

        return train_dataset, test_dataset
