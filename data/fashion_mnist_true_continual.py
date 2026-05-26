import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Task Split
TASKS = [
    [0, 1],  # T-shirt/top, Trouser
    [2, 3],  # Pullover, Dress
    [4, 5],  # Coat, Sandal
    [6, 7],  # Shirt, Sneaker
    [8, 9],  # Bag, Ankle boot
]

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

TASK_THEMES = {
    0: "T-shirt & Trouser",
    1: "Pullover & Dress", 
    2: "Coat & Sandal",
    3: "Shirt & Sneaker",
    4: "Bag & Ankle boot"
}

def get_transforms():
    """Training transforms for the current task split."""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def get_eval_transforms():
    """Evaluation transforms without random augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def _indices_for_classes(dataset, class_ids):
    """Return dataset indices whose labels are in class_ids without loading images."""
    targets = torch.as_tensor(dataset.targets)
    class_ids = torch.as_tensor(class_ids)
    mask = torch.isin(targets, class_ids)
    return mask.nonzero(as_tuple=False).flatten().tolist()

def get_task_loaders_true_continual(
    task_id,
    batch_size=128,
    root="./data",
    train_ratio=0.7,
    seed=42,
    verbose=True,
):
    """
    Build loaders for one true-continual Fashion-MNIST task.

    The training loader contains only the current task's new classes.
    Replay of previous classes is handled by the trainer's replay buffer.

    Args:
        task_id: 0-4
        batch_size: batch size
        root: data directory
        train_ratio: Ratio for train/val split (0.7 = 70% train, 30% val)
        seed: deterministic split seed
        verbose: print split summary

    Returns:
        train_loader, val_loader, test_loader, class_ids
    """
    if not 0 <= task_id < len(TASKS):
        raise ValueError(f"task_id must be 0-{len(TASKS) - 1}, got {task_id}")
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    os.makedirs(root, exist_ok=True)

    tf_train = get_transforms()
    tf_eval = get_eval_transforms()

    train_full = datasets.FashionMNIST(
        root,
        train=True,
        transform=tf_train,
        download=True,
    )
    val_full = datasets.FashionMNIST(
        root,
        train=True,
        transform=tf_eval,
        download=True,
    )
    test_full = datasets.FashionMNIST(
        root,
        train=False,
        transform=tf_eval,
        download=True,
    )
    
    current_task_classes = TASKS[task_id]
    train_indices = _indices_for_classes(train_full, current_task_classes)
    
    rng = random.Random(seed + task_id)
    rng.shuffle(train_indices)
    split_point = int(len(train_indices) * train_ratio)
    train_indices_split = train_indices[:split_point]
    val_indices_split = train_indices[split_point:]
    
    test_indices = _indices_for_classes(test_full, current_task_classes)

    train_subset = Subset(train_full, train_indices_split)
    val_subset = Subset(val_full, val_indices_split)
    test_subset = Subset(test_full, test_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    if verbose:
        print(
            f"[Data Split] Task {task_id}: "
            f"Train={len(train_indices_split)}, "
            f"Val={len(val_indices_split)}, "
            f"Test={len(test_indices)} samples"
        )
        print(f"[Classes] Task {task_id}: training on new classes {current_task_classes}")
    
    return train_loader, val_loader, test_loader, current_task_classes
