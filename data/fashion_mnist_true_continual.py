import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Task Split
TASKS = [
    [0, 1],  # T-shirt/top, Trouser
    [2, 3],  # Pullover, Dress
    [4, 5],  # Coat, Sandal
    [6, 7],  # Shirt, Sneaker
    [8, 9],  # Bag, Ankle boot
]

CLASS_NAMES= [
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
    return transforms.Compose([ # wrapper chains multiple transformations together so they execute sequentially on each image.
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees = 0, translate = (0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def _indices_for_classes(dataset, class_ids): # filter and return the location index of the specific classes in the dataset
    result = []
    for i, (_, y) in enumerate(dataset): # _ is placeholder for image, we dont use image so the leave the _ there
        if y in class_ids:
            result.append(i)
    return result

def get_task_loaders_true_continual(task_id, batch_size = 128, root = "./data", train_ratio=0.7):
    """
    Train only in current task's new classes
    Use ER buffer to rehearse old classes during training

    Args:
        task_id: 0-4
        batch_size: batch size
        root: data directory
        train_ratio: Ratio for train/val split (0.7 = 70% train, 30% val)

    Returns:
        train_loader, val_loader, test_loader, class_ids
    """
    assert 0 <= task_id < 5, f"task_id must be 0-4, got {task_id}" #  validates the task_id parameter before any processing begins.
    
    os.makedirs(root, exist_ok = True)

    tf_train = get_transforms()
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full datasets
    train_full = datasets.FashionMNIST(root, train = True, 
                                       transform = tf_train, 
                                       download = True)
    test_full = datasets.FashionMNIST(root,
                                      transform = tf_test,
                                      download = True)
    
    # Get ONLY current task's classes (Continual Learning)
    current_task_classes = TASKS[task_id]

    # Filter: ONLY current task for train (split into train/val)
    train_indices = _indices_for_classes(train_full, current_task_classes)
    
    # Split train into train (70%) and validation (30%)
    import random
    random.shuffle(train_indices)
    split_point = int(len(train_indices) * train_ratio)
    train_indices_split = train_indices[:split_point]
    val_indices_split = train_indices[split_point:]
    
    # Filter: ONLY current task for test
    test_indices = _indices_for_classes(test_full, current_task_classes)

    train_subset = Subset(train_full, train_indices_split)
    val_subset = Subset(train_full, val_indices_split)
    test_subset = Subset(test_full, test_indices)

    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
        pin_memory = False  # Disable to prevent CUDA deadlock on Windows
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
        pin_memory = False  # Disable to prevent CUDA deadlock on Windows
    )

    test_loader = DataLoader(
        test_subset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
        pin_memory = False  # Disable to prevent CUDA deadlock on Windows
    )

    print(f"[Data Split] Task {task_id}: Train={len(train_indices_split)}, Val={len(val_indices_split)}, Test={len(test_indices)} samples")
    print(f"[Classes] Task {task_id}: Training on new classes {current_task_classes}")
    
    return train_loader, val_loader, test_loader, current_task_classes