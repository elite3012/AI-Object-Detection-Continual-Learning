"""
Fashion-MNIST TRUE Continual Learning
Train ONLY on new classes per task, use ER/EWC to remember old ones
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Same task split
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
    """Data augmentation for Fashion-MNIST"""
    return transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

def _indices_for_classes(dataset, class_ids):
    """Get indices for specific classes"""
    return [i for i, (_, y) in enumerate(dataset) if y in class_ids]

def get_task_loaders_true_continual(task_id, batch_size=128, root="./data"):
    """
    TRUE Continual Learning: Train ONLY on current task's new classes
    Use ER buffer to rehearse old classes during training
    
    Args:
        task_id: 0-4
        batch_size: batch size
        root: data directory
        
    Returns:
        train_loader, test_loader, class_ids
    """
    assert 0 <= task_id < 5, f"task_id must be 0-4, got {task_id}"
    
    os.makedirs(root, exist_ok=True)
    
    tf_train = get_transforms()
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load full datasets
    train_full = datasets.FashionMNIST(root, train=True, transform=tf_train, download=True)
    test_full = datasets.FashionMNIST(root, train=False, transform=tf_test, download=True)
    
    # Get ONLY current task's classes (TRUE continual learning)
    current_task_classes = TASKS[task_id]
    
    # Filter: ONLY current task for both train and test
    train_indices = _indices_for_classes(train_full, current_task_classes)
    test_indices = _indices_for_classes(test_full, current_task_classes)
    
    train_subset = Subset(train_full, train_indices)
    test_subset = Subset(test_full, test_indices)
    
    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"[TRUE CL] Task {task_id}: Training ONLY on NEW classes {current_task_classes}")
    
    return train_loader, test_loader, current_task_classes
