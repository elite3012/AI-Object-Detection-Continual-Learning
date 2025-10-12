import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 5 nhiệm vụ, mỗi nhiệm vụ 20 lớp (0-19, 20-39, ...)
TASKS = [list(range(i*20, (i+1)*20)) for i in range(5)]

def get_transforms():
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])

def _indices_for_classes(dataset, class_ids):
    return [i for i, (_, y) in enumerate(dataset) if y in class_ids]

def get_task_loaders(task_id, batch_size=128, root="./data"):
    assert 0 <= task_id < 5
    tf = get_transforms()
    # Ensure the root directory exists so torchvision can write downloaded archives.
    try:
        os.makedirs(root, exist_ok=True)
    except Exception:
        # Fall back to a user-writable directory (home/.cache/cifar100)
        home = os.path.expanduser('~')
        fallback = os.path.join(home, '.cache', 'cifar100')
        try:
            os.makedirs(fallback, exist_ok=True)
            print(f"[WARN] Could not create '{root}'. Falling back to '{fallback}' for dataset downloads.")
            root = fallback
        except Exception:
            raise

    # Verify `root` is writable: try creating a temporary file inside it. If that fails,
    # fall back to a user cache directory where writes are allowed.
    try:
        import tempfile
        fd, tmp_path = tempfile.mkstemp(dir=root)
        os.close(fd)
        os.remove(tmp_path)
    except Exception:
        home = os.path.expanduser('~')
        fallback = os.path.join(home, '.cache', 'cifar100')
        os.makedirs(fallback, exist_ok=True)
        if os.path.abspath(fallback) != os.path.abspath(root):
            print(f"[WARN] '{root}' is not writable. Using fallback '{fallback}' for dataset downloads.")
        root = fallback
    train_set = datasets.CIFAR100(root=root, train=True, download=True, transform=tf)
    test_set  = datasets.CIFAR100(root=root, train=False, download=True, transform=tf)

    classes = TASKS[task_id]
    train_idx = _indices_for_classes(train_set, classes)
    test_idx  = _indices_for_classes(test_set, classes)

    train_sub = Subset(train_set, train_idx)
    test_sub  = Subset(test_set,  test_idx)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader, classes
