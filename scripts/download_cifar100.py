#!/usr/bin/env python3
"""Helper script to download CIFAR-100 into a specified root directory.

Usage:
  python scripts/download_cifar100.py --root ./data
  python scripts/download_cifar100.py --root C:\\Users\\Admin\\datasets\\cifar100

The script verifies the directory is writable and falls back to the user's cache dir if not.
"""
import argparse
import os
import sys
import tempfile
from torchvision import datasets


def ensure_writable_root(root):
    try:
        os.makedirs(root, exist_ok=True)
    except Exception:
        # fall back to user cache
        home = os.path.expanduser('~')
        fallback = os.path.join(home, '.cache', 'cifar100')
        os.makedirs(fallback, exist_ok=True)
        print(f"[WARN] Could not create '{root}'. Falling back to '{fallback}'")
        root = fallback

    # test writability
    try:
        fd, tmp_path = tempfile.mkstemp(dir=root)
        os.close(fd)
        os.remove(tmp_path)
    except Exception:
        home = os.path.expanduser('~')
        fallback = os.path.join(home, '.cache', 'cifar100')
        os.makedirs(fallback, exist_ok=True)
        print(f"[WARN] '{root}' is not writable. Using fallback '{fallback}'")
        root = fallback

    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='./data', help='Root dir to download CIFAR-100 into')
    p.add_argument('--train-only', action='store_true', help='Only download the train set')
    args = p.parse_args()

    root = ensure_writable_root(args.root)
    print(f"Downloading CIFAR-100 into: {root}")
    datasets.CIFAR100(root=root, train=True, download=True)
    if not args.train_only:
        datasets.CIFAR100(root=root, train=False, download=True)
    print('Done')


if __name__ == '__main__':
    main()
