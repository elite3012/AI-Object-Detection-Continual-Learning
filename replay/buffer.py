import random

import torch


class ReplayBuffer:
    def __init__(self, samples_per_class=500, max_classes=10):
        """
        Fixed-budget replay buffer with automatic per-class rebalancing.

        The total memory budget is samples_per_class * max_classes. As new
        classes appear, the same fixed budget is redistributed across all
        learned classes.
        """
        self.samples_per_class = samples_per_class
        self.m_per_class = samples_per_class  # Backward-compatible alias.
        self.max_classes = max_classes
        self.total_size = samples_per_class * max_classes
        self.data = {}
        self.total_added = 0

    def _get_capacity_per_class(self):
        """Calculate current capacity per class based on learned classes."""
        num_classes = len(self.data)
        if num_classes == 0:
            return self.total_size
        return max(1, self.total_size // num_classes)

    def add_batch(self, x, y):
        for xi, yi in zip(x, y):
            class_id = int(yi)
            is_new_class = class_id not in self.data

            bucket = self.data.setdefault(class_id, [])
            bucket.append((xi.detach().cpu(), yi.detach().cpu()))
            self.total_added += 1

            if is_new_class:
                self._redistribute_buffer()
            else:
                capacity_per_class = self._get_capacity_per_class()
                if len(bucket) > capacity_per_class:
                    del bucket[: len(bucket) - capacity_per_class]

    def _redistribute_buffer(self):
        """Trim every class to the current balanced capacity."""
        capacity_per_class = self._get_capacity_per_class()
        for bucket in self.data.values():
            if len(bucket) > capacity_per_class:
                del bucket[: len(bucket) - capacity_per_class]

    def sample(self, n):
        if n <= 0:
            return None, None

        non_empty_buckets = {
            class_id: pairs for class_id, pairs in self.data.items() if pairs
        }
        if not non_empty_buckets:
            return None, None

        target_n = min(n, sum(len(pairs) for pairs in non_empty_buckets.values()))
        class_ids = list(non_empty_buckets)
        random.shuffle(class_ids)

        base_quota = target_n // len(class_ids)
        remainder = target_n % len(class_ids)
        batch = []

        for idx, class_id in enumerate(class_ids):
            quota = base_quota + (1 if idx < remainder else 0)
            if quota <= 0:
                continue
            pairs = non_empty_buckets[class_id]
            batch.extend(random.sample(pairs, min(quota, len(pairs))))

        if len(batch) < target_n:
            chosen_ids = {id(pair) for pair in batch}
            remaining = [
                pair
                for pairs in non_empty_buckets.values()
                for pair in pairs
                if id(pair) not in chosen_ids
            ]
            batch.extend(random.sample(remaining, min(target_n - len(batch), len(remaining))))

        if not batch:
            return None, None

        random.shuffle(batch)
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys).long()

    def get_statistics(self):
        """Return buffer statistics for analysis and UI charts."""
        total_classes = len(self.data)
        total_samples = sum(len(samples) for samples in self.data.values())
        per_class_count = {
            class_id: len(samples) for class_id, samples in sorted(self.data.items())
        }
        capacity_per_class = self._get_capacity_per_class()

        return {
            "total_classes": total_classes,
            "total_samples": total_samples,
            "total_added_lifetime": self.total_added,
            "capacity_per_class": capacity_per_class,
            "total_capacity": self.total_size,
            "per_class_count": per_class_count,
            "utilization": (total_samples / self.total_size) * 100
            if self.total_size > 0
            else 0,
        }

    def analyze_buffer(self, class_names=None):
        """Print a compact replay-buffer report."""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("REPLAY BUFFER ANALYSIS")
        print("=" * 70)

        print(f"\n[Buffer Configuration - Fixed Total: {self.total_size} samples]")
        print(f"  Current classes: {stats['total_classes']}")
        print(f"  Target per class: {stats['capacity_per_class']} samples")
        print(
            f"  Total stored: {stats['total_samples']}/{self.total_size} "
            f"({stats['utilization']:.1f}%)"
        )

        print("\n[Per-Class Distribution]")
        if class_names is None:
            class_names = {i: f"Class {i}" for i in range(self.max_classes)}

        for class_id, count in stats["per_class_count"].items():
            if isinstance(class_names, list):
                class_name = (
                    class_names[class_id]
                    if class_id < len(class_names)
                    else f"Class {class_id}"
                )
            else:
                class_name = class_names.get(class_id, f"Class {class_id}")
            print(f"  Class {class_id} ({class_name:15s}): {count} samples")

        print("\n[Buffer Impact]")
        if stats["total_samples"] > 0:
            avg_samples_per_class = stats["total_samples"] / stats["total_classes"]
            print(f"  Average samples/class: {avg_samples_per_class:.1f}")
            print(f"  Fixed memory budget: {self.total_size} samples")
            print("  Sampling policy: class-balanced replay batch")
        else:
            print("  Buffer is empty.")

        print("=" * 70 + "\n")
        return stats
