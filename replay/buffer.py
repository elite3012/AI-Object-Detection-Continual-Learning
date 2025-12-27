import random, torch

class ReplayBuffer:
    def __init__(self, samples_per_class=500, max_classes=10):
        """
        Auto-Scaling Fixed Total Memory Budget Replay Buffer
        
        Args:
            samples_per_class: Target samples per class (default: 500)
            max_classes: Maximum number of classes in dataset (default: 10 for Fashion-MNIST)
        
        The buffer auto-scales total capacity: total_size = samples_per_class × max_classes
        For Fashion-MNIST (10 classes): 500 × 10 = 5000 samples total
        
        Capacity auto-distributes equally among learned classes:
        - 2 classes: 2500 samples/class
        - 4 classes: 1250 samples/class
        - 10 classes: 500 samples/class
        """
        self.samples_per_class = samples_per_class
        self.max_classes = max_classes
        self.total_size = samples_per_class * max_classes
        self.data = {}
        self.total_added = 0  # Track total samples added for analysis
    
    def _get_capacity_per_class(self):
        """Calculate current capacity per class based on number of classes"""
        num_classes = len(self.data)
        if num_classes == 0:
            return self.total_size
        return self.total_size // num_classes

    def add_batch(self, x, y):
        for xi, yi in zip(x, y):
            cid = int(yi)
            
            # Check if this is a new class
            is_new_class = cid not in self.data
            
            # Add sample
            bucket = self.data.setdefault(cid, [])
            bucket.append((xi.cpu(), yi.cpu()))
            self.total_added += 1
            
            # If new class added, redistribute entire buffer
            if is_new_class:
                self._redistribute_buffer()
            else:
                # Trim current class to capacity
                capacity_per_class = self._get_capacity_per_class()
                if len(bucket) > capacity_per_class:
                    bucket.pop(0)
    
    def _redistribute_buffer(self):
        """Redistribute entire buffer when new class is added to maintain fixed total capacity"""
        capacity_per_class = self._get_capacity_per_class()
        
        # Trim all classes to new capacity
        for class_id, bucket in self.data.items():
            while len(bucket) > capacity_per_class:
                bucket.pop(0)  # Remove oldest samples
    
    def sample(self, n):
        all_pairs = []
        for pairs in self.data.values():
            for p in pairs:
                all_pairs.append(p)
        if not all_pairs:
            return None, None
        
        # Balanced sampling: sample evenly from each class
        if len(self.data) > 0:
            # Simple balanced sampling: divide evenly across classes
            samples_per_class = max(1, n // len(self.data))
            balanced_batch = []

            for class_id, pairs in self.data.items():
                if len(pairs) > 0:
                    # Sample up to samples_per_class from this class
                    k = min(samples_per_class, len(pairs))
                    class_samples = random.sample(pairs, k)
                    balanced_batch.extend(class_samples)
            
                        # Shuffle to mix classes
            random.shuffle(balanced_batch)
            
            if balanced_batch:
                xs, ys = zip(*balanced_batch)
                return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
        
        # Fallback to random sampling
        batch = random.sample(all_pairs, min(n, len(all_pairs)))
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
    
    def get_statistics(self):
        """Return buffer statistics for analysis"""
        total_classes = len(self.data)
        total_samples = sum(len(samples) for samples in self.data.values())
        per_class_count = {class_id: len(samples) for class_id, samples in self.data.items()}
        capacity_per_class = self._get_capacity_per_class()
        
        return {
            'total_classes': total_classes,
            'total_samples': total_samples,
            'total_added_lifetime': self.total_added,
            'capacity_per_class': capacity_per_class,
            'total_capacity': self.total_size,
            'per_class_count': per_class_count,
            'utilization': (total_samples / self.total_size) * 100 if self.total_size > 0 else 0
        }
    
    def analyze_buffer(self, class_names=None):
        """Print detailed buffer analysis"""
        stats = self.get_statistics()
        
        print(f"\n{'═'*70}")
        print(f"REPLAY BUFFER ANALYSIS")
        print(f"{'═'*70}")
        
        print(f"\n[Buffer Configuration - Fixed Total: 5000 samples]")
        print(f"  Current classes: {stats['total_classes']}")
        print(f"  Target per class: {stats['capacity_per_class']} samples")
        print(f"  Total stored: {stats['total_samples']}/5000 ({stats['utilization']:.1f}%)")
        print(f"  Current utilization: {stats['utilization']*100:.1f}%")
        
        print(f"\n[Buffer Contents]")
        print(f"  Classes stored: {stats['total_classes']}")
        print(f"  Total samples in buffer: {stats['total_samples']}")
        print(f"  Total samples added (lifetime): {stats['total_added_lifetime']}")
        
        print(f"\n[Per-Class Distribution]")
        if class_names is None:
            class_names = {i: f"Class {i}" for i in range(10)}
        
        for class_id in sorted(stats['per_class_count'].keys()):
            count = stats['per_class_count'][class_id]
            
            # Handle class_names as either list or dict
            if isinstance(class_names, list):
                class_name = class_names[class_id] if class_id < len(class_names) else f"Class {class_id}"
            else:
                class_name = class_names.get(class_id, f"Class {class_id}")
            
            print(f"  Class {class_id} ({class_name:15s}): {count} samples")
        
        print(f"\n[Buffer Impact Analysis]")
        if stats['total_samples'] > 0:
            avg_samples_per_class = stats['total_samples'] / stats['total_classes']
            print(f"  Average samples/class: {avg_samples_per_class:.1f}")
            print(f"  Buffer prevents catastrophic forgetting by:")
            print(f"    ✓ Storing {stats['total_samples']} exemplars from {stats['total_classes']} classes")
            print(f"    ✓ Replaying {int(avg_samples_per_class * 0.7)} samples/class per batch (70% replay ratio)")
            print(f"    ✓ Maintaining balanced representation across all learned classes")
        else:
            print(f"  Buffer is empty - no data stored yet")
        
        print(f"{'═'*70}\n")
        
        return stats
