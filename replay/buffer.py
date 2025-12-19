import random, torch

class ReplayBuffer:
    def __init__(self, m_per_class = 200):
        self.m_per_class = m_per_class
        self.data = {}

    def add_batch(self, x, y):
        for xi, yi in zip(x, y):
            cid = int(yi)
            bucket = self.data.setdefault(cid, [])
            bucket.append((xi.cpu(), yi.cpu()))
            if len(bucket) > self.m_per_class:
                bucket.pop(0)
    
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
