import random, torch

class ReplayBuffer:
    def __init__(self, m_per_class=20):
        self.m_per_class = m_per_class
        self.data = {}  # {class_id: [(x,y), ...]}

    def add_batch(self, x, y):
        for xi, yi in zip(x, y):
            cid = int(yi)
            bucket = self.data.setdefault(cid, [])
            bucket.append((xi.cpu(), yi.cpu()))
            if len(bucket) > self.m_per_class:
                bucket.pop(0)

    def sample(self, n):
        all_pairs = [p for pairs in self.data.values() for p in pairs]
        if not all_pairs:
            return None, None
        batch = random.sample(all_pairs, min(n, len(all_pairs)))
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.tensor(ys)
