import torch

class EWC:
    def __init__(self):
        self.old_params = {}
        self.fisher = {}

    def store_old(self, model):
        self.old_params = {n: p.detach().clone() for n,p in model.named_parameters()}

    def compute_fisher(self, model, data_loader, device="cuda"):
        # Ngày 4–5: bổ sung tính Fisher (để tạm pass)
        self.fisher = {n: torch.zeros_like(p) for n,p in model.named_parameters()}

    def penalty(self, model, lam=50.0):
        reg = torch.tensor(0.0, device=next(model.parameters()).device)
        for n,p in model.named_parameters():
            if n in self.fisher:
                reg = reg + (self.fisher[n] * (p - self.old_params[n])**2).sum()
        return lam * reg
