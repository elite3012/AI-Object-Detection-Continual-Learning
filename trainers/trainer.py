import torch
from torch import nn, optim
from eval.metrics import accuracy
from eval.logger import log

def train_one_task(model, train_loader, test_loader, device="cuda", epochs=2, lr=0.1):
    """Train `model` on one task using a simple SGD loop.

    Returns the trained model.
    """
    model.to(device)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = accuracy(model, test_loader, device)
        log(f"Epoch {ep}: acc={acc:.4f}")
    return model
