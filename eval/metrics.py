import torch

@torch.no_grad()
def accuracy(model, loader, device = "cuda", debug = False):
    model.eval()
    correct = total = 0

    if debug:
        all_preds = []
        all_labels = []

    # Evaluate on full test set for accurate metrics
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()

        if debug:
            all_preds.extend(pred.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    
    if debug:
        print(f"[DEBUG] Sample predictions: {all_preds[:10]}")
        print(f"[DEBUG] Sample labels: {all_labels[:10]}")
        print(f"[DEBUG] Unique predictions: {set(all_preds)}")
        print(f"[DEBUG] Unique labels: {set(all_labels)}")
    
    return correct / max(total, 1)
   