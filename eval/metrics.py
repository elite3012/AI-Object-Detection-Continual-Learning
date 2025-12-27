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

@torch.no_grad()
def per_class_accuracy(model, loader, device="cuda", num_classes=10):
    """
    Calculate per-class accuracy
    
    Args:
        model: Model to evaluate
        loader: DataLoader
        device: Device
        num_classes: Total number of classes
    
    Returns:
        dict: {class_id: accuracy} for each class present in loader
    """
    model.eval()
    
    # Track correct and total per class
    class_correct = {i: 0 for i in range(num_classes)}
    class_total = {i: 0 for i in range(num_classes)}
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        
        # Count per class
        for label, prediction in zip(y, pred):
            label_item = label.item()
            class_total[label_item] += 1
            if prediction == label:
                class_correct[label_item] += 1
    
    # Calculate accuracy per class
    class_accuracies = {}
    for class_id in range(num_classes):
        if class_total[class_id] > 0:
            class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_id] = 0.0
    
    return class_accuracies