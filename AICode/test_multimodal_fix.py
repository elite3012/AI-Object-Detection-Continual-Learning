"""Test multimodal after fixing label leakage bug"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.multimodal_clip import create_multimodal_model
from trainers.multimodal_trainer import MultiModalContinualTrainer

def test_quick():
    """Quick test on Fashion-MNIST task 0"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load Fashion-MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split into tasks (5 tasks, 2 classes each)
    def get_task_data(dataset, task_id):
        classes = [task_id*2, task_id*2+1]
        indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
        return torch.utils.data.Subset(dataset, indices)
    
    # Test on Task 0 (classes 0-1)
    task_0_train = get_task_data(train_dataset, 0)
    task_0_test = get_task_data(test_dataset, 0)
    
    train_loader = torch.utils.data.DataLoader(task_0_train, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(task_0_test, batch_size=512, shuffle=False)
    
    print("=" * 70)
    print("TESTING CROSS-MODAL FUSION (Should NOT get 100% accuracy anymore)")
    print("=" * 70)
    
    # Create Fusion model
    model = create_multimodal_model(
        model_type='fusion',
        num_classes=10,
        embed_dim=192
    ).to(device)
    
    # Train for 2 epochs
    trainer = MultiModalContinualTrainer(
        model=model,
        device=str(device),
        buffer_size=500,
        use_amp=True
    )
    
    print("\nTraining for 2 epochs...")
    print("If still 100% acc + loss→0: Bug still exists")
    print("If ~85-90% acc + loss~0.3: Bug FIXED!\n")
    
    trainer._train_one_task(
        train_loader,
        epochs=2,
        lr=0.001
    )
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    print("\n" + "=" * 70)
    print("FINAL EVALUATION (Vision-only, no text cheating)")
    print("=" * 70)
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            # Generic dummy text (no label info)
            batch_size = images.size(0)
            dummy_tokens = torch.zeros(batch_size, 32, dtype=torch.long).to(device)
            dummy_tokens[:, 0] = 2  # START
            dummy_tokens[:, 1] = 4  # 'a'
            dummy_tokens[:, 2] = 18  # 'clothing'
            dummy_tokens[:, 3] = 3  # END
            
            logits = model(images, dummy_tokens)
            preds = logits.argmax(dim=1).cpu()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    acc = 100.0 * correct / total
    print(f"\nTask 0 Test Accuracy: {acc:.2f}%")
    
    if acc > 95:
        print("⚠️  STILL TOO HIGH! Bug may persist")
    else:
        print("✅ REALISTIC! Bug fixed")

if __name__ == '__main__':
    test_quick()
