"""
Test Phase 4: Hardware Optimization
Demonstrates quantization, pruning, and auto-optimization
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from models.vision_transformer import LightweightViT
from optimizers.quantization import quantize_model
from optimizers.pruning import prune_model
from optimizers.hardware_optimizer import AutoOptimizer, ModelProfiler
from optimizers.benchmark import benchmark_models, plot_optimization_comparison, create_efficiency_report


def load_fashion_mnist(batch_size=128):
    """Load Fashion-MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)
    
    return train_loader, test_loader


def train_baseline(model, train_loader, test_loader, device, epochs=3):
    """Quick training for baseline model"""
    print("\n[Training] Baseline model...")
    model.train()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100.0 * correct / total
        print(f"  Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.2f}%")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    test_acc = 100.0 * correct / total
    print(f"\n[Baseline] Test Accuracy: {test_acc:.2f}%\n")
    
    return model


def test_quantization(device='cpu'):
    """Test INT8 and FP16 quantization"""
    print("\n" + "="*80)
    print("TEST 1: QUANTIZATION")
    print("="*80)
    
    train_loader, test_loader = load_fashion_mnist()
    
    # Create and train baseline
    baseline_model = LightweightViT(num_classes=10, img_size=28, embed_dim=128, depth=4, num_heads=4)
    baseline_model = train_baseline(baseline_model, train_loader, test_loader, device, epochs=2)
    
    models = {'Baseline': baseline_model}
    
    # Test INT8 Quantization
    print("\n" + "-"*80)
    print("Testing INT8 Quantization")
    print("-"*80)
    
    int8_result = quantize_model(
        baseline_model,
        train_loader,
        dtype='qint8',
        qat_epochs=2,
        device=device
    )
    
    models['INT8 Quantized'] = int8_result['quantized_model']
    
    # Test FP16 Quantization (if GPU available)
    if device == 'cuda':
        print("\n" + "-"*80)
        print("Testing FP16 Quantization")
        print("-"*80)
        
        fp16_result = quantize_model(
            baseline_model,
            train_loader,
            dtype='float16',
            qat_epochs=2,
            device=device
        )
        
        models['FP16 Quantized'] = fp16_result['quantized_model']
    
    # Benchmark all models
    print("\n" + "="*80)
    print("QUANTIZATION BENCHMARK")
    print("="*80)
    
    results = benchmark_models(models, test_loader, device)
    
    # Plot comparison
    plot_optimization_comparison(results, save_path='docs/quantization_comparison.png')
    
    return models


def test_pruning(device='cpu'):
    """Test structured pruning"""
    print("\n" + "="*80)
    print("TEST 2: PRUNING")
    print("="*80)
    
    train_loader, test_loader = load_fashion_mnist()
    
    # Create and train baseline
    baseline_model = LightweightViT(num_classes=10, img_size=28, embed_dim=128, depth=4, num_heads=4)
    baseline_model = train_baseline(baseline_model, train_loader, test_loader, device, epochs=2)
    
    models = {'Baseline': baseline_model}
    
    # Test different sparsity levels
    for sparsity in [0.3, 0.5, 0.7]:
        print(f"\n" + "-"*80)
        print(f"Testing {sparsity*100:.0f}% Pruning")
        print("-"*80)
        
        def fine_tune_fn(model, s):
            """Quick fine-tuning"""
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            criterion = nn.CrossEntropyLoss()
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break  # Just 1 batch for speed
        
        prune_result = prune_model(
            baseline_model,
            target_sparsity=sparsity,
            method='channel',
            gradual=False,
            train_fn=fine_tune_fn
        )
        
        models[f'{sparsity*100:.0f}% Pruned'] = prune_result['pruned_model']
    
    # Benchmark all models
    print("\n" + "="*80)
    print("PRUNING BENCHMARK")
    print("="*80)
    
    results = benchmark_models(models, test_loader, device)
    
    # Plot comparison
    plot_optimization_comparison(results, save_path='docs/pruning_comparison.png')
    
    return models


def test_auto_optimizer(device='cpu'):
    """Test AutoOptimizer with different strategies"""
    print("\n" + "="*80)
    print("TEST 3: AUTO OPTIMIZER")
    print("="*80)
    
    train_loader, test_loader = load_fashion_mnist()
    
    # Create and train baseline
    baseline_model = LightweightViT(num_classes=10, img_size=28, embed_dim=128, depth=4, num_heads=4)
    baseline_model = train_baseline(baseline_model, train_loader, test_loader, device, epochs=2)
    
    # Test different optimization strategies
    for target_metric in ['balanced', 'speed', 'size']:
        print(f"\n" + "-"*80)
        print(f"Strategy: {target_metric.upper()}")
        print("-"*80)
        
        optimizer = AutoOptimizer(
            target_metric=target_metric,
            device=device
        )
        
        results = optimizer.optimize_model(
            baseline_model,
            train_loader,
            test_loader,
            auto_strategy=True
        )
        
        # Generate efficiency report
        create_efficiency_report(
            baseline_model,
            results['combined']['model'],
            test_loader,
            device,
            save_path=f'docs/efficiency_report_{target_metric}.md'
        )


def test_hardware_trainer(device='cpu'):
    """Test HardwareOptimizedTrainer"""
    print("\n" + "="*80)
    print("TEST 4: HARDWARE-OPTIMIZED CONTINUAL LEARNING")
    print("="*80)
    
    from trainers.hardware_trainer import HardwareOptimizedTrainer
    from data.fashion_mnist_true_continual import get_task_loaders
    
    # Create model
    model = LightweightViT(num_classes=10, img_size=28, embed_dim=128, depth=4, num_heads=4)
    
    # Create hardware-optimized trainer
    trainer = HardwareOptimizedTrainer(
        model=model,
        device=device,
        enable_quantization=True,
        quantization_dtype='float16' if device == 'cuda' else 'qint8',
        enable_pruning=True,
        target_sparsity=0.5,
        prune_per_task=True,
        track_efficiency=True
    )
    
    # Train on 3 tasks
    for task_id in range(3):
        print(f"\n{'='*80}")
        print(f"TASK {task_id}")
        print("="*80)
        
        train_loader, test_loader = get_task_loaders(task_id, batch_size=128, data_root='./data')
        
        trainer.train_task(
            task_id,
            train_loader,
            test_loader,
            epochs=5,
            lr=0.001,
            evaluate_every=2
        )
    
    # Print efficiency report
    trainer.print_efficiency_report()
    
    # Plot efficiency over tasks
    from optimizers.benchmark import plot_efficiency_over_tasks
    plot_efficiency_over_tasks(
        trainer.efficiency_history,
        save_path='docs/efficiency_over_tasks.png'
    )


def main():
    """Run all tests"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🚀 Testing Phase 4: Hardware Optimization")
    print(f"Device: {device}")
    print("="*80)
    
    try:
        # Test 1: Quantization
        test_quantization(device)
        
        # Test 2: Pruning
        test_pruning(device)
        
        # Test 3: Auto Optimizer
        test_auto_optimizer(device)
        
        # Test 4: Hardware-Optimized CL
        test_hardware_trainer(device)
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nGenerated files:")
        print("  - docs/quantization_comparison.png")
        print("  - docs/pruning_comparison.png")
        print("  - docs/efficiency_report_*.md")
        print("  - docs/efficiency_over_tasks.png")
        print("\nSee docs/PHASE4_HARDWARE_OPTIMIZATION.md for detailed documentation.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
