"""
Manual test script for HardwareOptimizedTrainer
Run this instead of running hardware_trainer.py directly
"""
import torch
import torch.nn as nn
from models.vision_transformer import LightweightViT
from trainers.hardware_trainer import HardwareOptimizedTrainer

def test_hardware_trainer():
    """Test hardware-optimized trainer"""
    print("Testing HardwareOptimizedTrainer...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = LightweightViT(num_classes=10, img_size=28, embed_dim=192, depth=6, num_heads=3).to(device)
    
    # Create trainer
    trainer = HardwareOptimizedTrainer(
        model=model,
        device=device,
        enable_quantization=True,
        quantization_dtype='float16',
        enable_pruning=True,
        target_sparsity=0.15,
        prune_per_task=False,
        track_efficiency=True
    )
    
    print(f"\n✅ Trainer created successfully!")
    print(f"  - Quantization: {'float16' if trainer.enable_quantization else 'Disabled'}")
    print(f"  - Pruning: {trainer.target_sparsity*100:.0f}% sparsity" if trainer.enable_pruning else "  - Pruning: Disabled")
    print(f"  - Efficiency tracking: {trainer.track_efficiency}")
    
    # Test training
    print(f"\n🚀 Starting training on Fashion-MNIST...")
    trainer.train_all_tasks(
        epochs_per_task=2,  # Short test
        batch_size=128,
        lr=0.001,
        data_root="./data"
    )
    
    # Get metrics
    metrics = trainer.get_metrics()
    print(f"\n📊 Results:")
    print(f"  - Average Accuracy: {metrics['average_accuracy']*100:.1f}%")
    print(f"  - Forgetting: {metrics['forgetting']*100:.1f}%")
    
    # Get efficiency summary
    if trainer.track_efficiency:
        summary = trainer.get_efficiency_summary()
        if summary:
            print(f"\n🚀 Hardware Optimization:")
            print(f"  - Size Reduction: {summary['total_size_reduction']:.1f}%")
            print(f"  - Speedup: {summary['total_speedup']:.2f}x")
            print(f"  - Final Model: {summary['final_size_mb']:.2f} MB")
            print(f"  - FPS: {summary['final_fps']:.0f}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_hardware_trainer()
