"""
Test script to verify model compression works correctly
"""

import torch
import os
from models.simple_cnn_multiclass import SimpleCNNMulticlass
from optimizers.pruning import prune_model

def get_checkpoint_size(checkpoint_path):
    """Get size of checkpoint file in MB"""
    size_bytes = os.path.getsize(checkpoint_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def test_compression():
    print("="*70)
    print("MODEL COMPRESSION TEST")
    print("="*70)
    
    # Create a simple model
    model = SimpleCNNMulticlass(num_classes=10)
    
    # Save original model
    torch.save(model.state_dict(), 'test_original.pt')
    original_size = get_checkpoint_size('test_original.pt')
    print(f"\n✅ Original Model: {original_size:.2f} MB")
    
    # Apply compression (50% sparsity for mobile)
    print("\n🔧 Applying compression (50% pruning + INT8 quantization)...")
    result = prune_model(model, sparsity=0.5, method='magnitude', structured=True)
    
    compressed_model = result['pruned_model']
    metrics = result['metrics']
    
    # Save compressed model
    torch.save(compressed_model.state_dict(), 'test_compressed.pt')
    compressed_size = get_checkpoint_size('test_compressed.pt')
    print(f"\n✅ Compressed Model: {compressed_size:.2f} MB")
    
    # Calculate actual compression
    actual_compression = original_size / compressed_size
    size_reduction = original_size - compressed_size
    size_reduction_percent = (size_reduction / original_size) * 100
    
    print("\n" + "="*70)
    print("COMPRESSION RESULTS")
    print("="*70)
    print(f"Original Size:      {original_size:.2f} MB")
    print(f"Compressed Size:    {compressed_size:.2f} MB")
    print(f"Size Reduction:     {size_reduction:.2f} MB ({size_reduction_percent:.1f}%)")
    print(f"Compression Ratio:  {actual_compression:.2f}x")
    print(f"\nParameter Metrics:")
    print(f"  Original params:  {metrics['baseline_params']:,}")
    print(f"  Non-zero params:  {metrics['nonzero_params']:,}")
    print(f"  Actual sparsity:  {metrics['actual_sparsity']*100:.1f}%")
    print("="*70)
    
    # Cleanup
    os.remove('test_original.pt')
    os.remove('test_compressed.pt')
    
    # Verify compression worked
    if actual_compression > 1.5:
        print(f"\n✅ COMPRESSION WORKS! Model size reduced by {actual_compression:.2f}x")
        return True
    else:
        print(f"\n❌ WARNING: Compression ratio too low. Expected >1.5x, got {actual_compression:.2f}x")
        return False

if __name__ == '__main__':
    test_compression()
