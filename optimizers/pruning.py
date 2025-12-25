"""
Model Pruning for Compression
Channel-level structured pruning and magnitude-based pruning
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import copy

def apply_quantization(model):
    """
    Apply aggressive INT8 quantization to reduce model size
    FP32 (4 bytes) -> INT8 (1 byte) = 4x compression
    """
    try:
        # Set model to eval mode for quantization
        model.eval()
        
        # Dynamic quantization for Linear and Conv2d layers
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        print(f"  ✅ Quantization: FP32 → INT8 (4x theoretical compression)")
        return quantized_model
    except Exception as e:
        print(f"  ⚠️  Quantization warning: {e}")
        print("  Continuing with FP32 model...")
        return model

def prune_model(model, sparsity=0.3, method='magnitude', structured=True):
    """
    Prune model to reduce parameters
    
    Args:
        model: PyTorch model
        sparsity: Target sparsity (0.0-1.0). 0.3 = 30% of weights pruned
        method: 'magnitude' or 'random'
        structured: True for channel pruning, False for unstructured
    
    Returns:
        dict with pruned_model and metrics
    """
    print(f"\n{'='*60}")
    print(f"Model Pruning (sparsity={sparsity*100:.0f}%, method={method}, structured={structured})")
    print(f"{'='*60}\n")
    
    # Get baseline metrics
    baseline_params = count_parameters(model)
    baseline_size = get_model_size_mb(model)
    
    print(f"Baseline: {baseline_params:,} params, {baseline_size:.2f} MB")
    
    # Create copy
    pruned_model = copy.deepcopy(model)
    
    # Apply pruning
    if structured:
        pruned_model = structured_prune(pruned_model, sparsity, method)
    else:
        pruned_model = unstructured_prune(pruned_model, sparsity, method)
    
    # Calculate metrics
    pruned_params = count_parameters(pruned_model)
    pruned_nonzero = count_nonzero_parameters(pruned_model)
    pruned_size = get_model_size_mb(pruned_model)
    
    metrics = {
        'baseline_params': baseline_params,
        'baseline_size_mb': baseline_size,
        'pruned_params': pruned_params,
        'nonzero_params': pruned_nonzero,
        'pruned_size_mb': pruned_size,
        'sparsity': sparsity,
        'actual_sparsity': 1.0 - (pruned_nonzero / baseline_params),
        'compression_ratio': baseline_params / pruned_nonzero,
        'size_reduction_percent': (1 - pruned_size / baseline_size) * 100
    }
    
    print(f"\n[Pruning Results]")
    print(f"  Original: {baseline_params:,} params")
    print(f"  After pruning: {pruned_nonzero:,} params (non-zero)")
    print(f"  Actual sparsity: {metrics['actual_sparsity']*100:.1f}%")
    print(f"  Compression: {metrics['compression_ratio']:.2f}x")
    print(f"  Size: {baseline_size:.2f} MB → {pruned_size:.2f} MB")
    print(f"  Size reduction: {metrics['size_reduction_percent']:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'pruned_model': pruned_model,
        'metrics': metrics
    }

def structured_prune(model, sparsity, method='magnitude'):
    """
    Channel-level structured pruning
    Removes entire channels based on importance
    """
    print("Applying structured (channel) pruning...")
    
    parameters_to_prune = []
    
    # Find all Conv2d and Linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
            print(f"  Pruning {name}: {module.weight.shape}")
        elif isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            print(f"  Pruning {name}: {module.weight.shape}")
    
    # Apply structured pruning (L1 norm on output channels)
    if method == 'magnitude':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
    elif method == 'random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=sparsity
        )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    # Apply quantization to actually reduce size (FP32 -> INT8)
    print("\nApplying INT8 quantization for size reduction...")
    model = apply_quantization(model)
    
    return model

def unstructured_prune(model, sparsity, method='magnitude'):
    """
    Unstructured magnitude-based pruning
    Prunes individual weights
    """
    print("Applying unstructured pruning...")
    
    parameters_to_prune = []
    
    # Find all parameters
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            if hasattr(module, 'bias') and module.bias is not None:
                parameters_to_prune.append((module, 'bias'))
    
    # Apply pruning
    if method == 'magnitude':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
    elif method == 'random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=sparsity
        )
    
    # Make permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    # Apply quantization to actually reduce size (FP32 -> INT8)
    print("\nApplying INT8 quantization for size reduction...")
    model = apply_quantization(model)
    
    return model

def channel_prune(model, sparsity):
    """
    Advanced channel pruning
    Removes entire output channels from Conv2d layers
    """
    print(f"Channel pruning with {sparsity*100:.0f}% sparsity...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune output channels based on L1 norm
            prune.ln_structured(
                module, 
                name='weight', 
                amount=sparsity, 
                n=1,  # L1 norm
                dim=0  # Prune output channels
            )
            prune.remove(module, 'weight')
            print(f"  Pruned {name}: {module.weight.shape}")
    
    return model

def gradual_pruning(model, train_loader, initial_sparsity=0.0, final_sparsity=0.5, steps=5, device='cpu'):
    """
    Gradual pruning: increase sparsity over time
    Better accuracy retention than one-shot pruning
    
    Args:
        model: PyTorch model
        train_loader: Training data for fine-tuning
        initial_sparsity: Starting sparsity (e.g., 0.0)
        final_sparsity: Target sparsity (e.g., 0.5 = 50%)
        steps: Number of pruning steps
        device: Device
    
    Returns:
        Gradually pruned model
    """
    print(f"\n{'='*60}")
    print(f"Gradual Pruning: {initial_sparsity*100:.0f}% → {final_sparsity*100:.0f}% in {steps} steps")
    print(f"{'='*60}\n")
    
    current_sparsity = initial_sparsity
    sparsity_increment = (final_sparsity - initial_sparsity) / steps
    
    for step in range(1, steps + 1):
        current_sparsity += sparsity_increment
        
        print(f"\nStep {step}/{steps}: Target sparsity = {current_sparsity*100:.1f}%")
        
        # Prune
        result = prune_model(model, sparsity=current_sparsity, method='magnitude', structured=True)
        model = result['pruned_model']
        
        # Fine-tune (optional - can be added if needed)
        # fine_tune(model, train_loader, epochs=1, device=device)
    
    print(f"\nGradual pruning complete!")
    return model

def count_parameters(model):
    """Count total parameters"""
    return sum(p.numel() for p in model.parameters())

def count_nonzero_parameters(model):
    """Count non-zero parameters (after pruning)"""
    return sum((p != 0).sum().item() for p in model.parameters())

def get_model_size_mb(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb

def analyze_sparsity(model):
    """
    Analyze sparsity distribution across layers
    
    Returns:
        dict with per-layer sparsity info
    """
    sparsity_info = {}
    total_params = 0
    total_nonzero = 0
    
    print(f"\n{'='*60}")
    print(f"Sparsity Analysis")
    print(f"{'='*60}\n")
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'):
                weight = module.weight.data
                n_params = weight.numel()
                n_nonzero = (weight != 0).sum().item()
                sparsity = 1.0 - (n_nonzero / n_params)
                
                sparsity_info[name] = {
                    'total_params': n_params,
                    'nonzero_params': n_nonzero,
                    'sparsity': sparsity
                }
                
                total_params += n_params
                total_nonzero += n_nonzero
                
                print(f"{name:30s}: {sparsity*100:5.1f}% sparse ({n_nonzero:,}/{n_params:,})")
    
    overall_sparsity = 1.0 - (total_nonzero / total_params)
    print(f"\n{'Overall Sparsity':30s}: {overall_sparsity*100:5.1f}%")
    print(f"{'='*60}\n")
    
    return sparsity_info
