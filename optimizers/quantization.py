"""
Quantization for Model Compression
INT8 and FP16 quantization with QAT (Quantization-Aware Training)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import get_default_qconfig, prepare_qat, convert
import copy

def convert_to_fp16(model):
    """
    Convert model to FP16 (half precision)
    Simple 2x compression with minimal accuracy loss
    
    Args:
        model: PyTorch model
    
    Returns:
        fp16_model: Model in half precision
        metrics: Size comparison
    """
    # Get baseline size
    baseline_size = get_model_size_mb(model)
    
    # Convert to FP16
    fp16_model = copy.deepcopy(model)
    fp16_model.half()
    
    # Get FP16 size
    fp16_size = get_model_size_mb(fp16_model)
    
    metrics = {
        'baseline_size_mb': baseline_size,
        'fp16_size_mb': fp16_size,
        'compression_ratio': baseline_size / fp16_size,
        'size_reduction_percent': (1 - fp16_size / baseline_size) * 100
    }
    
    print(f"\n[FP16 Conversion]")
    print(f"  Baseline: {baseline_size:.2f} MB")
    print(f"  FP16: {fp16_size:.2f} MB")
    print(f"  Compression: {metrics['compression_ratio']:.2f}x")
    print(f"  Size reduction: {metrics['size_reduction_percent']:.1f}%")
    
    return fp16_model, metrics

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

def quantize_model(model, train_loader, dtype='qint8', qat_epochs=3, device='cpu', lr=0.001):
    """
    Quantize model to INT8 or FP16
    
    Args:
        model: PyTorch model
        train_loader: Training data for QAT fine-tuning
        dtype: 'qint8' for INT8 (4x compression), 'float16' for FP16 (2x compression)
        qat_epochs: Epochs for quantization-aware training
        device: 'cpu' or 'cuda'
        lr: Learning rate for QAT
    
    Returns:
        dict with quantized_model and metrics
    """
    if dtype == 'float16':
        # FP16 is simpler - just convert
        return {'quantized_model': model.half(), 'metrics': {}}
    
    # INT8 quantization (more complex)
    print(f"\n{'='*60}")
    print(f"INT8 Quantization with QAT")
    print(f"{'='*60}\n")
    
    # Get baseline metrics
    baseline_size = get_model_size_mb(model)
    print(f"Baseline model size: {baseline_size:.2f} MB")
    
    # Prepare model for QAT
    model_qat = copy.deepcopy(model)
    model_qat.train()
    
    # Set quantization config
    model_qat.qconfig = get_default_qconfig('fbgemm' if device == 'cpu' else 'qnnpack')
    
    # Prepare QAT
    model_qat_prepared = prepare_qat(model_qat)
    
    # QAT Fine-tuning
    print(f"\nQAT Fine-tuning for {qat_epochs} epochs...")
    optimizer = optim.Adam(model_qat_prepared.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, qat_epochs + 1):
        model_qat_prepared.train()
        total_loss = 0
        n_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model_qat_prepared(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx >= 50:  # Limit batches for speed
                break
        
        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/{qat_epochs}: loss = {avg_loss:.4f}")
    
    # Convert to quantized model
    model_qat_prepared.eval()
    quantized_model = convert(model_qat_prepared)
    
    # Calculate metrics
    quantized_size = get_model_size_mb(quantized_model)
    
    metrics = {
        'baseline_size_mb': baseline_size,
        'quantized_size_mb': quantized_size,
        'compression_ratio': baseline_size / quantized_size,
        'size_reduction_percent': (1 - quantized_size / baseline_size) * 100,
        'dtype': 'int8'
    }
    
    print(f"\n[Quantization Results]")
    print(f"  Baseline: {baseline_size:.2f} MB")
    print(f"  Quantized (INT8): {quantized_size:.2f} MB")
    print(f"  Compression: {metrics['compression_ratio']:.2f}x")
    print(f"  Size reduction: {metrics['size_reduction_percent']:.1f}%")
    print(f"{'='*60}\n")
    
    return {
        'quantized_model': quantized_model,
        'metrics': metrics
    }

def quantization_aware_training(model, train_loader, test_loader, epochs=3, device='cpu'):
    """
    Standalone QAT function
    Fine-tune model with fake quantization
    
    Args:
        model: PyTorch model
        train_loader: Training data
        test_loader: Testing data
        epochs: QAT epochs
        device: Device
    
    Returns:
        QAT-trained model
    """
    model.train()
    model.qconfig = get_default_qconfig('fbgemm')
    model_qat = prepare_qat(model)
    
    optimizer = optim.Adam(model_qat.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs + 1):
        # Train
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model_qat(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Evaluate
        model_qat.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model_qat(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        acc = 100.0 * correct / total
        print(f"[QAT] Epoch {epoch}: loss={total_loss/len(train_loader):.4f}, acc={acc:.2f}%")
        model_qat.train()
    
    # Convert
    model_qat.eval()
    quantized_model = convert(model_qat)
    
    return quantized_model
