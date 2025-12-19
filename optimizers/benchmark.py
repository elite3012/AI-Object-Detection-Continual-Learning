"""
Benchmarking and Profiling Tools
Measure model size, speed, and accuracy
"""

import torch
import time
import numpy as np
from .pruning import get_model_size_mb, count_parameters, count_nonzero_parameters

def benchmark_model(model, test_loader, device='cpu', num_warmup=10, num_runs=100):
    """
    Comprehensive model benchmarking
    
    Args:
        model: PyTorch model
        test_loader: Test data
        device: Device
        num_warmup: Warmup iterations
        num_runs: Benchmark iterations
    
    Returns:
        dict with metrics: size, speed, accuracy
    """
    print(f"\n{'='*70}")
    print(f"Model Benchmark")
    print(f"{'='*70}\n")
    
    model.eval()
    model.to(device)
    
    # 1. Size metrics
    size_mb = get_model_size_mb(model)
    total_params = count_parameters(model)
    nonzero_params = count_nonzero_parameters(model)
    sparsity = 1.0 - (nonzero_params / total_params)
    
    print(f"[Size Metrics]")
    print(f"  Model size: {size_mb:.2f} MB")
    print(f"  Total params: {total_params:,}")
    print(f"  Non-zero params: {nonzero_params:,}")
    print(f"  Sparsity: {sparsity*100:.1f}%\n")
    
    # 2. Speed metrics
    print(f"[Speed Benchmark]")
    print(f"  Warmup: {num_warmup} iterations")
    print(f"  Benchmark: {num_runs} iterations")
    
    # Get sample batch
    sample_batch = next(iter(test_loader))[0].to(device)
    batch_size = sample_batch.size(0)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_batch)
    
    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(sample_batch)
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            latencies.append(end - start)
    
    latencies = np.array(latencies) * 1000  # Convert to ms
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = (batch_size * 1000) / avg_latency  # images/sec
    
    print(f"  Avg latency: {avg_latency:.2f} ± {std_latency:.2f} ms")
    print(f"  Throughput: {throughput:.1f} images/sec")
    print(f"  Batch size: {batch_size}\n")
    
    # 3. Accuracy metrics
    print(f"[Accuracy Evaluation]")
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Correct: {correct}/{total}\n")
    
    print(f"{'='*70}\n")
    
    return {
        'size_mb': size_mb,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'avg_latency_ms': avg_latency,
        'std_latency_ms': std_latency,
        'throughput_img_per_sec': throughput,
        'accuracy': accuracy,
        'batch_size': batch_size
    }

def compare_models(baseline_model, optimized_model, test_loader, device='cpu'):
    """
    Compare baseline vs optimized model
    
    Args:
        baseline_model: Original model
        optimized_model: Compressed model
        test_loader: Test data
        device: Device
    
    Returns:
        Comparison dict
    """
    print(f"\n{'='*70}")
    print(f"Model Comparison: Baseline vs Optimized")
    print(f"{'='*70}\n")
    
    # Benchmark both
    print("Benchmarking baseline model...")
    baseline_metrics = benchmark_model(baseline_model, test_loader, device, num_warmup=5, num_runs=50)
    
    print("\nBenchmarking optimized model...")
    optimized_metrics = benchmark_model(optimized_model, test_loader, device, num_warmup=5, num_runs=50)
    
    # Calculate improvements
    size_reduction = (1 - optimized_metrics['size_mb'] / baseline_metrics['size_mb']) * 100
    speedup = baseline_metrics['avg_latency_ms'] / optimized_metrics['avg_latency_ms']
    accuracy_drop = baseline_metrics['accuracy'] - optimized_metrics['accuracy']
    
    comparison = {
        'baseline': baseline_metrics,
        'optimized': optimized_metrics,
        'improvements': {
            'size_reduction_percent': size_reduction,
            'compression_ratio': baseline_metrics['size_mb'] / optimized_metrics['size_mb'],
            'speedup': speedup,
            'accuracy_drop': accuracy_drop
        }
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Comparison Summary")
    print(f"{'='*70}\n")
    
    print(f"Size:")
    print(f"  Baseline: {baseline_metrics['size_mb']:.2f} MB")
    print(f"  Optimized: {optimized_metrics['size_mb']:.2f} MB")
    print(f"  Reduction: {size_reduction:.1f}% ({comparison['improvements']['compression_ratio']:.2f}x)\n")
    
    print(f"Speed:")
    print(f"  Baseline: {baseline_metrics['avg_latency_ms']:.2f} ms")
    print(f"  Optimized: {optimized_metrics['avg_latency_ms']:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x\n")
    
    print(f"Accuracy:")
    print(f"  Baseline: {baseline_metrics['accuracy']:.2f}%")
    print(f"  Optimized: {optimized_metrics['accuracy']:.2f}%")
    print(f"  Drop: {accuracy_drop:.2f}%\n")
    
    print(f"{'='*70}\n")
    
    return comparison

def profile_memory(model, input_size=(1, 1, 28, 28), device='cpu'):
    """
    Profile memory usage
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device
    
    Returns:
        Memory metrics
    """
    import psutil
    import os
    
    print(f"\n{'='*70}")
    print(f"Memory Profiling")
    print(f"{'='*70}\n")
    
    model.to(device)
    
    # Get process
    process = psutil.Process(os.getpid())
    
    # Baseline memory
    baseline_mem = process.memory_info().rss / (1024 ** 2)  # MB
    
    # Model parameters memory
    param_mem = get_model_size_mb(model)
    
    # Forward pass memory
    dummy_input = torch.randn(*input_size).to(device)
    model.eval()
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
        allocated_mem = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        reserved_mem = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
        
        print(f"[GPU Memory]")
        print(f"  Allocated: {allocated_mem:.2f} MB")
        print(f"  Reserved: {reserved_mem:.2f} MB")
    else:
        allocated_mem = None
        reserved_mem = None
    
    current_mem = process.memory_info().rss / (1024 ** 2)
    inference_mem = current_mem - baseline_mem
    
    print(f"\n[CPU Memory]")
    print(f"  Baseline: {baseline_mem:.2f} MB")
    print(f"  Current: {current_mem:.2f} MB")
    print(f"  Inference overhead: {inference_mem:.2f} MB")
    print(f"  Model parameters: {param_mem:.2f} MB")
    
    print(f"\n{'='*70}\n")
    
    return {
        'param_memory_mb': param_mem,
        'cpu_baseline_mb': baseline_mem,
        'cpu_current_mb': current_mem,
        'cpu_inference_overhead_mb': inference_mem,
        'gpu_allocated_mb': allocated_mem,
        'gpu_reserved_mb': reserved_mem
    }

def quick_benchmark(model, test_loader, device='cpu'):
    """
    Quick benchmark - just essentials
    
    Args:
        model: PyTorch model
        test_loader: Test data
        device: Device
    
    Returns:
        Simple metrics dict
    """
    model.eval()
    model.to(device)
    
    # Size
    size_mb = get_model_size_mb(model)
    
    # Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100.0 * correct / total
    
    print(f"Quick Benchmark: Size={size_mb:.2f} MB, Accuracy={accuracy:.2f}%")
    
    return {
        'size_mb': size_mb,
        'accuracy': accuracy
    }
