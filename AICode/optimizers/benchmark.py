"""
Benchmark and visualize hardware optimization results
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import time

from optimizers.hardware_optimizer import ModelProfiler, AutoOptimizer


def benchmark_models(
    models: Dict[str, nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    input_shape: tuple = (1, 1, 28, 28)
):
    """
    Comprehensive benchmark comparing multiple models
    
    Args:
        models: Dict mapping name -> model
        test_loader: Test data
        device: Device to benchmark on
    
    Returns:
        Dict with all metrics
    """
    profiler = ModelProfiler(device)
    results = {}
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL BENCHMARK")
    print("="*80)
    
    for name, model in models.items():
        print(f"\nBenchmarking: {name}")
        metrics = profiler.profile_model(model, test_loader, input_shape)
        results[name] = metrics
        
        print(f"  ✓ Size: {metrics['model_size_mb']:.2f} MB")
        print(f"  ✓ Params: {metrics['total_params']:,}")
        print(f"  ✓ FPS: {metrics['fps']:.1f}")
        print(f"  ✓ Latency: {metrics['latency_ms']:.2f} ms")
        if 'accuracy' in metrics:
            print(f"  ✓ Accuracy: {metrics['accuracy']:.2f}%")
    
    return results


def plot_optimization_comparison(results: Dict, save_path: str = None):
    """
    Visualize optimization trade-offs
    
    Creates 2x2 subplot:
    - Size comparison
    - Speed comparison
    - Accuracy comparison
    - Efficiency scatter (size vs speed)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hardware Optimization Comparison', fontsize=16, fontweight='bold')
    
    names = list(results.keys())
    
    # Extract metrics
    sizes = [results[n]['model_size_mb'] for n in names]
    params = [results[n]['total_params'] for n in names]
    fps_values = [results[n]['fps'] for n in names]
    latencies = [results[n]['latency_ms'] for n in names]
    accuracies = [results[n].get('accuracy', 0) for n in names]
    
    # Color scheme
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # 1. Model Size Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(names, sizes, color=colors[:len(names)])
    ax1.set_ylabel('Model Size (MB)', fontweight='bold')
    ax1.set_title('Model Size Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    # 2. Inference Speed (FPS)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(names, fps_values, color=colors[:len(names)])
    ax2.set_ylabel('FPS (higher is better)', fontweight='bold')
    ax2.set_title('Inference Speed Comparison')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)
    
    # 3. Accuracy Comparison
    ax3 = axes[1, 0]
    bars3 = ax3.bar(names, accuracies, color=colors[:len(names)])
    ax3.set_ylabel('Accuracy (%)', fontweight='bold')
    ax3.set_title('Accuracy Comparison')
    ax3.set_ylim([min(accuracies)-5 if accuracies else 0, 100])
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=9)
    
    # 4. Efficiency Scatter: Size vs Speed
    ax4 = axes[1, 1]
    scatter = ax4.scatter(sizes, fps_values, s=[p/1000 for p in params], 
                         c=accuracies, cmap='RdYlGn', alpha=0.7,
                         edgecolors='black', linewidth=1.5)
    
    # Annotate points
    for i, name in enumerate(names):
        ax4.annotate(name, (sizes[i], fps_values[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, fontweight='bold')
    
    ax4.set_xlabel('Model Size (MB)', fontweight='bold')
    ax4.set_ylabel('FPS (higher is better)', fontweight='bold')
    ax4.set_title('Efficiency Trade-off (bubble size = params)')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar for accuracy
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Accuracy (%)', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()


def plot_efficiency_over_tasks(efficiency_history: List[Dict], save_path: str = None):
    """
    Plot how efficiency metrics change across continual learning tasks
    
    Args:
        efficiency_history: List of efficiency dicts from HardwareOptimizedTrainer
    """
    if not efficiency_history:
        print("No efficiency history to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hardware Efficiency Across Tasks', fontsize=16, fontweight='bold')
    
    tasks = [eff['task_id'] for eff in efficiency_history]
    
    # Extract metrics over tasks
    sizes = [eff['after']['model_size_mb'] for eff in efficiency_history]
    params = [eff['after']['total_params'] for eff in efficiency_history]
    fps_values = [eff['after']['fps'] for eff in efficiency_history]
    size_reductions = [eff['size_reduction'] for eff in efficiency_history]
    
    # 1. Model Size Over Tasks
    ax1 = axes[0, 0]
    ax1.plot(tasks, sizes, marker='o', linewidth=2, markersize=8, color='#3498db')
    ax1.fill_between(tasks, sizes, alpha=0.3, color='#3498db')
    ax1.set_xlabel('Task ID', fontweight='bold')
    ax1.set_ylabel('Model Size (MB)', fontweight='bold')
    ax1.set_title('Model Size Reduction Over Tasks')
    ax1.grid(True, alpha=0.3)
    
    # 2. Parameter Count Over Tasks
    ax2 = axes[0, 1]
    ax2.plot(tasks, params, marker='s', linewidth=2, markersize=8, color='#e74c3c')
    ax2.fill_between(tasks, params, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Task ID', fontweight='bold')
    ax2.set_ylabel('Parameters', fontweight='bold')
    ax2.set_title('Parameter Count Over Tasks')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='plain', axis='y')
    
    # 3. Inference Speed Over Tasks
    ax3 = axes[1, 0]
    ax3.plot(tasks, fps_values, marker='^', linewidth=2, markersize=8, color='#2ecc71')
    ax3.fill_between(tasks, fps_values, alpha=0.3, color='#2ecc71')
    ax3.set_xlabel('Task ID', fontweight='bold')
    ax3.set_ylabel('FPS (higher is better)', fontweight='bold')
    ax3.set_title('Inference Speed Over Tasks')
    ax3.grid(True, alpha=0.3)
    
    # 4. Cumulative Size Reduction
    ax4 = axes[1, 1]
    cumulative_reduction = np.array(size_reductions)
    ax4.bar(tasks, cumulative_reduction, color='#f39c12', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Task ID', fontweight='bold')
    ax4.set_ylabel('Size Reduction (%)', fontweight='bold')
    ax4.set_title('Size Reduction Per Task')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (t, val) in enumerate(zip(tasks, cumulative_reduction)):
        ax4.text(t, val, f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {save_path}")
    
    plt.show()


def create_efficiency_report(
    baseline_model: nn.Module,
    optimized_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu',
    save_path: str = None
) -> Dict:
    """
    Generate comprehensive efficiency report
    
    Returns markdown-formatted report and metrics dict
    """
    profiler = ModelProfiler(device)
    
    baseline_metrics = profiler.profile_model(baseline_model, test_loader)
    optimized_metrics = profiler.profile_model(optimized_model, test_loader)
    
    # Calculate improvements
    size_reduction = (1 - optimized_metrics['model_size_mb'] / baseline_metrics['model_size_mb']) * 100
    speedup = optimized_metrics['fps'] / baseline_metrics['fps']
    param_reduction = (1 - optimized_metrics['total_params'] / baseline_metrics['total_params']) * 100
    
    acc_drop = 0
    if 'accuracy' in baseline_metrics and 'accuracy' in optimized_metrics:
        acc_drop = baseline_metrics['accuracy'] - optimized_metrics['accuracy']
    
    # Create report
    report = f"""
# Hardware Optimization Efficiency Report

## Model Comparison

| Metric                | Baseline      | Optimized     | Improvement   |
|-----------------------|---------------|---------------|---------------|
| **Model Size**        | {baseline_metrics['model_size_mb']:.2f} MB | {optimized_metrics['model_size_mb']:.2f} MB | {size_reduction:.1f}% ↓ |
| **Parameters**        | {baseline_metrics['total_params']:,} | {optimized_metrics['total_params']:,} | {param_reduction:.1f}% ↓ |
| **Inference Speed**   | {baseline_metrics['fps']:.1f} FPS | {optimized_metrics['fps']:.1f} FPS | {speedup:.2f}x ↑ |
| **Latency**           | {baseline_metrics['latency_ms']:.2f} ms | {optimized_metrics['latency_ms']:.2f} ms | {(1-optimized_metrics['latency_ms']/baseline_metrics['latency_ms'])*100:.1f}% ↓ |
| **Peak Memory**       | {baseline_metrics['peak_memory_mb']:.1f} MB | {optimized_metrics['peak_memory_mb']:.1f} MB | {(1-optimized_metrics['peak_memory_mb']/baseline_metrics['peak_memory_mb'])*100:.1f}% ↓ |
| **Accuracy**          | {baseline_metrics.get('accuracy', 0):.2f}% | {optimized_metrics.get('accuracy', 0):.2f}% | {acc_drop:.2f}% ↓ |

## Summary

- ✅ **{size_reduction:.1f}% smaller** model size
- ✅ **{speedup:.2f}x faster** inference
- ✅ **{param_reduction:.1f}% fewer** parameters
- ⚠️ **{acc_drop:.2f}%** accuracy trade-off

## Deployment Benefits

1. **Storage**: Save {baseline_metrics['model_size_mb'] - optimized_metrics['model_size_mb']:.2f} MB per model
2. **Speed**: Process {optimized_metrics['fps'] - baseline_metrics['fps']:.1f} more images/sec
3. **Memory**: Use {baseline_metrics['peak_memory_mb'] - optimized_metrics['peak_memory_mb']:.1f} MB less RAM
4. **Energy**: Estimated {((1/speedup)-1)*100:.1f}% less energy per inference

---
*Generated by Phase 4: Hardware Optimization*
"""
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved to: {save_path}")
    
    print(report)
    
    return {
        'baseline': baseline_metrics,
        'optimized': optimized_metrics,
        'improvements': {
            'size_reduction_pct': size_reduction,
            'speedup': speedup,
            'param_reduction_pct': param_reduction,
            'accuracy_drop': acc_drop
        }
    }
