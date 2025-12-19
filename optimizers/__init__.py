"""
Hardware Optimization Package (Phase 4)
Model compression and acceleration for edge deployment
"""

# Quantization
from .quantization import quantize_model, convert_to_fp16, quantization_aware_training

# Pruning
from .pruning import (
    prune_model, 
    structured_prune, 
    channel_prune, 
    gradual_pruning,
    analyze_sparsity
)

# Hardware optimization
from .hardware_optimizer import HardwareOptimizer, auto_optimize

# Benchmarking
from .benchmark import (
    benchmark_model, 
    compare_models, 
    profile_memory,
    quick_benchmark
)

__all__ = [
    # Quantization
    'quantize_model',
    'convert_to_fp16',
    'quantization_aware_training',
    
    # Pruning
    'prune_model',
    'structured_prune',
    'channel_prune',
    'gradual_pruning',
    'analyze_sparsity',
    
    # Hardware optimization
    'HardwareOptimizer',
    'auto_optimize',
    
    # Benchmarking
    'benchmark_model',
    'compare_models',
    'profile_memory',
    'quick_benchmark',
]
