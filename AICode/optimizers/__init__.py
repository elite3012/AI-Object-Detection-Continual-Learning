"""
Hardware Optimization Package (Phase 4)

Provides model compression and acceleration tools:
- Quantization (INT8/FP16)
- Structured pruning
- Auto-optimization
- Benchmarking
"""

from .quantization import (
    quantize_model,
    QuantizationConfig,
    QuantizationAwareTrainer,
    convert_to_fp16
)

from .pruning import (
    prune_model,
    PruningConfig,
    ChannelPruner,
    MagnitudePruning
)

from .hardware_optimizer import (
    AutoOptimizer,
    ModelProfiler,
    HardwareProfile
)

from .benchmark import (
    benchmark_models,
    plot_optimization_comparison,
    plot_efficiency_over_tasks,
    create_efficiency_report
)

__all__ = [
    # Quantization
    'quantize_model',
    'QuantizationConfig',
    'QuantizationAwareTrainer',
    'convert_to_fp16',
    
    # Pruning
    'prune_model',
    'PruningConfig',
    'ChannelPruner',
    'MagnitudePruning',
    
    # Auto-optimization
    'AutoOptimizer',
    'ModelProfiler',
    'HardwareProfile',
    
    # Benchmarking
    'benchmark_models',
    'plot_optimization_comparison',
    'plot_efficiency_over_tasks',
    'create_efficiency_report',
]
