"""
Hardware-aware Optimizer
Automatically selects best optimization strategy for target hardware
"""

import torch
from .quantization import quantize_model, convert_to_fp16
from .pruning import prune_model, gradual_pruning

class HardwareOptimizer:
    """
    Auto-select optimization based on hardware constraints
    Combines quantization + pruning for maximum compression
    """
    
    PRESETS = {
        'mobile': {
            'quantization': 'int8',
            'sparsity': 0.5,
            'target_size_mb': 5.0,
            'priority': 'size',
            'description': 'Mobile deployment (ARM CPU, limited memory)'
        },
        'gpu': {
            'quantization': 'fp16',
            'sparsity': 0.3,
            'target_size_mb': 50.0,
            'priority': 'speed',
            'description': 'GPU deployment (NVIDIA GPU, high throughput)'
        },
        'edge': {
            'quantization': 'int8',
            'sparsity': 0.7,
            'target_size_mb': 2.0,
            'priority': 'size',
            'description': 'Edge device (Raspberry Pi, IoT)'
        },
        'cloud': {
            'quantization': 'fp16',
            'sparsity': 0.2,
            'target_size_mb': 100.0,
            'priority': 'accuracy',
            'description': 'Cloud deployment (server GPU, high accuracy)'
        },
        'custom': {
            'quantization': None,
            'sparsity': None,
            'target_size_mb': None,
            'priority': 'balanced',
            'description': 'Custom configuration'
        }
    }
    
    def __init__(self, preset='mobile'):
        """
        Initialize hardware optimizer
        
        Args:
            preset: 'mobile', 'gpu', 'edge', 'cloud', or 'custom'
        """
        if preset not in self.PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(self.PRESETS.keys())}")
        
        self.preset = preset
        self.config = self.PRESETS[preset].copy()
        
        print(f"\n{'='*70}")
        print(f"Hardware Optimizer: {preset.upper()}")
        print(f"  {self.config['description']}")
        print(f"  Quantization: {self.config['quantization']}")
        print(f"  Sparsity: {self.config['sparsity']*100 if self.config['sparsity'] else 'N/A'}%")
        print(f"  Target size: {self.config['target_size_mb']} MB")
        print(f"  Priority: {self.config['priority']}")
        print(f"{'='*70}\n")
    
    def optimize(self, model, train_loader=None, device='cpu'):
        """
        Apply hardware-optimized compression
        
        Args:
            model: PyTorch model
            train_loader: Training data for QAT (optional)
            device: Device
        
        Returns:
            dict with optimized_model, metrics, and strategy
        """
        print(f"\n{'='*70}")
        print(f"Starting Hardware Optimization")
        print(f"{'='*70}\n")
        
        optimized_model = model
        all_metrics = {
            'preset': self.preset,
            'steps': []
        }
        
        # Step 1: Pruning (if needed)
        if self.config['sparsity'] and self.config['sparsity'] > 0:
            print(f"[Step 1] Pruning with sparsity={self.config['sparsity']*100:.0f}%")
            prune_result = prune_model(
                optimized_model, 
                sparsity=self.config['sparsity'],
                method='magnitude',
                structured=True
            )
            optimized_model = prune_result['pruned_model']
            all_metrics['steps'].append({
                'type': 'pruning',
                'metrics': prune_result['metrics']
            })
        
        # Step 2: Quantization (if needed)
        if self.config['quantization']:
            print(f"[Step 2] Quantization to {self.config['quantization']}")
            
            if self.config['quantization'] == 'int8':
                if train_loader is None:
                    print("  Warning: No train_loader provided. Skipping QAT.")
                else:
                    quant_result = quantize_model(
                        optimized_model,
                        train_loader,
                        dtype='qint8',
                        qat_epochs=3,
                        device=device
                    )
                    optimized_model = quant_result['quantized_model']
                    all_metrics['steps'].append({
                        'type': 'quantization_int8',
                        'metrics': quant_result['metrics']
                    })
            
            elif self.config['quantization'] == 'fp16':
                optimized_model, fp16_metrics = convert_to_fp16(optimized_model)
                all_metrics['steps'].append({
                    'type': 'quantization_fp16',
                    'metrics': fp16_metrics
                })
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics(model, optimized_model, all_metrics)
        
        print(f"\n{'='*70}")
        print(f"Hardware Optimization Complete!")
        print(f"  Final compression: {final_metrics['total_compression_ratio']:.2f}x")
        print(f"  Final size: {final_metrics['final_size_mb']:.2f} MB")
        print(f"  Target achieved: {'✓' if final_metrics['target_achieved'] else '✗'}")
        print(f"{'='*70}\n")
        
        return {
            'optimized_model': optimized_model,
            'metrics': final_metrics,
            'strategy': self.config
        }
    
    def _calculate_final_metrics(self, original_model, optimized_model, all_metrics):
        """Calculate overall compression metrics"""
        from .pruning import get_model_size_mb, count_parameters, count_nonzero_parameters
        
        original_size = get_model_size_mb(original_model)
        final_size = get_model_size_mb(optimized_model)
        
        original_params = count_parameters(original_model)
        final_params = count_nonzero_parameters(optimized_model)
        
        total_compression = original_size / final_size
        target_achieved = final_size <= self.config['target_size_mb']
        
        return {
            'original_size_mb': original_size,
            'final_size_mb': final_size,
            'original_params': original_params,
            'final_params': final_params,
            'total_compression_ratio': total_compression,
            'size_reduction_percent': (1 - final_size / original_size) * 100,
            'param_reduction_percent': (1 - final_params / original_params) * 100,
            'target_size_mb': self.config['target_size_mb'],
            'target_achieved': target_achieved,
            'steps': all_metrics['steps']
        }
    
    @staticmethod
    def list_presets():
        """List all available hardware presets"""
        print(f"\n{'='*70}")
        print(f"Available Hardware Presets")
        print(f"{'='*70}\n")
        
        for name, config in HardwareOptimizer.PRESETS.items():
            print(f"{name.upper():10s}: {config['description']}")
            print(f"            Quantization: {config['quantization']}")
            print(f"            Sparsity: {config['sparsity']*100 if config['sparsity'] else 'N/A'}%")
            print(f"            Target size: {config['target_size_mb']} MB")
            print(f"            Priority: {config['priority']}")
            print()
        
        print(f"{'='*70}\n")
    
    def set_custom_config(self, quantization=None, sparsity=None, target_size_mb=None):
        """
        Set custom optimization configuration
        
        Args:
            quantization: 'int8', 'fp16', or None
            sparsity: 0.0-1.0 or None
            target_size_mb: Target model size
        """
        if self.preset != 'custom':
            print(f"Warning: Changing preset from '{self.preset}' to 'custom'")
            self.preset = 'custom'
        
        if quantization is not None:
            self.config['quantization'] = quantization
        
        if sparsity is not None:
            self.config['sparsity'] = sparsity
        
        if target_size_mb is not None:
            self.config['target_size_mb'] = target_size_mb
        
        print(f"\nCustom config updated:")
        print(f"  Quantization: {self.config['quantization']}")
        print(f"  Sparsity: {self.config['sparsity']*100 if self.config['sparsity'] else 'N/A'}%")
        print(f"  Target size: {self.config['target_size_mb']} MB")

def auto_optimize(model, train_loader=None, target_hardware='mobile', device='cpu'):
    """
    Convenience function for one-shot hardware optimization
    
    Args:
        model: PyTorch model
        train_loader: Training data (optional, needed for INT8 QAT)
        target_hardware: 'mobile', 'gpu', 'edge', or 'cloud'
        device: Device
    
    Returns:
        Optimized model
    
    Example:
        >>> optimized_model = auto_optimize(model, train_loader, target_hardware='mobile')
    """
    optimizer = HardwareOptimizer(preset=target_hardware)
    result = optimizer.optimize(model, train_loader, device)
    return result['optimized_model']
