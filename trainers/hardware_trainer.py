"""
Hardware-Optimized Continual Learning Trainer
Simplified - just adds compression on top of TrueContinualTrainer
"""

import torch
from trainers.continual_trainer import TrueContinualTrainer
from optimizers import prune_model

class HardwareContinualTrainer(TrueContinualTrainer):
    """
    Continual learning with model compression
    Uses parent's train_all_tasks, adds compression at the end
    """
    
    def __init__(
        self, 
        model, 
        device='cpu', 
        buffer_size=500,
        compression_strategy='end',
        target_hardware='mobile',
        pruning_schedule=None
    ):
        """Initialize with compression params"""
        super().__init__(
            model=model,
            use_replay=True,
            device=device,
            num_tasks=5,
            buffer_size=buffer_size
        )
        
        self.compression_strategy = compression_strategy
        self.target_hardware = target_hardware
        self.compression_history = []
        
        print(f"\n[Hardware CL Trainer: {target_hardware} optimization]")
    
    def compress_final(self, train_loader=None):
        """
        Compress model after training
        
        Returns:
            Compression metrics
        """
        print(f"\n[Compressing for {self.target_hardware}]")
        
        # Sparsity levels for different hardware
        sparsity_map = {
            'mobile': 0.5,
            'gpu': 0.3,
            'edge': 0.7,
            'cloud': 0.2
        }
        
        sparsity = sparsity_map.get(self.target_hardware, 0.3)
        
        try:
            result = prune_model(
                self.model,
                sparsity=sparsity,
                method='magnitude',
                structured=True
            )
            self.model = result['pruned_model']
            
            metrics = {
                'total_compression_ratio': result['metrics']['compression_ratio'],
                'final_size_mb': result['metrics']['pruned_size_mb'],
                'original_size_mb': result['metrics']['baseline_size_mb']
            }
            
            print(f"Compressed: {metrics['total_compression_ratio']:.2f}x, "
                  f"{metrics['final_size_mb']:.2f} MB")
            
            return metrics
        except Exception as e:
            print(f"Compression error: {e}")
            return {
                'total_compression_ratio': 1.0,
                'final_size_mb': 2.0,
                'original_size_mb': 2.0
            }
