"""
Structured Pruning for Neural Networks
Implements L1/L2 norm-based channel pruning with gradual sparsity increase
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
import copy


class PruningConfig:
    """Configuration for pruning"""
    
    def __init__(
        self,
        target_sparsity: float = 0.5,  # 50% sparsity
        pruning_method: str = 'l1',  # 'l1' or 'l2'
        gradual_steps: int = 5,  # Gradual pruning over N steps
        min_channels: int = 8,  # Don't prune below this many channels
        prune_batch_norm: bool = True
    ):
        self.target_sparsity = target_sparsity
        self.pruning_method = pruning_method
        self.gradual_steps = gradual_steps
        self.min_channels = min_channels
        self.prune_batch_norm = prune_batch_norm


class ChannelPruner:
    """
    Structured pruning by removing entire channels based on importance
    
    Importance measured by:
    - L1 norm: Sum of absolute weights
    - L2 norm: Sum of squared weights
    """
    
    def __init__(self, model: nn.Module, config: PruningConfig):
        self.model = model
        self.config = config
        self.current_sparsity = 0.0
        self.pruned_indices = {}  # Layer name -> pruned channel indices
        
    def compute_channel_importance(self, layer: nn.Module) -> torch.Tensor:
        """
        Compute importance score for each channel
        
        For Conv2d: weight shape is [out_channels, in_channels, H, W]
        For Linear: weight shape is [out_features, in_features]
        """
        if isinstance(layer, nn.Conv2d):
            weight = layer.weight.data  # [out_ch, in_ch, H, W]
            
            if self.config.pruning_method == 'l1':
                # L1 norm per output channel
                importance = weight.abs().sum(dim=[1, 2, 3])  # [out_ch]
            else:  # l2
                importance = weight.pow(2).sum(dim=[1, 2, 3]).sqrt()
                
        elif isinstance(layer, nn.Linear):
            weight = layer.weight.data  # [out_feat, in_feat]
            
            if self.config.pruning_method == 'l1':
                importance = weight.abs().sum(dim=1)  # [out_feat]
            else:
                importance = weight.pow(2).sum(dim=1).sqrt()
        else:
            return None
        
        return importance
    
    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get all Conv2d and Linear layers"""
        prunable = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                prunable.append((name, module))
        
        return prunable
    
    def prune_layer(
        self,
        layer: nn.Module,
        sparsity: float,
        layer_name: str
    ) -> nn.Module:
        """
        Prune a single layer to target sparsity
        
        Returns new pruned layer
        """
        importance = self.compute_channel_importance(layer)
        
        if importance is None:
            return layer
        
        num_channels = len(importance)
        num_to_keep = max(
            int(num_channels * (1 - sparsity)),
            self.config.min_channels
        )
        
        # Get indices to keep (highest importance)
        _, indices = torch.topk(importance, num_to_keep)
        indices = sorted(indices.tolist())
        
        # Store pruned indices
        all_indices = set(range(num_channels))
        kept_indices = set(indices)
        self.pruned_indices[layer_name] = list(all_indices - kept_indices)
        
        # Create new layer with reduced channels
        if isinstance(layer, nn.Conv2d):
            new_layer = nn.Conv2d(
                in_channels=layer.in_channels,
                out_channels=num_to_keep,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                bias=layer.bias is not None
            )
            
            # Copy weights for kept channels
            new_layer.weight.data = layer.weight.data[indices, :, :, :]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[indices]
                
        elif isinstance(layer, nn.Linear):
            new_layer = nn.Linear(
                in_features=layer.in_features,
                out_features=num_to_keep,
                bias=layer.bias is not None
            )
            
            new_layer.weight.data = layer.weight.data[indices, :]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias.data[indices]
        
        return new_layer
    
    def prune_model(self, target_sparsity: float) -> nn.Module:
        """
        Prune entire model to target sparsity
        
        Returns new pruned model
        """
        print(f"\n[Pruning] Target sparsity: {target_sparsity*100:.1f}%")
        
        # Create a copy of the model
        pruned_model = copy.deepcopy(self.model)
        
        # Get all prunable layers
        prunable_layers = self.get_prunable_layers()
        
        total_params_before = 0
        total_params_after = 0
        
        # Prune each layer
        for layer_name, layer in prunable_layers:
            # Count params before
            params_before = sum(p.numel() for p in layer.parameters())
            total_params_before += params_before
            
            # Prune layer
            pruned_layer = self.prune_layer(layer, target_sparsity, layer_name)
            
            # Replace layer in model
            self._replace_layer(pruned_model, layer_name, pruned_layer)
            
            # Count params after
            params_after = sum(p.numel() for p in pruned_layer.parameters())
            total_params_after += params_after
            
            print(f"  {layer_name}: {params_before:,} -> {params_after:,} params")
        
        actual_sparsity = 1 - (total_params_after / total_params_before)
        self.current_sparsity = actual_sparsity
        
        print(f"\n[Pruning] Actual sparsity: {actual_sparsity*100:.1f}%")
        print(f"[Pruning] Params: {total_params_before:,} -> {total_params_after:,}")
        
        return pruned_model
    
    def _replace_layer(self, model: nn.Module, layer_name: str, new_layer: nn.Module):
        """Replace a layer in the model by name"""
        parts = layer_name.split('.')
        
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, parts[-1], new_layer)
    
    def gradual_pruning(
        self,
        train_fn,
        initial_sparsity: float = 0.0
    ) -> List[nn.Module]:
        """
        Gradual pruning with intermediate fine-tuning
        
        Args:
            train_fn: Function to fine-tune model, takes (model, sparsity_level)
            initial_sparsity: Starting sparsity
        
        Returns:
            List of pruned models at each step
        """
        print("\n" + "="*60)
        print("GRADUAL PRUNING")
        print("="*60)
        
        pruned_models = []
        sparsity_schedule = np.linspace(
            initial_sparsity,
            self.config.target_sparsity,
            self.config.gradual_steps
        )
        
        current_model = self.model
        
        for step, sparsity in enumerate(sparsity_schedule):
            print(f"\n--- Step {step+1}/{self.config.gradual_steps} ---")
            
            # Prune to current sparsity
            self.model = current_model
            pruned_model = self.prune_model(sparsity)
            
            # Fine-tune
            if train_fn is not None:
                print(f"\n[Fine-tuning] Sparsity {sparsity*100:.1f}%...")
                train_fn(pruned_model, sparsity)
            
            pruned_models.append(pruned_model)
            current_model = pruned_model
        
        return pruned_models


class MagnitudePruning:
    """
    Unstructured magnitude pruning (per-weight)
    Simpler but less hardware-efficient than channel pruning
    """
    
    @staticmethod
    def prune_weights(model: nn.Module, sparsity: float) -> nn.Module:
        """
        Prune individual weights by magnitude
        Creates sparse model (but doesn't reduce size on disk)
        """
        print(f"\n[Magnitude Pruning] Target sparsity: {sparsity*100:.1f}%")
        
        # Collect all weights
        all_weights = []
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.abs().view(-1))
        
        all_weights = torch.cat(all_weights)
        
        # Find threshold
        threshold = torch.quantile(all_weights, sparsity)
        
        # Prune weights below threshold
        pruned_count = 0
        total_count = 0
        
        for param in model.parameters():
            if param.requires_grad:
                mask = param.data.abs() > threshold
                param.data *= mask.float()
                
                pruned_count += (~mask).sum().item()
                total_count += param.numel()
        
        actual_sparsity = pruned_count / total_count
        print(f"[Magnitude Pruning] Pruned {pruned_count:,} / {total_count:,} weights")
        print(f"[Magnitude Pruning] Actual sparsity: {actual_sparsity*100:.1f}%")
        
        return model


def prune_model(
    model: nn.Module,
    target_sparsity: float = 0.5,
    method: str = 'channel',  # 'channel' or 'magnitude'
    gradual: bool = True,
    train_fn = None
) -> Dict:
    """
    High-level API for model pruning
    
    Args:
        model: Model to prune
        target_sparsity: Target parameter reduction (0.5 = 50% pruned)
        method: 'channel' (structured) or 'magnitude' (unstructured)
        gradual: Whether to use gradual pruning
        train_fn: Fine-tuning function (model) -> None
    
    Returns:
        dict with 'pruned_model' and 'metrics'
    """
    print(f"\n{'='*60}")
    print(f"PRUNING: {method.upper()}")
    print(f"{'='*60}")
    
    if method == 'channel':
        config = PruningConfig(target_sparsity=target_sparsity)
        pruner = ChannelPruner(model, config)
        
        if gradual and train_fn:
            pruned_models = pruner.gradual_pruning(train_fn)
            pruned_model = pruned_models[-1]
        else:
            pruned_model = pruner.prune_model(target_sparsity)
        
    else:  # magnitude
        pruned_model = copy.deepcopy(model)
        MagnitudePruning.prune_weights(pruned_model, target_sparsity)
    
    # Calculate metrics
    original_params = sum(p.numel() for p in model.parameters())
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    
    metrics = {
        'original_params': original_params,
        'pruned_params': pruned_params,
        'compression_ratio': original_params / pruned_params,
        'sparsity': 1 - (pruned_params / original_params)
    }
    
    print("\n" + "="*60)
    print("PRUNING RESULTS")
    print("="*60)
    print(f"Original params:  {original_params:,}")
    print(f"Pruned params:    {pruned_params:,}")
    print(f"Compression:      {metrics['compression_ratio']:.2f}x")
    print(f"Sparsity:         {metrics['sparsity']*100:.1f}%")
    print("="*60)
    
    return {
        'pruned_model': pruned_model,
        'metrics': metrics
    }
