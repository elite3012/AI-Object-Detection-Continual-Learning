"""
Parameter-Efficient Fine-Tuning: LoRA (Low-Rank Adaptation)
Implements LoRA layers for efficient continual learning
"""
import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation layer
    Decomposes weight updates as: W + BA (where B and A are low-rank)
    
    Only trains B and A matrices, keeping W frozen
    Reduces trainable params by ~100-1000x
    """
    def __init__(self, in_features, out_features, rank=4, alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank decomposition matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        """Apply LoRA: x @ (A @ B) * scaling"""
        # x: [batch, in_features]
        # lora_A: [in_features, rank]
        # lora_B: [rank, out_features]
        result = x @ self.lora_A  # [batch, rank]
        result = self.dropout(result)
        result = result @ self.lora_B  # [batch, out_features]
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation
    Combines frozen pre-trained weights with learnable low-rank updates
    """
    def __init__(self, linear_layer, rank=4, alpha=16, dropout=0.1):
        super().__init__()
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        # Freeze original linear layer
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)
        
        # Expose attributes for compatibility with PyTorch modules
        self.in_features = in_features
        self.out_features = out_features
    
    @property
    def weight(self):
        """Expose weight for compatibility with MultiheadAttention"""
        return self.linear.weight
    
    @property
    def bias(self):
        """Expose bias for compatibility"""
        return self.linear.bias
    
    def forward(self, x):
        """Forward: original(x) + lora(x)"""
        return self.linear(x) + self.lora(x)
    
    def merge_weights(self):
        """Merge LoRA weights into original layer (for deployment)"""
        with torch.no_grad():
            # W_new = W_old + (B @ A) * scaling
            delta_w = (self.lora.lora_B @ self.lora.lora_A.T) * self.lora.scaling
            self.linear.weight.data += delta_w.T
            
            # Reset LoRA to zeros after merging
            self.lora.lora_A.zero_()
            self.lora.lora_B.zero_()


class Conv2dWithLoRA(nn.Module):
    """
    Conv2d layer with LoRA adaptation
    Applies low-rank adaptation to convolutional kernels
    """
    def __init__(self, conv_layer, rank=4, alpha=16, dropout=0.1):
        super().__init__()
        # Freeze original conv layer
        self.conv = conv_layer
        for param in self.conv.parameters():
            param.requires_grad = False
        
        # Extract conv parameters
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size[0]
        
        # LoRA for conv: reshape kernel to 2D, apply LoRA, reshape back
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices for conv kernels
        kernel_flat = in_channels * kernel_size * kernel_size
        self.lora_A = nn.Parameter(torch.zeros(kernel_flat, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_channels))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Store kernel shape for reshaping
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        """Forward: conv(x) + lora_conv(x)"""
        # Original conv
        out = self.conv(x)
        
        # LoRA adaptation
        batch_size = x.shape[0]
        # Extract patches and apply LoRA
        # For simplicity, we apply LoRA as additional channels
        # Full implementation would use im2col for efficiency
        lora_out = self._apply_lora_conv(x)
        
        return out + lora_out
    
    def _apply_lora_conv(self, x):
        """Apply LoRA as low-rank conv (simplified version)"""
        # Simplified: apply as 1x1 conv after global pool for efficiency
        # Full LoRA conv would require im2col transformation
        batch, c, h, w = x.shape
        
        # Use adaptive pooling to get fixed size, apply LoRA, then upsample
        # This is a simplified approximation for demonstration
        pooled = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        pooled = pooled.view(batch, -1)
        
        # Apply LoRA
        lora_feat = pooled @ self.lora_A
        lora_feat = self.dropout(lora_feat)
        lora_feat = lora_feat @ self.lora_B
        lora_feat = lora_feat * self.scaling
        
        # Broadcast back to spatial dimensions
        lora_out = lora_feat.view(batch, self.out_channels, 1, 1)
        lora_out = lora_out.expand(-1, -1, x.shape[2], x.shape[3])
        
        return lora_out


def apply_lora_to_model(model, rank=4, alpha=16, target_modules=None, dropout=0.1):
    """
    Apply LoRA to all Linear layers in a model
    
    Args:
        model: PyTorch model
        rank: LoRA rank (lower = more compression, typical: 4-64)
        alpha: LoRA scaling (typical: 16-32)
        target_modules: List of module name patterns to apply LoRA (None = all Linear)
        dropout: Dropout rate for LoRA layers
        
    Returns:
        model with LoRA layers, trainable param count, total param count
    """
    # Get model device
    model_device = next(model.parameters()).device
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA to target layers
    lora_count = 0
    for name, module in model.named_modules():
        # Check if this module should get LoRA
        if target_modules is not None:
            if not any(pattern in name for pattern in target_modules):
                continue
        
        # Replace Linear layers with LoRALinear
        if isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            # Replace with LoRA version and move to model device
            lora_layer = LinearWithLoRA(module, rank, alpha, dropout).to(model_device)
            setattr(parent, child_name, lora_layer)
            lora_count += 1
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[LoRA] Applied to {lora_count} Linear layers")
    print(f"[LoRA] Total params: {total_params:,}")
    print(f"[LoRA] Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"[LoRA] Reduction: {total_params/trainable_params:.1f}x fewer trainable params")
    
    return model, trainable_params, total_params


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_lora_parameters(model):
    """Get only LoRA parameters for optimizer"""
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and param.requires_grad:
            lora_params.append(param)
    return lora_params
