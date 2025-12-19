"""
LoRA (Low-Rank Adaptation) Implementation
Parameter-Efficient Fine-Tuning for Continual Learning
"""
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation layer
    
    Instead of fine-tuning W (d x k), we train two small matrices:
    - A (d x r): down-projection
    - B (r x k): up-projection
    where r << min(d, k)
    
    Forward: h = W0*x + (B*A)*x
    - W0: frozen pre-trained weights
    - B*A: trainable low-rank adaptation
    """
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.0):
        """
        Args:
            original_layer: nn.Linear or nn.Conv2d to adapt
            rank: LoRA rank (r), typically 4-32
            alpha: LoRA scaling factor (typically 2*rank)
            dropout: Dropout rate for LoRA
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices for Linear layer
        if isinstance(original_layer, nn.Linear):
            in_features = original_layer.in_features
            out_features = original_layer.out_features
            
            # A: (in_features, rank) - Gaussian init
            self.lora_A = nn.Parameter(torch.randn(in_features, rank) / math.sqrt(rank))
            # B: (rank, out_features) - Zero init (important!)
            self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
            
        # LoRA matrices for Conv2d layer
        elif isinstance(original_layer, nn.Conv2d):
            in_channels = original_layer.in_channels
            out_channels = original_layer.out_channels
            kernel_size = original_layer.kernel_size[0]
            
            # For Conv2d, we adapt with 1x1 convolutions
            self.lora_A = nn.Parameter(torch.randn(in_channels, rank) / math.sqrt(rank))
            self.lora_B = nn.Parameter(torch.zeros(rank, out_channels))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x):
        # Original output (frozen)
        output = self.original_layer(x)
        
        # LoRA adaptation
        if isinstance(self.original_layer, nn.Linear):
            # x: (batch, in_features)
            # lora_A: (in_features, rank)
            # lora_B: (rank, out_features)
            lora_out = x @ self.lora_A @ self.lora_B  # (batch, out_features)
            
        elif isinstance(self.original_layer, nn.Conv2d):
            # x: (batch, in_channels, H, W)
            # Apply lora as 1x1 conv
            batch, in_c, h, w = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, in_c)  # (batch*H*W, in_c)
            lora_out = x_flat @ self.lora_A @ self.lora_B  # (batch*H*W, out_c)
            lora_out = lora_out.reshape(batch, h, w, -1).permute(0, 3, 1, 2)  # (batch, out_c, H, W)
        
        if self.dropout is not None:
            lora_out = self.dropout(lora_out)
        
        # Combine: W0*x + scaling * (B*A)*x
        return output + self.scaling * lora_out


def apply_lora_to_model(model, rank=8, alpha=16, target_modules=None, dropout=0.0):
    """
    Apply LoRA to all Linear/Conv2d layers in model
    
    Args:
        model: PyTorch model
        rank: LoRA rank (4-32, higher = more capacity)
        alpha: LoRA alpha (typically 2*rank for scaling)
        target_modules: List of module names to apply LoRA (None = all Linear/Conv2d)
        dropout: Dropout rate
    
    Returns:
        model: Model with LoRA layers
        trainable_params: Number of trainable parameters
        total_params: Total number of parameters
    """
    # Count original parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Apply LoRA to target layers
    def apply_lora_recursive(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if should apply LoRA
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                # Skip if target_modules specified and this module not in list
                if target_modules is not None:
                    if not any(target in full_name for target in target_modules):
                        apply_lora_recursive(child, full_name)
                        continue
                
                # Replace with LoRA layer
                lora_layer = LoRALayer(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(module, name, lora_layer)
                print(f"[LoRA] Applied to {full_name}: rank={rank}, alpha={alpha}")
            else:
                # Recursively apply to children
                apply_lora_recursive(child, full_name)
    
    apply_lora_recursive(model)
    
    # Count trainable parameters (only LoRA)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n[LoRA] Trainable params: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    return model, trainable_params, total_params


def get_lora_parameters(model):
    """Get only LoRA trainable parameters"""
    lora_params = []
    for name, param in model.named_parameters():
        if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model):
    """
    Merge LoRA weights into original weights (for deployment)
    After merging, model becomes regular model without LoRA overhead
    """
    def merge_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, LoRALayer):
                # Get original layer
                original = child.original_layer
                
                # Compute LoRA delta: scaling * B * A
                if isinstance(original, nn.Linear):
                    delta = child.scaling * (child.lora_A @ child.lora_B).T
                    original.weight.data += delta
                
                elif isinstance(original, nn.Conv2d):
                    delta = child.scaling * (child.lora_A @ child.lora_B).T
                    # Expand delta to match conv weight shape
                    delta = delta.unsqueeze(-1).unsqueeze(-1)
                    original.weight.data += delta
                
                # Replace LoRA layer with original (now merged)
                original.requires_grad_(True)
                setattr(module, name, original)
                print(f"[LoRA] Merged weights in {name}")
            else:
                merge_recursive(child)
    
    merge_recursive(model)
    print("[LoRA] All weights merged. Model is now regular (no LoRA overhead).")
    return model
