"""
Vision Transformer (ViT) for Continual Learning
Pre-trained ViT backbone with classification head for Fashion-MNIST
"""
import torch
import torch.nn as nn
from torchvision import models


class ViTForContinualLearning(nn.Module):
    """
    Vision Transformer adapted for continual learning
    Uses pre-trained ViT-B/16 as backbone with custom classifier
    """
    def __init__(self, num_classes=10, pretrained=True, freeze_backbone=False):
        super().__init__()
        
        # Load pre-trained ViT-B/16 (ImageNet weights)
        # Note: ViT expects 224x224 RGB, we'll need to adapt for 28x28 grayscale
        try:
            # Try to load ViT from torchvision (requires torchvision >= 0.13)
            self.backbone = models.vit_b_16(pretrained=pretrained)
            self.has_vit = True
        except AttributeError:
            # Fallback to ResNet if ViT not available
            print("[WARNING] ViT not available, using ResNet18 as fallback")
            self.backbone = models.resnet18(pretrained=pretrained)
            self.has_vit = False
        
        if self.has_vit:
            # ViT architecture
            hidden_dim = self.backbone.heads.head.in_features
            
            # Remove original classification head
            self.backbone.heads.head = nn.Identity()
            
            # Adapt patch embedding for 28x28 input
            # Original: 224x224 with patch_size=16 -> 14x14 patches
            # Fashion-MNIST: 28x28 with patch_size=4 -> 7x7 patches
            original_patch_embed = self.backbone.conv_proj
            self.backbone.conv_proj = nn.Conv2d(
                1,  # Grayscale input
                original_patch_embed.out_channels,
                kernel_size=4,  # Smaller patch for 28x28
                stride=4,
                padding=0
            )
            
            # Initialize new patch embedding
            with torch.no_grad():
                # Average RGB channels to grayscale for initialization
                weight = original_patch_embed.weight.data.mean(dim=1, keepdim=True)
                # Interpolate to new patch size
                self.backbone.conv_proj.weight.data = nn.functional.interpolate(
                    weight, 
                    size=(4, 4), 
                    mode='bilinear', 
                    align_corners=False
                )
            
        else:
            # ResNet fallback
            hidden_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
            # Adapt first conv for grayscale
            original_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                1,  # Grayscale
                original_conv.out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
            
            # Initialize with averaged RGB weights
            with torch.no_grad():
                self.backbone.conv1.weight.data = original_conv.weight.data.mean(dim=1, keepdim=True)
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch, 1, 28, 28] Fashion-MNIST images
        Returns:
            logits: [batch, num_classes]
        """
        # Upsample to minimum size for ViT/ResNet
        if x.shape[-1] == 28:
            # Upsample 28x28 -> 224x224 for pre-trained weights compatibility
            # Alternative: use adaptive pooling in backbone
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x):
        """Extract features without classification"""
        if x.shape[-1] == 28:
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


class LightweightViT(nn.Module):
    """
    Lightweight Vision Transformer for Fashion-MNIST
    Built from scratch for 28x28 images (no upsampling needed)
    """
    def __init__(self, num_classes=10, img_size=28, patch_size=4, embed_dim=192, 
                 depth=6, num_heads=3, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 7x7 = 49 patches
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Store original num_classes to prevent it from being changed by pruning
        self._original_num_classes = num_classes
    
    def adapt_to_pruning(self):
        """
        Adapt model layers after pruning has modified embedding dimension.
        Call this after pruning to rebuild transformer and norm layers.
        """
        # Detect actual embedding dimension from patch_embed output channels
        actual_embed_dim = self.patch_embed.out_channels
        
        if actual_embed_dim != self.embed_dim:
            print(f"[ViT] Adapting to pruned embed_dim: {self.embed_dim} -> {actual_embed_dim}")
            
            # Update stored embed_dim
            self.embed_dim = actual_embed_dim
            
            # Find valid number of heads that divides embed_dim
            # Try common values: 1, 2, 3, 4, 6, 8, 12
            possible_heads = [1, 2, 3, 4, 6, 8, 12]
            num_heads = 1
            for h in possible_heads:
                if actual_embed_dim % h == 0:
                    num_heads = h
                else:
                    break  # Use the largest valid number
            
            mlp_ratio = 2.0
            dropout = 0.1
            depth = len(list(self.transformer.layers))
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=actual_embed_dim,
                nhead=num_heads,
                dim_feedforward=int(actual_embed_dim * mlp_ratio),
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            
            # Copy device and dtype from old transformer
            device = next(self.transformer.parameters()).device
            dtype = next(self.transformer.parameters()).dtype
            
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth).to(device=device, dtype=dtype)
            
            # Rebuild norm layer with correct dtype
            self.norm = nn.LayerNorm(actual_embed_dim).to(device=device, dtype=dtype)
            
            # Preserve the ORIGINAL number of classes (don't let pruning change this!)
            target_num_classes = getattr(self, '_original_num_classes', 10)
            
            # Adjust head input dimension - preserve weights if possible
            old_head_weight = self.head.weight.data.clone()  # [old_num_classes, old_embed_dim]
            old_head_bias = self.head.bias.data.clone() if self.head.bias is not None else None
            
            # Create new head with ORIGINAL num_classes (not old head's potentially pruned output)
            self.head = nn.Linear(actual_embed_dim, target_num_classes).to(device=device, dtype=dtype)
            
            # Initialize new head with truncated old weights
            with torch.no_grad():
                # Copy weights up to the new embedding dimension and available classes
                min_dim = min(old_head_weight.shape[1], actual_embed_dim)
                min_classes = min(old_head_weight.shape[0], target_num_classes)
                self.head.weight.data[:min_classes, :min_dim] = old_head_weight[:min_classes, :min_dim]
                
                # Copy bias if it exists
                if old_head_bias is not None:
                    self.head.bias.data[:min_classes] = old_head_bias[:min_classes]
                else:
                    self.head.bias.data.zero_()
            
            # Update num_classes to match actual head
            self.num_classes = target_num_classes
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: [batch, 1, 28, 28]
        Returns:
            logits: [batch, num_classes]
        """
        batch_size = x.shape[0]
        
        # Convert input to match model dtype (handles float16 quantization)
        if next(self.parameters()).dtype != x.dtype:
            x = x.to(next(self.parameters()).dtype)
        
        # Patch embedding: [B, 1, 28, 28] -> [B, embed_dim, 7, 7] -> [B, 49, embed_dim]
        x = self.patch_embed(x)  # [B, embed_dim, 7, 7]
        x = x.flatten(2).transpose(1, 2)  # [B, 49, embed_dim]
        
        # Get actual embedding dimension (may differ from self.embed_dim after pruning/quantization)
        actual_embed_dim = x.shape[-1]
        
        # Dynamically slice cls_token and pos_embed to match actual dimension
        # This handles any changes from pruning, quantization, or other optimizations
        cls_tokens = self.cls_token[:, :, :actual_embed_dim].expand(batch_size, -1, -1)  # [B, 1, actual_embed_dim]
        pos_embed = self.pos_embed[:, :, :actual_embed_dim]  # [1, 50, actual_embed_dim]
        
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 50, actual_embed_dim]
        
        # Add positional embedding
        x = x + pos_embed
        
        # Check if transformer needs adaptation
        if actual_embed_dim != self.embed_dim:
            # Rebuild transformer layers on-the-fly if dimension changed
            self.adapt_to_pruning()
        
        # Transformer
        x = self.transformer(x)
        
        # Classification: use CLS token
        x = self.norm(x[:, 0])  # [B, actual_embed_dim]
        logits = self.head(x)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x):
        """Extract CLS token features"""
        batch_size = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.norm(x[:, 0])


def count_parameters(model):
    """Count parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
