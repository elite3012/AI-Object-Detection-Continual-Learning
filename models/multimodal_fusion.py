"""
Multi-Modal Fusion Layer
Combines visual and text features for joint classification
Supports multiple fusion strategies: concatenation, cross-attention, gated fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between vision and text features
    Vision attends to text, text attends to vision
    """
    def __init__(self, vision_dim, text_dim, hidden_dim=256, num_heads=4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project to same dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Cross-attention: vision queries text
        self.v2t_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Cross-attention: text queries vision
        self.t2v_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, vision_feat, text_feat):
        """
        Args:
            vision_feat: (batch, vision_dim)
            text_feat: (batch, text_dim)
        
        Returns:
            fused: (batch, hidden_dim * 2)
        """
        # Project
        v = self.vision_proj(vision_feat).unsqueeze(1)  # (batch, 1, hidden)
        t = self.text_proj(text_feat).unsqueeze(1)      # (batch, 1, hidden)
        
        # Cross-attend
        v2t, _ = self.v2t_attention(v, t, t)  # vision queries text
        t2v, _ = self.t2v_attention(t, v, v)  # text queries vision
        
        # Residual + norm
        v_out = self.norm1(v + v2t).squeeze(1)
        t_out = self.norm2(t + t2v).squeeze(1)
        
        # Concatenate
        fused = torch.cat([v_out, t_out], dim=1)
        
        return fused

class GatedFusion(nn.Module):
    """
    Gated fusion: Learn dynamic weights for vision and text
    """
    def __init__(self, vision_dim, text_dim, hidden_dim=256):
        super().__init__()
        
        # Project to same dimension
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, vision_feat, text_feat):
        """
        Args:
            vision_feat: (batch, vision_dim)
            text_feat: (batch, text_dim)
        
        Returns:
            fused: (batch, hidden_dim)
        """
        # Project
        v = self.vision_proj(vision_feat)
        t = self.text_proj(text_feat)
        
        # Compute gates
        concat = torch.cat([v, t], dim=1)
        gates = self.gate(concat)  # (batch, 2)
        
        # Weighted sum
        weighted = gates[:, 0:1] * v + gates[:, 1:2] * t
        
        # Fusion
        fused = self.fusion(weighted)
        
        return fused

class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation fusion (baseline)
    """
    def __init__(self, vision_dim, text_dim, hidden_dim=256):
        super().__init__()
        
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, vision_feat, text_feat):
        """
        Args:
            vision_feat: (batch, vision_dim)
            text_feat: (batch, text_dim)
        
        Returns:
            fused: (batch, hidden_dim)
        """
        concat = torch.cat([vision_feat, text_feat], dim=1)
        fused = self.fusion(concat)
        return fused

class MultiModalClassifier(nn.Module):
    """
    Complete multi-modal classifier
    Vision encoder + Text encoder + Fusion + Classifier
    """
    def __init__(self, vision_encoder, text_encoder, num_classes=10, 
                 fusion_type='concat', hidden_dim=256):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        
        # Get feature dimensions
        # Vision: SimpleCNN outputs from classifier (we'll extract features before final layer)
        # Text: from text_encoder.embed_dim
        vision_dim = 256  # SimpleCNN has 256 hidden units before final layer
        text_dim = text_encoder.embed_dim
        
        # Fusion layer
        if fusion_type == 'cross_attention':
            self.fusion = CrossAttentionFusion(vision_dim, text_dim, hidden_dim)
            fusion_output_dim = hidden_dim * 2
        elif fusion_type == 'gated':
            self.fusion = GatedFusion(vision_dim, text_dim, hidden_dim)
            fusion_output_dim = hidden_dim
        else:  # concat
            self.fusion = SimpleConcatFusion(vision_dim, text_dim, hidden_dim)
            fusion_output_dim = hidden_dim
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.num_classes = num_classes
        
    def extract_vision_features(self, images):
        """Extract visual features (before final classification layer)"""
        # Pass through CNN backbone
        x = self.vision_encoder.features(images)
        x = torch.flatten(x, 1)
        
        # Get features from first part of classifier (before final linear)
        # SimpleCNN classifier: Flatten -> Linear(1152, 256) -> BN -> ReLU -> Dropout -> Linear(256, 10)
        # We want output after first Linear+BN+ReLU
        x = self.vision_encoder.classifier[0](x)  # Flatten (already done above, but keeping structure)
        x = self.vision_encoder.classifier[1](x)  # Linear 1152->256
        x = self.vision_encoder.classifier[2](x)  # BatchNorm
        x = self.vision_encoder.classifier[3](x)  # ReLU
        
        return x  # (batch, 256)
    
    def forward(self, images, input_ids, attention_mask):
        """
        Args:
            images: (batch, 1, 28, 28)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Extract features
        vision_feat = self.extract_vision_features(images)
        text_feat = self.text_encoder(input_ids, attention_mask)
        
        # Fuse
        fused_feat = self.fusion(vision_feat, text_feat)
        
        # Classify
        logits = self.classifier(fused_feat)
        
        return logits
    
    def forward_vision_only(self, images):
        """Vision-only inference (when text not available)"""
        return self.vision_encoder(images)
    
    def forward_text_only(self, input_ids, attention_mask):
        """Text-only inference (for debugging)"""
        text_feat = self.text_encoder(input_ids, attention_mask)
        # Use zero vision features
        batch_size = text_feat.size(0)
        vision_feat = torch.zeros(batch_size, 256, device=text_feat.device)
        fused_feat = self.fusion(vision_feat, text_feat)
        logits = self.classifier(fused_feat)
        return logits
