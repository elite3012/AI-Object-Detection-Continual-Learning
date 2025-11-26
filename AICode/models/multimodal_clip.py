"""
Multi-Modal Continual Learning with CLIP-style Architecture
Vision + Language for Fashion-MNIST with text descriptions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vision_transformer import LightweightViT


class TextEncoder(nn.Module):
    """
    Simple text encoder for fashion item descriptions
    Uses word embeddings + Transformer (optimized for speed)
    """
    def __init__(self, vocab_size=1000, embed_dim=192, num_heads=3, num_layers=2, max_seq_len=32):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Word embeddings
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        
        # Transformer encoder for text (reduced layers for speed)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 3,  # Reduced from 4x to 3x
            dropout=0.05,  # Reduced dropout
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.ln = nn.LayerNorm(embed_dim)
        
        # Initialize
        nn.init.normal_(self.word_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, text_tokens):
        """
        Args:
            text_tokens: [batch, seq_len] token indices
        Returns:
            text_features: [batch, embed_dim]
        """
        batch_size, seq_len = text_tokens.shape
        
        # Word embeddings + positional embeddings
        x = self.word_embed(text_tokens)  # [batch, seq_len, embed_dim]
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Transformer encoding
        x = self.transformer(x)  # [batch, seq_len, embed_dim]
        
        # Pool: use [CLS] token (first token) or mean pooling
        # Here we use mean pooling for simplicity
        text_features = x.mean(dim=1)  # [batch, embed_dim]
        text_features = self.ln(text_features)
        
        return text_features


class CLIPStyleModel(nn.Module):
    """
    CLIP-style multi-modal model for Fashion-MNIST
    Joint vision-language embedding space
    """
    def __init__(self, num_classes=10, embed_dim=192, temperature=0.07):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.temperature = nn.Parameter(torch.ones([]) * temperature)
        
        # Vision encoder (lightweight ViT - optimized)
        self.vision_encoder = LightweightViT(
            num_classes=embed_dim,  # Output embedding instead of logits
            img_size=28,
            patch_size=4,
            embed_dim=embed_dim,
            depth=5,  # Reduced from 6 to 5 layers
            num_heads=3,
            dropout=0.05  # Reduced dropout
        )
        
        # Replace final classification head with projection
        self.vision_encoder.head = nn.Linear(embed_dim, embed_dim)
        
        # Text encoder (optimized: 2 layers instead of 3)
        self.text_encoder = TextEncoder(
            vocab_size=1000,
            embed_dim=embed_dim,
            num_heads=3,
            num_layers=2,
            max_seq_len=32
        )
        
        # Learnable class text embeddings (for zero-shot classification)
        self.class_text_embeds = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.normal_(self.class_text_embeds, std=0.02)
        
        # Cache for text embeddings (avoid re-encoding same text)
        self.register_buffer('_text_cache', torch.zeros(0, embed_dim))
        self.register_buffer('_text_cache_keys', torch.zeros(0, 32, dtype=torch.long))
    
    def encode_image(self, images):
        """Encode images to embedding space"""
        image_features = self.vision_encoder(images)
        # L2 normalize
        image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def encode_text(self, text_tokens, use_cache=True):
        """Encode text to embedding space with optional caching"""
        if use_cache and self.training and self._text_cache.size(0) > 0:
            # Check cache for exact matches
            batch_size = text_tokens.size(0)
            cached_features = []
            uncached_indices = []
            uncached_tokens = []
            
            for i in range(batch_size):
                # Simple cache lookup (exact token match)
                match_idx = (self._text_cache_keys == text_tokens[i].unsqueeze(0)).all(dim=1).nonzero(as_tuple=True)[0]
                if len(match_idx) > 0:
                    cached_features.append(self._text_cache[match_idx[0]])
                else:
                    uncached_indices.append(i)
                    uncached_tokens.append(text_tokens[i])
            
            if len(uncached_indices) == 0:
                # All cached
                return torch.stack(cached_features)
            
            # Encode uncached
            if len(uncached_tokens) > 0:
                uncached_tokens = torch.stack(uncached_tokens)
                new_features = self.text_encoder(uncached_tokens)
                new_features = F.normalize(new_features, dim=-1)
                
                # Merge cached and new
                all_features = torch.zeros(batch_size, self.embed_dim, device=text_tokens.device)
                cached_idx = [i for i in range(batch_size) if i not in uncached_indices]
                if len(cached_idx) > 0:
                    all_features[cached_idx] = torch.stack(cached_features)
                all_features[uncached_indices] = new_features
                return all_features
        
        # Standard encoding (no cache or eval mode)
        text_features = self.text_encoder(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        return text_features
    
    def forward(self, images, text_tokens=None, return_embeddings=False):
        """
        Forward pass with optional text
        
        Args:
            images: [batch, 1, 28, 28]
            text_tokens: [batch, seq_len] or None
            return_embeddings: return features instead of logits
            
        Returns:
            logits or (image_features, text_features)
        """
        # Encode images
        image_features = self.encode_image(images)
        
        if return_embeddings:
            if text_tokens is not None:
                text_features = self.encode_text(text_tokens)
                return image_features, text_features
            return image_features
        
        # Classification using class text embeddings
        # Compute similarity to each class text embedding
        class_embeds = F.normalize(self.class_text_embeds, dim=-1)
        logits = image_features @ class_embeds.t()  # [batch, num_classes]
        logits = logits / self.temperature
        
        return logits
    
    def contrastive_loss(self, image_features, text_features):
        """
        Compute contrastive loss (like CLIP)
        
        Args:
            image_features: [batch, embed_dim]
            text_features: [batch, embed_dim]
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity matrix
        logits_per_image = image_features @ text_features.t() / self.temperature
        logits_per_text = logits_per_image.t()
        
        # Labels: diagonal elements (image i matches text i)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)
        
        # Cross-entropy loss (symmetric)
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


class MultiModalFusionModel(nn.Module):
    """
    Multi-modal fusion for continual learning (optimized for speed)
    Combines vision + text features with cross-modal attention
    """
    def __init__(self, num_classes=10, embed_dim=192):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Vision encoder (optimized: 5 layers)
        self.vision_encoder = LightweightViT(
            num_classes=embed_dim,
            img_size=28,
            patch_size=4,
            embed_dim=embed_dim,
            depth=5,
            num_heads=3,
            dropout=0.05
        )
        self.vision_encoder.head = nn.Linear(embed_dim, embed_dim)
        
        # Text encoder (optimized: 2 layers)
        self.text_encoder = TextEncoder(
            vocab_size=1000,
            embed_dim=embed_dim,
            num_heads=3,
            num_layers=2
        )
        
        # Cross-modal attention (vision attends to text)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=3,
            dropout=0.1,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
    
    def forward(self, images, text_tokens):
        """
        Forward with vision-text fusion
        
        Args:
            images: [batch, 1, 28, 28]
            text_tokens: [batch, seq_len]
        """
        # Encode both modalities
        vision_feat = self.vision_encoder(images)  # [batch, embed_dim]
        text_feat = self.text_encoder(text_tokens)  # [batch, embed_dim]
        
        # Add sequence dimension for attention
        vision_feat = vision_feat.unsqueeze(1)  # [batch, 1, embed_dim]
        text_feat = text_feat.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Cross-attention: vision queries text
        fused_feat, _ = self.cross_attention(
            query=vision_feat,
            key=text_feat,
            value=text_feat
        )
        fused_feat = fused_feat.squeeze(1)  # [batch, embed_dim]
        
        # Residual connection + fusion
        fused_feat = fused_feat + vision_feat.squeeze(1)
        fused_feat = self.fusion(fused_feat)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        return logits


def create_multimodal_model(model_type='clip', num_classes=10, embed_dim=192):
    """
    Factory function for multi-modal models
    
    Args:
        model_type: 'clip' or 'fusion'
        num_classes: Number of output classes
        embed_dim: Embedding dimension
    """
    if model_type == 'clip':
        return CLIPStyleModel(num_classes, embed_dim)
    elif model_type == 'fusion':
        return MultiModalFusionModel(num_classes, embed_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
