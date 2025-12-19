"""
Lightweight Text Encoder for Multi-Modal Learning
Mini-BERT style transformer for encoding Fashion-MNIST text descriptions
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SimpleTextEncoder(nn.Module):
    """
    Lightweight text encoder for Fashion-MNIST descriptions
    Mini-BERT style with 2-4 transformer layers
    
    Input: Tokenized text (batch, seq_len)
    Output: Text embeddings (batch, embed_dim)
    """
    def __init__(self, vocab_size=5000, embed_dim=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Token embedding + positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Projection head (optional, for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch, seq_len) - tokenized text
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            text_features: (batch, embed_dim)
        """
        # Embed tokens
        x = self.token_embedding(input_ids)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Create attention mask for transformer (True = ignore)
        if attention_mask is not None:
            # Invert: 0 -> True (padding), 1 -> False (real)
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling over sequence length
        if attention_mask is not None:
            # Mask out padding before pooling
            x = x * attention_mask.unsqueeze(-1)
            pooled = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = x.mean(dim=1)
        
        # Normalize
        pooled = self.norm(pooled)
        
        # Optional projection
        projected = self.projection(pooled)
        
        return projected

class CharacterTokenizer:
    """
    Simple character-level tokenizer for Fashion-MNIST text
    Lightweight alternative to WordPiece/BPE
    """
    def __init__(self, max_length=64):
        self.max_length = max_length
        
        # Character vocabulary (a-z, space, punctuation)
        chars = 'abcdefghijklmnopqrstuvwxyz -.,/():'
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(chars)}  # 0 reserved for padding
        self.char_to_idx['<PAD>'] = 0
        self.char_to_idx['<UNK>'] = len(self.char_to_idx)
        
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def encode(self, text):
        """
        Encode text to token IDs
        
        Args:
            text: String or list of strings
        
        Returns:
            input_ids: (batch, max_length) tensor
            attention_mask: (batch, max_length) tensor
        """
        if isinstance(text, str):
            text = [text]
        
        batch_size = len(text)
        input_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        
        for i, sentence in enumerate(text):
            sentence = sentence.lower()
            tokens = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in sentence]
            
            # Truncate or pad
            length = min(len(tokens), self.max_length)
            input_ids[i, :length] = torch.tensor(tokens[:length])
            attention_mask[i, :length] = 1
        
        return input_ids, attention_mask
    
    def decode(self, token_ids):
        """Decode token IDs back to text"""
        if len(token_ids.shape) == 1:
            token_ids = token_ids.unsqueeze(0)
        
        texts = []
        for sequence in token_ids:
            chars = [self.idx_to_char.get(int(idx), '') for idx in sequence if int(idx) != 0]
            texts.append(''.join(chars))
        
        return texts[0] if len(texts) == 1 else texts

# Pre-initialized tokenizer singleton
_tokenizer = None

def get_tokenizer():
    """Get or create tokenizer instance"""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = CharacterTokenizer(max_length=64)
    return _tokenizer

def encode_texts(texts, device='cpu'):
    """
    Convenience function to encode texts
    
    Args:
        texts: String or list of strings
        device: torch device
    
    Returns:
        input_ids, attention_mask tensors on device
    """
    tokenizer = get_tokenizer()
    input_ids, attention_mask = tokenizer.encode(texts)
    return input_ids.to(device), attention_mask.to(device)
