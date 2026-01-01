"""
FT-Transformer: Feature Tokenizer + Transformer.

Paper: "Revisiting Deep Learning Models for Tabular Data"
https://arxiv.org/abs/2106.11959

Key features:
- Each feature becomes a token
- [CLS] token for classification
- Self-attention learns feature interactions
- State-of-the-art for tabular data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FeatureTokenizer(nn.Module):
    """
    Convert numerical features to embeddings.

    Each feature gets its own embedding: x_i * w_i + b_i
    """

    def __init__(self, num_features, d_token):
        super(FeatureTokenizer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_features, d_token))

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: (batch, num_features)
        # output: (batch, num_features, d_token)
        x = x.unsqueeze(-1)  # (batch, num_features, 1)
        return x * self.weight + self.bias


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.size()

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)

        if return_attention:
            return output, attn_weights
        return output


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with Pre-LN (Layer Norm before attention).

    Pre-LN is more stable for training than Post-LN.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-LN: normalize before attention
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FTTransformer(nn.Module):
    """
    FT-Transformer: Feature Tokenizer + Transformer.

    Args:
        num_features: Number of input features
        d_token: Dimension of token embeddings
        n_blocks: Number of transformer blocks
        n_heads: Number of attention heads
        d_ff_multiplier: Multiplier for FFN hidden dimension
        dropout: Dropout rate
        attention_dropout: Dropout rate for attention (if different)
    """

    def __init__(self, num_features, d_token=64, n_blocks=3, n_heads=4,
                 d_ff_multiplier=2, dropout=0.2, attention_dropout=None):
        super(FTTransformer, self).__init__()

        self.num_features = num_features
        self.d_token = d_token
        self.n_blocks = n_blocks

        if attention_dropout is None:
            attention_dropout = dropout

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(num_features, d_token)

        # [CLS] token - learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer blocks
        d_ff = d_token * d_ff_multiplier
        self.blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ff, dropout)
            for _ in range(n_blocks)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_token)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1)
        )

    def forward(self, x, return_embeddings=False):
        batch_size = x.size(0)

        # Tokenize features: (batch, num_features, d_token)
        tokens = self.tokenizer(x)

        # Add [CLS] token: (batch, 1 + num_features, d_token)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            tokens = block(tokens)

        # Normalize
        tokens = self.norm(tokens)

        # Use [CLS] token for classification
        cls_output = tokens[:, 0]

        if return_embeddings:
            return self.head(cls_output).squeeze(-1), cls_output

        return self.head(cls_output).squeeze(-1)

    def get_attention_weights(self, x):
        """
        Get attention weights from all layers.

        Useful for interpretability.
        """
        batch_size = x.size(0)
        tokens = self.tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        all_attention_weights = []

        for block in self.blocks:
            # Get attention weights
            normed = block.norm1(tokens)
            _, attn_weights = block.attention(normed, return_attention=True)
            all_attention_weights.append(attn_weights)

            # Continue forward pass
            tokens = tokens + block.attention(normed)
            tokens = tokens + block.ffn(block.norm2(tokens))

        return all_attention_weights

    def get_feature_importance(self, x):
        """
        Get feature importance based on attention to [CLS] token.

        Returns attention from [CLS] to each feature, averaged across heads and layers.
        """
        attention_weights = self.get_attention_weights(x)

        # attention_weights: list of (batch, n_heads, seq_len, seq_len)
        # We want attention from CLS (position 0) to features (positions 1:)

        importance = []
        for attn in attention_weights:
            # attn[:, :, 0, 1:] -> attention from CLS to features
            cls_attention = attn[:, :, 0, 1:]  # (batch, n_heads, num_features)
            importance.append(cls_attention.mean(dim=1))  # Average over heads

        # Stack and average over layers and batch
        importance = torch.stack(importance, dim=0)  # (n_layers, batch, num_features)
        importance = importance.mean(dim=[0, 1])  # (num_features,)

        return importance.detach().cpu().numpy()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"FTTransformer(num_features={self.num_features}, d_token={self.d_token}, "
                f"n_blocks={self.n_blocks}, params={self.get_num_params():,})")