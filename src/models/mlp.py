"""
MLP (Multi-Layer Perceptron) for Tabular Data.

A simple but effective baseline neural network for tabular data.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Tabular Data.

    Architecture:
    - Input → BatchNorm → Dense layers with dropout
    - GELU activation (smoother than ReLU)
    - Residual connections optional

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        use_residual: Whether to use residual connections (when dims match)
    """

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3, use_residual=False):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual

        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)

        # Build hidden layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(self._make_block(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _make_block(self, in_dim, out_dim, dropout):
        """Create a single hidden block."""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_bn(x)

        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x.squeeze(-1)

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"MLP(input_dim={self.input_dim}, "
                f"hidden_dims={self.hidden_dims}, "
                f"params={self.get_num_params():,})")