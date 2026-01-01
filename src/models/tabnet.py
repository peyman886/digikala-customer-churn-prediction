"""
TabNet: Attentive Interpretable Tabular Learning.

Paper: https://arxiv.org/abs/1908.07442

Key features:
- Sequential attention for feature selection
- Instance-wise feature selection
- Interpretable (can see which features are important)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLUBlock(nn.Module):
    """
    Gated Linear Unit Block.

    GLU(x) = x * sigmoid(gate)
    """

    def __init__(self, input_dim, output_dim):
        super(GLUBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.bn = nn.BatchNorm1d(output_dim * 2)
        self.output_dim = output_dim

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class AttentiveTransformer(nn.Module):
    """
    Attention mechanism for feature selection.

    Learns which features to focus on at each step.
    """

    def __init__(self, input_dim, output_dim):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x, priors):
        x = self.fc(x)
        x = self.bn(x)
        x = x * priors
        return F.softmax(x, dim=-1)


class FeatureTransformer(nn.Module):
    """Feature transformer with shared and step-specific layers."""

    def __init__(self, input_dim, output_dim, shared_layers=None, n_independent=2):
        super(FeatureTransformer, self).__init__()

        self.shared = shared_layers

        # Step-specific layers
        self.independent = nn.ModuleList([
            GLUBlock(input_dim if i == 0 else output_dim, output_dim)
            for i in range(n_independent)
        ])

    def forward(self, x):
        if self.shared is not None:
            x = self.shared(x)

        for layer in self.independent:
            x = layer(x)

        return x


class TabNet(nn.Module):
    """
    TabNet: Attentive Interpretable Tabular Learning.

    Args:
        input_dim: Number of input features
        n_steps: Number of decision steps
        n_d: Dimension of decision layer
        n_a: Dimension of attention layer
        gamma: Coefficient for feature reuse (relaxation factor)
        momentum: Momentum for batch normalization
    """

    def __init__(self, input_dim, n_steps=3, n_d=64, n_a=64,
                 gamma=1.3, momentum=0.02):
        super(TabNet, self).__init__()

        self.input_dim = input_dim
        self.n_steps = n_steps
        self.n_d = n_d
        self.n_a = n_a
        self.gamma = gamma

        # Initial batch normalization
        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=momentum)

        # Shared layers across steps
        self.shared_fc = nn.Linear(input_dim, n_d + n_a)
        self.shared_bn = nn.BatchNorm1d(n_d + n_a, momentum=momentum)

        # Step-specific layers
        self.step_attentions = nn.ModuleList([
            AttentiveTransformer(n_a, input_dim)
            for _ in range(n_steps)
        ])

        self.step_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, n_d + n_a),
                nn.BatchNorm1d(n_d + n_a, momentum=momentum),
                nn.GELU()
            )
            for _ in range(n_steps)
        ])

        # Final mapping
        self.final_fc = nn.Linear(n_d, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_attention=False):
        batch_size = x.size(0)

        # Initial transform
        x = self.initial_bn(x)
        x_init = x

        # Prior scales (initialized to 1)
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)

        # Output aggregation
        output_aggregate = torch.zeros(batch_size, self.n_d, device=x.device)

        # Store attention masks for interpretability
        attention_masks = []

        # Initial shared transform
        h = self.shared_fc(x_init)
        h = self.shared_bn(h)
        h = F.gelu(h)

        for step in range(self.n_steps):
            # Split into decision and attention
            d, a = h[:, :self.n_d], h[:, self.n_d:]

            # Attention mask for this step
            mask = self.step_attentions[step](a, prior_scales)
            attention_masks.append(mask)

            # Update prior scales (decrease importance of already-used features)
            prior_scales = prior_scales * (self.gamma - mask)

            # Masked features
            masked_x = x_init * mask

            # Transform for next step
            h = self.step_transforms[step](masked_x)

            # Aggregate decisions (with ReLU for sparsity)
            output_aggregate = output_aggregate + F.relu(d)

        # Final output
        out = self.final_fc(output_aggregate)

        if return_attention:
            return out.squeeze(-1), attention_masks

        return out.squeeze(-1)

    def get_feature_importance(self, x):
        """
        Get feature importance scores based on attention masks.

        Returns average attention across all steps.
        """
        _, attention_masks = self.forward(x, return_attention=True)

        # Stack and average attention masks
        stacked = torch.stack(attention_masks, dim=0)  # (n_steps, batch, input_dim)
        importance = stacked.mean(dim=[0, 1])  # (input_dim,)

        return importance.detach().cpu().numpy()

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (f"TabNet(input_dim={self.input_dim}, n_steps={self.n_steps}, "
                f"n_d={self.n_d}, n_a={self.n_a}, params={self.get_num_params():,})")