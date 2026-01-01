"""
Training Module.

Provides training utilities including:
- Trainer: Main training loop with segment-wise tracking
- FocalLoss: For imbalanced data
- RecallFocusedLoss: Emphasizes recall
- EarlyStopping: Based on weighted recall
"""

from .trainer import (
    Trainer,
    TrainingConfig,
    EarlyStopping,
    FocalLoss,
    RecallFocusedLoss
)

__all__ = [
    'Trainer',
    'TrainingConfig',
    'EarlyStopping',
    'FocalLoss',
    'RecallFocusedLoss'
]