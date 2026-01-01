"""
Training Module.

Provides training loop with:
- Early stopping based on weighted recall
- Per-segment recall tracking each epoch
- Learning rate scheduling
- Gradient clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import time


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 15
    min_delta: float = 0.001
    gradient_clip: float = 1.0
    pos_weight: float = 2.0  # Weight for positive class in BCE
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Segment weights for weighted recall
    segment_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.segment_weights is None:
            self.segment_weights = {
                '2-4 Orders': 1.0,
                '5-10 Orders': 1.5,
                '11-30 Orders': 2.0,
                '30+ Orders': 3.0
            }


class EarlyStopping:
    """
    Early stopping based on weighted recall (or other metric).

    Stops training when metric doesn't improve for `patience` epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = 'max', restore_best: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics where higher is better, 'min' for loss
            restore_best: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best

        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if should stop training.

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self._save_weights(model)
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self._save_weights(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False

    def _save_weights(self, model: nn.Module):
        """Save model weights."""
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def restore(self, model: nn.Module):
        """Restore best weights to model."""
        if self.restore_best and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    FL(p) = -α * (1-p)^γ * log(p) for positive class

    Focuses training on hard examples by down-weighting easy ones.
    Better than simple class weighting for imbalanced data.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weight for positive class (0-1)
            gamma: Focusing parameter. Higher = more focus on hard examples
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Raw logits (before sigmoid)
            targets: Binary targets (0 or 1)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # p_t = p for positive, 1-p for negative
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * ce_loss

        return loss.mean()


class RecallFocusedLoss(nn.Module):
    """
    Custom loss that emphasizes recall.

    Combines BCE with an extra penalty for false negatives.
    """

    def __init__(self, fn_weight: float = 2.0, pos_weight: float = 1.5):
        """
        Args:
            fn_weight: Extra weight for false negatives (misclassified positives)
            pos_weight: Weight for positive class in BCE
        """
        super(RecallFocusedLoss, self).__init__()
        self.fn_weight = fn_weight
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard BCE with pos_weight
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets,
            pos_weight=torch.tensor([self.pos_weight], device=inputs.device)
        )

        # Extra penalty for false negatives
        # When target=1 and prediction is low, add extra loss
        p = torch.sigmoid(inputs)
        fn_penalty = targets * (1 - p) ** 2  # Higher when missing true positives

        return bce + self.fn_weight * fn_penalty.mean()


class Trainer:
    """
    Main trainer class with segment-wise recall tracking.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig,
                 loss_fn: str = 'focal'):
        """
        Args:
            model: PyTorch model to train
            config: Training configuration
            loss_fn: Loss function type: 'bce', 'focal', 'recall_focused'
        """
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device

        # Setup loss function
        if loss_fn == 'focal':
            self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif loss_fn == 'recall_focused':
            self.criterion = RecallFocusedLoss(fn_weight=2.0, pos_weight=config.pos_weight)
        else:  # bce
            pos_weight = torch.tensor([config.pos_weight]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max',
            restore_best=True
        )

        # Segment weights for weighted recall
        self.segment_weights = config.segment_weights

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_weighted_recall': [],
            'val_segment_recall': {},
            'learning_rate': []
        }

    def _train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()

            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _evaluate(self, data_loader, segments: np.ndarray = None) -> Dict:
        """
        Evaluate model with segment-wise metrics.

        Args:
            data_loader: DataLoader for evaluation
            segments: Segment labels for each sample (optional)

        Returns:
            Dictionary with loss, predictions, probabilities, and segment metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)

        # Calculate segment-wise recall
        segment_recalls = {}
        if segments is not None:
            unique_segments = np.unique(segments)
            for seg in unique_segments:
                mask = segments == seg
                if mask.sum() > 0 and all_targets[mask].sum() > 0:
                    recall = (all_preds[mask] * all_targets[mask]).sum() / all_targets[mask].sum()
                    segment_recalls[seg] = float(recall)

        # Calculate weighted recall
        weighted_recall = self._calculate_weighted_recall(segment_recalls)

        # Overall recall
        overall_recall = all_preds[all_targets == 1].sum() / all_targets.sum() if all_targets.sum() > 0 else 0

        return {
            'loss': total_loss / len(data_loader),
            'predictions': all_preds,
            'probabilities': all_probs,
            'targets': all_targets,
            'segment_recalls': segment_recalls,
            'weighted_recall': weighted_recall,
            'overall_recall': float(overall_recall)
        }

    def _calculate_weighted_recall(self, segment_recalls: Dict[str, float]) -> float:
        """Calculate weighted average recall."""
        if not segment_recalls:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for segment, recall in segment_recalls.items():
            weight = self.segment_weights.get(segment, 1.0)
            weighted_sum += recall * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _format_segment_recalls(self, segment_recalls: Dict[str, float]) -> str:
        """Format segment recalls for printing."""
        parts = [f"{seg}: {recall:.4f}" for seg, recall in sorted(segment_recalls.items())]
        return " | ".join(parts)

    def train(self, train_loader, val_loader, val_segments: np.ndarray,
              print_every: int = 1) -> Dict:
        """
        Full training loop with segment-wise recall tracking.

        Args:
            train_loader: DataLoader for training
            val_loader: DataLoader for validation
            val_segments: Segment labels for validation data
            print_every: Print metrics every N epochs

        Returns:
            Training history
        """
        print(f"\n{'=' * 80}")
        print(f"🚀 Starting Training")
        print(f"{'=' * 80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.epochs}")
        print(f"Patience: {self.config.patience}")
        print(f"Segment Weights: {self.segment_weights}")
        print(f"{'=' * 80}\n")

        start_time = time.time()

        for epoch in range(self.config.epochs):
            # Train
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_results = self._evaluate(val_loader, val_segments)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_weighted_recall'].append(val_results['weighted_recall'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])

            for seg, recall in val_results['segment_recalls'].items():
                if seg not in self.history['val_segment_recall']:
                    self.history['val_segment_recall'][seg] = []
                self.history['val_segment_recall'][seg].append(recall)

            # Scheduler step (based on weighted recall)
            self.scheduler.step(val_results['weighted_recall'])

            # Print progress
            if (epoch + 1) % print_every == 0 or epoch == 0:
                seg_str = self._format_segment_recalls(val_results['segment_recalls'])
                print(f"Epoch {epoch + 1:3d}/{self.config.epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_results['loss']:.4f}")
                print(f"           Segment Recalls: {seg_str}")
                print(f"           Weighted Recall: {val_results['weighted_recall']:.4f} | "
                      f"Overall Recall: {val_results['overall_recall']:.4f}")
                print("-" * 80)

            # Early stopping (based on weighted recall)
            if self.early_stopping(val_results['weighted_recall'], self.model, epoch):
                print(f"\n⚠️ Early stopping at epoch {epoch + 1}")
                print(f"   Best epoch: {self.early_stopping.best_epoch + 1}")
                print(f"   Best weighted recall: {self.early_stopping.best_score:.4f}")
                break

        # Restore best weights
        self.early_stopping.restore(self.model)

        elapsed = time.time() - start_time
        print(f"\n{'=' * 80}")
        print(f"✅ Training Complete in {elapsed:.1f}s")
        print(f"   Best Epoch: {self.early_stopping.best_epoch + 1}")
        print(f"   Best Weighted Recall: {self.early_stopping.best_score:.4f}")
        print(f"{'=' * 80}")

        return self.history

    def predict(self, data_loader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.

        Returns:
            (predictions, probabilities)
        """
        self.model.eval()
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds), np.array(all_probs)