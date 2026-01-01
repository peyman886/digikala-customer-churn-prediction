"""
Evaluation Metrics Module.

Provides comprehensive metrics calculation including:
- Standard metrics (accuracy, precision, recall, F1, AUC)
- Segment-wise metrics
- Weighted aggregate metrics with custom weights
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SegmentWeights:
    """
    Weights for each segment for weighted average calculations.

    Higher weight = more important segment.
    Weights are normalized internally.
    """
    weights: Dict[str, float] = field(default_factory=lambda: {
        '2-4 Orders': 1.0,  # Base weight
        '5-10 Orders': 1.5,  # More valuable users
        '11-30 Orders': 2.0,  # Loyal users
        '30+ Orders': 3.0  # VIP users - most important to retain!
    })

    def get_normalized(self) -> Dict[str, float]:
        """Return normalized weights that sum to 1."""
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}

    def update(self, segment: str, weight: float):
        """Update weight for a segment."""
        self.weights[segment] = weight


class MetricsCalculator:
    """
    Calculate comprehensive metrics including segment-wise breakdown.
    """

    def __init__(self, segment_weights: Optional[SegmentWeights] = None):
        """
        Args:
            segment_weights: Custom weights for weighted average.
                            If None, uses default weights.
        """
        self.segment_weights = segment_weights or SegmentWeights()

    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_prob: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard classification metrics.

        Returns:
            Dictionary with accuracy, precision, recall, f1, roc_auc, pr_auc
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
            'pr_auc': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0,
            'support': len(y_true),
            'positive_rate': y_true.mean()
        }
        return metrics

    def calculate_segment_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_prob: np.ndarray, segments: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for each segment separately.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            segments: Segment labels for each sample

        Returns:
            Dictionary mapping segment name to metrics dict
        """
        segment_metrics = {}
        unique_segments = np.unique(segments)

        for segment in unique_segments:
            mask = segments == segment
            if mask.sum() == 0:
                continue

            seg_metrics = self.calculate_basic_metrics(
                y_true[mask], y_pred[mask], y_prob[mask]
            )
            segment_metrics[segment] = seg_metrics

        return segment_metrics

    def calculate_weighted_recall(self, segment_metrics: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate weighted average recall across segments.

        Uses segment_weights to weight the importance of each segment.

        Args:
            segment_metrics: Output from calculate_segment_metrics

        Returns:
            Weighted average recall
        """
        normalized_weights = self.segment_weights.get_normalized()

        weighted_sum = 0.0
        total_weight = 0.0

        for segment, metrics in segment_metrics.items():
            weight = normalized_weights.get(segment, 1.0)  # Default weight if not specified
            weighted_sum += metrics['recall'] * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_prob: np.ndarray, segments: np.ndarray) -> Dict:
        """
        Calculate all metrics: overall, per-segment, and weighted.

        Returns:
            Dictionary with 'overall', 'per_segment', 'weighted_recall'
        """
        overall = self.calculate_basic_metrics(y_true, y_pred, y_prob)
        per_segment = self.calculate_segment_metrics(y_true, y_pred, y_prob, segments)
        weighted_recall = self.calculate_weighted_recall(per_segment)

        return {
            'overall': overall,
            'per_segment': per_segment,
            'weighted_recall': weighted_recall
        }

    def format_epoch_metrics(self, segment_metrics: Dict[str, Dict[str, float]],
                             weighted_recall: float, epoch: int) -> str:
        """
        Format metrics for epoch logging.

        Returns nicely formatted string showing recall per segment.
        """
        lines = [f"Epoch {epoch:3d} | Recall per segment:"]

        for segment in sorted(segment_metrics.keys()):
            recall = segment_metrics[segment]['recall']
            support = segment_metrics[segment]['support']
            lines.append(f"  {segment:15s}: {recall:.4f} (n={support:,})")

        lines.append(f"  {'Weighted Avg':15s}: {weighted_recall:.4f}")

        return '\n'.join(lines)


def get_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                          metric: str = 'f1') -> Tuple[float, float]:
    """
    Find optimal classification threshold for a given metric.

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        metric: One of 'f1', 'recall', 'precision', 'youden' (TPR - FPR)

    Returns:
        (optimal_threshold, best_score)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'youden':
            # Youden's J statistic: TPR - FPR
            tn = ((y_true == 0) & (y_pred == 0)).sum()
            fp = ((y_true == 0) & (y_pred == 1)).sum()
            tp = ((y_true == 1) & (y_pred == 1)).sum()
            fn = ((y_true == 1) & (y_pred == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            score = tpr - fpr
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score


def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                segment_metrics: Optional[Dict] = None):
    """
    Print comprehensive classification report.
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=['Active', 'Churned']))

    if segment_metrics:
        print("\n" + "=" * 60)
        print("PER-SEGMENT METRICS")
        print("=" * 60)

        df = pd.DataFrame(segment_metrics).T
        df = df[['recall', 'precision', 'f1', 'roc_auc', 'support']]
        print(df.round(4).to_string())


class RecallFocusedMetrics:
    """
    Metrics tracker that prioritizes recall.

    Useful when recall is the most important metric.
    """

    def __init__(self, segment_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            segment_weights: Custom weights for weighted recall.
        """
        self.calculator = MetricsCalculator(
            SegmentWeights(segment_weights) if segment_weights else None
        )
        self.history = {
            'epoch': [],
            'weighted_recall': [],
            'segment_recalls': {}
        }

    def update(self, epoch: int, y_true: np.ndarray, y_pred: np.ndarray,
               y_prob: np.ndarray, segments: np.ndarray):
        """
        Update history with new epoch metrics.
        """
        all_metrics = self.calculator.calculate_all_metrics(y_true, y_pred, y_prob, segments)

        self.history['epoch'].append(epoch)
        self.history['weighted_recall'].append(all_metrics['weighted_recall'])

        for segment, metrics in all_metrics['per_segment'].items():
            if segment not in self.history['segment_recalls']:
                self.history['segment_recalls'][segment] = []
            self.history['segment_recalls'][segment].append(metrics['recall'])

        return all_metrics

    def get_best_epoch(self) -> int:
        """Get epoch with best weighted recall."""
        if not self.history['weighted_recall']:
            return 0
        return self.history['epoch'][np.argmax(self.history['weighted_recall'])]

    def get_history_df(self) -> pd.DataFrame:
        """Get history as DataFrame."""
        data = {'epoch': self.history['epoch'],
                'weighted_recall': self.history['weighted_recall']}

        for segment, recalls in self.history['segment_recalls'].items():
            data[f'recall_{segment}'] = recalls

        return pd.DataFrame(data)