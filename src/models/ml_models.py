"""
Traditional Machine Learning Models for Churn Prediction.

Provides:
- Model factory for creating different classifiers
- Rule-based baseline model
- Model comparison utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


@dataclass
class ModelConfig:
    """Configuration for ML models."""
    n_estimators: int = 5000
    max_depth: int = 8
    learning_rate: float = 0.03
    subsample: float = 0.9
    gamma: float = 0.01
    random_state: int = 42
    early_stopping_rounds: int = 200


class RuleBasedBaseline:
    """
    Rule-based baseline model using recency thresholds.

    Simple rule: If days since last order > threshold, predict churn.
    Each segment can have its own threshold.
    """

    def __init__(self, thresholds: Dict[str, int] = None):
        """
        Args:
            thresholds: Dict mapping segment name to recency threshold (days).
                       If None, uses default thresholds.
        """
        self.thresholds = thresholds or {
            '1 Order': 45,
            '2-4 Orders': 60,
            '5-10 Orders': 30,
            '11-30 Orders': 16,
            '30+ Orders': 14
        }
        self.default_threshold = 30  # For unknown segments

    def predict(self, recency: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """
        Predict churn based on recency thresholds.

        Args:
            recency: Array of days since last order
            segments: Array of segment labels

        Returns:
            Binary predictions (1=churn, 0=active)
        """
        predictions = np.zeros(len(recency), dtype=int)

        for i, (rec, seg) in enumerate(zip(recency, segments)):
            threshold = self.thresholds.get(seg, self.default_threshold)
            predictions[i] = 1 if rec > threshold else 0

        return predictions

    def predict_proba(self, recency: np.ndarray, segments: np.ndarray) -> np.ndarray:
        """
        Generate probability-like scores based on recency ratio.

        Score = recency / threshold (capped at 1.0)
        """
        probas = np.zeros(len(recency))

        for i, (rec, seg) in enumerate(zip(recency, segments)):
            threshold = self.thresholds.get(seg, self.default_threshold)
            # Normalize: recency/threshold, but cap at 1.5 * threshold
            probas[i] = min(rec / threshold, 1.5) / 1.5

        return probas

    def get_thresholds(self) -> Dict[str, int]:
        """Get current thresholds."""
        return self.thresholds.copy()

    def set_thresholds(self, thresholds: Dict[str, int]):
        """Update thresholds."""
        self.thresholds.update(thresholds)


class MLModelFactory:
    """
    Factory class to create different ML models with consistent interface.
    """

    AVAILABLE_MODELS = ['XGBoost', 'LightGBM', 'RandomForest', 'LogisticRegression', 'GradientBoosting']

    @staticmethod
    def create_model(model_name: str,
                     config: ModelConfig = None,
                     is_imbalanced: bool = False,
                     scale_pos_weight: float = 1.0) -> Any:
        """
        Create a model instance.

        Args:
            model_name: One of AVAILABLE_MODELS
            config: Model configuration
            is_imbalanced: Whether to use class weighting
            scale_pos_weight: Weight for positive class

        Returns:
            Instantiated model
        """
        config = config or ModelConfig()

        if model_name == 'XGBoost':
            params = {
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'subsample': config.subsample,
                'colsample_bytree': 0.8,
                'random_state': config.random_state,
                'eval_metric': 'aucpr',  # Better for imbalanced
                'early_stopping_rounds': config.early_stopping_rounds,
                'gamma': config.gamma,
            }
            if is_imbalanced:
                params['scale_pos_weight'] = scale_pos_weight
            return XGBClassifier(**params)

        elif model_name == 'LightGBM':
            params = {
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'subsample': config.subsample,
                'colsample_bytree': 0.8,
                'random_state': config.random_state,
                'verbose': -1,
            }
            if is_imbalanced:
                params['scale_pos_weight'] = scale_pos_weight
            return LGBMClassifier(**params)

        elif model_name == 'RandomForest':
            params = {
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth + 4,  # RF usually needs deeper trees
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': config.random_state,
                'n_jobs': -1,
            }
            if is_imbalanced:
                params['class_weight'] = 'balanced'
            return RandomForestClassifier(**params)

        elif model_name == 'GradientBoosting':
            params = {
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'subsample': config.subsample,
                'random_state': config.random_state,
            }
            return GradientBoostingClassifier(**params)

        elif model_name == 'LogisticRegression':
            params = {
                'random_state': config.random_state,
                'max_iter': 1000,
                'solver': 'lbfgs',
            }
            if is_imbalanced:
                params['class_weight'] = 'balanced'
            return LogisticRegression(**params)

        else:
            raise ValueError(f"Unknown model: {model_name}. "
                           f"Available: {MLModelFactory.AVAILABLE_MODELS}")

    @staticmethod
    def get_model_params(model_name: str) -> Dict:
        """Get default parameters for a model type."""
        defaults = {
            'XGBoost': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
            'LightGBM': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
            'RandomForest': {'n_estimators': 200, 'max_depth': 10},
            'GradientBoosting': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1},
            'LogisticRegression': {'max_iter': 1000}
        }
        return defaults.get(model_name, {})


class Segmenter:
    """
    Handles different segmentation strategies for users.
    """

    def __init__(self, method: str = 'manual', config: Dict = None):
        """
        Args:
            method: 'manual' or 'clustering'
            config: Configuration for segmentation
        """
        self.method = method
        self.config = config or {}
        self.kmeans = None
        self.scaler = None

        # Default manual segments
        if method == 'manual' and 'segments' not in self.config:
            self.config['segments'] = {
                '1 Order': {'min': 1, 'max': 1},
                '2-4 Orders': {'min': 2, 'max': 4},
                '5-10 Orders': {'min': 5, 'max': 10},
                '11-30 Orders': {'min': 11, 'max': 30},
                '30+ Orders': {'min': 31, 'max': 99999}
            }

    def fit_transform(self, df: pd.DataFrame, order_col: str = 'total_orders') -> pd.DataFrame:
        """
        Assign segments to users.

        Args:
            df: DataFrame with user data
            order_col: Column name for order count

        Returns:
            DataFrame with 'segment' column added
        """
        df = df.copy()

        if self.method == 'manual':
            df['segment'] = df[order_col].apply(self._assign_manual_segment)
        elif self.method == 'clustering':
            df = self._clustering_segmentation(df, fit=True)

        return df

    def transform(self, df: pd.DataFrame, order_col: str = 'total_orders') -> pd.DataFrame:
        """Apply fitted segmentation to new data."""
        df = df.copy()

        if self.method == 'manual':
            df['segment'] = df[order_col].apply(self._assign_manual_segment)
        elif self.method == 'clustering':
            df = self._clustering_segmentation(df, fit=False)

        return df

    def _assign_manual_segment(self, order_count: int) -> str:
        """Assign segment based on order count."""
        for seg_name, bounds in self.config['segments'].items():
            if bounds['min'] <= order_count <= bounds['max']:
                return seg_name
        return 'Unknown'

    def _clustering_segmentation(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Segment using K-Means clustering."""
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        features = self.config.get('features', ['total_orders', 'recency', 'tenure_days'])
        n_clusters = self.config.get('n_clusters', 5)

        X = df[features].fillna(0).values

        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = self.kmeans.fit_predict(X_scaled)
        else:
            X_scaled = self.scaler.transform(X)
            clusters = self.kmeans.predict(X_scaled)

        df['segment'] = [f'Cluster_{c}' for c in clusters]
        return df

    def get_segment_names(self) -> List[str]:
        """Get list of segment names."""
        if self.method == 'manual':
            return list(self.config['segments'].keys())
        elif self.method == 'clustering':
            n = self.config.get('n_clusters', 5)
            return [f'Cluster_{i}' for i in range(n)]
        return []

    @staticmethod
    def create_custom_segments(segment_definitions: Dict[str, Dict[str, int]]) -> 'Segmenter':
        """
        Create segmenter with custom segment definitions.

        Example:
            segments = {
                'New': {'min': 1, 'max': 2},
                'Growing': {'min': 3, 'max': 10},
                'Loyal': {'min': 11, 'max': 99999}
            }
            segmenter = Segmenter.create_custom_segments(segments)
        """
        return Segmenter(method='manual', config={'segments': segment_definitions})


def train_model_with_early_stopping(model, X_train, y_train, X_val, y_val,
                                    model_name: str) -> Any:
    """
    Train a model with early stopping (for models that support it).

    Args:
        model: The model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name of the model type

    Returns:
        Trained model
    """
    if model_name in ['XGBoost', 'LightGBM']:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
    elif model_name == 'GradientBoosting':
        # GradientBoosting uses sample_weight for imbalance
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)

    return model


def calculate_class_weight(y: np.ndarray) -> float:
    """Calculate scale_pos_weight for imbalanced data."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    return n_neg / n_pos if n_pos > 0 else 1.0


# =============================================================================
# Per-Segment Training Utilities
# =============================================================================

def train_per_segment_models(
    model_name: str,
    segments: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    segments_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    segments_test: pd.Series,
    imbalanced_segments: List[str] = None,
    model_config: ModelConfig = None
) -> Dict:
    """
    Train a separate model for each segment.

    Args:
        model_name: Name of the model to train ('XGBoost', 'LightGBM', etc.)
        segments: List of segment names
        X_train, y_train: Training features and labels
        segments_train: Segment labels for training data
        X_test, y_test: Test features and labels
        segments_test: Segment labels for test data
        imbalanced_segments: List of segments with imbalanced classes
        model_config: Optional ModelConfig for hyperparameters

    Returns:
        Dict with keys:
            - 'models': Dict[segment, trained_model]
            - 'predictions': Dict[segment, {'y_true', 'y_pred', 'y_prob'}]
            - 'segment_metrics': Dict[segment, {'recall', 'precision', 'f1', 'roc_auc', 'support'}]
    """
    from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

    imbalanced_segments = imbalanced_segments or []
    model_config = model_config or ModelConfig()

    models = {}
    predictions = {}
    segment_metrics = {}

    for segment in segments:
        # Filter data for this segment
        train_mask = segments_train == segment
        test_mask = segments_test == segment

        X_train_seg = X_train[train_mask]
        y_train_seg = y_train[train_mask]
        X_test_seg = X_test[test_mask]
        y_test_seg = y_test[test_mask]

        if len(X_train_seg) == 0 or len(X_test_seg) == 0:
            continue

        # Calculate class weight for this segment
        scale_pos_weight = calculate_class_weight(y_train_seg.values)
        is_imbalanced = segment in imbalanced_segments

        # Create and train model
        model = MLModelFactory.create_model(
            model_name,
            config=model_config,
            is_imbalanced=is_imbalanced,
            scale_pos_weight=scale_pos_weight
        )

        # Fit with early stopping for boosting models
        if model_name in ['XGBoost', 'LightGBM']:
            model.fit(X_train_seg, y_train_seg, eval_set=[(X_test_seg, y_test_seg)])
        else:
            model.fit(X_train_seg, y_train_seg)

        # Predict
        y_pred = model.predict(X_test_seg)
        y_prob = model.predict_proba(X_test_seg)[:, 1]

        # Calculate metrics
        recall = recall_score(y_test_seg, y_pred, zero_division=0)
        precision = precision_score(y_test_seg, y_pred, zero_division=0)
        f1 = f1_score(y_test_seg, y_pred, zero_division=0)
        auc = roc_auc_score(y_test_seg, y_prob) if len(np.unique(y_test_seg)) > 1 else 0

        models[segment] = model
        predictions[segment] = {
            'y_true': y_test_seg.values,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        segment_metrics[segment] = {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': auc,
            'support': len(y_test_seg)
        }

    return {
        'models': models,
        'predictions': predictions,
        'segment_metrics': segment_metrics
    }


def calculate_weighted_recall(segment_metrics: Dict, weights: Dict) -> float:
    """
    Calculate weighted average recall across segments.

    Args:
        segment_metrics: Dict with segment metrics (must have 'recall' key)
        weights: Dict mapping segment names to their weights

    Returns:
        Weighted recall score
    """
    total_weight = 0
    weighted_sum = 0

    for segment, metrics in segment_metrics.items():
        weight = weights.get(segment, 1.0)
        weighted_sum += metrics['recall'] * weight
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0


def combine_segment_predictions(predictions: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine predictions from all segments into single arrays.

    Args:
        predictions: Dict[segment, {'y_true', 'y_pred', 'y_prob'}]

    Returns:
        Tuple of (all_y_true, all_y_pred, all_y_prob, all_segments)
    """
    all_y_true = np.concatenate([p['y_true'] for p in predictions.values()])
    all_y_pred = np.concatenate([p['y_pred'] for p in predictions.values()])
    all_y_prob = np.concatenate([p['y_prob'] for p in predictions.values()])
    all_segments = np.concatenate([[seg] * len(p['y_true'])
                                   for seg, p in predictions.items()])
    return all_y_true, all_y_pred, all_y_prob, all_segments


def print_segment_results(segment_metrics: Dict, weights: Dict, title: str = '') -> float:
    """
    Print formatted results for each segment.

    Args:
        segment_metrics: Dict with segment metrics
        weights: Segment weights for weighted recall
        title: Optional title to print

    Returns:
        Weighted recall value
    """
    if title:
        print(f'\n{title}')
    print('-' * 70)
    print(f'{"Segment":20s} {"Recall":>10s} {"Precision":>10s} {"F1":>10s} {"AUC":>10s}')
    print('-' * 70)

    for segment in sorted(segment_metrics.keys()):
        m = segment_metrics[segment]
        print(f'{segment:20s} {m["recall"]:>10.4f} {m["precision"]:>10.4f} '
              f'{m["f1"]:>10.4f} {m["roc_auc"]:>10.4f}')

    weighted_recall = calculate_weighted_recall(segment_metrics, weights)
    print('-' * 70)
    print(f'{"WEIGHTED RECALL":20s} {weighted_recall:>10.4f}')

    return weighted_recall


def find_optimal_threshold_per_segment(
    predictions: Dict,
    weights: Dict,
    min_precision: float = 0.25,
    min_recall: float = 0.5
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """
    Find optimal threshold for each segment that maximizes weighted recall
    while maintaining reasonable precision.

    The optimization balances:
    1. Maximizing recall (catching churners)
    2. Keeping precision above minimum (avoiding too many false positives)
    3. Using Youden's J statistic (TPR - FPR) as tiebreaker

    Args:
        predictions: Dict[segment, {'y_true', 'y_prob'}]
        weights: Segment weights
        min_precision: Minimum acceptable precision (default 0.25 = 25%)
        min_recall: Minimum acceptable recall (default 0.5 = 50%)

    Returns:
        Tuple of (optimal_thresholds, threshold_details)
    """
    from sklearn.metrics import recall_score, precision_score, f1_score

    optimal_thresholds = {}
    threshold_details = {}

    for segment, pred_data in predictions.items():
        y_true = pred_data['y_true']
        y_prob = pred_data['y_prob']

        # Default threshold
        y_pred_default = (y_prob >= 0.5).astype(int)
        recall_default = recall_score(y_true, y_pred_default, zero_division=0)
        precision_default = precision_score(y_true, y_pred_default, zero_division=0)
        f1_default = f1_score(y_true, y_pred_default, zero_division=0)

        # Search for optimal threshold
        best_threshold = 0.5
        best_score = -1  # Youden's J or weighted score
        best_metrics = {'recall': recall_default, 'precision': precision_default, 'f1': f1_default}

        # Search in reasonable range (0.2 to 0.7)
        for threshold in np.linspace(0.2, 0.7, 51):
            y_pred_t = (y_prob >= threshold).astype(int)

            # Calculate metrics
            recall_t = recall_score(y_true, y_pred_t, zero_division=0)
            precision_t = precision_score(y_true, y_pred_t, zero_division=0)
            f1_t = f1_score(y_true, y_pred_t, zero_division=0)

            # Calculate FPR (False Positive Rate)
            n_neg = (y_true == 0).sum()
            fp = ((y_pred_t == 1) & (y_true == 0)).sum()
            fpr = fp / n_neg if n_neg > 0 else 0

            # Youden's J statistic: TPR - FPR = Recall - FPR
            youden_j = recall_t - fpr

            # Constraints: precision and recall must meet minimums
            if precision_t >= min_precision and recall_t >= min_recall:
                # Score: prioritize Youden's J (balances sensitivity and specificity)
                # With bonus for higher F1
                score = youden_j + 0.2 * f1_t

                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_metrics = {
                        'recall': recall_t,
                        'precision': precision_t,
                        'f1': f1_t,
                        'fpr': fpr,
                        'youden_j': youden_j
                    }

        optimal_thresholds[segment] = best_threshold
        threshold_details[segment] = {
            'default_threshold': 0.5,
            'default_recall': recall_default,
            'default_precision': precision_default,
            'default_f1': f1_default,
            'optimal_threshold': best_threshold,
            'optimal_recall': best_metrics['recall'],
            'optimal_precision': best_metrics['precision'],
            'optimal_f1': best_metrics['f1'],
            'recall_improvement': best_metrics['recall'] - recall_default
        }

    return optimal_thresholds, threshold_details