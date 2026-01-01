"""
Services for User Data and Statistics.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from .models.predictor import ChurnPredictor

logger = logging.getLogger(__name__)


# Feature display names
FEATURE_DISPLAY_NAMES = {
    'recency': 'Days Since Last Order',
    'total_orders': 'Total Orders',
    'orders_last_30d': 'Orders (Last 30 Days)',
    'avg_days_between_orders': 'Avg Days Between Orders',
    'otd_rate': 'On-Time Delivery Rate',
    'late_rate': 'Late Delivery Rate',
    'cancellation_rate': 'Cancellation Rate',
    'total_crm_requests': 'Total CRM Requests',
    'crm_request_rate': 'CRM Request Rate',
    'avg_rate_shop': 'Avg Shop Rating',
    'avg_rate_courier': 'Avg Courier Rating',
    'low_shop_rating_ratio': 'Low Shop Rating %',
    'low_courier_rating_ratio': 'Low Courier Rating %',
    'total_comments': 'Total Comments',
    'comment_rate': 'Comment Rate',
    'tenure_days': 'Customer Tenure (Days)',
    'orders_per_month': 'Orders Per Month',
    'last_order_was_late': 'Last Order Was Late',
    'first_order_had_issue': 'First Order Had Issue',
}


class UserDataService:
    """Manages user data and predictions cache."""

    def __init__(self, features_path: Path, predictor: ChurnPredictor):
        self.features_path = features_path
        self.predictor = predictor

        self.df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = []
        self.cache: Optional[pd.DataFrame] = None

    def load_data(self) -> int:
        """Load user features. Returns user count."""
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features not found: {self.features_path}")

        logger.info(f"Loading features from {self.features_path}")

        self.df = pd.read_csv(self.features_path)
        self.df['user_id'] = self.df['user_id'].astype(str)

        # Feature columns (exclude user_id, is_churned, segment, frequency_segment)
        exclude_cols = ['user_id', 'is_churned', 'segment', 'frequency_segment']
        self.feature_cols = [
            c for c in self.df.columns
            if c not in exclude_cols
        ]

        n = len(self.df)
        logger.info(f"Loaded {n:,} users, {len(self.feature_cols)} features")
        return n

    def cache_predictions(self) -> None:
        """Pre-compute predictions for all users."""
        if self.df is None:
            raise RuntimeError("Data not loaded")

        logger.info("Computing predictions...")

        features = self.df[self.feature_cols].values
        total_orders = self.df['total_orders'].values

        # Validate feature count
        expected = self.predictor.num_features
        actual = features.shape[1]
        if actual != expected:
            raise ValueError(
                f"Feature shape mismatch, expected: {expected}, got {actual}. "
                f"Your user_features.csv has {actual} features but model expects {expected}. "
                f"Use data/processed/train_features.csv or run: python scripts/prepare_user_features.py"
            )

        # Log info about the batch processing
        n_multi = (total_orders > 1).sum()
        batch_size = self.predictor.inference_batch_size
        n_batches = (n_multi + batch_size - 1) // batch_size
        logger.info(f"Processing {n_multi:,} multi-order users in ~{n_batches} batches (batch_size={batch_size})")

        probs, _, risk_levels, models = self.predictor.predict_batch(features, total_orders)

        self.cache = pd.DataFrame({
            'user_id': self.df['user_id'],
            'probability': probs,
            'risk_level': risk_levels,
            'model_used': models,
            'recency': self.df['recency'],
            'total_orders': self.df['total_orders'],
        })

        logger.info(f"✅ Cached predictions for {len(self.cache):,} users")

    def get_user(self, user_id: str) -> Optional[pd.Series]:
        """Get user features row."""
        if self.df is None:
            return None
        row = self.df[self.df['user_id'] == user_id]
        return row.iloc[0] if not row.empty else None

    def get_prediction(self, user_id: str) -> Optional[pd.Series]:
        """Get cached prediction."""
        if self.cache is None:
            return None
        row = self.cache[self.cache['user_id'] == user_id]
        return row.iloc[0] if not row.empty else None

    def is_ready(self) -> bool:
        return self.df is not None and self.cache is not None


class StatsService:
    """Dashboard statistics."""

    def __init__(self, user_service: UserDataService):
        self.user_service = user_service

    def get_overview(self) -> Dict[str, Any]:
        cache = self.user_service.cache
        if cache is None:
            raise RuntimeError("Cache not ready")

        total = len(cache)
        low = (cache['risk_level'] == 'LOW').sum()
        med = (cache['risk_level'] == 'MEDIUM').sum()
        high = (cache['risk_level'] == 'HIGH').sum()

        return {
            'total_users': total,
            'low_risk': int(low),
            'medium_risk': int(med),
            'high_risk': int(high),
            'low_risk_pct': round(low / total * 100, 1),
            'medium_risk_pct': round(med / total * 100, 1),
            'high_risk_pct': round(high / total * 100, 1),
            'avg_churn_probability': round(cache['probability'].mean(), 4),
        }

    def get_users_at_risk(
        self,
        risk_level: Optional[str] = None,
        min_probability: Optional[float] = None,
        min_days_inactive: Optional[int] = None,
        limit: int = 50,
        page: int = 1,
    ) -> Dict[str, Any]:
        cache = self.user_service.cache
        if cache is None:
            raise RuntimeError("Cache not ready")

        df = cache.copy()

        if risk_level:
            df = df[df['risk_level'] == risk_level.upper()]
        if min_probability is not None:
            df = df[df['probability'] >= min_probability]
        if min_days_inactive is not None:
            df = df[df['recency'] >= min_days_inactive]

        df = df.sort_values('probability', ascending=False)

        total = len(df)
        start = (page - 1) * limit
        df_page = df.iloc[start:start+limit]

        users = [
            {
                'user_id': r['user_id'],
                'probability': round(float(r['probability']), 4),
                'risk_level': r['risk_level'],
                'recency': int(r['recency']),
                'total_orders': int(r['total_orders']),
            }
            for _, r in df_page.iterrows()
        ]

        return {'total': total, 'page': page, 'limit': limit, 'users': users}

    def get_risk_distribution(self) -> Dict[str, Any]:
        cache = self.user_service.cache
        if cache is None:
            raise RuntimeError("Cache not ready")

        bins = np.arange(0, 1.05, 0.05)
        hist, edges = np.histogram(cache['probability'], bins=bins)

        return {
            'bins': [round(float(e), 2) for e in edges[:-1]],
            'counts': [int(c) for c in hist],
        }

    def get_feature_importance(self) -> List[Dict[str, Any]]:
        predictor = self.user_service.predictor
        model = predictor.xgboost

        if model is None or not hasattr(model, 'feature_importances_'):
            raise RuntimeError("XGBoost model not available")

        cols = self.user_service.feature_cols
        imps = model.feature_importances_

        result = []
        for feat, imp in sorted(zip(cols, imps), key=lambda x: -x[1])[:15]:
            result.append({
                'feature': feat,
                'importance': round(float(imp), 4),
                'display_name': FEATURE_DISPLAY_NAMES.get(feat, feat),
            })

        return result

    def export_high_risk(self, risk_level: str = "HIGH", limit: int = 1000) -> Dict[str, Any]:
        cache = self.user_service.cache
        if cache is None:
            raise RuntimeError("Cache not ready")

        df = cache[cache['risk_level'] == risk_level.upper()]
        df = df.sort_values('probability', ascending=False).head(limit)

        return {'count': len(df), 'data': df.to_dict('records')}