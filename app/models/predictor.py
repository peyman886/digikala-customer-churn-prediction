"""
Churn Predictor

Handles model loading and prediction routing.
Strategy:
    - 1-Order users: XGBoost
    - 2+ Orders users: FT-Transformer
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import joblib
import torch

from .ft_transformer import FTTransformer

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Unified churn predictor with multi-model routing.

    Routing:
        total_orders == 1  ->  XGBoost
        total_orders >= 2  ->  FT-Transformer (with scaler)
    """

    # Risk thresholds
    THRESHOLDS = {"HIGH": 0.7, "MEDIUM": 0.4}

    def __init__(
        self,
        xgboost_path: Path,
        transformer_path: Path,
        scaler_path: Path,
        device: torch.device,
        # FT-Transformer architecture params
        num_features: int = 98,
        d_token: int = 64,
        n_blocks: int = 3,
        n_heads: int = 4,
        d_ff_multiplier: int = 2,
        dropout: float = 0.2,
        # Inference settings
        inference_batch_size: int = 1024,
    ):
        self.device = device
        self.xgboost_path = xgboost_path
        self.transformer_path = transformer_path
        self.scaler_path = scaler_path

        # Architecture config
        self.num_features = num_features
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.d_ff_multiplier = d_ff_multiplier
        self.dropout = dropout

        # Inference config
        self.inference_batch_size = inference_batch_size

        # Models (loaded later)
        self.xgboost = None
        self.transformer = None
        self.scaler = None

        # Status
        self._xgboost_loaded = False
        self._transformer_loaded = False

    def load_models(self) -> dict:
        """Load all models. Returns status dict."""
        status = {"xgboost": False, "transformer": False}

        # Load XGBoost
        try:
            if self.xgboost_path.exists():
                self.xgboost = joblib.load(self.xgboost_path)
                self._xgboost_loaded = True
                status["xgboost"] = True
                logger.info(f"✅ XGBoost loaded from {self.xgboost_path}")
            else:
                logger.error(f"❌ XGBoost not found: {self.xgboost_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load XGBoost: {e}")

        # Load FT-Transformer
        try:
            if self.transformer_path.exists() and self.scaler_path.exists():
                # Create model architecture
                self.transformer = FTTransformer(
                    num_features=self.num_features,
                    d_token=self.d_token,
                    n_blocks=self.n_blocks,
                    n_heads=self.n_heads,
                    d_ff_multiplier=self.d_ff_multiplier,
                    dropout=self.dropout,
                )

                # Load weights
                state_dict = torch.load(self.transformer_path, map_location=self.device)
                self.transformer.load_state_dict(state_dict)
                self.transformer.to(self.device)
                self.transformer.eval()

                # Load scaler
                self.scaler = joblib.load(self.scaler_path)

                self._transformer_loaded = True
                status["transformer"] = True
                logger.info(f"✅ FT-Transformer loaded: {self.transformer}")
                logger.info(f"✅ Scaler loaded from {self.scaler_path}")
            else:
                if not self.transformer_path.exists():
                    logger.error(f"❌ Transformer not found: {self.transformer_path}")
                if not self.scaler_path.exists():
                    logger.error(f"❌ Scaler not found: {self.scaler_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load Transformer: {e}")

        return status

    def predict_single(
        self,
        features: np.ndarray,
        total_orders: int
    ) -> Tuple[float, bool, str, str]:
        """
        Predict for a single user.

        Args:
            features: Feature array (1D or 2D)
            total_orders: User's total order count

        Returns:
            (probability, will_churn, risk_level, model_used)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Route to appropriate model
        if total_orders <= 1:
            if not self._xgboost_loaded:
                raise RuntimeError("XGBoost model not loaded")
            prob = float(self.xgboost.predict_proba(features)[0, 1])
            model_used = "xgboost"
        else:
            if not self._transformer_loaded:
                raise RuntimeError("FT-Transformer not loaded")
            prob = self._predict_transformer(features)[0]
            model_used = "ft_transformer"

        will_churn = prob >= 0.5
        risk_level = self._get_risk_level(prob)

        return prob, will_churn, risk_level, model_used

    def predict_batch(
        self,
        features: np.ndarray,
        total_orders: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, list, list]:
        """
        Batch prediction with automatic routing.

        Returns:
            (probabilities, will_churn, risk_levels, models_used)
        """
        n = len(features)
        probs = np.zeros(n)
        models = [""] * n

        # Masks
        mask_1order = total_orders <= 1
        mask_multi = ~mask_1order

        # XGBoost for 1-order users
        if mask_1order.any() and self._xgboost_loaded:
            probs[mask_1order] = self.xgboost.predict_proba(features[mask_1order])[:, 1]
            for i in np.where(mask_1order)[0]:
                models[i] = "xgboost"

        # Transformer for multi-order users
        if mask_multi.any() and self._transformer_loaded:
            probs[mask_multi] = self._predict_transformer(
                features[mask_multi],
                batch_size=self.inference_batch_size
            )
            for i in np.where(mask_multi)[0]:
                models[i] = "ft_transformer"

        will_churn = probs >= 0.5
        risk_levels = [self._get_risk_level(p) for p in probs]

        return probs, will_churn, risk_levels, models

    def _predict_transformer(self, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Run transformer inference with scaling and batching.

        Args:
            features: Feature array
            batch_size: Batch size for GPU inference (default 1024 to avoid OOM)
        """
        # Validate feature dimensions
        expected_features = self.num_features
        actual_features = features.shape[1]

        if actual_features != expected_features:
            raise ValueError(
                f"Feature shape mismatch, expected: {expected_features}, got {actual_features}. "
                f"Make sure user_features.csv has the same features used during training. "
                f"Run: python scripts/prepare_user_features.py"
            )

        # Scale all at once (CPU operation, memory efficient)
        scaled = self.scaler.transform(features)

        n_samples = len(scaled)
        all_probs = []

        # Process in batches to avoid GPU OOM
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = scaled[start:end]

            # To tensor
            x = torch.tensor(batch, dtype=torch.float32, device=self.device)

            # Inference
            with torch.no_grad():
                logits = self.transformer(x)
                probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())

            # Clear GPU cache periodically
            if self.device.type == 'cuda' and (start // batch_size) % 10 == 0:
                torch.cuda.empty_cache()

        return np.concatenate(all_probs)

    def _get_risk_level(self, prob: float) -> str:
        if prob >= self.THRESHOLDS["HIGH"]:
            return "HIGH"
        elif prob >= self.THRESHOLDS["MEDIUM"]:
            return "MEDIUM"
        return "LOW"

    def get_status(self) -> dict:
        return {
            "xgboost": self._xgboost_loaded,
            "ft_transformer": self._transformer_loaded,
        }

    def get_device_info(self) -> str:
        if self.device.type == "cuda":
            return f"cuda ({torch.cuda.get_device_name(0)})"
        return "cpu"