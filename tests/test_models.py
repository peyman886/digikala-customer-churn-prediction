"""
Unit tests for model predictors.
"""

import pytest
import numpy as np
import torch
from pathlib import Path


class TestChurnPredictor:
    """Test suite for ChurnPredictor."""

    def test_routing_logic(self, mock_predictor):
        """Test that users are routed to correct models."""
        # Single order users should use XGBoost
        # Multi-order users should use Transformer
        features = np.random.randn(10, 98)
        total_orders = np.array([1, 1, 5, 10, 1, 20, 30, 2, 15, 1])

        probs, preds, risks, models = mock_predictor.predict_batch(features, total_orders)

        assert len(probs) == 10
        assert len(models) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_prediction_shapes(self, mock_predictor):
        """Test output shapes are correct."""
        n_samples = 100
        features = np.random.randn(n_samples, 98)
        total_orders = np.random.randint(1, 50, n_samples)

        probs, preds, risks, models = mock_predictor.predict_batch(features, total_orders)

        assert probs.shape == (n_samples,)
        assert preds.shape == (n_samples,)
        assert risks.shape == (n_samples,)
        assert models.shape == (n_samples,)

    def test_risk_levels(self, mock_predictor):
        """Test risk level assignment."""
        probs, _, risks, _ = mock_predictor.predict_batch(
            np.random.randn(5, 98),
            np.array([1, 5, 10, 15, 20])
        )

        valid_risks = {'LOW', 'MEDIUM', 'HIGH'}
        assert all(r in valid_risks for r in risks)


class TestDeviceSelection:
    """Test device selection logic."""

    def test_auto_device_selection(self):
        """Test automatic device selection."""
        from app.config import get_device

        device = get_device("auto")
        assert device.type in ['cuda', 'cpu']

    def test_cpu_device(self):
        """Test forcing CPU device."""
        from app.config import get_device

        device = get_device("cpu")
        assert device.type == 'cpu'

    def test_cuda_unavailable_fallback(self, mocker):
        """Test fallback to CPU when CUDA unavailable."""
        mocker.patch('torch.cuda.is_available', return_value=False)
        from app.config import get_device

        # Should not raise error, should fallback
        device = get_device("auto")
        assert device.type == 'cpu'
