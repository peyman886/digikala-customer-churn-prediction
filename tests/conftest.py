"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def sample_features_df():
    """Create sample user features for testing."""
    return pd.DataFrame({
        'user_id': ['1001', '1002', '1003', '1004', '1005'],
        'total_orders': [1, 5, 15, 30, 2],
        'recency': [10, 5, 3, 2, 45],
        'tenure_days': [100, 200, 300, 400, 50],
        'otd_rate': [0.9, 0.95, 0.98, 1.0, 0.8],
        'avg_rate_shop': [4.5, 4.8, 4.9, 5.0, 4.0],
        'avg_rate_courier': [4.3, 4.7, 4.8, 4.9, 3.8],
        'total_crm_requests': [0, 1, 2, 0, 3],
        'orders_last_30d': [1, 2, 5, 10, 0],
        'total_comments': [0, 2, 5, 10, 1],
        # Add more features to match your 98 features
    })


@pytest.fixture(scope="session")
def sample_feature_array():
    """Sample feature array matching model input (98 features)."""
    return np.random.randn(5, 98)


@pytest.fixture
def mock_predictor(mocker):
    """Mock ChurnPredictor for isolated testing."""
    predictor = mocker.MagicMock()
    predictor.predict_batch.return_value = (
        np.array([0.2, 0.5, 0.8, 0.9, 0.3]),  # probabilities
        np.array([0, 1, 1, 1, 0]),  # predictions
        np.array(['LOW', 'MEDIUM', 'HIGH', 'HIGH', 'LOW']),  # risk levels
        np.array(['xgboost', 'transformer', 'transformer', 'transformer', 'xgboost'])
    )
    predictor.num_features = 98
    predictor.get_device_info.return_value = "cpu"
    predictor.xgboost = mocker.MagicMock()
    predictor.xgboost.feature_importances_ = np.random.rand(98)
    return predictor


@pytest.fixture
def temp_features_csv(tmp_path, sample_features_df):
    """Create temporary features CSV file."""
    csv_path = tmp_path / "user_features.csv"

    # Expand to 98 features
    for i in range(88):
        sample_features_df[f'feature_{i}'] = np.random.rand(len(sample_features_df))

    sample_features_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="module")
def test_client(temp_features_csv, mock_predictor, mocker):
    """Create FastAPI test client with mocked dependencies."""
    from app.config import settings

    # Mock settings
    mocker.patch.object(settings, 'features_path', temp_features_csv)

    # Import app after patching
    from app.main import app

    client = TestClient(app)
    yield client
