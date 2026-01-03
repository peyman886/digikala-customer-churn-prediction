"""
Unit tests for service layer.
"""

import pytest
import pandas as pd
import numpy as np
from app.services import UserDataService, StatsService


class TestUserDataService:
    """Test UserDataService functionality."""

    def test_load_data(self, temp_features_csv, mock_predictor):
        """Test data loading."""
        service = UserDataService(temp_features_csv, mock_predictor)
        n_users = service.load_data()

        assert n_users == 5
        assert service.df is not None
        assert len(service.feature_cols) == 98

    def test_cache_predictions(self, temp_features_csv, mock_predictor):
        """Test prediction caching."""
        service = UserDataService(temp_features_csv, mock_predictor)
        service.load_data()
        service.cache_predictions()

        assert service.cache is not None
        assert len(service.cache) == 5
        assert 'probability' in service.cache.columns
        assert 'risk_level' in service.cache.columns
        assert 'model_used' in service.cache.columns

    def test_get_user(self, temp_features_csv, mock_predictor):
        """Test getting user data."""
        service = UserDataService(temp_features_csv, mock_predictor)
        service.load_data()

        user = service.get_user('1001')
        assert user is not None
        assert user['user_id'] == '1001'

        # Non-existent user
        user = service.get_user('9999')
        assert user is None

    def test_get_prediction(self, temp_features_csv, mock_predictor):
        """Test getting cached prediction."""
        service = UserDataService(temp_features_csv, mock_predictor)
        service.load_data()
        service.cache_predictions()

        pred = service.get_prediction('1001')
        assert pred is not None
        assert 'probability' in pred
        assert 'risk_level' in pred

    def test_feature_mismatch_error(self, temp_features_csv, mock_predictor):
        """Test error when feature count doesn't match."""
        # Modify predictor to expect different feature count
        mock_predictor.num_features = 50

        service = UserDataService(temp_features_csv, mock_predictor)
        service.load_data()

        with pytest.raises(ValueError, match="Feature shape mismatch"):
            service.cache_predictions()

    def test_is_ready(self, temp_features_csv, mock_predictor):
        """Test service ready status."""
        service = UserDataService(temp_features_csv, mock_predictor)

        assert not service.is_ready()

        service.load_data()
        assert not service.is_ready()

        service.cache_predictions()
        assert service.is_ready()


class TestStatsService:
    """Test StatsService functionality."""

    @pytest.fixture
    def stats_service(self, temp_features_csv, mock_predictor):
        """Create StatsService with loaded data."""
        user_service = UserDataService(temp_features_csv, mock_predictor)
        user_service.load_data()
        user_service.cache_predictions()
        return StatsService(user_service)

    def test_get_overview(self, stats_service):
        """Test overview statistics."""
        overview = stats_service.get_overview()

        assert 'total_users' in overview
        assert 'low_risk' in overview
        assert 'medium_risk' in overview
        assert 'high_risk' in overview
        assert overview['total_users'] == 5
        assert overview['low_risk'] + overview['medium_risk'] + overview['high_risk'] == 5

    def test_get_users_at_risk(self, stats_service):
        """Test filtered user list."""
        result = stats_service.get_users_at_risk(
            risk_level='HIGH',
            limit=10,
            page=1
        )

        assert 'total' in result
        assert 'users' in result
        assert isinstance(result['users'], list)

    def test_get_risk_distribution(self, stats_service):
        """Test risk distribution histogram."""
        dist = stats_service.get_risk_distribution()

        assert 'bins' in dist
        assert 'counts' in dist
        assert len(dist['bins']) == len(dist['counts'])

    def test_get_feature_importance(self, stats_service):
        """Test feature importance extraction."""
        importance = stats_service.get_feature_importance()

        assert isinstance(importance, list)
        assert len(importance) == 15  # Top 15
        assert all('feature' in f for f in importance)
        assert all('importance' in f for f in importance)

    def test_pagination(self, stats_service):
        """Test pagination logic."""
        # Page 1
        result1 = stats_service.get_users_at_risk(limit=2, page=1)
        # Page 2
        result2 = stats_service.get_users_at_risk(limit=2, page=2)

        # Should have different users
        users1 = [u['user_id'] for u in result1['users']]
        users2 = [u['user_id'] for u in result2['users']]

        if len(users1) > 0 and len(users2) > 0:
            assert users1[0] != users2[0]
