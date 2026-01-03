"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns API info."""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert 'name' in data
        assert 'version' in data
        assert data['version'] == "2.0.0"

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'models_loaded' in data
        assert 'device' in data


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_predict_single_user(self, test_client):
        """Test single user prediction."""
        response = test_client.post(
            "/api/predict",
            json={"user_id": "1001"}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'user_id' in data
        assert 'probability' in data
        assert 'risk_level' in data
        assert 'model_used' in data
        assert 0 <= data['probability'] <= 1

    def test_predict_nonexistent_user(self, test_client):
        """Test prediction for non-existent user."""
        response = test_client.post(
            "/api/predict",
            json={"user_id": "9999"}
        )
        assert response.status_code == 404

    def test_predict_invalid_request(self, test_client):
        """Test prediction with invalid request."""
        response = test_client.post("/api/predict", json={})
        assert response.status_code == 422


class TestUserEndpoints:
    """Test user profile endpoints."""

    def test_get_user_profile(self, test_client):
        """Test getting user profile."""
        response = test_client.get("/api/user/1001/profile")
        assert response.status_code == 200
        data = response.json()
        assert 'user_id' in data
        assert 'total_orders' in data
        assert 'recency' in data

    def test_get_nonexistent_profile(self, test_client):
        """Test getting non-existent user profile."""
        response = test_client.get("/api/user/9999/profile")
        assert response.status_code == 404


class TestDashboardEndpoints:
    """Test dashboard statistics endpoints."""

    def test_get_overview(self, test_client):
        """Test overview statistics."""
        response = test_client.get("/api/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert 'total_users' in data
        assert 'low_risk' in data
        assert 'high_risk' in data

    def test_get_risk_distribution(self, test_client):
        """Test risk distribution."""
        response = test_client.get("/api/stats/risk-distribution")
        assert response.status_code == 200
        data = response.json()
        assert 'bins' in data
        assert 'counts' in data

    def test_get_feature_importance(self, test_client):
        """Test feature importance."""
        response = test_client.get("/api/stats/feature-importance")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestUsersListEndpoint:
    """Test users at risk listing endpoint."""

    def test_get_users_at_risk_default(self, test_client):
        """Test default users at risk."""
        response = test_client.get("/api/users/at-risk")
        assert response.status_code == 200
        data = response.json()
        assert 'total' in data
        assert 'users' in data
        assert 'page' in data

    def test_get_users_at_risk_filtered(self, test_client):
        """Test filtered users at risk."""
        response = test_client.get(
            "/api/users/at-risk",
            params={"risk_level": "HIGH", "limit": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert all(u['risk_level'] == 'HIGH' for u in data['users'])

    def test_pagination_params(self, test_client):
        """Test pagination parameters."""
        response = test_client.get(
            "/api/users/at-risk",
            params={"limit": 2, "page": 1}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['limit'] == 2
        assert data['page'] == 1

    def test_invalid_pagination(self, test_client):
        """Test invalid pagination parameters."""
        response = test_client.get(
            "/api/users/at-risk",
            params={"limit": 1000, "page": 0}  # Invalid: limit>500, page<1
        )
        assert response.status_code == 422


class TestExportEndpoint:
    """Test export functionality."""

    def test_export_high_risk(self, test_client):
        """Test exporting high risk users."""
        response = test_client.get(
            "/api/export/high-risk",
            params={"risk_level": "HIGH", "limit": 100}
        )
        assert response.status_code == 200
        data = response.json()
        assert 'count' in data
        assert 'data' in data
        assert isinstance(data['data'], list)
