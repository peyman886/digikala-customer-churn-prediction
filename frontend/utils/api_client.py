"""
API Client Module

Handles all communication with the backend API.
"""

import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

import requests


API_URL = os.getenv("API_URL", "http://localhost:8000")
TIMEOUT = 15


@dataclass
class APIResponse:
    """Standardized API response."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class APIClient:
    """Client for backend API communication."""
    
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url
    
    def _get(self, endpoint: str, params: Optional[Dict] = None) -> APIResponse:
        """Make GET request."""
        try:
            r = requests.get(
                f"{self.base_url}{endpoint}",
                params=params,
                timeout=TIMEOUT
            )
            if r.status_code == 200:
                return APIResponse(success=True, data=r.json())
            elif r.status_code == 404:
                return APIResponse(success=False, error="not_found")
            else:
                return APIResponse(success=False, error=f"API Error: {r.status_code}")
        except requests.exceptions.ConnectionError:
            return APIResponse(success=False, error="connection_error")
        except requests.exceptions.Timeout:
            return APIResponse(success=False, error="timeout")
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    def _post(self, endpoint: str, data: Dict) -> APIResponse:
        """Make POST request."""
        try:
            r = requests.post(
                f"{self.base_url}{endpoint}",
                json=data,
                timeout=TIMEOUT
            )
            if r.status_code == 200:
                return APIResponse(success=True, data=r.json())
            elif r.status_code == 404:
                return APIResponse(success=False, error="not_found")
            else:
                return APIResponse(success=False, error=f"API Error: {r.status_code}")
        except requests.exceptions.ConnectionError:
            return APIResponse(success=False, error="connection_error")
        except requests.exceptions.Timeout:
            return APIResponse(success=False, error="timeout")
        except Exception as e:
            return APIResponse(success=False, error=str(e))
    
    # =========================================================================
    # Prediction Endpoints
    # =========================================================================
    
    def predict_user(self, user_id: str) -> APIResponse:
        """Get churn prediction for a user."""
        return self._post("/api/predict", {"user_id": user_id})
    
    def get_user_profile(self, user_id: str) -> APIResponse:
        """Get user profile details."""
        return self._get(f"/api/user/{user_id}/profile")
    
    # =========================================================================
    # Statistics Endpoints
    # =========================================================================
    
    def get_overview(self) -> APIResponse:
        """Get overview statistics."""
        return self._get("/api/stats/overview")
    
    def get_risk_distribution(self) -> APIResponse:
        """Get risk probability distribution."""
        return self._get("/api/stats/risk-distribution")
    
    def get_feature_importance(self) -> APIResponse:
        """Get feature importance from model."""
        return self._get("/api/stats/feature-importance")
    
    def get_segment_stats(self) -> APIResponse:
        """Get statistics by segment."""
        return self._get("/api/stats/segments")
    
    # =========================================================================
    # User List Endpoints
    # =========================================================================
    
    def get_users_at_risk(
        self,
        risk_level: Optional[str] = None,
        segment: Optional[str] = None,
        min_days_inactive: Optional[int] = None,
        limit: int = 50,
        page: int = 1
    ) -> APIResponse:
        """Get paginated list of at-risk users."""
        params = {"limit": limit, "page": page}
        
        if risk_level and risk_level != "All":
            params["risk_level"] = risk_level
        if segment and segment != "All":
            params["segment"] = segment
        if min_days_inactive and min_days_inactive > 0:
            params["min_days_inactive"] = min_days_inactive
        
        return self._get("/api/users/at-risk", params)
    
    def export_high_risk(self, risk_level: str = "HIGH", limit: int = 1000) -> APIResponse:
        """Export high risk users."""
        return self._get("/api/export/high-risk", {"risk_level": risk_level, "limit": limit})


# Singleton instance
api_client = APIClient()
