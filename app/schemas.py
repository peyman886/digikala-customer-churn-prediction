"""
Pydantic Schemas for API.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


# Requests
class PredictRequest(BaseModel):
    user_id: str = Field(..., example="1385028")


# Responses
class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    user_id: str
    will_churn: bool
    probability: float
    risk_level: str
    model_used: str


class BatchPredictRequest(BaseModel):
    user_ids: List[str] = Field(..., example=["1385028", "54227"], max_length=100)


class BatchPredictResponse(BaseModel):
    total: int
    successful: int
    failed: int
    predictions: List[PredictResponse]
    errors: List[dict]  # {"user_id": "123", "error": "not found"}


class UserProfile(BaseModel):
    user_id: str
    total_orders: int
    recency: int  # days_since_last_order
    tenure_days: int
    otd_rate: float
    avg_rate_shop: float
    avg_rate_courier: float
    total_crm_requests: int
    # orders_last_30d: int
    total_comments: int


class OverviewStats(BaseModel):
    total_users: int
    low_risk: int
    medium_risk: int
    high_risk: int
    low_risk_pct: float
    medium_risk_pct: float
    high_risk_pct: float
    avg_churn_probability: float


class UserRisk(BaseModel):
    user_id: str
    probability: float
    risk_level: str
    recency: int
    total_orders: int


class UsersAtRiskResponse(BaseModel):
    total: int
    page: int
    limit: int
    users: List[UserRisk]


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    display_name: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    users_loaded: int
    predictions_cached: bool
    device: str