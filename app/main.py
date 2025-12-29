"""
Churn Prediction API - Extended Version

Endpoints for dashboard, analytics, and user management.

Author: Peyman
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================
MODEL_PATH = Path(os.getenv('MODEL_PATH', './model.pkl'))
FEATURES_PATH = Path(os.getenv('FEATURES_PATH', './user_features.csv'))

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Churn Prediction API",
    description="Complete API for Churn Prediction Dashboard",
    version="2.0.0",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State
# =============================================================================
model = None
user_features_df = None
feature_columns = None
predictions_cache = None  # Cache all predictions for speed

# =============================================================================
# Pydantic Models
# =============================================================================

class PredictRequest(BaseModel):
    user_id: str = Field(..., example="1385028")

class PredictResponse(BaseModel):
    user_id: str
    will_churn: bool
    probability: float
    risk_level: str

class UserProfile(BaseModel):
    user_id: str
    total_orders: int
    days_since_last_order: int
    days_since_first_order: int
    customer_tenure_days: int
    on_time_ratio: float
    avg_shop_rating: float
    avg_courier_rating: float
    total_complaints: int
    orders_last_30d: int
    orders_last_7d: int
    comment_count: int

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
    days_since_last_order: int
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
    model_loaded: bool
    users_loaded: int
    predictions_cached: bool

# =============================================================================
# Startup
# =============================================================================

@app.on_event("startup")
async def startup():
    global model, user_features_df, feature_columns, predictions_cache

    logger.info("=" * 50)
    logger.info("Starting Churn Prediction API v2.0...")

    # Load model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded: {MODEL_PATH}")
    else:
        logger.error(f"❌ Model not found: {MODEL_PATH}")
        return

    # Load features
    if FEATURES_PATH.exists():
        user_features_df = pd.read_csv(FEATURES_PATH)
        user_features_df['user_id'] = user_features_df['user_id'].astype(str)
        feature_columns = [c for c in user_features_df.columns
                          if c not in ['user_id', 'is_churned']]
        logger.info(f"✅ Loaded {len(user_features_df):,} users")

        # Pre-compute all predictions (for dashboard speed)
        logger.info("⏳ Pre-computing predictions...")
        X = user_features_df[feature_columns].values
        probabilities = model.predict_proba(X)[:, 1]

        predictions_cache = pd.DataFrame({
            'user_id': user_features_df['user_id'],
            'probability': probabilities,
            'risk_level': pd.cut(probabilities,
                                bins=[0, 0.4, 0.7, 1.0],
                                labels=['LOW', 'MEDIUM', 'HIGH']),
            'days_since_last_order': user_features_df['days_since_last_order'],
            'total_orders': user_features_df['total_orders']
        })
        logger.info(f"✅ Predictions cached for {len(predictions_cache):,} users")
    else:
        logger.error(f"❌ Features not found: {FEATURES_PATH}")

    logger.info("=" * 50)

# =============================================================================
# Helper Functions
# =============================================================================

def get_risk_level(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    return "LOW"

# Feature display names (English -> Persian friendly)
FEATURE_DISPLAY_NAMES = {
    'days_since_last_order': 'Days Since Last Order',
    'orders_last_30d': 'Orders (Last 30 Days)',
    'orders_last_7d': 'Orders (Last 7 Days)',
    'total_orders': 'Total Orders',
    'avg_order_gap_days': 'Avg Days Between Orders',
    'on_time_ratio': 'On-Time Delivery Rate',
    'late_delivery_count': 'Late Deliveries',
    'total_complaints': 'Total Complaints',
    'complaints_per_order': 'Complaints per Order',
    'avg_shop_rating': 'Avg Shop Rating',
    'avg_courier_rating': 'Avg Courier Rating',
    'min_shop_rating': 'Min Shop Rating',
    'has_low_shop_rating': 'Has Low Rating',
    'comment_count': 'Number of Comments',
    'customer_tenure_days': 'Customer Tenure (Days)',
    'days_since_first_order': 'Days Since First Order',
}

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Churn Prediction API",
        "version": "2.0.0",
        "endpoints": {
            "dashboard": "/api/stats/overview",
            "predict": "/api/predict",
            "profile": "/api/user/{user_id}/profile",
            "at_risk": "/api/users/at-risk",
            "features": "/api/stats/feature-importance"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        users_loaded=len(user_features_df) if user_features_df is not None else 0,
        predictions_cached=predictions_cache is not None
    )


# -----------------------------------------------------------------------------
# Dashboard Stats
# -----------------------------------------------------------------------------

@app.get("/api/stats/overview", response_model=OverviewStats, tags=["Dashboard"])
async def get_overview_stats():
    """Get overall churn risk statistics for dashboard."""
    if predictions_cache is None:
        raise HTTPException(503, "Data not loaded")

    total = len(predictions_cache)
    low = (predictions_cache['risk_level'] == 'LOW').sum()
    medium = (predictions_cache['risk_level'] == 'MEDIUM').sum()
    high = (predictions_cache['risk_level'] == 'HIGH').sum()

    return OverviewStats(
        total_users=total,
        low_risk=int(low),
        medium_risk=int(medium),
        high_risk=int(high),
        low_risk_pct=round(low / total * 100, 1),
        medium_risk_pct=round(medium / total * 100, 1),
        high_risk_pct=round(high / total * 100, 1),
        avg_churn_probability=round(predictions_cache['probability'].mean(), 4)
    )


# -----------------------------------------------------------------------------
# Single User Prediction
# -----------------------------------------------------------------------------

@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_churn(request: PredictRequest):
    """Predict churn for a single user."""
    user_id = request.user_id

    if predictions_cache is None:
        raise HTTPException(503, "Data not loaded")

    user = predictions_cache[predictions_cache['user_id'] == user_id]

    if user.empty:
        raise HTTPException(404, f"User {user_id} not found")

    row = user.iloc[0]
    return PredictResponse(
        user_id=user_id,
        will_churn=row['probability'] >= 0.5,
        probability=round(float(row['probability']), 4),
        risk_level=row['risk_level']
    )


@app.get("/api/user/{user_id}/profile", response_model=UserProfile, tags=["User"])
async def get_user_profile(user_id: str):
    """Get detailed profile for a user."""
    if user_features_df is None:
        raise HTTPException(503, "Data not loaded")

    user = user_features_df[user_features_df['user_id'] == user_id]

    if user.empty:
        raise HTTPException(404, f"User {user_id} not found")

    row = user.iloc[0]
    return UserProfile(
        user_id=user_id,
        total_orders=int(row['total_orders']),
        days_since_last_order=int(row['days_since_last_order']),
        days_since_first_order=int(row['days_since_first_order']),
        customer_tenure_days=int(row['customer_tenure_days']),
        on_time_ratio=round(float(row['on_time_ratio']), 3),
        avg_shop_rating=round(float(row['avg_shop_rating']), 2),
        avg_courier_rating=round(float(row['avg_courier_rating']), 2),
        total_complaints=int(row['total_complaints']),
        orders_last_30d=int(row['orders_last_30d']),
        orders_last_7d=int(row['orders_last_7d']),
        comment_count=int(row['comment_count'])
    )


# -----------------------------------------------------------------------------
# High Risk Users List
# -----------------------------------------------------------------------------

@app.get("/api/users/at-risk", response_model=UsersAtRiskResponse, tags=["Users"])
async def get_users_at_risk(
    risk_level: Optional[str] = Query(None, description="Filter by risk: LOW, MEDIUM, HIGH"),
    min_probability: Optional[float] = Query(None, ge=0, le=1),
    min_days_inactive: Optional[int] = Query(None, ge=0),
    limit: int = Query(50, ge=1, le=500),
    page: int = Query(1, ge=1)
):
    """Get list of users at risk with filters and pagination."""
    if predictions_cache is None:
        raise HTTPException(503, "Data not loaded")

    # Start with all data
    df = predictions_cache.copy()

    # Apply filters
    if risk_level:
        df = df[df['risk_level'] == risk_level.upper()]

    if min_probability is not None:
        df = df[df['probability'] >= min_probability]

    if min_days_inactive is not None:
        df = df[df['days_since_last_order'] >= min_days_inactive]

    # Sort by probability (highest first)
    df = df.sort_values('probability', ascending=False)

    # Pagination
    total = len(df)
    start = (page - 1) * limit
    end = start + limit
    df_page = df.iloc[start:end]

    users = [
        UserRisk(
            user_id=row['user_id'],
            probability=round(float(row['probability']), 4),
            risk_level=row['risk_level'],
            days_since_last_order=int(row['days_since_last_order']),
            total_orders=int(row['total_orders'])
        )
        for _, row in df_page.iterrows()
    ]

    return UsersAtRiskResponse(
        total=total,
        page=page,
        limit=limit,
        users=users
    )


# -----------------------------------------------------------------------------
# Analytics
# -----------------------------------------------------------------------------

@app.get("/api/stats/feature-importance", response_model=List[FeatureImportance], tags=["Analytics"])
async def get_feature_importance():
    """Get model feature importance for analytics."""
    if model is None or feature_columns is None:
        raise HTTPException(503, "Model not loaded")

    # Get importance from model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise HTTPException(500, "Model doesn't support feature importance")

    # Create sorted list
    features = []
    for feat, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        features.append(FeatureImportance(
            feature=feat,
            importance=round(float(imp), 4),
            display_name=FEATURE_DISPLAY_NAMES.get(feat, feat)
        ))

    return features[:15]  # Top 15


@app.get("/api/stats/risk-distribution", tags=["Analytics"])
async def get_risk_distribution():
    """Get probability distribution for histogram."""
    if predictions_cache is None:
        raise HTTPException(503, "Data not loaded")

    # Create histogram bins
    bins = np.arange(0, 1.05, 0.05)
    hist, edges = np.histogram(predictions_cache['probability'], bins=bins)

    return {
        "bins": [round(float(e), 2) for e in edges[:-1]],
        "counts": [int(c) for c in hist]
    }


# -----------------------------------------------------------------------------
# Export (for CSV download)
# -----------------------------------------------------------------------------

@app.get("/api/export/high-risk", tags=["Export"])
async def export_high_risk_users(
    risk_level: str = Query("HIGH"),
    limit: int = Query(1000, le=10000)
):
    """Export high-risk users as JSON (frontend can convert to CSV)."""
    if predictions_cache is None:
        raise HTTPException(503, "Data not loaded")

    df = predictions_cache[predictions_cache['risk_level'] == risk_level.upper()]
    df = df.sort_values('probability', ascending=False).head(limit)

    return {
        "count": len(df),
        "data": df.to_dict('records')
    }


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)