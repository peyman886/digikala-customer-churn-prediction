"""
Churn Prediction API

FastAPI service for predicting customer churn probability.

Endpoints:
    GET  /           - API info
    GET  /health     - Health check
    POST /predict    - Predict churn for a user

Author: Peyman
"""

import os
import logging
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# Logging Configuration
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
    description="Predict customer churn probability for Digikala",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global State (loaded at startup)
# =============================================================================
model = None
user_features_df = None
feature_columns = None


# =============================================================================
# Pydantic Models (Request/Response Schemas)
# =============================================================================

class PredictRequest(BaseModel):
    """Request body for /predict endpoint."""
    user_id: str = Field(..., description="User ID to predict churn for", example="1385028")


class PredictResponse(BaseModel):
    """Response body for /predict endpoint."""
    user_id: str
    will_churn: bool
    probability: float = Field(..., ge=0, le=1, description="Churn probability (0-1)")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH")

    class Config:
        schema_extra = {
            "example": {
                "user_id": "1385028",
                "will_churn": True,
                "probability": 0.82,
                "risk_level": "HIGH"
            }
        }


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str
    model_loaded: bool
    users_loaded: int
    timestamp: str


# =============================================================================
# Startup / Shutdown Events
# =============================================================================

@app.on_event("startup")
async def load_model():
    """Load model and user features on startup."""
    global model, user_features_df, feature_columns

    logger.info("=" * 50)
    logger.info("Starting Churn Prediction API...")

    # Load model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded from {MODEL_PATH}")
    else:
        logger.error(f"❌ Model not found: {MODEL_PATH}")

    # Load user features
    if FEATURES_PATH.exists():
        user_features_df = pd.read_csv(FEATURES_PATH)
        # Convert user_id to string for consistent lookup
        user_features_df['user_id'] = user_features_df['user_id'].astype(str)

        # Get feature columns (exclude user_id and target)
        feature_columns = [c for c in user_features_df.columns
                           if c not in ['user_id', 'is_churned']]

        logger.info(f"✅ Loaded features for {len(user_features_df):,} users")
        logger.info(f"   Features: {len(feature_columns)}")
    else:
        logger.error(f"❌ Features not found: {FEATURES_PATH}")

    logger.info("=" * 50)


# =============================================================================
# Helper Functions
# =============================================================================

def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """API information."""
    return {
        "name": "Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "predict": "POST /predict",
            "docs": "GET /docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        users_loaded=len(user_features_df) if user_features_df is not None else 0,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_churn(request: PredictRequest):
    """
    Predict churn probability for a user.

    - **user_id**: The user ID to predict churn for

    Returns churn prediction with probability and risk level.
    """
    user_id = request.user_id
    logger.info(f"Prediction request for user: {user_id}")

    # Check model loaded
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check features loaded
    if user_features_df is None:
        logger.error("User features not loaded")
        raise HTTPException(status_code=503, detail="User features not loaded")

    # Find user
    user_data = user_features_df[user_features_df['user_id'] == user_id]

    if user_data.empty:
        logger.warning(f"User not found: {user_id}")
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found. Available users: {len(user_features_df):,}"
        )

    # Get features and predict
    X = user_data[feature_columns].values
    probability = float(model.predict_proba(X)[0][1])
    will_churn = probability >= 0.5
    risk_level = get_risk_level(probability)

    logger.info(f"Prediction: user={user_id}, prob={probability:.4f}, risk={risk_level}")

    return PredictResponse(
        user_id=user_id,
        will_churn=will_churn,
        probability=round(probability, 4),
        risk_level=risk_level
    )


# =============================================================================
# Run (for local development)
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)