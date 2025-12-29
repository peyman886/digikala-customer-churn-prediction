#!/usr/bin/env python3
"""
Churn Prediction API Service

FastAPI application for predicting customer churn probability.
Provides REST endpoints for real-time predictions.

Author: Peyman
Date: 2025
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import os
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===================================================================
# Configuration
# ===================================================================

MODEL_PATH = os.getenv('MODEL_PATH', './model.pkl')
SCALER_PATH = os.getenv('SCALER_PATH', './scaler.pkl')
FEATURES_PATH = os.getenv('FEATURES_PATH', './user_features.csv')
API_TITLE = os.getenv('API_TITLE', 'Churn Prediction API')
API_VERSION = os.getenv('API_VERSION', '1.0.0')

# ===================================================================
# Initialize FastAPI App
# ===================================================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="🎯 AI-powered customer churn prediction service for Digikala",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================================================================
# Global Variables (Loaded at Startup)
# ===================================================================

model = None
scaler = None
user_features_df = None
feature_columns = None

# ===================================================================
# Pydantic Models (Request/Response Schemas)
# ===================================================================

class PredictionRequest(BaseModel):
    """Request schema for churn prediction."""
    user_id: str = Field(..., description="Unique user identifier", example="12345")
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError('user_id cannot be empty')
        return v.strip()


class PredictionResponse(BaseModel):
    """Response schema for churn prediction."""
    user_id: str = Field(..., description="Unique user identifier")
    will_churn: bool = Field(..., description="Predicted churn status")
    probability: float = Field(..., description="Churn probability (0-1)")
    risk_level: str = Field(..., description="Risk category: LOW, MEDIUM, HIGH")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "12345",
                "will_churn": True,
                "probability": 0.8723,
                "risk_level": "HIGH",
                "prediction_timestamp": "2025-12-29T01:30:00"
            }
        }


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    features_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    error_code: str
    timestamp: str

# ===================================================================
# Startup and Shutdown Events
# ===================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and data on application startup."""
    global model, scaler, user_features_df, feature_columns
    
    logger.info("🚀 Starting Churn Prediction API...")
    
    try:
        # Load trained model
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model file not found: {MODEL_PATH}")
        
        # Load scaler (if exists)
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"✅ Scaler loaded from {SCALER_PATH}")
        else:
            logger.info("ℹ️ No scaler file found (not required for tree-based models)")
        
        # Load user features
        if os.path.exists(FEATURES_PATH):
            user_features_df = pd.read_csv(FEATURES_PATH)
            logger.info(f"✅ Loaded features for {len(user_features_df)} users")
            
            # Determine feature columns
            feature_columns = [col for col in user_features_df.columns 
                             if col not in ['user_id', 'is_churned']]
            logger.info(f"📈 Feature columns: {len(feature_columns)}")
        else:
            logger.warning(f"⚠️ Features file not found: {FEATURES_PATH}")
        
        logger.info("✅ Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("🛑 Shutting down Churn Prediction API...")

# ===================================================================
# Helper Functions
# ===================================================================

def get_risk_level(probability: float) -> str:
    """
    Categorize churn probability into risk levels.
    
    Args:
        probability: Churn probability (0-1)
        
    Returns:
        Risk level: LOW, MEDIUM, or HIGH
    """
    if probability >= 0.7:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"

# ===================================================================
# API Endpoints
# ===================================================================

@app.get("/", tags=["Root"])
async def root() -> Dict:
    """
    Root endpoint with API information.
    """
    return {
        "message": "🎯 Digikala Churn Prediction API",
        "version": API_VERSION,
        "status": "active",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service health status and component availability.
    """
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        features_loaded=user_features_df is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        404: {"description": "User not found"},
        500: {"description": "Internal server error"}
    }
)
async def predict_churn(request: PredictionRequest) -> PredictionResponse:
    """
    Predict churn probability for a given user.
    
    Args:
        request: PredictionRequest containing user_id
        
    Returns:
        PredictionResponse with churn prediction details
        
    Raises:
        HTTPException: If user not found or prediction fails
    """
    user_id = request.user_id
    logger.info(f"🔍 Prediction request for user_id: {user_id}")
    
    # Check if model is loaded
    if model is None:
        logger.error("❌ Model not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded. Please contact administrator."
        )
    
    # Check if features are loaded
    if user_features_df is None:
        logger.error("❌ Features not loaded")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Feature data not loaded. Please contact administrator."
        )
    
    try:
        # Get user features
        user_data = user_features_df[user_features_df['user_id'] == user_id]
        
        if user_data.empty:
            logger.warning(f"⚠️ User {user_id} not found in features")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found in database"
            )
        
        # Extract features
        X = user_data[feature_columns].values
        
        # Apply scaling if scaler exists
        if scaler is not None:
            X = scaler.transform(X)
        
        # Make prediction
        churn_proba = model.predict_proba(X)[0][1]  # Probability of class 1 (churn)
        will_churn = bool(churn_proba >= 0.5)
        risk_level = get_risk_level(churn_proba)
        
        logger.info(
            f"✅ Prediction for user {user_id}: "
            f"churn={will_churn}, probability={churn_proba:.4f}, risk={risk_level}"
        )
        
        return PredictionResponse(
            user_id=user_id,
            will_churn=will_churn,
            probability=round(float(churn_proba), 4),
            risk_level=risk_level,
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction error for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/users/high-risk", tags=["Analytics"])
async def get_high_risk_users(limit: int = 10) -> Dict:
    """
    Get list of users with highest churn risk.
    
    Args:
        limit: Maximum number of users to return
        
    Returns:
        List of high-risk users with their probabilities
    """
    if user_features_df is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        # Get all features
        X = user_features_df[feature_columns].values
        if scaler is not None:
            X = scaler.transform(X)
        
        # Predict for all users
        probabilities = model.predict_proba(X)[:, 1]
        
        # Create results dataframe
        results = pd.DataFrame({
            'user_id': user_features_df['user_id'].values,
            'churn_probability': probabilities
        })
        
        # Sort by probability and get top N
        high_risk = results.nlargest(limit, 'churn_probability')
        
        return {
            "count": len(high_risk),
            "users": high_risk.to_dict('records')
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting high-risk users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/stats", tags=["Analytics"])
async def get_statistics() -> Dict:
    """
    Get overall churn statistics.
    
    Returns:
        Dictionary with churn statistics
    """
    if user_features_df is None or model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )
    
    try:
        # Get all predictions
        X = user_features_df[feature_columns].values
        if scaler is not None:
            X = scaler.transform(X)
        
        probabilities = model.predict_proba(X)[:, 1]
        predictions = probabilities >= 0.5
        
        # Calculate statistics
        total_users = len(user_features_df)
        predicted_churners = int(predictions.sum())
        high_risk_users = int((probabilities >= 0.7).sum())
        medium_risk_users = int(((probabilities >= 0.4) & (probabilities < 0.7)).sum())
        low_risk_users = int((probabilities < 0.4).sum())
        
        return {
            "total_users": total_users,
            "predicted_churners": predicted_churners,
            "churn_rate": round(predicted_churners / total_users, 4),
            "risk_distribution": {
                "high": high_risk_users,
                "medium": medium_risk_users,
                "low": low_risk_users
            },
            "average_churn_probability": round(float(probabilities.mean()), 4)
        }
        
    except Exception as e:
        logger.error(f"❌ Error calculating statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ===================================================================
# Exception Handlers
# ===================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )

# ===================================================================
# Run Application (for development)
# ===================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv('API_PORT', 8000)),
        reload=True,
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )
