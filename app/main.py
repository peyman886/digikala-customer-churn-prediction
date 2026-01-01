"""
Churn Prediction API v2.0

Multi-model prediction:
    - 1-Order users: XGBoost
    - 2+ Orders users: FT-Transformer
"""

import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import (
    PredictRequest,
    PredictResponse,
    UserProfile,
    OverviewStats,
    UsersAtRiskResponse,
    UserRisk,
    FeatureImportance,
    HealthResponse,
)
from .models import ChurnPredictor
from .services import UserDataService, StatsService

# =============================================================================
# Logging
# =============================================================================
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Global State
# =============================================================================
predictor: Optional[ChurnPredictor] = None
user_service: Optional[UserDataService] = None
stats_service: Optional[StatsService] = None
models_status: dict = {}


# =============================================================================
# Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, user_service, stats_service, models_status

    logger.info("=" * 60)
    logger.info("Starting Churn Prediction API v2.0")
    logger.info("=" * 60)

    # Initialize predictor
    device = settings.get_device()
    logger.info(f"Device: {device}")

    predictor = ChurnPredictor(
        xgboost_path=settings.xgboost_path,
        transformer_path=settings.transformer_path,
        scaler_path=settings.scaler_path,
        device=device,
        num_features=settings.num_features,
        d_token=settings.d_token,
        n_blocks=settings.n_blocks,
        n_heads=settings.n_heads,
        d_ff_multiplier=settings.d_ff_multiplier,
        dropout=settings.dropout,
    )

    # Load models
    models_status = predictor.load_models()
    logger.info(f"Models: {models_status}")

    # User data service
    user_service = UserDataService(settings.features_path, predictor)

    try:
        n_users = user_service.load_data()
        user_service.cache_predictions()
        logger.info(f"✅ {n_users:,} users loaded and cached")
    except FileNotFoundError as e:
        logger.error(f"❌ {e}")
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")

    # Stats service
    stats_service = StatsService(user_service)

    logger.info("=" * 60)
    logger.info("Startup complete!")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down...")


# =============================================================================
# App
# =============================================================================
app = FastAPI(
    title="Churn Prediction API",
    description="Multi-model churn prediction (XGBoost + FT-Transformer)",
    version="2.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_ready():
    if not user_service or not user_service.is_ready():
        raise HTTPException(503, "Service not ready")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    return {
        "name": "Churn Prediction API",
        "version": "2.0.0",
        "strategy": {
            "1_order": "XGBoost",
            "2+_orders": "FT-Transformer",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy" if predictor and all(models_status.values()) else "degraded",
        models_loaded=models_status,
        users_loaded=len(user_service.df) if user_service and user_service.df is not None else 0,
        predictions_cached=user_service.cache is not None if user_service else False,
        device=predictor.get_device_info() if predictor else "unknown",
    )


# Dashboard
@app.get("/api/stats/overview", response_model=OverviewStats, tags=["Dashboard"])
async def get_overview():
    ensure_ready()
    return OverviewStats(**stats_service.get_overview())


@app.get("/api/stats/risk-distribution", tags=["Dashboard"])
async def get_risk_distribution():
    ensure_ready()
    return stats_service.get_risk_distribution()


@app.get("/api/stats/feature-importance", response_model=List[FeatureImportance], tags=["Analytics"])
async def get_feature_importance():
    ensure_ready()
    return [FeatureImportance(**f) for f in stats_service.get_feature_importance()]


# Prediction
@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    ensure_ready()

    cached = user_service.get_prediction(request.user_id)
    if cached is None:
        raise HTTPException(404, f"User {request.user_id} not found")

    return PredictResponse(
        user_id=request.user_id,
        will_churn=cached['probability'] >= 0.5,
        probability=round(float(cached['probability']), 4),
        risk_level=cached['risk_level'],
        model_used=cached['model_used'],
    )


@app.get("/api/user/{user_id}/profile", response_model=UserProfile, tags=["User"])
async def get_profile(user_id: str):
    ensure_ready()

    user = user_service.get_user(user_id)
    if user is None:
        raise HTTPException(404, f"User {user_id} not found")

    return UserProfile(
        user_id=user_id,
        total_orders=int(user['total_orders']),
        recency=int(user['recency']),
        tenure_days=int(user['tenure_days']),
        otd_rate=round(float(user['otd_rate']), 3),
        avg_rate_shop=round(float(user['avg_rate_shop']), 2),
        avg_rate_courier=round(float(user['avg_rate_courier']), 2),
        total_crm_requests=int(user['total_crm_requests']),
        orders_last_30d=int(user['orders_last_30d']),
        total_comments=int(user['total_comments']),
    )


# Users list
@app.get("/api/users/at-risk", response_model=UsersAtRiskResponse, tags=["Users"])
async def get_users_at_risk(
    risk_level: Optional[str] = Query(None),
    min_probability: Optional[float] = Query(None, ge=0, le=1),
    min_days_inactive: Optional[int] = Query(None, ge=0),
    limit: int = Query(50, ge=1, le=500),
    page: int = Query(1, ge=1),
):
    ensure_ready()

    result = stats_service.get_users_at_risk(
        risk_level=risk_level,
        min_probability=min_probability,
        min_days_inactive=min_days_inactive,
        limit=limit,
        page=page,
    )

    return UsersAtRiskResponse(
        total=result['total'],
        page=result['page'],
        limit=result['limit'],
        users=[UserRisk(**u) for u in result['users']],
    )


# Export
@app.get("/api/export/high-risk", tags=["Export"])
async def export_high_risk(
    risk_level: str = Query("HIGH"),
    limit: int = Query(1000, le=10000),
):
    ensure_ready()
    return stats_service.export_high_risk(risk_level, limit)


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("main:app", host="0.0.0.0", port=settings.api_port, reload=True)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)