"""
MLOps Configuration

All paths and default settings in one place.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

# Project root (one level up from mlops/)
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "user_features.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
PRODUCTION_DIR = MODELS_DIR / "production"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"

# MLflow paths
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"file://{MLFLOW_DIR.absolute()}"

# App paths (for deployment)
APP_DIR = PROJECT_ROOT / "app"
APP_MODEL_PATH = APP_DIR / "model.pkl"
APP_FEATURES_PATH = APP_DIR / "user_features.csv"

# =============================================================================
# MLflow Settings
# =============================================================================

EXPERIMENT_NAME = "churn-prediction"
MLFLOW_PORT = 5000

# =============================================================================
# Model Defaults
# =============================================================================

DEFAULT_MODEL_TYPE = "xgboost"
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model hyperparameters defaults
MODEL_DEFAULTS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
    },
    "rf": {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "gbm": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE,
    },
    "logistic": {
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
}

# =============================================================================
# Metrics
# =============================================================================

# Primary metric for model comparison
PRIMARY_METRIC = "roc_auc"

# All metrics to track
METRICS_TO_TRACK = [
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "train_time_seconds",
    "inference_time_ms",
]

# =============================================================================
# Ensure directories exist
# =============================================================================

def ensure_dirs():
    """Create all necessary directories."""
    for dir_path in [DATA_DIR, MODELS_DIR, PRODUCTION_DIR, EXPERIMENTS_DIR, MLFLOW_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

# Auto-create on import
ensure_dirs()