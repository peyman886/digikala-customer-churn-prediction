"""
MLOps Module for Churn Prediction

Simple and clean experiment tracking with MLflow.

Quick Start (in Notebook):

    from mlops import ExperimentTracker

    tracker = ExperimentTracker()

    with tracker.run("my_experiment"):
        model.fit(X_train, y_train)
        tracker.log_metrics(model, X_test, y_test)
        tracker.log_model(model, feature_names=feature_cols)

    # Compare experiments
    tracker.compare()

    # Promote best model to production (app/model.pkl)
    tracker.promote("my_experiment")

CLI Usage:

    # Compare all experiments
    python mlops/compare.py

    # Promote best model
    python mlops/compare.py --promote best

    # Start MLflow UI
    mlflow ui --port 5000 --backend-store-uri ./mlruns
"""

from mlops.tracker import (
    ExperimentTracker,
    quick_compare,
    promote_best,
)

from mlops.config import (
    EXPERIMENT_NAME,
    FEATURES_PATH,
    PRODUCTION_DIR,
    PRIMARY_METRIC,
)

__all__ = [
    # Main class
    "ExperimentTracker",

    # Quick functions
    "quick_compare",
    "promote_best",

    # Config
    "EXPERIMENT_NAME",
    "FEATURES_PATH",
    "PRODUCTION_DIR",
    "PRIMARY_METRIC",
]

__version__ = "1.0.0"