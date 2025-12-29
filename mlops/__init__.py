"""
MLOps Module for Churn Prediction

Components:
    - ExperimentTracker: ردیاب آزمایش‌ها
    - train_model: تابع آموزش با tracking
    - compare_experiments: مقایسه runs

Usage:
    from mlops import ExperimentTracker, train_model

    # Quick training
    result = train_model(
        run_name="my_experiment",
        model_type="xgboost",
        max_depth=10
    )

    # Custom tracking
    tracker = ExperimentTracker("churn-prediction")
    with tracker.start_run("my_run"):
        tracker.log_params({"lr": 0.1})
        ...
"""

from mlops.experiment import ExperimentTracker, quick_compare
from mlops.train import train_model, compare_experiments

__all__ = [
    "ExperimentTracker",
    "train_model",
    "compare_experiments",
    "quick_compare"
]