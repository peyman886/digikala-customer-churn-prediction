"""
MLOps Experiment Tracker

این ماژول برای ثبت و مقایسه آزمایش‌های مختلف مدل استفاده میشه.

Features:
    - ثبت خودکار metrics, parameters, artifacts
    - اندازه‌گیری زمان training و inference
    - مقایسه آسان بین experiments
    - Model versioning با MLflow Model Registry

Usage:
    from mlops.experiment import ExperimentTracker

    tracker = ExperimentTracker("churn-prediction")

    with tracker.start_run(run_name="baseline_v1"):
        tracker.log_params({"n_estimators": 100, "max_depth": 10})
        tracker.log_features(feature_list)

        # Train with timing
        model, train_time = tracker.train_with_timing(model, X_train, y_train)

        # Evaluate with timing
        metrics, inf_time = tracker.evaluate_with_timing(model, X_test, y_test)

        # Log everything
        tracker.log_metrics(metrics)
        tracker.log_model(model, "xgboost-model")

"""

import os
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import pandas as pd
import joblib

# MLflow imports
try:
    import mlflow
    from mlflow.tracking import MlflowClient

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed. Using local tracking only.")

# Sklearn metrics
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExperimentResult:
    """نتایج یک آزمایش"""
    run_id: str
    run_name: str
    timestamp: str

    # Model info
    model_type: str
    model_params: Dict[str, Any]

    # Features
    features: List[str]
    feature_count: int
    feature_hash: str  # For detecting feature changes

    # Metrics
    roc_auc: float
    f1: float
    precision: float
    recall: float
    accuracy: float

    # Timing
    train_time_seconds: float
    inference_time_ms: float  # Per sample

    # Dataset info
    train_samples: int
    test_samples: int
    churn_rate: float

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# Experiment Tracker
# =============================================================================

class ExperimentTracker:
    """
    ردیاب آزمایش‌ها - مدیریت کامل MLOps

    Example:
        tracker = ExperimentTracker("churn-prediction")

        with tracker.start_run("baseline"):
            tracker.log_params(params)
            tracker.log_features(features)
            model, train_time = tracker.train_with_timing(...)
            metrics, inf_time = tracker.evaluate_with_timing(...)
            tracker.log_model(model)
    """

    def __init__(
            self,
            experiment_name: str = "churn-prediction",
            tracking_uri: Optional[str] = None,
            local_dir: str = "./mlruns"
    ):
        """
        Initialize tracker.

        Args:
            experiment_name: نام آزمایش (گروه‌بندی runs)
            tracking_uri: آدرس MLflow server (None = local)
            local_dir: مسیر ذخیره لوکال
        """
        self.experiment_name = experiment_name
        self.local_dir = Path(local_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)

        # Current run state
        self._current_run = None
        self._run_data = {}

        # Setup MLflow if available
        if MLFLOW_AVAILABLE:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            else:
                mlflow.set_tracking_uri(f"file://{self.local_dir.absolute()}")

            mlflow.set_experiment(experiment_name)
            self.client = MlflowClient()
            print(f"✅ MLflow initialized: {experiment_name}")
        else:
            print("⚠️ MLflow not available. Using local JSON tracking.")

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    def start_run(self, run_name: str, tags: Optional[Dict] = None):
        """Start a new experiment run."""
        self._run_data = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "params": {},
            "metrics": {},
            "features": [],
            "tags": tags or {}
        }

        if MLFLOW_AVAILABLE:
            self._current_run = mlflow.start_run(run_name=run_name)
            if tags:
                mlflow.set_tags(tags)

        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def end_run(self):
        """End current run and save results."""
        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.end_run()

        # Save to local JSON as backup
        self._save_local()
        self._current_run = None

    # -------------------------------------------------------------------------
    # Logging Methods
    # -------------------------------------------------------------------------

    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        self._run_data["params"].update(params)

        if MLFLOW_AVAILABLE and self._current_run:
            # MLflow needs string values
            for k, v in params.items():
                mlflow.log_param(k, str(v))

    def log_metrics(self, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        self._run_data["metrics"].update(metrics)

        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.log_metrics(metrics)

    def log_features(self, features: List[str]):
        """Log feature list with hash for change detection."""
        self._run_data["features"] = features
        self._run_data["feature_count"] = len(features)

        # Create hash to detect feature changes
        feature_str = ",".join(sorted(features))
        feature_hash = hashlib.md5(feature_str.encode()).hexdigest()[:8]
        self._run_data["feature_hash"] = feature_hash

        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.log_param("feature_count", len(features))
            mlflow.log_param("feature_hash", feature_hash)
            # Log feature list as artifact
            feature_file = self.local_dir / "temp_features.json"
            with open(feature_file, 'w') as f:
                json.dump(features, f, indent=2)
            mlflow.log_artifact(str(feature_file), "features")
            feature_file.unlink()

    def log_dataset_info(self, X_train, X_test, y_train, y_test):
        """Log dataset statistics."""
        info = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test": float(y_test.mean()),
        }
        self._run_data.update(info)

        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.log_params({
                "train_samples": info["train_samples"],
                "test_samples": info["test_samples"],
            })
            mlflow.log_metrics({
                "churn_rate_train": info["churn_rate_train"],
                "churn_rate_test": info["churn_rate_test"],
            })

    # -------------------------------------------------------------------------
    # Training & Evaluation with Timing
    # -------------------------------------------------------------------------

    def train_with_timing(self, model, X_train, y_train) -> Tuple[Any, float]:
        """
        Train model and measure time.

        Returns:
            (trained_model, training_time_seconds)
        """
        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start

        self._run_data["train_time_seconds"] = round(train_time, 3)

        if MLFLOW_AVAILABLE and self._current_run:
            mlflow.log_metric("train_time_seconds", train_time)

        print(f"⏱️ Training time: {train_time:.2f}s")
        return model, train_time

    def evaluate_with_timing(
            self,
            model,
            X_test,
            y_test,
            n_inference_samples: int = 1000
    ) -> Tuple[Dict[str, float], float]:
        """
        Evaluate model and measure inference time.

        Returns:
            (metrics_dict, inference_time_per_sample_ms)
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        # Round metrics
        metrics = {k: round(v, 4) for k, v in metrics.items()}

        # Measure inference time (average over multiple samples)
        sample_size = min(n_inference_samples, len(X_test))
        X_sample = X_test[:sample_size]

        start = time.perf_counter()
        for _ in range(10):  # 10 iterations for stability
            _ = model.predict_proba(X_sample)
        inference_time = (time.perf_counter() - start) / 10

        # Time per sample in milliseconds
        inference_time_ms = (inference_time / sample_size) * 1000

        # Log timing
        metrics["inference_time_ms"] = round(inference_time_ms, 4)
        self._run_data["inference_time_ms"] = inference_time_ms

        # Log to MLflow
        self.log_metrics(metrics)

        print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"📊 F1: {metrics['f1']:.4f}")
        print(f"⏱️ Inference: {inference_time_ms:.4f}ms per sample")

        return metrics, inference_time_ms

    # -------------------------------------------------------------------------
    # Model Management
    # -------------------------------------------------------------------------

    def log_model(
            self,
            model,
            model_name: str = "model",
            register: bool = False
    ):
        """
        Log trained model.

        Args:
            model: Trained model object
            model_name: Name for the model
            register: If True, register in Model Registry
        """
        # Get model type
        model_type = type(model).__name__
        self._run_data["model_type"] = model_type

        if MLFLOW_AVAILABLE and self._current_run:
            # Log model
            if "XGB" in model_type:
                mlflow.xgboost.log_model(model, model_name)
            else:
                mlflow.sklearn.log_model(model, model_name)

            # Register if requested
            if register:
                run_id = self._current_run.info.run_id
                model_uri = f"runs:/{run_id}/{model_name}"
                mlflow.register_model(model_uri, self.experiment_name)
                print(f"📦 Model registered: {self.experiment_name}")

        # Always save locally
        local_path = self.local_dir / f"{model_name}.pkl"
        joblib.dump(model, local_path)
        print(f"💾 Model saved: {local_path}")

    # -------------------------------------------------------------------------
    # Comparison & Analysis
    # -------------------------------------------------------------------------

    def get_all_runs(self) -> pd.DataFrame:
        """Get all experiment runs as DataFrame."""
        if MLFLOW_AVAILABLE:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                return runs

        # Fallback to local JSON
        return self._load_local_runs()

    def compare_runs(
            self,
            metric: str = "roc_auc",
            top_n: int = 10
    ) -> pd.DataFrame:
        """
        Compare top runs by a metric.

        Args:
            metric: Metric to sort by
            top_n: Number of top runs to show
        """
        runs = self.get_all_runs()

        if runs.empty:
            print("No runs found.")
            return runs

        # Find metric column (MLflow adds 'metrics.' prefix)
        metric_col = f"metrics.{metric}" if f"metrics.{metric}" in runs.columns else metric

        if metric_col not in runs.columns:
            print(f"Metric '{metric}' not found.")
            return runs

        # Sort and select top
        runs = runs.sort_values(metric_col, ascending=False).head(top_n)

        # Select relevant columns
        cols = ["run_id", "tags.mlflow.runName", metric_col]
        timing_cols = ["metrics.train_time_seconds", "metrics.inference_time_ms"]
        param_cols = [c for c in runs.columns if c.startswith("params.")]

        available_cols = [c for c in cols + timing_cols + param_cols[:5] if c in runs.columns]

        return runs[available_cols]

    def get_best_run(self, metric: str = "roc_auc") -> Optional[str]:
        """Get run_id of best performing run."""
        runs = self.compare_runs(metric=metric, top_n=1)
        if not runs.empty:
            return runs.iloc[0]["run_id"]
        return None

    # -------------------------------------------------------------------------
    # Local Storage (Backup)
    # -------------------------------------------------------------------------

    def _save_local(self):
        """Save run data to local JSON."""
        if not self._run_data.get("run_name"):
            return

        runs_file = self.local_dir / "runs.json"

        # Load existing runs
        if runs_file.exists():
            with open(runs_file) as f:
                all_runs = json.load(f)
        else:
            all_runs = []

        # Add current run
        all_runs.append(self._run_data)

        # Save
        with open(runs_file, 'w') as f:
            json.dump(all_runs, f, indent=2, default=str)

    def _load_local_runs(self) -> pd.DataFrame:
        """Load runs from local JSON."""
        runs_file = self.local_dir / "runs.json"

        if not runs_file.exists():
            return pd.DataFrame()

        with open(runs_file) as f:
            runs = json.load(f)

        return pd.DataFrame(runs)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_compare(experiment_name: str = "churn-prediction") -> pd.DataFrame:
    """Quickly compare all runs in an experiment."""
    tracker = ExperimentTracker(experiment_name)
    return tracker.compare_runs()


def get_production_model(experiment_name: str = "churn-prediction"):
    """Load the best model from registry."""
    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow required for model registry")

    model_uri = f"models:/{experiment_name}/Production"
    return mlflow.pyfunc.load_model(model_uri)


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MLOps Experiment Tracker")
    parser.add_argument("--compare", action="store_true", help="Compare all runs")
    parser.add_argument("--metric", default="roc_auc", help="Metric to sort by")
    parser.add_argument("--top", type=int, default=10, help="Top N runs")

    args = parser.parse_args()

    if args.compare:
        df = quick_compare()
        print(df.to_string())