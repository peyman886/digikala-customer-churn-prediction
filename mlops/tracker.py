"""
Experiment Tracker for Churn Prediction

Simple and clean API for tracking ML experiments.

Usage in Notebook:
    from mlops import ExperimentTracker

    tracker = ExperimentTracker()

    with tracker.run("added_sentiment_features"):
        # Train your model
        model.fit(X_train, y_train)

        # Log metrics
        tracker.log_metrics(model, X_test, y_test)

        # Save model
        tracker.log_model(model)

    # Compare all experiments
    tracker.compare()

    # Promote best model to production
    tracker.promote("added_sentiment_features")
"""

import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from contextlib import contextmanager

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, accuracy_score
)

# MLflow
import mlflow
from mlflow.tracking import MlflowClient

# Local config
from mlops.config import (
    EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MLFLOW_DIR,
    PRODUCTION_DIR, EXPERIMENTS_DIR, APP_MODEL_PATH, APP_FEATURES_PATH,
    FEATURES_PATH, PRIMARY_METRIC
)


class ExperimentTracker:
    """
    Simple experiment tracker with MLflow backend.

    Designed for easy use in Jupyter notebooks.

    Example:
        tracker = ExperimentTracker()

        with tracker.run("my_experiment"):
            model = XGBClassifier()
            model.fit(X_train, y_train)
            tracker.log_metrics(model, X_test, y_test)
            tracker.log_model(model, feature_names)

        # See all experiments
        tracker.compare()

        # Deploy best model
        tracker.promote("my_experiment")
    """

    def __init__(self, experiment_name: str = EXPERIMENT_NAME):
        """
        Initialize tracker.

        Args:
            experiment_name: Name for grouping experiments (default: "churn-prediction")
        """
        self.experiment_name = experiment_name
        self._current_run = None
        self._run_name = None
        self._start_time = None

        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

        print(f"✅ Tracker ready: {experiment_name}")
        print(f"   MLflow UI: mlflow ui --port 5000 --backend-store-uri {MLFLOW_DIR}")

    # =========================================================================
    # Context Manager for Runs
    # =========================================================================

    @contextmanager
    def run(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Start an experiment run.

        Args:
            name: Descriptive name for this run (e.g., "added_sentiment_features")
            tags: Optional tags for filtering

        Usage:
            with tracker.run("my_experiment"):
                # your training code
                tracker.log_metrics(...)
        """
        self._run_name = name
        self._start_time = time.perf_counter()

        # Default tags
        all_tags = {
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            **(tags or {})
        }

        print(f"\n{'=' * 60}")
        print(f"🚀 Starting: {name}")
        print(f"{'=' * 60}")

        try:
            with mlflow.start_run(run_name=name) as run:
                self._current_run = run
                mlflow.set_tags(all_tags)
                yield self

        finally:
            elapsed = time.perf_counter() - self._start_time
            print(f"\n{'=' * 60}")
            print(f"✅ Completed: {name} ({elapsed:.1f}s)")
            print(f"{'=' * 60}\n")
            self._current_run = None
            self._run_name = None

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names and values
        """
        if not self._current_run:
            raise RuntimeError("Must be inside a run() context")

        # MLflow needs string values for params
        for key, value in params.items():
            mlflow.log_param(key, str(value))

        print(f"📝 Logged {len(params)} params")

    def log_metrics(
            self,
            model,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            X_train: Optional[pd.DataFrame] = None,
            y_train: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model and log all metrics.

        Args:
            model: Trained model with predict() and predict_proba()
            X_test: Test features
            y_test: Test labels
            X_train: Optional train features (for logging sizes)
            y_train: Optional train labels (for logging churn rate)

        Returns:
            Dictionary of computed metrics
        """
        if not self._current_run:
            raise RuntimeError("Must be inside a run() context")

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Core metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }

        # Timing: measure inference speed
        inference_time = self._measure_inference_time(model, X_test)
        metrics["inference_time_ms"] = inference_time

        # Training time (total elapsed so far)
        if self._start_time:
            metrics["train_time_seconds"] = time.perf_counter() - self._start_time

        # Dataset info
        metrics["test_samples"] = len(X_test)
        metrics["feature_count"] = X_test.shape[1]

        if X_train is not None:
            metrics["train_samples"] = len(X_train)
        if y_train is not None:
            metrics["churn_rate_train"] = float(y_train.mean())
        metrics["churn_rate_test"] = float(y_test.mean())

        # Log to MLflow
        mlflow.log_metrics(metrics)

        # Print summary
        print(f"\n📊 Metrics:")
        print(f"   ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"   F1:         {metrics['f1']:.4f}")
        print(f"   Precision:  {metrics['precision']:.4f}")
        print(f"   Recall:     {metrics['recall']:.4f}")
        print(f"   Inference:  {metrics['inference_time_ms']:.4f} ms/sample")

        return metrics

    def log_model(
            self,
            model,
            feature_names: Optional[List[str]] = None,
            model_type: Optional[str] = None,
    ):
        """
        Save model artifact.

        Args:
            model: Trained model object
            feature_names: List of feature column names
            model_type: Optional model type string (auto-detected if not provided)
        """
        if not self._current_run:
            raise RuntimeError("Must be inside a run() context")

        # Auto-detect model type
        if model_type is None:
            model_type = type(model).__name__

        mlflow.log_param("model_type", model_type)

        # Log model based on type
        if "XGB" in model_type:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        # Save feature names
        if feature_names:
            mlflow.log_param("feature_count", len(feature_names))

            # Save as artifact
            features_file = MLFLOW_DIR / "temp_features.json"
            with open(features_file, 'w') as f:
                json.dump(feature_names, f)
            mlflow.log_artifact(str(features_file), "metadata")
            features_file.unlink()

        # Also save locally in experiments folder
        run_id = self._current_run.info.run_id
        local_path = EXPERIMENTS_DIR / f"{self._run_name}_{run_id[:8]}.pkl"
        joblib.dump(model, local_path)

        print(f"💾 Model saved: {local_path.name}")

    def log_figure(self, fig, name: str):
        """
        Log a matplotlib/plotly figure.

        Args:
            fig: Figure object
            name: Name for the figure file
        """
        if not self._current_run:
            raise RuntimeError("Must be inside a run() context")

        # Save temporarily and log
        temp_path = MLFLOW_DIR / f"temp_{name}.png"
        fig.savefig(temp_path, dpi=100, bbox_inches='tight')
        mlflow.log_artifact(str(temp_path), "figures")
        temp_path.unlink()

        print(f"📈 Figure saved: {name}")

    # =========================================================================
    # Comparison Methods
    # =========================================================================

    def compare(
            self,
            metric: str = PRIMARY_METRIC,
            top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Compare all experiment runs.

        Args:
            metric: Metric to sort by (default: roc_auc)
            top_n: Number of top runs to show

        Returns:
            DataFrame with comparison
        """
        # Get all runs
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            print("❌ No experiments found")
            return pd.DataFrame()

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
        )

        if runs.empty:
            print("❌ No runs found")
            return runs

        # Select and rename columns
        col_map = {
            "tags.mlflow.runName": "Name",
            "params.model_type": "Model",
            "metrics.roc_auc": "ROC-AUC",
            "metrics.f1": "F1",
            "metrics.precision": "Precision",
            "metrics.recall": "Recall",
            "metrics.train_time_seconds": "Train(s)",
            "metrics.inference_time_ms": "Infer(ms)",
            "start_time": "Date",
        }

        available = [c for c in col_map.keys() if c in runs.columns]
        df = runs[available].head(top_n).copy()
        df.columns = [col_map[c] for c in available]

        # Format numbers
        for col in ["ROC-AUC", "F1", "Precision", "Recall"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")

        if "Train(s)" in df.columns:
            df["Train(s)"] = df["Train(s)"].apply(lambda x: f"{x:.1f}" if pd.notnull(x) else "-")

        if "Infer(ms)" in df.columns:
            df["Infer(ms)"] = df["Infer(ms)"].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "-")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m-%d %H:%M")

        print(f"\n📊 Top {len(df)} experiments (by {metric}):\n")
        print(df.to_string(index=False))

        # Show best
        if len(df) > 0:
            best_name = df.iloc[0]["Name"]
            best_score = df.iloc[0].get("ROC-AUC", df.iloc[0].get("F1", "N/A"))
            print(f"\n🏆 Best: {best_name} ({metric}={best_score})")

        return df

    def get_best_run(self, metric: str = PRIMARY_METRIC) -> Optional[str]:
        """Get run name of best performing experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )

        if runs.empty:
            return None

        return runs.iloc[0].get("tags.mlflow.runName")

    # =========================================================================
    # Production Deployment
    # =========================================================================

    def promote(
            self,
            run_name: str,
            copy_to_app: bool = True,
    ) -> bool:
        """
        Promote an experiment to production.

        This copies the model to:
        1. models/production/ (versioned backup)
        2. app/model.pkl (for FastAPI to use)

        Args:
            run_name: Name of the run to promote
            copy_to_app: Also copy to app/ folder for FastAPI

        Returns:
            True if successful
        """
        # Find the run
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if not experiment:
            print("❌ Experiment not found")
            return False

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{run_name}'",
        )

        if runs.empty:
            print(f"❌ Run not found: {run_name}")
            return False

        run_id = runs.iloc[0]["run_id"]

        # Load the model from MLflow
        model_uri = f"runs:/{run_id}/model"

        try:
            # Try loading as XGBoost first, then sklearn
            try:
                model = mlflow.xgboost.load_model(model_uri)
            except:
                model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

        # Get metrics for metadata
        metrics = {
            "roc_auc": runs.iloc[0].get("metrics.roc_auc"),
            "f1": runs.iloc[0].get("metrics.f1"),
            "promoted_at": datetime.now().isoformat(),
            "run_id": run_id,
            "run_name": run_name,
        }

        # Save to production folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prod_model_path = PRODUCTION_DIR / f"model_{timestamp}.pkl"
        prod_meta_path = PRODUCTION_DIR / f"metadata_{timestamp}.json"

        joblib.dump(model, prod_model_path)
        with open(prod_meta_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"✅ Saved to production: {prod_model_path.name}")

        # Copy to app folder
        if copy_to_app:
            # Ensure app directory exists
            APP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

            # Copy model
            shutil.copy(prod_model_path, APP_MODEL_PATH)
            print(f"✅ Copied to app: {APP_MODEL_PATH}")

            # Copy features if exists
            if FEATURES_PATH.exists():
                shutil.copy(FEATURES_PATH, APP_FEATURES_PATH)
                print(f"✅ Copied features: {APP_FEATURES_PATH}")

            # Save metadata in app
            app_meta_path = APP_MODEL_PATH.parent / "model_metadata.json"
            with open(app_meta_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        print(f"\n🚀 Model '{run_name}' is now in production!")
        print(f"   ROC-AUC: {metrics.get('roc_auc', 'N/A'):.4f}")

        return True

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _measure_inference_time(
            self,
            model,
            X: pd.DataFrame,
            n_iterations: int = 10,
            sample_size: int = 1000,
    ) -> float:
        """
        Measure inference time per sample in milliseconds.
        """
        sample_size = min(sample_size, len(X))
        X_sample = X.iloc[:sample_size]

        # Warm up
        _ = model.predict_proba(X_sample)

        # Measure
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = model.predict_proba(X_sample)
        elapsed = time.perf_counter() - start

        # Time per sample in ms
        time_per_sample_ms = (elapsed / n_iterations / sample_size) * 1000

        return round(time_per_sample_ms, 4)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_compare(metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Quick comparison of all experiments."""
    tracker = ExperimentTracker()
    return tracker.compare(metric=metric)


def promote_best(metric: str = PRIMARY_METRIC) -> bool:
    """Promote the best model to production."""
    tracker = ExperimentTracker()
    best_run = tracker.get_best_run(metric=metric)

    if best_run:
        return tracker.promote(best_run)

    print("❌ No runs found to promote")
    return False