"""
Model Training Script with MLOps Tracking

این اسکریپت برای آموزش مدل با ثبت کامل آزمایش استفاده میشه.

Usage:
    # Train baseline model
    python mlops/train.py --name baseline_v1

    # Train with specific model
    python mlops/train.py --name xgb_tuned --model xgboost --max-depth 15

    # Train and register as production
    python mlops/train.py --name production_v1 --register

    # Compare all runs
    python mlops/train.py --compare

"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.experiment import ExperimentTracker

# =============================================================================
# Configuration
# =============================================================================

FEATURES_PATH = Path("data/user_features.csv")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_type: str, **kwargs):
    """
    Create model based on type.

    Args:
        model_type: One of 'logistic', 'rf', 'xgboost', 'gbm'
        **kwargs: Model-specific hyperparameters
    """
    models = {
        "logistic": lambda: LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            **kwargs
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 6),
            learning_rate=kwargs.get('learning_rate', 0.1),
            scale_pos_weight=kwargs.get('scale_pos_weight', 1.0),
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        ),
        "gbm": lambda: GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=42
        )
    }

    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Choose from {list(models.keys())}")

    return models[model_type]()


# =============================================================================
# Data Loading
# =============================================================================

def load_data(features_path: Path = FEATURES_PATH):
    """Load and prepare data for training."""
    print(f"📂 Loading data from {features_path}...")

    df = pd.read_csv(features_path)

    # Separate features and target
    feature_cols = [c for c in df.columns if c not in ['user_id', 'is_churned']]

    X = df[feature_cols]
    y = df['is_churned']

    print(f"   Samples: {len(df):,}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Churn rate: {y.mean():.1%}")

    return X, y, feature_cols


def prepare_data(X, y, test_size: float = 0.2):
    """Split data into train/test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# Training Pipeline
# =============================================================================

def train_model(
        run_name: str,
        model_type: str = "xgboost",
        features_path: Path = FEATURES_PATH,
        register: bool = False,
        tags: dict = None,
        **model_params
):
    """
    Complete training pipeline with MLOps tracking.

    Args:
        run_name: Name for this experiment run
        model_type: Type of model to train
        features_path: Path to features CSV
        register: If True, register model in registry
        tags: Additional tags for the run
        **model_params: Hyperparameters for the model

    Returns:
        dict with run results
    """
    print("=" * 60)
    print(f"🚀 Training: {run_name}")
    print(f"   Model: {model_type}")
    print("=" * 60)

    # Load data
    X, y, feature_cols = load_data(features_path)
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    # Calculate scale_pos_weight for XGBoost
    if model_type == "xgboost" and "scale_pos_weight" not in model_params:
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        model_params["scale_pos_weight"] = neg_count / pos_count

    # Create model
    model = get_model(model_type, **model_params)

    # Get actual model parameters
    actual_params = model.get_params()
    relevant_params = {
        k: v for k, v in actual_params.items()
        if k in ['n_estimators', 'max_depth', 'learning_rate', 'C', 'scale_pos_weight']
           and v is not None
    }
    relevant_params["model_type"] = model_type

    # Initialize tracker
    tracker = ExperimentTracker("churn-prediction")

    # Tags
    run_tags = {
        "model_type": model_type,
        "run_date": datetime.now().strftime("%Y-%m-%d"),
        **(tags or {})
    }

    # Start tracking
    with tracker.start_run(run_name=run_name, tags=run_tags):
        # Log everything
        tracker.log_params(relevant_params)
        tracker.log_features(feature_cols)
        tracker.log_dataset_info(X_train, X_test, y_train, y_test)

        # Train with timing
        print("\n⏳ Training...")
        model, train_time = tracker.train_with_timing(model, X_train, y_train)

        # Evaluate with timing
        print("\n📊 Evaluating...")
        metrics, inference_time = tracker.evaluate_with_timing(model, X_test, y_test)

        # Log model
        tracker.log_model(model, model_name="model", register=register)

    # Save to models directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{run_name}_{timestamp}.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Saved: {model_path}")

    # If this is the best, save as production model
    if register:
        prod_path = MODELS_DIR / "model.pkl"
        joblib.dump(model, prod_path)

        # Also save feature names
        with open(MODELS_DIR / "feature_names.txt", 'w') as f:
            f.write("\n".join(feature_cols))

        # Copy user_features for API
        import shutil
        shutil.copy(features_path, MODELS_DIR / "user_features.csv")

        print(f"🏭 Production model saved: {prod_path}")

    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("=" * 60)

    return {
        "run_name": run_name,
        "model_type": model_type,
        "metrics": metrics,
        "train_time": train_time,
        "inference_time_ms": inference_time,
        "model_path": str(model_path)
    }


# =============================================================================
# Comparison
# =============================================================================

def compare_experiments(metric: str = "roc_auc", top_n: int = 10):
    """Compare all experiment runs."""
    print("=" * 60)
    print("📊 Experiment Comparison")
    print("=" * 60)

    tracker = ExperimentTracker("churn-prediction")
    df = tracker.compare_runs(metric=metric, top_n=top_n)

    if df.empty:
        print("No experiments found.")
        return

    print(df.to_string())

    # Best run
    best_run_id = tracker.get_best_run(metric=metric)
    if best_run_id:
        print(f"\n🏆 Best run (by {metric}): {best_run_id}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train churn prediction model with MLOps tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train baseline XGBoost
    python mlops/train.py --name baseline_v1

    # Train with specific hyperparameters
    python mlops/train.py --name xgb_deep --model xgboost --max-depth 15 --n-estimators 200

    # Train and register as production
    python mlops/train.py --name production_v1 --register

    # Compare all experiments
    python mlops/train.py --compare

    # Compare by F1 score
    python mlops/train.py --compare --metric f1
        """
    )

    # Run configuration
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (e.g., 'baseline_v1', 'with_sentiment')")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=["logistic", "rf", "xgboost", "gbm"],
                        help="Model type")
    parser.add_argument("--register", action="store_true",
                        help="Register as production model")

    # Model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)

    # Data
    parser.add_argument("--features", type=str, default="data/user_features.csv",
                        help="Path to features CSV")

    # Comparison
    parser.add_argument("--compare", action="store_true",
                        help="Compare all experiments")
    parser.add_argument("--metric", type=str, default="roc_auc",
                        help="Metric for comparison")
    parser.add_argument("--top", type=int, default=10,
                        help="Top N runs to show")

    # Tags
    parser.add_argument("--tag", action="append", nargs=2, metavar=("KEY", "VALUE"),
                        help="Add tag (can be used multiple times)")

    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        compare_experiments(metric=args.metric, top_n=args.top)
        return

    # Training mode
    if not args.name:
        # Generate name if not provided
        args.name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Parse tags
    tags = dict(args.tag) if args.tag else None

    # Train
    result = train_model(
        run_name=args.name,
        model_type=args.model,
        features_path=Path(args.features),
        register=args.register,
        tags=tags,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate
    )

    print(f"\n📝 Results Summary:")
    print(f"   ROC-AUC: {result['metrics']['roc_auc']:.4f}")
    print(f"   F1: {result['metrics']['f1']:.4f}")
    print(f"   Train Time: {result['train_time']:.2f}s")
    print(f"   Inference: {result['inference_time_ms']:.4f}ms/sample")


if __name__ == "__main__":
    main()