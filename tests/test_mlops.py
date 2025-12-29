"""
Tests for MLOps Module

Run with: pytest tests/ -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExperimentTracker:
    """Tests for ExperimentTracker class."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        from mlops.experiment import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test-experiment",
                local_dir=tmpdir
            )

            assert tracker.experiment_name == "test-experiment"
            assert tracker.local_dir.exists()

    def test_start_run(self):
        """Test starting a run."""
        from mlops.experiment import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test",
                local_dir=tmpdir
            )

            with tracker.start_run(run_name="test_run"):
                assert tracker._run_data["run_name"] == "test_run"

    def test_log_params(self):
        """Test logging parameters."""
        from mlops.experiment import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test",
                local_dir=tmpdir
            )

            with tracker.start_run(run_name="test_run"):
                tracker.log_params({"learning_rate": 0.1, "max_depth": 10})

                assert tracker._run_data["params"]["learning_rate"] == 0.1
                assert tracker._run_data["params"]["max_depth"] == 10

    def test_log_features(self):
        """Test logging feature list."""
        from mlops.experiment import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test",
                local_dir=tmpdir
            )

            features = ["feature_a", "feature_b", "feature_c"]

            with tracker.start_run(run_name="test_run"):
                tracker.log_features(features)

                assert tracker._run_data["features"] == features
                assert tracker._run_data["feature_count"] == 3
                assert "feature_hash" in tracker._run_data

    def test_feature_hash_changes(self):
        """Test that feature hash changes when features change."""
        from mlops.experiment import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test",
                local_dir=tmpdir
            )

            with tracker.start_run(run_name="run1"):
                tracker.log_features(["a", "b", "c"])
                hash1 = tracker._run_data["feature_hash"]

            with tracker.start_run(run_name="run2"):
                tracker.log_features(["a", "b", "c", "d"])
                hash2 = tracker._run_data["feature_hash"]

            assert hash1 != hash2

    def test_local_save(self):
        """Test saving runs to local JSON."""
        from mlops.experiment import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(
                experiment_name="test",
                local_dir=tmpdir
            )

            with tracker.start_run(run_name="test_run"):
                tracker.log_params({"lr": 0.1})
                tracker.log_metrics({"accuracy": 0.95})

            # Check local file exists
            runs_file = Path(tmpdir) / "runs.json"
            assert runs_file.exists()

            with open(runs_file) as f:
                runs = json.load(f)

            assert len(runs) == 1
            assert runs[0]["run_name"] == "test_run"


class TestGetRiskLevel:
    """Tests for risk level function."""

    def test_high_risk(self):
        """Test high risk classification."""
        from mlops.experiment import ExperimentTracker

        # Create a mock or test the helper function
        # In main.py the function is standalone
        pass

    def test_medium_risk(self):
        """Test medium risk classification."""
        pass

    def test_low_risk(self):
        """Test low risk classification."""
        pass


class TestModelFactory:
    """Tests for model creation."""

    def test_create_xgboost(self):
        """Test XGBoost model creation."""
        from mlops.train import get_model

        model = get_model("xgboost", n_estimators=50, max_depth=3)

        assert model is not None
        assert model.get_params()["n_estimators"] == 50
        assert model.get_params()["max_depth"] == 3

    def test_create_random_forest(self):
        """Test Random Forest model creation."""
        from mlops.train import get_model

        model = get_model("rf", n_estimators=50, max_depth=5)

        assert model is not None
        assert model.get_params()["n_estimators"] == 50

    def test_invalid_model(self):
        """Test invalid model type raises error."""
        from mlops.train import get_model

        with pytest.raises(ValueError):
            get_model("invalid_model_type")


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_data_missing_file(self):
        """Test handling of missing data file."""
        from mlops.train import load_data

        with pytest.raises(Exception):
            load_data(Path("nonexistent_file.csv"))


# =============================================================================
# Integration Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.slow
    def test_full_training_pipeline(self):
        """Test complete training pipeline (slow)."""
        # This would require actual data
        # Skip in CI unless data is available
        pass


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])