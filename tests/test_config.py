"""
Tests for configuration module.
"""

import pytest
import os
from pathlib import Path


class TestConfiguration:
    """Test configuration settings."""

    def test_settings_defaults(self):
        """Test default settings values."""
        from app.config import Settings

        settings = Settings()
        assert settings.api_port == 9000
        assert settings.log_level == "INFO"
        assert not settings.debug

    def test_environment_override(self, monkeypatch):
        """Test environment variable override."""
        monkeypatch.setenv("API_PORT", "8080")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("DEBUG", "true")

        from app.config import Settings
        settings = Settings()

        assert settings.api_port == 8080
        assert settings.log_level == "DEBUG"
        assert settings.debug

    def test_model_paths(self):
        """Test model path properties."""
        from app.config import Settings

        settings = Settings()
        assert settings.xgboost_path.name == "xgboost_1order.pkl"
        assert settings.transformer_path.name == "ft_transformer_model.pt"
        assert settings.scaler_path.name == "scaler.pkl"
