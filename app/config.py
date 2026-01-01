"""
Application Configuration

Loads settings from environment variables.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal

import torch


def get_device(preference: Literal["auto", "cpu", "cuda"] = "auto") -> torch.device:
    """Get PyTorch device based on preference."""
    if preference == "cpu":
        return torch.device("cpu")
    elif preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    else:  # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Settings:
    """Application settings from environment variables."""

    # API
    api_port: int = int(os.getenv("API_PORT", "9000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Model paths
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./models_v2"))
    xgboost_filename: str = "xgboost_1order.pkl"
    transformer_filename: str = "ft_transformer_model.pt"
    scaler_filename: str = "scaler.pkl"

    # Data
    features_path: Path = Path(os.getenv("FEATURES_PATH", "./user_features.csv"))

    # Device & Performance
    device_preference: str = os.getenv("DEVICE", "auto")
    inference_batch_size: int = int(os.getenv("INFERENCE_BATCH_SIZE", "1024"))

    # FT-Transformer architecture (must match training)
    num_features: int = 98
    d_token: int = 64
    n_blocks: int = 3
    n_heads: int = 4
    d_ff_multiplier: int = 2
    dropout: float = 0.2

    @property
    def xgboost_path(self) -> Path:
        return self.model_dir / self.xgboost_filename

    @property
    def transformer_path(self) -> Path:
        return self.model_dir / self.transformer_filename

    @property
    def scaler_path(self) -> Path:
        return self.model_dir / self.scaler_filename

    def get_device(self) -> torch.device:
        return get_device(self.device_preference)


# Global settings instance
settings = Settings()