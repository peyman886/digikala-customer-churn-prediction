"""Configuration module for frontend."""

from .i18n import get_translations, Translations, SEGMENT_INFO, MODEL_METRICS, FEATURE_IMPORTANCE, EN, FA

__all__ = [
    "get_translations",
    "Translations", 
    "SEGMENT_INFO",
    "MODEL_METRICS",
    "FEATURE_IMPORTANCE",
    "EN",
    "FA",
]
