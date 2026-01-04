"""Utility modules for frontend."""

from .api_client import api_client, APIClient, APIResponse
from .helpers import (
    COLORS, RISK_COLORS, SEGMENT_COLORS,
    get_risk_color, get_risk_emoji, get_segment_color,
    format_percentage, format_number, format_days,
    styled_metric_card, risk_level_badge, segment_badge,
    info_box, gradient_header,
    init_session_state, get_language, set_language,
    apply_rtl_style, page_config, custom_css,
)

__all__ = [
    "api_client", "APIClient", "APIResponse",
    "COLORS", "RISK_COLORS", "SEGMENT_COLORS",
    "get_risk_color", "get_risk_emoji", "get_segment_color",
    "format_percentage", "format_number", "format_days",
    "styled_metric_card", "risk_level_badge", "segment_badge",
    "info_box", "gradient_header",
    "init_session_state", "get_language", "set_language",
    "apply_rtl_style", "page_config", "custom_css",
]
