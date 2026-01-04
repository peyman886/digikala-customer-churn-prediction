"""
UI Helper Functions

Common utilities for Streamlit UI components.
"""

from typing import Dict, Any, Optional
import streamlit as st


# =============================================================================
# Color Schemes
# =============================================================================

COLORS = {
    "primary": "#2563eb",
    "secondary": "#64748b",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "purple": "#8b5cf6",
    "cyan": "#06b6d4",
    "dark": "#1e293b",
    "light": "#f8fafc",
}

RISK_COLORS = {
    "LOW": "#10b981",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
}

SEGMENT_COLORS = {
    "1 Order": "#ef4444",
    "2-4 Orders": "#f59e0b",
    "5-10 Orders": "#10b981",
    "11-30 Orders": "#2563eb",
    "30+ Orders": "#8b5cf6",
}


# =============================================================================
# Risk Level Utilities
# =============================================================================

def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    return RISK_COLORS.get(risk_level.upper(), COLORS["secondary"])


def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level."""
    emojis = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
    return emojis.get(risk_level.upper(), "⚪")


def get_segment_color(segment: str) -> str:
    """Get color for segment."""
    return SEGMENT_COLORS.get(segment, COLORS["secondary"])


# =============================================================================
# Formatting Utilities
# =============================================================================

def format_percentage(value: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: int) -> str:
    """Format number with thousand separators."""
    return f"{value:,}"


def format_days(value: int, lang: str = "en") -> str:
    """Format days value."""
    suffix = "روز" if lang == "fa" else "days"
    return f"{value} {suffix}"


# =============================================================================
# UI Components
# =============================================================================

def styled_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    color: str = "primary",
    icon: str = ""
) -> None:
    """Display a styled metric card."""
    bg_color = COLORS.get(color, COLORS["primary"])
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {bg_color}22 0%, {bg_color}11 100%);
        border-left: 4px solid {bg_color};
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 10px;
    '>
        <p style='margin: 0; color: #64748b; font-size: 0.85em;'>{icon} {label}</p>
        <h2 style='margin: 5px 0 0 0; color: {bg_color};'>{value}</h2>
        {f"<p style='margin: 5px 0 0 0; color: #64748b; font-size: 0.8em;'>{delta}</p>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def risk_level_badge(risk_level: str, lang: str = "en") -> str:
    """Generate HTML badge for risk level."""
    color = get_risk_color(risk_level)
    emoji = get_risk_emoji(risk_level)
    
    labels = {
        "LOW": ("Low" if lang == "en" else "پایین"),
        "MEDIUM": ("Medium" if lang == "en" else "متوسط"),
        "HIGH": ("High" if lang == "en" else "بالا"),
    }
    label = labels.get(risk_level.upper(), risk_level)
    
    return f"""
    <span style='
        background-color: {color}22;
        color: {color};
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        border: 1px solid {color}44;
    '>{emoji} {label}</span>
    """


def segment_badge(segment: str) -> str:
    """Generate HTML badge for segment."""
    color = get_segment_color(segment)
    
    return f"""
    <span style='
        background-color: {color}22;
        color: {color};
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        border: 1px solid {color}44;
    '>{segment}</span>
    """


def info_box(title: str, content: str, color: str = "primary") -> None:
    """Display an info box."""
    bg_color = COLORS.get(color, COLORS["primary"])
    
    st.markdown(f"""
    <div style='
        background: linear-gradient(90deg, {bg_color}22 0%, transparent 100%);
        border-left: 4px solid {bg_color};
        padding: 15px 20px;
        border-radius: 5px;
        margin: 10px 0;
    '>
        <h4 style='margin: 0 0 10px 0; color: {bg_color};'>{title}</h4>
        <p style='margin: 0; color: #334155;'>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def gradient_header(title: str, subtitle: str = "") -> None:
    """Display a gradient header."""
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
    '>
        <h1 style='margin: 0; font-size: 2.2em;'>{title}</h1>
        {f"<p style='margin: 10px 0 0 0; opacity: 0.9;'>{subtitle}</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Session State Helpers
# =============================================================================

def init_session_state(key: str, default: Any) -> None:
    """Initialize session state if not exists."""
    if key not in st.session_state:
        st.session_state[key] = default


def get_language() -> str:
    """Get current language from session state."""
    return st.session_state.get("language", "en")


def set_language(lang: str) -> None:
    """Set current language in session state."""
    st.session_state["language"] = lang


# =============================================================================
# Layout Helpers
# =============================================================================

def apply_rtl_style() -> None:
    """Apply RTL styling for Persian language."""
    st.markdown("""
    <style>
        .rtl {
            direction: rtl;
            text-align: right;
        }
        .rtl h1, .rtl h2, .rtl h3, .rtl h4, .rtl p {
            direction: rtl;
            text-align: right;
        }
    </style>
    """, unsafe_allow_html=True)


def page_config(title: str, icon: str = "🔮", layout: str = "wide") -> None:
    """Set page configuration."""
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )


def custom_css() -> None:
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
        /* Main styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Metric cards */
        .stMetric {
            background-color: #f8fafc;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
        }
        
        /* Risk colors */
        .risk-high { color: #ef4444; }
        .risk-medium { color: #f59e0b; }
        .risk-low { color: #10b981; }
        
        /* Cards */
        .metric-card {
            background-color: #f8fafc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e2e8f0;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }
        ::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }
    </style>
    """, unsafe_allow_html=True)
