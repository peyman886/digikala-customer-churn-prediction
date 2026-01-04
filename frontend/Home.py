"""
Churn Prediction Dashboard - Home Page

Main dashboard showing overview statistics and quick actions.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_translations, SEGMENT_INFO, MODEL_METRICS
from utils import (
    api_client, custom_css, init_session_state, get_language, set_language,
    gradient_header, COLORS, SEGMENT_COLORS, format_percentage, format_number
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

custom_css()

# Initialize language
init_session_state("language", "en")


# =============================================================================
# Sidebar - Language Selection
# =============================================================================

with st.sidebar:
    lang = st.selectbox(
        "🌐 Language / زبان",
        options=["en", "fa"],
        format_func=lambda x: "English" if x == "en" else "فارسی",
        index=0 if get_language() == "en" else 1,
        key="lang_selector"
    )
    set_language(lang)

# Get translations
t = get_translations(lang)
is_rtl = lang == "fa"


# =============================================================================
# Main Content
# =============================================================================

# Header
gradient_header(t.app_title, t.app_subtitle)

st.markdown("---")


# =============================================================================
# Fetch Data
# =============================================================================

@st.cache_data(ttl=300)
def fetch_dashboard_data():
    """Fetch all dashboard data with caching."""
    overview = api_client.get_overview()
    distribution = api_client.get_risk_distribution()
    return overview, distribution


overview_resp, dist_resp = fetch_dashboard_data()

if not overview_resp.success:
    st.warning(f"⚠️ {t.api_error}")
    st.code("docker-compose up -d api", language="bash")
    
    # Show static segment info instead
    st.subheader(t.segment_analysis)
    
    segments_df = pd.DataFrame([
        {
            "Segment" if lang == "en" else "سگمنت": seg,
            "Threshold (days)" if lang == "en" else "آستانه (روز)": info["threshold_days"],
            "Churn Rate" if lang == "en" else "نرخ ریزش": f"{info['churn_rate']*100:.1f}%",
            "% of Users" if lang == "en" else "% کاربران": f"{info['weight']*100:.1f}%",
        }
        for seg, info in SEGMENT_INFO.items()
    ])
    st.dataframe(segments_df, use_container_width=True, hide_index=True)
    
else:
    data = overview_resp.data
    
    # =========================================================================
    # Row 1: Key Metrics
    # =========================================================================
    
    st.subheader(t.overview_title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=t.total_users,
            value=format_number(data['total_users'])
        )
    
    with col2:
        st.metric(
            label=t.risk_low,
            value=format_number(data['low_risk']),
            delta=f"{data['low_risk_pct']}%"
        )
    
    with col3:
        st.metric(
            label=t.risk_medium,
            value=format_number(data['medium_risk']),
            delta=f"{data['medium_risk_pct']}%"
        )
    
    with col4:
        st.metric(
            label=t.risk_high,
            value=format_number(data['high_risk']),
            delta=f"{data['high_risk_pct']}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # =========================================================================
    # Row 2: Charts
    # =========================================================================
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 " + (t.churn_by_segment if lang == "fa" else "Churn by Segment"))
        
        # Segment churn rate chart
        seg_df = pd.DataFrame([
            {"Segment": seg, "Churn Rate": info["churn_rate"] * 100, "Threshold": info["threshold_days"]}
            for seg, info in SEGMENT_INFO.items()
        ])
        
        fig_seg = go.Figure()
        
        fig_seg.add_trace(go.Bar(
            x=seg_df['Segment'],
            y=seg_df['Churn Rate'],
            marker_color=[SEGMENT_COLORS[s] for s in seg_df['Segment']],
            text=[f"{r:.1f}%" for r in seg_df['Churn Rate']],
            textposition='outside',
            name='Churn Rate' if lang == "en" else 'نرخ ریزش'
        ))
        
        fig_seg.update_layout(
            height=400,
            margin=dict(t=20, b=40, l=40, r=20),
            yaxis_title="Churn Rate (%)" if lang == "en" else "نرخ ریزش (%)",
            showlegend=False,
            yaxis=dict(range=[0, 85])
        )
        
        st.plotly_chart(fig_seg, use_container_width=True)
    
    with col_right:
        st.subheader("📈 " + ("Risk Distribution" if lang == "en" else "توزیع ریسک"))
        
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=[t.risk_low, t.risk_medium, t.risk_high],
            values=[data['low_risk'], data['medium_risk'], data['high_risk']],
            hole=0.4,
            marker_colors=[COLORS['success'], COLORS['warning'], COLORS['danger']],
            textinfo='label+percent',
            textposition='outside',
            pull=[0, 0, 0.05]
        )])
        
        fig_pie.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1)
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # =========================================================================
    # Row 3: Probability Distribution
    # =========================================================================
    
    if dist_resp.success:
        st.subheader("📊 " + ("Churn Probability Distribution" if lang == "en" else "توزیع احتمال ریزش"))
        
        dist_data = dist_resp.data
        
        # Color bars based on risk threshold
        colors = []
        for b in dist_data['bins']:
            if b < 0.4:
                colors.append(COLORS['success'])
            elif b < 0.7:
                colors.append(COLORS['warning'])
            else:
                colors.append(COLORS['danger'])
        
        fig_dist = go.Figure(data=[go.Bar(
            x=dist_data['bins'],
            y=dist_data['counts'],
            marker_color=colors
        )])
        
        # Add threshold lines
        fig_dist.add_vline(x=0.4, line_dash="dash", line_color=COLORS['warning'],
                          annotation_text="Medium" if lang == "en" else "متوسط")
        fig_dist.add_vline(x=0.7, line_dash="dash", line_color=COLORS['danger'],
                          annotation_text="High" if lang == "en" else "بالا")
        
        fig_dist.update_layout(
            height=350,
            xaxis_title="Churn Probability" if lang == "en" else "احتمال ریزش",
            yaxis_title="Number of Users" if lang == "en" else "تعداد کاربران",
            margin=dict(t=20, b=40, l=40, r=20)
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")


# =============================================================================
# Row 4: Model Performance
# =============================================================================

st.subheader("🤖 " + t.model_performance)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {COLORS["danger"]}22 0%, {COLORS["danger"]}11 100%);
        border: 1px solid {COLORS["danger"]}44;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h4 style='margin: 0; color: {COLORS["danger"]};'>{t.model_xgboost}</h4>
        <p style='margin: 10px 0 0 0; color: #64748b;'>Recall: <b>{MODEL_METRICS["xgboost_1order"]["recall"]:.1%}</b></p>
        <p style='margin: 5px 0 0 0; color: #64748b;'>ROC-AUC: <b>{MODEL_METRICS["xgboost_1order"]["roc_auc"]:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {COLORS["purple"]}22 0%, {COLORS["purple"]}11 100%);
        border: 1px solid {COLORS["purple"]}44;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h4 style='margin: 0; color: {COLORS["purple"]};'>{t.model_ft_transformer}</h4>
        <p style='margin: 10px 0 0 0; color: #64748b;'>Recall: <b>{MODEL_METRICS["ft_transformer"]["overall_recall"]:.1%}</b></p>
        <p style='margin: 5px 0 0 0; color: #64748b;'>Weighted Recall: <b>{MODEL_METRICS["ft_transformer"]["weighted_recall"]:.1%}</b></p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, {COLORS["primary"]}22 0%, {COLORS["primary"]}11 100%);
        border: 1px solid {COLORS["primary"]}44;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h4 style='margin: 0; color: {COLORS["primary"]};'>{t.model_combined}</h4>
        <p style='margin: 10px 0 0 0; color: #64748b;'>Overall Recall: <b>{MODEL_METRICS["combined"]["overall_recall"]:.1%}</b></p>
        <p style='margin: 5px 0 0 0; color: #64748b;'>F1: <b>{MODEL_METRICS["combined"]["f1"]:.2f}</b></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# =============================================================================
# Row 5: Quick Actions
# =============================================================================

st.subheader("🎯 " + ("Quick Actions" if lang == "en" else "دسترسی سریع"))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"### 👤 {t.nav_prediction}")
    desc = "Look up churn risk for a specific user" if lang == "en" else "بررسی ریسک ریزش یک کاربر خاص"
    st.write(desc)
    if st.button("→ " + t.nav_prediction, key="btn_predict", use_container_width=True):
        st.switch_page("pages/1_👤_User_Prediction.py")

with col2:
    st.markdown(f"### 🚨 {t.nav_high_risk}")
    desc = "View all high-risk users" if lang == "en" else "مشاهده کاربران پرریسک"
    st.write(desc)
    if st.button("→ " + t.nav_high_risk, key="btn_highrisk", use_container_width=True):
        st.switch_page("pages/2_🚨_High_Risk_Users.py")

with col3:
    st.markdown(f"### 📊 {t.nav_analytics}")
    desc = "Explore churn insights" if lang == "en" else "کاوش در تحلیل‌ها"
    st.write(desc)
    if st.button("→ " + t.nav_analytics, key="btn_analytics", use_container_width=True):
        st.switch_page("pages/3_📊_Analytics.py")

with col4:
    st.markdown(f"### 📄 {t.nav_report}")
    desc = "View full report" if lang == "en" else "مشاهده گزارش کامل"
    st.write(desc)
    if st.button("→ " + t.nav_report, key="btn_report", use_container_width=True):
        st.switch_page("pages/4_📄_Report.py")


# =============================================================================
# Footer
# =============================================================================

st.markdown("---")

footer_text = """
<div style='text-align: center; color: #94a3b8; padding: 20px;'>
    <p>Churn Prediction System v3.0 | Segment-Based Approach</p>
    <p>Models: XGBoost (1-Order) + FT-Transformer (2+ Orders)</p>
    <p>Combined Recall: 81.8% | Weighted Recall: 65%</p>
</div>
"""
st.markdown(footer_text, unsafe_allow_html=True)


# =============================================================================
# Sidebar Info
# =============================================================================

with st.sidebar:
    st.markdown("---")
    st.markdown("### ℹ️ " + ("About" if lang == "en" else "درباره"))
    
    if lang == "en":
        st.markdown("""
        This system predicts customer churn using a **segment-based approach**:
        
        - **5 Customer Segments** based on order frequency
        - **Segment-specific churn thresholds** (14-45 days)
        - **Dual model strategy**: XGBoost + FT-Transformer
        
        **Churn Definition:**
        Users who don't return within their segment's threshold.
        """)
    else:
        st.markdown("""
        این سیستم با **رویکرد سگمنت‌محور** ریزش مشتری را پیش‌بینی می‌کند:
        
        - **۵ سگمنت مشتری** بر اساس فراوانی سفارش
        - **آستانه‌های ریزش مختص هر سگمنت** (۱۴ تا ۴۵ روز)
        - **استراتژی دو مدلی**: XGBoost + FT-Transformer
        
        **تعریف ریزش:**
        کاربرانی که در آستانه زمانی سگمنت خود برنگردند.
        """)
    
    st.markdown("---")
    st.markdown("### 📊 " + ("Segments" if lang == "en" else "سگمنت‌ها"))
    
    for seg, info in SEGMENT_INFO.items():
        st.markdown(f"**{seg}**: {info['threshold_days']}d → {info['churn_rate']*100:.0f}%")
