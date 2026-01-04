"""
User Churn Prediction Page

Check churn risk for individual users with detailed insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_translations, SEGMENT_INFO, MODEL_METRICS
from utils import (
    api_client, custom_css, init_session_state, get_language,
    COLORS, SEGMENT_COLORS, get_risk_color, get_risk_emoji,
    risk_level_badge, segment_badge, format_percentage, format_number
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="User Prediction",
    page_icon="👤",
    layout="wide"
)

custom_css()
init_session_state("language", "en")
lang = get_language()
t = get_translations(lang)


# =============================================================================
# Helper Functions
# =============================================================================

def get_segment_from_orders(total_orders: int) -> str:
    """Determine segment from order count."""
    if total_orders == 1:
        return "1 Order"
    elif total_orders <= 4:
        return "2-4 Orders"
    elif total_orders <= 10:
        return "5-10 Orders"
    elif total_orders <= 30:
        return "11-30 Orders"
    else:
        return "30+ Orders"


def create_gauge_chart(probability: float, risk_level: str) -> go.Figure:
    """Create probability gauge chart."""
    risk_color = get_risk_color(risk_level)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': '%', 'font': {'size': 40, 'color': risk_color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': risk_color, 'thickness': 0.8},
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': '#e2e8f0',
            'steps': [
                {'range': [0, 40], 'color': f"{COLORS['success']}22"},
                {'range': [40, 70], 'color': f"{COLORS['warning']}22"},
                {'range': [70, 100], 'color': f"{COLORS['danger']}22"}
            ],
            'threshold': {
                'line': {'color': risk_color, 'width': 4},
                'thickness': 0.8,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': COLORS['dark']}
    )
    
    return fig


def create_user_radar_chart(profile: dict, lang: str = "en") -> go.Figure:
    """Create radar chart for user profile."""
    # Normalize metrics to 0-100 scale
    categories = [
        ("Orders" if lang == "en" else "سفارشات", min(profile.get('total_orders', 0) / 50 * 100, 100)),
        ("Tenure" if lang == "en" else "عمر", min(profile.get('customer_tenure_days', 0) / 365 * 100, 100)),
        ("OTD Rate" if lang == "en" else "تحویل به‌موقع", profile.get('on_time_ratio', 0) * 100),
        ("Shop Rating" if lang == "en" else "امتیاز فروشگاه", profile.get('avg_shop_rating', 0) / 5 * 100),
        ("Engagement" if lang == "en" else "تعامل", min(profile.get('comment_count', 0) / 10 * 100, 100)),
    ]
    
    labels = [c[0] for c in categories]
    values = [c[1] for c in categories]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the shape
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor=f"{COLORS['primary']}33",
        line=dict(color=COLORS['primary'], width=2),
        name='User Profile'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=300,
        margin=dict(l=60, r=60, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


def create_recency_comparison(user_recency: int, segment: str) -> go.Figure:
    """Create bar chart comparing user recency to segment threshold."""
    threshold = SEGMENT_INFO[segment]['threshold_days']
    
    colors = [
        COLORS['danger'] if user_recency > threshold else COLORS['success'],
        COLORS['secondary']
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[
                "User Recency" if lang == "en" else "رسنسی کاربر",
                "Threshold" if lang == "en" else "آستانه"
            ],
            y=[user_recency, threshold],
            marker_color=colors,
            text=[f"{user_recency}d", f"{threshold}d"],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=40),
        yaxis_title="Days" if lang == "en" else "روز",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig


# =============================================================================
# Main Content
# =============================================================================

st.title(t.user_prediction_title)
st.markdown(t.user_prediction_desc)

st.markdown("---")

# Input form
col1, col2 = st.columns([3, 1])

with col1:
    user_id = st.text_input(
        t.enter_user_id,
        placeholder="e.g., 1385028",
        help="Enter the numeric user ID"
    )

with col2:
    st.write("")
    st.write("")
    predict_btn = st.button(t.predict_button, type="primary", use_container_width=True)

# Sample IDs
st.caption(f"{t.sample_ids}: 1385028, 54227, 30492532")

st.markdown("---")

# =============================================================================
# Prediction Result
# =============================================================================

if predict_btn and user_id:
    with st.spinner(t.loading):
        prediction_resp = api_client.predict_user(user_id.strip())
        profile_resp = api_client.get_user_profile(user_id.strip())
    
    if not prediction_resp.success:
        if prediction_resp.error == "not_found":
            st.error(f"❌ {t.user_not_found}: {user_id}")
        else:
            st.error(f"❌ {prediction_resp.error}")
    else:
        prediction = prediction_resp.data
        profile = profile_resp.data if profile_resp.success else {}
        
        # Determine segment
        total_orders = profile.get('total_orders', prediction.get('total_orders', 1))
        segment = get_segment_from_orders(total_orders)
        threshold = SEGMENT_INFO[segment]['threshold_days']
        
        risk_level = prediction['risk_level']
        risk_color = get_risk_color(risk_level)
        risk_emoji = get_risk_emoji(risk_level)
        
        # =====================================================================
        # Result Header
        # =====================================================================
        
        st.subheader(t.prediction_result)
        
        # Main result box
        st.markdown(f"""
        <div style='
            background: linear-gradient(90deg, {risk_color}22 0%, transparent 100%);
            border-left: 5px solid {risk_color};
            padding: 25px;
            border-radius: 10px;
            margin: 20px 0;
        '>
            <div style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
                <div>
                    <h2 style='margin: 0; color: {risk_color};'>
                        {risk_emoji} {t.risk_level}: {risk_level}
                    </h2>
                    <p style='margin: 10px 0 0 0; color: #64748b;'>
                        User ID: <b>{prediction['user_id']}</b> | 
                        Segment: {segment_badge(segment)}
                    </p>
                </div>
                <div style='text-align: right;'>
                    <h1 style='margin: 0; color: {risk_color}; font-size: 2.5em;'>
                        {prediction['probability']*100:.1f}%
                    </h1>
                    <p style='margin: 0; color: #64748b;'>{t.churn_probability}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # =====================================================================
        # Metrics Row
        # =====================================================================
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            churn_text = (t.yes + " ⚠️") if prediction['will_churn'] else (t.no + " ✅")
            st.metric(t.will_churn, churn_text)
        
        with col2:
            st.metric(t.user_segment, segment)
        
        with col3:
            st.metric(t.churn_threshold, f"{threshold} {t.days}")
        
        with col4:
            model = prediction.get('model_used', 'XGBoost' if total_orders == 1 else 'FT-Transformer')
            st.metric(t.model_used, model)
        
        st.markdown("---")
        
        # =====================================================================
        # Visualizations
        # =====================================================================
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📊 " + t.churn_probability)
            fig_gauge = create_gauge_chart(prediction['probability'], risk_level)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col_right:
            st.subheader("📅 " + ("Recency vs Threshold" if lang == "en" else "رسنسی در مقابل آستانه"))
            recency = profile.get('days_since_last_order', prediction.get('recency', 0))
            fig_recency = create_recency_comparison(recency, segment)
            st.plotly_chart(fig_recency, use_container_width=True)
        
        # =====================================================================
        # User Profile
        # =====================================================================
        
        if profile:
            st.markdown("---")
            st.subheader(t.user_profile)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Profile metrics in grid
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                
                with m_col1:
                    st.metric(t.total_orders, profile.get('total_orders', 'N/A'))
                    st.metric(t.recency_days, profile.get('days_since_last_order', 'N/A'))
                
                with m_col2:
                    tenure = profile.get('customer_tenure_days', 0)
                    st.metric(t.tenure_days, f"{tenure} {t.days}")
                    st.metric("📆 " + ("Last 30d" if lang == "en" else "۳۰ روز اخیر"), 
                             profile.get('orders_last_30d', 'N/A'))
                
                with m_col3:
                    otd = profile.get('on_time_ratio', 0)
                    st.metric(t.otd_rate, format_percentage(otd))
                    st.metric(t.avg_shop_rating, f"{profile.get('avg_shop_rating', 0):.1f} ⭐")
                
                with m_col4:
                    st.metric(t.crm_requests, profile.get('total_complaints', 0))
                    st.metric(t.comment_count, profile.get('comment_count', 0))
            
            with col2:
                # Radar chart
                fig_radar = create_user_radar_chart(profile, lang)
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # =====================================================================
        # Segment Context
        # =====================================================================
        
        st.markdown("---")
        st.subheader("📈 " + ("Segment Context" if lang == "en" else "متن سگمنت"))
        
        seg_info = SEGMENT_INFO[segment]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='
                background: {SEGMENT_COLORS[segment]}11;
                border: 1px solid {SEGMENT_COLORS[segment]}44;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            '>
                <p style='margin: 0; color: #64748b;'>{"Segment Churn Rate" if lang == "en" else "نرخ ریزش سگمنت"}</p>
                <h2 style='margin: 5px 0; color: {SEGMENT_COLORS[segment]};'>{seg_info['churn_rate']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='
                background: {COLORS['secondary']}11;
                border: 1px solid {COLORS['secondary']}44;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            '>
                <p style='margin: 0; color: #64748b;'>{"% of All Users" if lang == "en" else "% از کل کاربران"}</p>
                <h2 style='margin: 5px 0; color: {COLORS['secondary']};'>{seg_info['weight']*100:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # Compare user to segment
            user_prob = prediction['probability']
            seg_rate = seg_info['churn_rate']
            diff = user_prob - seg_rate
            diff_color = COLORS['danger'] if diff > 0 else COLORS['success']
            
            st.markdown(f"""
            <div style='
                background: {diff_color}11;
                border: 1px solid {diff_color}44;
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            '>
                <p style='margin: 0; color: #64748b;'>{"vs Segment Avg" if lang == "en" else "نسبت به میانگین"}</p>
                <h2 style='margin: 5px 0; color: {diff_color};'>{diff*100:+.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # =====================================================================
        # Recommendations
        # =====================================================================
        
        st.markdown("---")
        st.subheader(t.recommendations)
        
        if risk_level == "HIGH":
            st.error(t.high_risk_action)
        elif risk_level == "MEDIUM":
            st.warning(t.medium_risk_action)
        else:
            st.success(t.low_risk_action)

elif predict_btn and not user_id:
    st.warning("⚠️ " + ("Please enter a user ID" if lang == "en" else "لطفاً شناسه کاربر را وارد کنید"))


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### ℹ️ " + ("About" if lang == "en" else "درباره"))
    
    if lang == "en":
        st.markdown("""
        This page predicts the likelihood of a customer churning 
        based on their **segment-specific threshold**.
        
        **Risk Levels:**
        - 🟢 **Low**: < 40% probability
        - 🟡 **Medium**: 40-70% probability
        - 🔴 **High**: > 70% probability
        
        **Models Used:**
        - **1-Order Users**: XGBoost
        - **2+ Orders**: FT-Transformer
        """)
    else:
        st.markdown("""
        این صفحه احتمال ریزش مشتری را بر اساس 
        **آستانه مختص سگمنت** پیش‌بینی می‌کند.
        
        **سطوح ریسک:**
        - 🟢 **پایین**: کمتر از ۴۰٪
        - 🟡 **متوسط**: ۴۰ تا ۷۰٪
        - 🔴 **بالا**: بیش از ۷۰٪
        
        **مدل‌های استفاده‌شده:**
        - **کاربران ۱ سفارش**: XGBoost
        - **۲+ سفارش**: FT-Transformer
        """)
    
    st.markdown("---")
    st.markdown("### 📊 " + ("Segment Thresholds" if lang == "en" else "آستانه‌های سگمنت"))
    
    for seg, info in SEGMENT_INFO.items():
        st.markdown(f"**{seg}**: {info['threshold_days']} days")
