"""
Analytics & Insights Page

Explore churn factors and model insights.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_translations, SEGMENT_INFO, MODEL_METRICS, FEATURE_IMPORTANCE
from utils import (
    api_client, custom_css, init_session_state, get_language,
    COLORS, SEGMENT_COLORS, format_percentage, format_number
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

custom_css()
init_session_state("language", "en")
lang = get_language()
t = get_translations(lang)


# =============================================================================
# Main Content
# =============================================================================

st.title(t.analytics_title)
st.markdown(t.analytics_desc)

st.markdown("---")


# =============================================================================
# Row 1: Key Insights
# =============================================================================

st.subheader(t.key_insights)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    '>
        <h2 style='margin: 0; font-size: 2.5em;'>44.9%</h2>
        <p style='margin: 10px 0 0 0; opacity: 0.9;'>
            {"of users are one-time buyers" if lang == "en" else "کاربران یک‌بار خریدار"}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    '>
        <h2 style='margin: 0; font-size: 2.5em;'>74.4%</h2>
        <p style='margin: 10px 0 0 0; opacity: 0.9;'>
            {"churn rate for 1-order users" if lang == "en" else "نرخ ریزش کاربران ۱ سفارش"}
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
    '>
        <h2 style='margin: 0; font-size: 2.5em;'>12.5%</h2>
        <p style='margin: 10px 0 0 0; opacity: 0.9;'>
            {"churn rate for VIP users (30+)" if lang == "en" else "نرخ ریزش کاربران VIP"}
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")


# =============================================================================
# Row 2: Feature Importance
# =============================================================================

st.subheader(t.feature_importance)

# Use local feature importance data
features_df = pd.DataFrame(FEATURE_IMPORTANCE)

# Create horizontal bar chart
fig_importance = go.Figure()

# Color gradient based on importance
max_imp = features_df['importance'].max()
colors = [f'rgba(99, 102, 241, {0.4 + 0.6 * (imp / max_imp)})' for imp in features_df['importance']]

# Highlight top feature
colors[0] = COLORS['danger']

display_col = 'display_fa' if lang == 'fa' else 'display_en'

fig_importance.add_trace(go.Bar(
    y=features_df[display_col][::-1],
    x=features_df['importance'][::-1],
    orientation='h',
    marker_color=colors[::-1],
    text=[f"{x*100:.1f}%" for x in features_df['importance'][::-1]],
    textposition='outside'
))

fig_importance.update_layout(
    height=450,
    margin=dict(l=20, r=100, t=20, b=20),
    xaxis_title="Importance Score" if lang == "en" else "امتیاز اهمیت",
    yaxis_title="",
    showlegend=False,
    xaxis=dict(range=[0, max_imp * 1.2])
)

st.plotly_chart(fig_importance, use_container_width=True)

# Feature interpretation
with st.expander("📖 " + ("Feature Interpretation" if lang == "en" else "تفسیر ویژگی‌ها")):
    if lang == "en":
        st.markdown("""
        | Feature | Description | Why It Matters |
        |---------|-------------|----------------|
        | **Recency/Tenure Ratio** | Days since last order ÷ customer age | Strongest predictor - shows relative inactivity |
        | **Recency** | Days since last order | Direct measure of current engagement |
        | **Rating Engagement** | How actively user rates shops/couriers | Engaged users rate more |
        | **Shop Rating** | Average rating given to shops | Low ratings = dissatisfaction |
        | **Tenure** | How long user has been a customer | Longer tenure = more loyal |
        | **Order Interval CV** | Variability in order timing | Inconsistent = disengaged |
        """)
    else:
        st.markdown("""
        | ویژگی | توضیحات | چرا مهم است |
        |-------|---------|-------------|
        | **نسبت رسنسی/عمر** | روز از آخرین سفارش ÷ عمر مشتری | قوی‌ترین پیش‌بین - غیرفعالی نسبی |
        | **رسنسی** | روز از آخرین سفارش | اندازه‌گیری مستقیم تعامل |
        | **تعامل امتیازدهی** | میزان فعالیت در امتیازدهی | کاربران فعال بیشتر امتیاز می‌دهند |
        | **امتیاز فروشگاه** | میانگین امتیاز داده‌شده | امتیاز کم = نارضایتی |
        | **عمر** | مدت زمان مشتری بودن | عمر بیشتر = وفاداری بیشتر |
        """)

st.markdown("---")


# =============================================================================
# Row 3: Segment Analysis
# =============================================================================

st.subheader(t.segment_analysis)

col1, col2 = st.columns(2)

with col1:
    # Churn rate by segment
    seg_df = pd.DataFrame([
        {
            "Segment": seg,
            "Churn Rate": info["churn_rate"] * 100,
            "Threshold": info["threshold_days"],
            "Users %": info["weight"] * 100
        }
        for seg, info in SEGMENT_INFO.items()
    ])
    
    fig_churn = go.Figure()
    
    fig_churn.add_trace(go.Bar(
        x=seg_df['Segment'],
        y=seg_df['Churn Rate'],
        marker_color=[SEGMENT_COLORS[s] for s in seg_df['Segment']],
        text=[f"{r:.1f}%" for r in seg_df['Churn Rate']],
        textposition='outside',
        name='Churn Rate'
    ))
    
    # Add threshold line
    fig_churn.add_trace(go.Scatter(
        x=seg_df['Segment'],
        y=seg_df['Threshold'],
        mode='lines+markers',
        name='Threshold (days)',
        yaxis='y2',
        line=dict(color=COLORS['dark'], dash='dash'),
        marker=dict(size=10)
    ))
    
    fig_churn.update_layout(
        height=400,
        yaxis_title="Churn Rate (%)" if lang == "en" else "نرخ ریزش (%)",
        yaxis2=dict(
            title="Threshold (days)" if lang == "en" else "آستانه (روز)",
            overlaying='y',
            side='right',
            range=[0, 60]
        ),
        yaxis=dict(range=[0, 85]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, b=40)
    )
    
    st.plotly_chart(fig_churn, use_container_width=True)

with col2:
    # User distribution by segment
    fig_dist = go.Figure(data=[go.Pie(
        labels=seg_df['Segment'],
        values=seg_df['Users %'],
        hole=0.4,
        marker_colors=[SEGMENT_COLORS[s] for s in seg_df['Segment']],
        textinfo='label+percent',
        textposition='outside',
        pull=[0.05, 0, 0, 0, 0]
    )])
    
    fig_dist.update_layout(
        height=400,
        title="User Distribution by Segment" if lang == "en" else "توزیع کاربران بر اساس سگمنت",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
        margin=dict(t=60, b=40)
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")


# =============================================================================
# Row 4: Model Performance
# =============================================================================

st.subheader(t.model_performance)

col1, col2 = st.columns(2)

with col1:
    # Model comparison chart
    models_data = [
        {"Model": "XGBoost\n(1-Order)", "Recall": 0.7484, "Type": "1-Order"},
        {"Model": "MLP\n(2+ Orders)", "Recall": 0.5901, "Type": "Neural"},
        {"Model": "TabNet\n(2+ Orders)", "Recall": 0.5387, "Type": "Neural"},
        {"Model": "FT-Trans\n(2+ Orders)", "Recall": 0.6482, "Type": "Neural"},
    ]
    
    models_df = pd.DataFrame(models_data)
    
    colors = [
        COLORS['danger'],  # XGBoost
        COLORS['primary'],  # MLP
        COLORS['success'],  # TabNet
        COLORS['purple'],   # FT-Trans
    ]
    
    fig_models = go.Figure(data=[go.Bar(
        x=models_df['Model'],
        y=models_df['Recall'],
        marker_color=colors,
        text=[f"{r:.1%}" for r in models_df['Recall']],
        textposition='outside'
    )])
    
    fig_models.update_layout(
        height=350,
        title="Weighted Recall Comparison" if lang == "en" else "مقایسه Weighted Recall",
        yaxis_title="Weighted Recall",
        yaxis=dict(range=[0, 0.9]),
        showlegend=False,
        margin=dict(t=60, b=40)
    )
    
    st.plotly_chart(fig_models, use_container_width=True)

with col2:
    # Per-segment recall for FT-Transformer
    ft_recalls = MODEL_METRICS['ft_transformer']['segment_recall']
    
    fig_seg_recall = go.Figure(data=[go.Bar(
        x=list(ft_recalls.keys()),
        y=list(ft_recalls.values()),
        marker_color=[SEGMENT_COLORS.get(s, COLORS['primary']) for s in ft_recalls.keys()],
        text=[f"{r:.1%}" for r in ft_recalls.values()],
        textposition='outside'
    )])
    
    fig_seg_recall.update_layout(
        height=350,
        title="FT-Transformer Recall by Segment" if lang == "en" else "Recall مدل FT-Transformer به تفکیک سگمنت",
        yaxis_title="Recall",
        yaxis=dict(range=[0, 1.1]),
        showlegend=False,
        margin=dict(t=60, b=40)
    )
    
    st.plotly_chart(fig_seg_recall, use_container_width=True)

# Model metrics table
st.markdown("#### 📋 " + ("Detailed Metrics" if lang == "en" else "متریک‌های تفصیلی"))

metrics_table = pd.DataFrame([
    {
        "Model": "XGBoost (1-Order)",
        "Recall": "74.8%",
        "Precision": "92.3%",
        "ROC-AUC": "0.65"
    },
    {
        "Model": "FT-Transformer (2+ Orders)",
        "Recall": "90.3%",
        "Weighted Recall": "64.8%",
        "ROC-AUC": "0.76"
    },
    {
        "Model": "Combined (Final)",
        "Overall Recall": "81.8%",
        "F1": "0.73",
        "ROC-AUC": "0.63"
    }
])

st.dataframe(metrics_table, use_container_width=True, hide_index=True)

st.markdown("---")


# =============================================================================
# Row 5: Recommendations
# =============================================================================

st.subheader("💡 " + ("Actionable Recommendations" if lang == "en" else "پیشنهادات عملی"))

col1, col2 = st.columns(2)

with col1:
    if lang == "en":
        st.markdown("""
        ### 🎯 For High-Risk Users
        
        1. **Immediate Outreach** - Send personalized retention emails
        2. **Special Offers** - Provide exclusive discounts
        3. **Customer Success** - Personal call for high-value users
        4. **Feedback Survey** - Understand their concerns
        
        ### 📊 For Medium-Risk Users
        
        1. **Re-engagement Campaign** - Remind of new products
        2. **Loyalty Points** - Incentivize next purchase
        3. **Personalized Recommendations** - Based on history
        """)
    else:
        st.markdown("""
        ### 🎯 برای کاربران پرریسک
        
        ۱. **تماس فوری** - ارسال ایمیل شخصی‌سازی‌شده
        ۲. **پیشنهاد ویژه** - تخفیف اختصاصی
        ۳. **موفقیت مشتری** - تماس شخصی برای کاربران با ارزش
        ۴. **نظرسنجی** - درک نگرانی‌های آن‌ها
        
        ### 📊 برای کاربران با ریسک متوسط
        
        ۱. **کمپین بازگشت** - یادآوری محصولات جدید
        ۲. **امتیاز وفاداری** - تشویق خرید بعدی
        ۳. **پیشنهاد شخصی‌سازی‌شده** - بر اساس سابقه
        """)

with col2:
    if lang == "en":
        st.markdown("""
        ### 🔧 System Improvements
        
        Based on analysis, focus on:
        
        1. **First Purchase Experience** - 74.4% of 1-order users churn
        2. **VIP Retention** - Focus on 30+ order users (5% of users, lowest churn)
        3. **Re-engagement at Threshold** - Trigger before segment threshold
        4. **Rating Follow-up** - Low ratings predict churn
        
        ### 📈 KPIs to Track
        
        - Monthly Active Users (MAU)
        - Average Order Frequency
        - Segment Migration Rate
        - Churn Rate by Segment
        """)
    else:
        st.markdown("""
        ### 🔧 بهبودهای سیستمی
        
        بر اساس تحلیل، تمرکز بر:
        
        ۱. **تجربه خرید اول** - ۷۴.۴٪ کاربران ۱ سفارش ریزش می‌کنند
        ۲. **حفظ VIP** - تمرکز بر کاربران ۳۰+ سفارش
        ۳. **بازگشت قبل از آستانه** - فعال‌سازی قبل از رسیدن به آستانه
        ۴. **پیگیری امتیازات** - امتیازات پایین پیش‌بین ریزش است
        
        ### 📈 KPIهای قابل پیگیری
        
        - کاربران فعال ماهانه (MAU)
        - میانگین فراوانی سفارش
        - نرخ مهاجرت سگمنت
        - نرخ ریزش به تفکیک سگمنت
        """)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 📊 " + ("Quick Stats" if lang == "en" else "آمار سریع"))
    
    st.metric("Total Users", "375,998")
    st.metric("Overall Churn", "54.7%")
    st.metric("VIP Churn", "12.5%")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ " + ("About" if lang == "en" else "درباره"))
    
    if lang == "en":
        st.markdown("""
        This page provides insights into:
        
        - **Feature Importance** - What drives churn
        - **Segment Analysis** - Churn patterns by segment
        - **Model Performance** - Accuracy metrics
        - **Recommendations** - Actionable steps
        """)
    else:
        st.markdown("""
        این صفحه اطلاعاتی درباره:
        
        - **اهمیت ویژگی‌ها** - چه چیزی ریزش را تعیین می‌کند
        - **تحلیل سگمنت** - الگوهای ریزش
        - **عملکرد مدل** - متریک‌های دقت
        - **پیشنهادات** - اقدامات عملی
        """)
