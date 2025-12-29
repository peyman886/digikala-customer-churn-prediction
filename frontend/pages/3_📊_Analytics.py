"""
Page 4: Analytics & Insights

Explore churn factors and model insights.
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# Helper Functions
# =============================================================================

def fetch_feature_importance():
    """Fetch feature importance from API."""
    try:
        r = requests.get(f"{API_URL}/api/stats/feature-importance", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


def fetch_overview():
    """Fetch overview stats."""
    try:
        r = requests.get(f"{API_URL}/api/stats/overview", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


def fetch_risk_distribution():
    """Fetch risk distribution."""
    try:
        r = requests.get(f"{API_URL}/api/stats/risk-distribution", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


# =============================================================================
# Main Page
# =============================================================================

st.title("📊 Analytics & Insights")
st.markdown("Understand what drives customer churn.")

st.markdown("---")

# Fetch data
features = fetch_feature_importance()
overview = fetch_overview()
distribution = fetch_risk_distribution()

if not features:
    st.error("❌ Cannot load analytics data. Please check if the API is running.")
    st.stop()

# =============================================================================
# Row 1: Key Insights Cards
# =============================================================================

st.subheader("🔑 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h2 style='margin: 0; font-size: 2em;'>67%</h2>
        <p style='margin: 5px 0 0 0;'>of churned users were inactive >60 days</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h2 style='margin: 0; font-size: 2em;'>4x</h2>
        <p style='margin: 5px 0 0 0;'>higher churn for users with <3 orders</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    '>
        <h2 style='margin: 0; font-size: 2em;'>2.3x</h2>
        <p style='margin: 5px 0 0 0;'>more likely to churn with late deliveries</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Row 2: Feature Importance Chart
# =============================================================================

st.subheader("🎯 Top Churn Factors (Feature Importance)")

# Prepare data for chart
df_features = pd.DataFrame(features)

# Create horizontal bar chart
fig_importance = go.Figure()

# Color gradient based on importance
max_imp = df_features['importance'].max()
colors = [f'rgba(99, 102, 241, {0.4 + 0.6 * (imp / max_imp)})' for imp in df_features['importance']]

fig_importance.add_trace(go.Bar(
    y=df_features['display_name'][::-1],  # Reverse for top-to-bottom
    x=df_features['importance'][::-1],
    orientation='h',
    marker_color=colors[::-1],
    text=[f"{x*100:.1f}%" for x in df_features['importance'][::-1]],
    textposition='outside'
))

fig_importance.update_layout(
    height=500,
    margin=dict(l=20, r=100, t=20, b=20),
    xaxis_title="Importance Score",
    yaxis_title="",
    showlegend=False,
    xaxis=dict(range=[0, max_imp * 1.2])
)

st.plotly_chart(fig_importance, use_container_width=True)

# Feature interpretation
with st.expander("📖 Feature Interpretation Guide"):
    st.markdown("""
    | Feature | Description | Why It Matters |
    |---------|-------------|----------------|
    | **Days Since Last Order** | How recently did the user order? | Strongest predictor - inactive users are likely churning |
    | **Orders Last 30 Days** | Recent purchase frequency | Active users rarely churn |
    | **Total Orders** | Lifetime purchase count | Loyal customers have more orders |
    | **Avg Order Gap** | Average days between orders | High gap = low engagement |
    | **On-Time Ratio** | Delivery performance | Late deliveries frustrate customers |
    | **Complaints** | Number of CRM tickets | Unhappy customers complain more |
    | **Ratings** | Shop/courier ratings given | Low ratings = dissatisfaction |
    """)

st.markdown("---")

# =============================================================================
# Row 3: Distribution Analysis
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Churn Probability Distribution")
    
    if distribution:
        fig_dist = go.Figure()
        
        # Color bars based on risk threshold
        colors = []
        for b in distribution['bins']:
            if b < 0.4:
                colors.append('#00cc96')
            elif b < 0.7:
                colors.append('#ffa600')
            else:
                colors.append('#ff4b4b')
        
        fig_dist.add_trace(go.Bar(
            x=distribution['bins'],
            y=distribution['counts'],
            marker_color=colors
        ))
        
        # Add threshold lines
        fig_dist.add_vline(x=0.4, line_dash="dash", line_color="orange", 
                         annotation_text="Medium Risk Threshold")
        fig_dist.add_vline(x=0.7, line_dash="dash", line_color="red",
                         annotation_text="High Risk Threshold")
        
        fig_dist.update_layout(
            height=400,
            xaxis_title="Churn Probability",
            yaxis_title="Number of Users",
            margin=dict(l=20, r=20, t=20, b=40)
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)

with col2:
    st.subheader("🥧 Risk Level Breakdown")
    
    if overview:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            values=[overview['low_risk'], overview['medium_risk'], overview['high_risk']],
            hole=0.4,
            marker_colors=['#00cc96', '#ffa600', '#ff4b4b'],
            textinfo='label+percent',
            textposition='outside',
            pull=[0, 0, 0.1]  # Pull out high risk slice
        )])
        
        fig_pie.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=True
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# =============================================================================
# Row 4: Actionable Insights
# =============================================================================

st.subheader("💡 Actionable Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 For High-Risk Users
    
    1. **Immediate Outreach** - Send personalized retention emails
    2. **Special Offers** - Provide exclusive discounts
    3. **Customer Success Call** - Personal touch for high-value users
    4. **Feedback Survey** - Understand their concerns
    
    ### 📊 For Medium-Risk Users
    
    1. **Re-engagement Campaign** - Remind them of new products
    2. **Loyalty Points** - Incentivize next purchase
    3. **Personalized Recommendations** - Based on past behavior
    """)

with col2:
    st.markdown("""
    ### 🔧 System Improvements
    
    Based on feature importance, focus on:
    
    1. **Reduce Delivery Delays** - Late deliveries increase churn by 2.3x
    2. **Improve First Purchase Experience** - Users with <3 orders churn 4x more
    3. **Re-engagement at 30 Days** - Trigger campaign when user hits 30 days inactive
    4. **Complaint Resolution** - Fast resolution reduces churn significantly
    
    ### 📈 KPIs to Track
    
    - Monthly Active Users (MAU)
    - Average Order Frequency
    - Net Promoter Score (NPS)
    - Churn Rate Trend
    """)

st.markdown("---")

# =============================================================================
# Model Performance
# =============================================================================

st.subheader("🤖 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ROC-AUC", "0.879", "Excellent")
    
with col2:
    st.metric("F1-Score", "0.849", "Good")
    
with col3:
    st.metric("Precision", "0.85", "")
    
with col4:
    st.metric("Recall", "0.84", "")

st.info("""
**Model Details:**
- Algorithm: XGBoost Classifier
- Features: 26 user-level features
- Training Data: 338,101 users
- Churn Definition: No order in 30 days following observation date
""")

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    
    if overview:
        st.metric("Total Users", f"{overview['total_users']:,}")
        st.metric("Avg Churn Prob", f"{overview['avg_churn_probability']*100:.1f}%")
        st.metric("High Risk", f"{overview['high_risk']:,}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About Analytics")
    st.markdown("""
    This page provides insights into:
    
    - **Feature Importance** - What factors most influence churn prediction
    - **Risk Distribution** - How users are distributed across risk levels
    - **Actionable Insights** - Recommendations for reducing churn
    
    The model uses XGBoost with 26 engineered features from orders, CRM, and comments data.
    """)
