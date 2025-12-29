"""
Churn Prediction Dashboard - Main Page (Dashboard)

Streamlit multi-page application for customer churn prediction.

Author: Peyman
"""

import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .risk-high { color: #ff4b4b; }
    .risk-medium { color: #ffa600; }
    .risk-low { color: #00cc96; }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Helper Functions
# =============================================================================

def fetch_overview():
    """Fetch overview stats from API."""
    try:
        r = requests.get(f"{API_URL}/api/stats/overview", timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error: {r.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the backend is running.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def fetch_risk_distribution():
    """Fetch risk distribution for histogram."""
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

def main():
    # Header
    st.markdown('<h1 class="main-header">🔮 Customer Churn Prediction Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fetch data
    data = fetch_overview()
    
    if data is None:
        st.warning("⚠️ Unable to load data. Please check if the API is running.")
        st.code("docker-compose up -d api", language="bash")
        return
    
    # ==========================================================================
    # Row 1: Key Metrics
    # ==========================================================================
    st.subheader("📊 Overview Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="👥 Total Users",
            value=f"{data['total_users']:,}"
        )
    
    with col2:
        st.metric(
            label="🟢 Low Risk",
            value=f"{data['low_risk']:,}",
            delta=f"{data['low_risk_pct']}%"
        )
    
    with col3:
        st.metric(
            label="🟡 Medium Risk",
            value=f"{data['medium_risk']:,}",
            delta=f"{data['medium_risk_pct']}%"
        )
    
    with col4:
        st.metric(
            label="🔴 High Risk",
            value=f"{data['high_risk']:,}",
            delta=f"{data['high_risk_pct']}%",
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # ==========================================================================
    # Row 2: Charts
    # ==========================================================================
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Risk Distribution")
        
        # Pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            values=[data['low_risk'], data['medium_risk'], data['high_risk']],
            hole=0.4,
            marker_colors=['#00cc96', '#ffa600', '#ff4b4b'],
            textinfo='label+percent',
            textposition='outside'
        )])
        fig_pie.update_layout(
            showlegend=True,
            height=400,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        st.subheader("📊 Probability Distribution")
        
        # Histogram
        dist_data = fetch_risk_distribution()
        if dist_data:
            fig_hist = go.Figure(data=[go.Bar(
                x=dist_data['bins'],
                y=dist_data['counts'],
                marker_color='#1f77b4'
            )])
            fig_hist.update_layout(
                xaxis_title="Churn Probability",
                yaxis_title="Number of Users",
                height=400,
                margin=dict(t=20, b=40, l=40, r=20)
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Distribution data not available")
    
    st.markdown("---")
    
    # ==========================================================================
    # Row 3: Quick Actions
    # ==========================================================================
    st.subheader("🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Check User")
        st.write("Look up churn risk for a specific user")
        if st.button("→ Go to User Prediction", key="btn_predict"):
            st.switch_page("pages/1_👤_User_Prediction.py")
    
    with col2:
        st.markdown("### 🚨 High Risk Users")
        st.write(f"View all {data['high_risk']:,} high-risk users")
        if st.button("→ View High Risk List", key="btn_highrisk"):
            st.switch_page("pages/2_🚨_High_Risk_Users.py")
    
    with col3:
        st.markdown("### 📊 Analytics")
        st.write("Explore churn factors and insights")
        if st.button("→ View Analytics", key="btn_analytics"):
            st.switch_page("pages/3_📊_Analytics.py")
    
    # ==========================================================================
    # Footer
    # ==========================================================================
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888;'>
            <p>Churn Prediction System v2.0 | Built with FastAPI + Streamlit</p>
            <p>Model: XGBoost | ROC-AUC: 0.879 | F1: 0.849</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
