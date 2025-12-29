"""
Page 2: Single User Prediction

Check churn risk for individual users.
"""

import streamlit as st
import requests
import plotly.graph_objects as go

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="User Prediction",
    page_icon="👤",
    layout="wide"
)

# =============================================================================
# Helper Functions
# =============================================================================

def predict_user(user_id: str):
    """Get prediction for a user."""
    try:
        r = requests.post(
            f"{API_URL}/api/predict",
            json={"user_id": user_id},
            timeout=10
        )
        if r.status_code == 200:
            return r.json(), None
        elif r.status_code == 404:
            return None, "User not found"
        else:
            return None, f"API Error: {r.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API"
    except Exception as e:
        return None, str(e)


def get_user_profile(user_id: str):
    """Get user profile details."""
    try:
        r = requests.get(f"{API_URL}/api/user/{user_id}/profile", timeout=10)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        "LOW": "#00cc96",
        "MEDIUM": "#ffa600",
        "HIGH": "#ff4b4b"
    }
    return colors.get(risk_level, "#888888")


def get_risk_emoji(risk_level: str) -> str:
    """Get emoji for risk level."""
    emojis = {
        "LOW": "🟢",
        "MEDIUM": "🟡",
        "HIGH": "🔴"
    }
    return emojis.get(risk_level, "⚪")


# =============================================================================
# Main Page
# =============================================================================

st.title("👤 User Churn Prediction")
st.markdown("Enter a user ID to check their churn risk and profile.")

st.markdown("---")

# Input form
col1, col2 = st.columns([3, 1])

with col1:
    user_id = st.text_input(
        "User ID",
        placeholder="Enter user ID (e.g., 1385028)",
        help="Enter the numeric user ID"
    )

with col2:
    st.write("")  # Spacing
    st.write("")
    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

# Sample user IDs
st.caption("💡 Sample user IDs: 1385028, 54227, 30492532")

st.markdown("---")

# Prediction result
if predict_btn and user_id:
    with st.spinner("Analyzing user..."):
        prediction, error = predict_user(user_id.strip())
    
    if error:
        st.error(f"❌ {error}")
    elif prediction:
        # Get profile
        profile = get_user_profile(user_id.strip())
        
        # =================================================================
        # Prediction Result Card
        # =================================================================
        risk_color = get_risk_color(prediction['risk_level'])
        risk_emoji = get_risk_emoji(prediction['risk_level'])
        
        st.subheader("🎯 Prediction Result")
        
        # Main result box
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="User ID",
                value=prediction['user_id']
            )
        
        with col2:
            churn_text = "Yes ⚠️" if prediction['will_churn'] else "No ✅"
            st.metric(
                label="Will Churn?",
                value=churn_text
            )
        
        with col3:
            st.metric(
                label="Churn Probability",
                value=f"{prediction['probability']*100:.1f}%"
            )
        
        # Risk level display
        st.markdown(
            f"""
            <div style='
                background: linear-gradient(90deg, {risk_color}22 0%, transparent 100%);
                border-left: 5px solid {risk_color};
                padding: 20px;
                border-radius: 5px;
                margin: 20px 0;
            '>
                <h2 style='margin: 0; color: {risk_color};'>
                    {risk_emoji} Risk Level: {prediction['risk_level']}
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction['probability'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 40], 'color': "#00cc9622"},
                    {'range': [40, 70], 'color': "#ffa60022"},
                    {'range': [70, 100], 'color': "#ff4b4b22"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction['probability'] * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # =================================================================
        # User Profile
        # =================================================================
        if profile:
            st.subheader("📋 User Profile")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📦 Total Orders", profile['total_orders'])
                st.metric("📅 Days Since Last Order", profile['days_since_last_order'])
            
            with col2:
                st.metric("🕐 Customer Tenure", f"{profile['customer_tenure_days']} days")
                st.metric("📆 Orders (Last 30d)", profile['orders_last_30d'])
            
            with col3:
                st.metric("✅ On-Time Delivery", f"{profile['on_time_ratio']*100:.0f}%")
                st.metric("⭐ Avg Shop Rating", f"{profile['avg_shop_rating']:.1f}")
            
            with col4:
                st.metric("📢 Complaints", profile['total_complaints'])
                st.metric("💬 Comments", profile['comment_count'])
            
            st.markdown("---")
            
            # =================================================================
            # Recommended Action
            # =================================================================
            st.subheader("💡 Recommended Action")
            
            if prediction['risk_level'] == "HIGH":
                st.error("""
                🚨 **HIGH RISK - Immediate Action Required**
                
                - Send personalized retention offer
                - Assign to customer success team
                - Schedule follow-up call
                - Offer special discount or loyalty reward
                """)
            elif prediction['risk_level'] == "MEDIUM":
                st.warning("""
                ⚠️ **MEDIUM RISK - Monitor Closely**
                
                - Send engagement email campaign
                - Offer small incentive for next purchase
                - Monitor activity in next 2 weeks
                """)
            else:
                st.success("""
                ✅ **LOW RISK - No Immediate Action Needed**
                
                - Continue standard engagement
                - Include in loyalty program communications
                - Monitor for any changes
                """)

elif predict_btn and not user_id:
    st.warning("Please enter a user ID")

# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This page predicts the likelihood of a customer churning 
    (not making a purchase in the next 30 days).
    
    **Risk Levels:**
    - 🟢 **Low**: < 40% probability
    - 🟡 **Medium**: 40-70% probability
    - 🔴 **High**: > 70% probability
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Algorithm:** XGBoost
    - **ROC-AUC:** 0.879
    - **F1-Score:** 0.849
    """)
