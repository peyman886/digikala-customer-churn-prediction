"""
Page 3: High Risk Users List

View and filter users at risk of churning.
"""

import streamlit as st
import requests
import pandas as pd

import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="High Risk Users",
    page_icon="🚨",
    layout="wide"
)

# =============================================================================
# Helper Functions
# =============================================================================

def fetch_users_at_risk(risk_level=None, min_days=None, limit=50, page=1):
    """Fetch users at risk from API."""
    try:
        params = {"limit": limit, "page": page}
        if risk_level and risk_level != "All":
            params["risk_level"] = risk_level
        if min_days and min_days > 0:
            params["min_days_inactive"] = min_days
        
        r = requests.get(f"{API_URL}/api/users/at-risk", params=params, timeout=15)
        if r.status_code == 200:
            return r.json(), None
        else:
            return None, f"API Error: {r.status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API"
    except Exception as e:
        return None, str(e)


def get_risk_badge(risk_level: str) -> str:
    """Get HTML badge for risk level."""
    colors = {
        "LOW": ("#00cc96", "Low"),
        "MEDIUM": ("#ffa600", "Medium"),
        "HIGH": ("#ff4b4b", "High")
    }
    color, label = colors.get(risk_level, ("#888", risk_level))
    return f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">{label}</span>'


# =============================================================================
# Main Page
# =============================================================================

st.title("🚨 High Risk Users")
st.markdown("View and manage users at risk of churning.")

st.markdown("---")

# =============================================================================
# Filters
# =============================================================================

col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    risk_filter = st.selectbox(
        "Risk Level",
        options=["All", "HIGH", "MEDIUM", "LOW"],
        index=1  # Default to HIGH
    )

with col2:
    min_days_inactive = st.number_input(
        "Min Days Inactive",
        min_value=0,
        max_value=180,
        value=0,
        step=10,
        help="Filter users inactive for at least X days"
    )

with col3:
    rows_per_page = st.selectbox(
        "Rows per page",
        options=[25, 50, 100, 200],
        index=1
    )

with col4:
    st.write("")
    st.write("")
    apply_filter = st.button("🔍 Apply", use_container_width=True)

# Initialize session state for pagination
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

# Reset page when filters change
if apply_filter:
    st.session_state.current_page = 1

st.markdown("---")

# =============================================================================
# Fetch and Display Data
# =============================================================================

# Fetch data
with st.spinner("Loading users..."):
    data, error = fetch_users_at_risk(
        risk_level=risk_filter if risk_filter != "All" else None,
        min_days=min_days_inactive if min_days_inactive > 0 else None,
        limit=rows_per_page,
        page=st.session_state.current_page
    )

if error:
    st.error(f"❌ {error}")
elif data:
    # Summary stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Matching Users", f"{data['total']:,}")
    
    with col2:
        st.metric("📄 Current Page", f"{data['page']} of {max(1, (data['total'] + rows_per_page - 1) // rows_per_page)}")
    
    with col3:
        st.metric("📋 Showing", f"{len(data['users'])} users")
    
    st.markdown("---")
    
    # ==========================================================================
    # Data Table
    # ==========================================================================
    
    if data['users']:
        # Convert to DataFrame
        df = pd.DataFrame(data['users'])
        
        # Rename columns for display
        df = df.rename(columns={
            'user_id': 'User ID',
            'probability': 'Churn Probability',
            'risk_level': 'Risk Level',
            'days_since_last_order': 'Days Inactive',
            'total_orders': 'Total Orders'
        })
        
        # Format probability as percentage
        df['Churn Probability'] = df['Churn Probability'].apply(lambda x: f"{x*100:.1f}%")
        
        # Add risk emoji
        def add_risk_emoji(level):
            emojis = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
            return f"{emojis.get(level, '')} {level}"
        
        df['Risk Level'] = df['Risk Level'].apply(add_risk_emoji)
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "User ID": st.column_config.TextColumn("User ID", width="medium"),
                "Churn Probability": st.column_config.TextColumn("Probability", width="small"),
                "Risk Level": st.column_config.TextColumn("Risk", width="small"),
                "Days Inactive": st.column_config.NumberColumn("Days Inactive", width="small"),
                "Total Orders": st.column_config.NumberColumn("Orders", width="small"),
            }
        )
        
        # ==========================================================================
        # Pagination
        # ==========================================================================
        
        total_pages = max(1, (data['total'] + rows_per_page - 1) // rows_per_page)
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First", disabled=st.session_state.current_page == 1):
                st.session_state.current_page = 1
                st.rerun()
        
        with col2:
            if st.button("◀️ Prev", disabled=st.session_state.current_page == 1):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col3:
            st.markdown(
                f"<div style='text-align: center; padding: 10px;'>Page {st.session_state.current_page} of {total_pages}</div>",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button("Next ▶️", disabled=st.session_state.current_page >= total_pages):
                st.session_state.current_page += 1
                st.rerun()
        
        with col5:
            if st.button("Last ⏭️", disabled=st.session_state.current_page >= total_pages):
                st.session_state.current_page = total_pages
                st.rerun()
        
        # ==========================================================================
        # Export
        # ==========================================================================
        
        st.markdown("---")
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export current page as CSV
            csv = pd.DataFrame(data['users']).to_csv(index=False)
            st.download_button(
                label="📄 Download Current Page (CSV)",
                data=csv,
                file_name=f"high_risk_users_page{st.session_state.current_page}.csv",
                mime="text/csv"
            )
        
        with col2:
            st.info(f"💡 To export all {data['total']:,} users, use the API directly: `/api/export/high-risk`")
    
    else:
        st.info("No users found matching the criteria.")

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 🎯 Quick Filters")
    
    if st.button("🔴 Show High Risk Only", use_container_width=True):
        st.session_state.current_page = 1
        st.rerun()
    
    if st.button("🟡 Show Medium Risk", use_container_width=True):
        st.session_state.current_page = 1
        st.rerun()
    
    if st.button("📅 Inactive > 30 Days", use_container_width=True):
        st.session_state.current_page = 1
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This page shows users sorted by churn probability (highest first).
    
    **Risk Levels:**
    - 🔴 **High**: > 70% churn probability
    - 🟡 **Medium**: 40-70% probability
    - 🟢 **Low**: < 40% probability
    
    **Actions:**
    - Filter by risk level
    - Filter by days inactive
    - Export to CSV for campaigns
    """)
