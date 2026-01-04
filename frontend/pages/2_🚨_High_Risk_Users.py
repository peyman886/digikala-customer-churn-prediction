"""
High Risk Users Page

View and manage users at risk of churning.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_translations, SEGMENT_INFO
from utils import (
    api_client, custom_css, init_session_state, get_language,
    COLORS, SEGMENT_COLORS, get_risk_emoji, format_number
)


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="High Risk Users",
    page_icon="🚨",
    layout="wide"
)

custom_css()
init_session_state("language", "en")
init_session_state("hr_page", 1)

lang = get_language()
t = get_translations(lang)


# =============================================================================
# Main Content
# =============================================================================

st.title(t.high_risk_title)
st.markdown(t.high_risk_desc)

st.markdown("---")

# =============================================================================
# Filters
# =============================================================================

col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 1, 1])

with col1:
    risk_filter = st.selectbox(
        t.filter_risk,
        options=["All", "HIGH", "MEDIUM", "LOW"],
        index=1,
        format_func=lambda x: {
            "All": "All" if lang == "en" else "همه",
            "HIGH": "🔴 High" if lang == "en" else "🔴 بالا",
            "MEDIUM": "🟡 Medium" if lang == "en" else "🟡 متوسط",
            "LOW": "🟢 Low" if lang == "en" else "🟢 پایین"
        }.get(x, x)
    )

with col2:
    segment_filter = st.selectbox(
        t.filter_segment,
        options=["All"] + list(SEGMENT_INFO.keys()),
        index=0
    )

with col3:
    min_days = st.number_input(
        t.filter_days_inactive,
        min_value=0,
        max_value=180,
        value=0,
        step=10
    )

with col4:
    rows_per_page = st.selectbox(
        t.rows_per_page,
        options=[25, 50, 100],
        index=1
    )

with col5:
    st.write("")
    st.write("")
    if st.button(t.apply_filter, use_container_width=True):
        st.session_state.hr_page = 1

st.markdown("---")


# =============================================================================
# Fetch Data
# =============================================================================

with st.spinner(t.loading):
    response = api_client.get_users_at_risk(
        risk_level=risk_filter if risk_filter != "All" else None,
        segment=segment_filter if segment_filter != "All" else None,
        min_days_inactive=min_days if min_days > 0 else None,
        limit=rows_per_page,
        page=st.session_state.hr_page
    )

if not response.success:
    st.error(f"❌ {t.api_error}")
    st.code("docker-compose up -d api", language="bash")
else:
    data = response.data
    
    # =========================================================================
    # Summary Stats
    # =========================================================================
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📊 " + ("Total Matching" if lang == "en" else "کل منطبق"),
            format_number(data['total'])
        )
    
    with col2:
        total_pages = max(1, (data['total'] + rows_per_page - 1) // rows_per_page)
        st.metric(
            "📄 " + ("Page" if lang == "en" else "صفحه"),
            f"{data['page']} / {total_pages}"
        )
    
    with col3:
        st.metric(
            "📋 " + ("Showing" if lang == "en" else "نمایش"),
            f"{len(data['users'])} " + t.users
        )
    
    st.markdown("---")
    
    # =========================================================================
    # Data Table
    # =========================================================================
    
    if data['users']:
        # Convert to DataFrame
        df = pd.DataFrame(data['users'])
        
        # Add risk emoji
        def add_risk_emoji(level):
            return f"{get_risk_emoji(level)} {level}"
        
        df['risk_level'] = df['risk_level'].apply(add_risk_emoji)
        
        # Format probability
        df['probability'] = df['probability'].apply(lambda x: f"{x*100:.1f}%")
        
        # Rename columns based on language
        if lang == "en":
            df = df.rename(columns={
                'user_id': 'User ID',
                'probability': 'Churn Probability',
                'risk_level': 'Risk Level',
                'recency': 'Days Inactive',
                'total_orders': 'Total Orders'
            })
        else:
            df = df.rename(columns={
                'user_id': 'شناسه کاربر',
                'probability': 'احتمال ریزش',
                'risk_level': 'سطح ریسک',
                'recency': 'روز غیرفعال',
                'total_orders': 'کل سفارشات'
            })
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # =====================================================================
        # Pagination
        # =====================================================================
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ " + ("First" if lang == "en" else "اول"), 
                        disabled=st.session_state.hr_page == 1,
                        use_container_width=True):
                st.session_state.hr_page = 1
                st.rerun()
        
        with col2:
            if st.button("◀️ " + ("Prev" if lang == "en" else "قبلی"),
                        disabled=st.session_state.hr_page == 1,
                        use_container_width=True):
                st.session_state.hr_page -= 1
                st.rerun()
        
        with col3:
            st.markdown(
                f"<div style='text-align: center; padding: 10px; color: #64748b;'>"
                f"{'Page' if lang == 'en' else 'صفحه'} {st.session_state.hr_page} / {total_pages}"
                f"</div>",
                unsafe_allow_html=True
            )
        
        with col4:
            if st.button(("Next" if lang == "en" else "بعدی") + " ▶️",
                        disabled=st.session_state.hr_page >= total_pages,
                        use_container_width=True):
                st.session_state.hr_page += 1
                st.rerun()
        
        with col5:
            if st.button(("Last" if lang == "en" else "آخر") + " ⏭️",
                        disabled=st.session_state.hr_page >= total_pages,
                        use_container_width=True):
                st.session_state.hr_page = total_pages
                st.rerun()
        
        # =====================================================================
        # Export
        # =====================================================================
        
        st.markdown("---")
        st.subheader("📥 " + ("Export Data" if lang == "en" else "خروجی داده"))
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = pd.DataFrame(data['users']).to_csv(index=False)
            st.download_button(
                label="📄 " + ("Download CSV" if lang == "en" else "دانلود CSV"),
                data=csv,
                file_name=f"high_risk_users_page{st.session_state.hr_page}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.info(
                f"💡 " + (f"Total {data['total']:,} users match current filters" 
                         if lang == "en" 
                         else f"در مجموع {data['total']:,} کاربر با فیلترها منطبق است")
            )
    
    else:
        st.info("📭 " + ("No users found matching the criteria" 
                        if lang == "en" 
                        else "کاربری با این معیارها یافت نشد"))


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 🎯 " + ("Quick Filters" if lang == "en" else "فیلترهای سریع"))
    
    if st.button("🔴 " + ("High Risk Only" if lang == "en" else "فقط ریسک بالا"), 
                use_container_width=True):
        st.session_state.hr_page = 1
        st.rerun()
    
    if st.button("🟡 " + ("Medium Risk" if lang == "en" else "ریسک متوسط"), 
                use_container_width=True):
        st.session_state.hr_page = 1
        st.rerun()
    
    if st.button("📅 " + ("Inactive > 30 Days" if lang == "en" else "غیرفعال > ۳۰ روز"), 
                use_container_width=True):
        st.session_state.hr_page = 1
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ " + ("About" if lang == "en" else "درباره"))
    
    if lang == "en":
        st.markdown("""
        This page shows users sorted by churn probability.
        
        **Risk Levels:**
        - 🔴 **High**: > 70%
        - 🟡 **Medium**: 40-70%
        - 🟢 **Low**: < 40%
        
        **Filters:**
        - Risk level
        - User segment
        - Days inactive
        
        **Export:**
        Download filtered results as CSV for marketing campaigns.
        """)
    else:
        st.markdown("""
        این صفحه کاربران را بر اساس احتمال ریزش نشان می‌دهد.
        
        **سطوح ریسک:**
        - 🔴 **بالا**: بیش از ۷۰٪
        - 🟡 **متوسط**: ۴۰ تا ۷۰٪
        - 🟢 **پایین**: کمتر از ۴۰٪
        
        **فیلترها:**
        - سطح ریسک
        - سگمنت کاربر
        - روز غیرفعال
        
        **خروجی:**
        دانلود نتایج فیلترشده برای کمپین‌های بازاریابی.
        """)
