"""
Report Page

Display comprehensive churn analysis report.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_translations
from utils import custom_css, init_session_state, get_language, COLORS


# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Report",
    page_icon="📄",
    layout="wide"
)

custom_css()
init_session_state("language", "en")
lang = get_language()
t = get_translations(lang)


# =============================================================================
# Report Content
# =============================================================================

st.title(t.report_title)
st.markdown(t.report_desc)

st.markdown("---")

# Report selection
report_lang = st.radio(
    "📄 " + ("Select Report Language" if lang == "en" else "انتخاب زبان گزارش"),
    options=["fa", "en"],
    format_func=lambda x: "فارسی" if x == "fa" else "English",
    horizontal=True
)

st.markdown("---")

# =============================================================================
# Display Report
# =============================================================================

# Try to load report from reports directory
reports_dir = Path(__file__).parent.parent.parent / "reports"
report_file = reports_dir / ("Gozaresh.html" if report_lang == "fa" else "Report_EN.html")

# Fallback to embedded reports
if not report_file.exists():
    # Show embedded summary
    if report_lang == "fa":
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
        
        # 📊 گزارش پیش‌بینی ریزش مشتری
        
        ## ۱. خلاصه اجرایی
        
        این پروژه یک سیستم پیش‌بینی ریزش مشتری با رویکرد **سگمنت‌محور** توسعه داده است.
        
        ### یافته‌های کلیدی:
        - **۴۴.۹٪** کاربران فقط یک سفارش دارند
        - **۵ سگمنت** با آستانه‌های ریزش متفاوت (۱۴ تا ۴۵ روز)
        - **Recall کلی: ۸۱.۸٪** با ترکیب XGBoost و FT-Transformer
        
        ## ۲. تعریف ریزش
        
        | سگمنت | آستانه (روز) | نرخ ریزش |
        |--------|-------------|----------|
        | ۱ سفارش | ۴۵ | ۷۴.۴٪ |
        | ۲-۴ سفارش | ۳۹ | ۵۴.۴٪ |
        | ۵-۱۰ سفارش | ۳۵ | ۳۱.۶٪ |
        | ۱۱-۳۰ سفارش | ۱۷ | ۳۰.۲٪ |
        | ۳۰+ سفارش | ۱۴ | ۱۲.۵٪ |
        
        ## ۳. مدل‌سازی
        
        - **کاربران ۱ سفارش**: XGBoost (Recall: 74.8%)
        - **کاربران ۲+ سفارش**: FT-Transformer (Weighted Recall: 64.8%)
        
        ## ۴. مهم‌ترین ویژگی‌ها
        
        1. `recency_tenure_ratio` (21.5%)
        2. `recency` (21.1%)
        3. `rating_engagement` (17.5%)
        
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        # 📊 Customer Churn Prediction Report
        
        ## 1. Executive Summary
        
        This project developed a customer churn prediction system using a **segment-based approach**.
        
        ### Key Findings:
        - **44.9%** of users have only 1 order (one-time buyers)
        - **5 segments** with different churn thresholds (14-45 days)
        - **Overall Recall: 81.8%** with combined XGBoost + FT-Transformer
        
        ## 2. Churn Definition
        
        | Segment | Threshold (days) | Churn Rate |
        |---------|------------------|------------|
        | 1 Order | 45 | 74.4% |
        | 2-4 Orders | 39 | 54.4% |
        | 5-10 Orders | 35 | 31.6% |
        | 11-30 Orders | 17 | 30.2% |
        | 30+ Orders | 14 | 12.5% |
        
        ## 3. Modeling Strategy
        
        - **1-Order Users**: XGBoost (Recall: 74.8%)
        - **2+ Order Users**: FT-Transformer (Weighted Recall: 64.8%)
        
        ## 4. Top Features
        
        1. `recency_tenure_ratio` (21.5%)
        2. `recency` (21.1%)
        3. `rating_engagement` (17.5%)
        4. `last_order_rate_to_shop_filled` (6.1%)
        5. `delivered_orders` (4.4%)
        
        ## 5. Model Performance
        
        ### Combined Model:
        - **Overall Recall**: 81.8%
        - **Weighted Recall**: 65%
        - **F1 Score**: 0.73
        
        ### Per-Segment Performance (FT-Transformer):
        
        | Segment | Recall |
        |---------|--------|
        | 2-4 Orders | 99.6% |
        | 5-10 Orders | 88.7% |
        | 11-30 Orders | 68.8% |
        | 30+ Orders | 51.8% |
        
        ## 6. Recommendations
        
        1. **First Purchase Experience** - Focus on converting 1-order users
        2. **Segment-Specific Thresholds** - Use appropriate re-engagement timing
        3. **VIP Protection** - Prioritize high-value segment retention
        4. **Rating Follow-up** - Low ratings predict churn
        
        ## 7. Technical Stack
        
        - **Models**: XGBoost, FT-Transformer
        - **Features**: 98 engineered features
        - **Training**: Rolling window (60-day history, 30-day prediction)
        - **API**: FastAPI + Streamlit
        - **Deployment**: Docker Compose
        """)

else:
    # Load and display actual HTML report
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Display in iframe
        st.components.v1.html(html_content, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error loading report: {e}")


st.markdown("---")

# =============================================================================
# Download Section
# =============================================================================

st.subheader("📥 " + t.download_report)

col1, col2, col3 = st.columns(3)

with col1:
    # Download HTML report
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        st.download_button(
            label="📄 HTML Report",
            data=html_content,
            file_name=f"churn_report_{report_lang}.html",
            mime="text/html",
            use_container_width=True
        )
    else:
        st.info("HTML report not available")

with col2:
    # Download Markdown
    md_file = reports_dir / ("Gozaresh.md" if report_lang == "fa" else "Report_EN.md")
    if md_file.exists():
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        st.download_button(
            label="📝 Markdown Report",
            data=md_content,
            file_name=f"churn_report_{report_lang}.md",
            mime="text/markdown",
            use_container_width=True
        )
    else:
        st.info("Markdown report not available")

with col3:
    # Download ER Diagram
    er_file = reports_dir / "er_diagram.svg"
    if er_file.exists():
        with open(er_file, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        
        st.download_button(
            label="🗂️ ER Diagram (SVG)",
            data=svg_content,
            file_name="er_diagram.svg",
            mime="image/svg+xml",
            use_container_width=True
        )
    else:
        st.info("ER diagram not available")


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### 📋 " + ("Report Sections" if lang == "en" else "بخش‌های گزارش"))
    
    sections = [
        ("1. Executive Summary", "۱. خلاصه اجرایی"),
        ("2. Problem Definition", "۲. تعریف مسئله"),
        ("3. Dataset Overview", "۳. معرفی دیتاست"),
        ("4. EDA", "۴. تحلیل اکتشافی"),
        ("5. Business Analysis", "۵. تحلیل بیزینسی"),
        ("6. Segmentation", "۶. سگمنت‌بندی"),
        ("7. Churn Definition", "۷. تعریف ریزش"),
        ("8. Preprocessing", "۸. پیش‌پردازش"),
        ("9. Feature Engineering", "۹. مهندسی ویژگی"),
        ("10. Modeling", "۱۰. مدل‌سازی"),
        ("11. Results", "۱۱. نتایج"),
        ("12. Recommendations", "۱۲. پیشنهادات"),
    ]
    
    for en, fa in sections:
        st.markdown(f"- {fa if lang == 'fa' else en}")
    
    st.markdown("---")
    
    st.markdown("### ℹ️ " + ("About" if lang == "en" else "درباره"))
    
    if lang == "en":
        st.markdown("""
        This comprehensive report includes:
        
        - Methodology explanation
        - EDA visualizations
        - Model architecture
        - Performance metrics
        - Actionable recommendations
        
        **Report available in:**
        - 🇮🇷 Persian (فارسی)
        - 🇬🇧 English
        """)
    else:
        st.markdown("""
        این گزارش جامع شامل:
        
        - توضیح روش‌شناسی
        - نمودارهای EDA
        - معماری مدل
        - متریک‌های عملکرد
        - پیشنهادات عملی
        
        **زبان‌های موجود:**
        - 🇮🇷 فارسی
        - 🇬🇧 English
        """)
