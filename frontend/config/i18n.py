"""
Internationalization (i18n) Configuration

Provides bilingual support for Persian (fa) and English (en).
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Translations:
    """Store translations for both languages."""
    
    # App-wide
    app_title: str
    app_subtitle: str
    language_label: str
    
    # Navigation
    nav_home: str
    nav_prediction: str
    nav_high_risk: str
    nav_analytics: str
    nav_report: str
    
    # Dashboard
    overview_title: str
    total_users: str
    total_churned: str
    churn_rate: str
    avg_probability: str
    
    # Segments
    segment_1_order: str
    segment_2_4_orders: str
    segment_5_10_orders: str
    segment_11_30_orders: str
    segment_30_plus_orders: str
    
    # Risk levels
    risk_low: str
    risk_medium: str
    risk_high: str
    
    # User Prediction
    user_prediction_title: str
    user_prediction_desc: str
    enter_user_id: str
    predict_button: str
    sample_ids: str
    prediction_result: str
    will_churn: str
    churn_probability: str
    risk_level: str
    user_segment: str
    churn_threshold: str
    model_used: str
    
    # User Profile
    user_profile: str
    total_orders: str
    recency_days: str
    tenure_days: str
    otd_rate: str
    late_rate: str
    avg_shop_rating: str
    avg_courier_rating: str
    crm_requests: str
    comment_count: str
    
    # Recommendations
    recommendations: str
    high_risk_action: str
    medium_risk_action: str
    low_risk_action: str
    
    # High Risk Users
    high_risk_title: str
    high_risk_desc: str
    filter_risk: str
    filter_segment: str
    filter_days_inactive: str
    rows_per_page: str
    apply_filter: str
    export_csv: str
    
    # Analytics
    analytics_title: str
    analytics_desc: str
    key_insights: str
    feature_importance: str
    model_performance: str
    segment_analysis: str
    churn_by_segment: str
    
    # Report
    report_title: str
    report_desc: str
    download_report: str
    
    # Model info
    model_info: str
    model_xgboost: str
    model_ft_transformer: str
    model_combined: str
    overall_recall: str
    weighted_recall: str
    f1_score: str
    roc_auc: str
    
    # Common
    loading: str
    error: str
    success: str
    not_found: str
    user_not_found: str
    api_error: str
    yes: str
    no: str
    days: str
    users: str
    orders: str


# English translations
EN = Translations(
    # App-wide
    app_title="🔮 Customer Churn Prediction",
    app_subtitle="Predict and prevent customer churn with AI-powered insights",
    language_label="🌐 Language",
    
    # Navigation
    nav_home="Home",
    nav_prediction="User Prediction",
    nav_high_risk="High Risk Users",
    nav_analytics="Analytics",
    nav_report="Report",
    
    # Dashboard
    overview_title="📊 Overview Statistics",
    total_users="👥 Total Users",
    total_churned="📉 Total Churned",
    churn_rate="📊 Churn Rate",
    avg_probability="📈 Avg Probability",
    
    # Segments
    segment_1_order="1 Order",
    segment_2_4_orders="2-4 Orders",
    segment_5_10_orders="5-10 Orders",
    segment_11_30_orders="11-30 Orders",
    segment_30_plus_orders="30+ Orders",
    
    # Risk levels
    risk_low="🟢 Low Risk",
    risk_medium="🟡 Medium Risk",
    risk_high="🔴 High Risk",
    
    # User Prediction
    user_prediction_title="👤 User Churn Prediction",
    user_prediction_desc="Enter a user ID to check their churn risk and profile",
    enter_user_id="User ID",
    predict_button="🔍 Predict",
    sample_ids="💡 Sample user IDs",
    prediction_result="🎯 Prediction Result",
    will_churn="Will Churn?",
    churn_probability="Churn Probability",
    risk_level="Risk Level",
    user_segment="User Segment",
    churn_threshold="Churn Threshold",
    model_used="Model Used",
    
    # User Profile
    user_profile="📋 User Profile",
    total_orders="📦 Total Orders",
    recency_days="📅 Days Since Last Order",
    tenure_days="🕐 Customer Tenure",
    otd_rate="✅ On-Time Delivery",
    late_rate="⏰ Late Delivery Rate",
    avg_shop_rating="⭐ Avg Shop Rating",
    avg_courier_rating="🛵 Avg Courier Rating",
    crm_requests="📢 CRM Requests",
    comment_count="💬 Comments",
    
    # Recommendations
    recommendations="💡 Recommended Actions",
    high_risk_action="""
    🚨 **HIGH RISK - Immediate Action Required**
    
    - Send personalized retention offer
    - Assign to customer success team
    - Schedule follow-up call
    - Offer special discount or loyalty reward
    """,
    medium_risk_action="""
    ⚠️ **MEDIUM RISK - Monitor Closely**
    
    - Send engagement email campaign
    - Offer small incentive for next purchase
    - Monitor activity in next 2 weeks
    """,
    low_risk_action="""
    ✅ **LOW RISK - No Immediate Action Needed**
    
    - Continue standard engagement
    - Include in loyalty program communications
    - Monitor for any changes
    """,
    
    # High Risk Users
    high_risk_title="🚨 High Risk Users",
    high_risk_desc="View and manage users at risk of churning",
    filter_risk="Risk Level",
    filter_segment="Segment",
    filter_days_inactive="Min Days Inactive",
    rows_per_page="Rows per page",
    apply_filter="🔍 Apply",
    export_csv="📥 Export CSV",
    
    # Analytics
    analytics_title="📊 Analytics & Insights",
    analytics_desc="Understand what drives customer churn",
    key_insights="🔑 Key Insights",
    feature_importance="🎯 Feature Importance",
    model_performance="🤖 Model Performance",
    segment_analysis="📈 Segment Analysis",
    churn_by_segment="Churn Rate by Segment",
    
    # Report
    report_title="📄 Churn Analysis Report",
    report_desc="Comprehensive analysis report with methodology and findings",
    download_report="📥 Download Report",
    
    # Model info
    model_info="Model Information",
    model_xgboost="XGBoost (1-Order Users)",
    model_ft_transformer="FT-Transformer (2+ Orders)",
    model_combined="Combined Model",
    overall_recall="Overall Recall",
    weighted_recall="Weighted Recall",
    f1_score="F1 Score",
    roc_auc="ROC-AUC",
    
    # Common
    loading="Loading...",
    error="Error",
    success="Success",
    not_found="Not Found",
    user_not_found="User not found",
    api_error="Cannot connect to API",
    yes="Yes",
    no="No",
    days="days",
    users="users",
    orders="orders",
)


# Persian translations
FA = Translations(
    # App-wide
    app_title="🔮 پیش‌بینی ریزش مشتری",
    app_subtitle="پیش‌بینی و جلوگیری از ریزش مشتری با هوش مصنوعی",
    language_label="🌐 زبان",
    
    # Navigation
    nav_home="خانه",
    nav_prediction="پیش‌بینی کاربر",
    nav_high_risk="کاربران پرریسک",
    nav_analytics="تحلیل‌ها",
    nav_report="گزارش",
    
    # Dashboard
    overview_title="📊 آمار کلی",
    total_users="👥 کل کاربران",
    total_churned="📉 کاربران ریزش‌کرده",
    churn_rate="📊 نرخ ریزش",
    avg_probability="📈 میانگین احتمال",
    
    # Segments
    segment_1_order="۱ سفارش",
    segment_2_4_orders="۲-۴ سفارش",
    segment_5_10_orders="۵-۱۰ سفارش",
    segment_11_30_orders="۱۱-۳۰ سفارش",
    segment_30_plus_orders="۳۰+ سفارش",
    
    # Risk levels
    risk_low="🟢 ریسک پایین",
    risk_medium="🟡 ریسک متوسط",
    risk_high="🔴 ریسک بالا",
    
    # User Prediction
    user_prediction_title="👤 پیش‌بینی ریزش کاربر",
    user_prediction_desc="شناسه کاربر را وارد کنید تا ریسک ریزش و پروفایل او را ببینید",
    enter_user_id="شناسه کاربر",
    predict_button="🔍 پیش‌بینی",
    sample_ids="💡 نمونه شناسه‌ها",
    prediction_result="🎯 نتیجه پیش‌بینی",
    will_churn="آیا ریزش می‌کند؟",
    churn_probability="احتمال ریزش",
    risk_level="سطح ریسک",
    user_segment="سگمنت کاربر",
    churn_threshold="آستانه ریزش",
    model_used="مدل استفاده‌شده",
    
    # User Profile
    user_profile="📋 پروفایل کاربر",
    total_orders="📦 کل سفارشات",
    recency_days="📅 روز از آخرین سفارش",
    tenure_days="🕐 عمر مشتری",
    otd_rate="✅ تحویل به‌موقع",
    late_rate="⏰ نرخ تأخیر",
    avg_shop_rating="⭐ میانگین امتیاز فروشگاه",
    avg_courier_rating="🛵 میانگین امتیاز پیک",
    crm_requests="📢 درخواست‌های پشتیبانی",
    comment_count="💬 نظرات",
    
    # Recommendations
    recommendations="💡 اقدامات پیشنهادی",
    high_risk_action="""
    🚨 **ریسک بالا - نیاز به اقدام فوری**
    
    - ارسال پیشنهاد شخصی‌سازی‌شده
    - ارجاع به تیم موفقیت مشتری
    - برنامه‌ریزی تماس پیگیری
    - ارائه تخفیف ویژه یا پاداش وفاداری
    """,
    medium_risk_action="""
    ⚠️ **ریسک متوسط - نظارت دقیق**
    
    - ارسال کمپین ایمیل تعاملی
    - ارائه مشوق کوچک برای خرید بعدی
    - پایش فعالیت در ۲ هفته آینده
    """,
    low_risk_action="""
    ✅ **ریسک پایین - نیازی به اقدام فوری نیست**
    
    - ادامه تعامل استاندارد
    - گنجاندن در برنامه‌های وفاداری
    - نظارت بر تغییرات احتمالی
    """,
    
    # High Risk Users
    high_risk_title="🚨 کاربران پرریسک",
    high_risk_desc="مشاهده و مدیریت کاربران در معرض ریزش",
    filter_risk="سطح ریسک",
    filter_segment="سگمنت",
    filter_days_inactive="حداقل روز غیرفعال",
    rows_per_page="تعداد در صفحه",
    apply_filter="🔍 اعمال",
    export_csv="📥 خروجی CSV",
    
    # Analytics
    analytics_title="📊 تحلیل‌ها و بینش‌ها",
    analytics_desc="درک عوامل مؤثر بر ریزش مشتری",
    key_insights="🔑 بینش‌های کلیدی",
    feature_importance="🎯 اهمیت ویژگی‌ها",
    model_performance="🤖 عملکرد مدل",
    segment_analysis="📈 تحلیل سگمنت‌ها",
    churn_by_segment="نرخ ریزش به تفکیک سگمنت",
    
    # Report
    report_title="📄 گزارش تحلیل ریزش",
    report_desc="گزارش جامع شامل روش‌شناسی و یافته‌ها",
    download_report="📥 دانلود گزارش",
    
    # Model info
    model_info="اطلاعات مدل",
    model_xgboost="XGBoost (کاربران ۱ سفارش)",
    model_ft_transformer="FT-Transformer (۲+ سفارش)",
    model_combined="مدل ترکیبی",
    overall_recall="Recall کلی",
    weighted_recall="Weighted Recall",
    f1_score="F1 Score",
    roc_auc="ROC-AUC",
    
    # Common
    loading="در حال بارگذاری...",
    error="خطا",
    success="موفق",
    not_found="یافت نشد",
    user_not_found="کاربر یافت نشد",
    api_error="اتصال به API امکان‌پذیر نیست",
    yes="بله",
    no="خیر",
    days="روز",
    users="کاربر",
    orders="سفارش",
)


def get_translations(lang: str = "en") -> Translations:
    """Get translations for specified language."""
    return FA if lang == "fa" else EN


# Segment information with thresholds
SEGMENT_INFO = {
    "1 Order": {"threshold_days": 45, "churn_rate": 0.744, "weight": 0.449},
    "2-4 Orders": {"threshold_days": 39, "churn_rate": 0.544, "weight": 0.222},
    "5-10 Orders": {"threshold_days": 35, "churn_rate": 0.316, "weight": 0.149},
    "11-30 Orders": {"threshold_days": 17, "churn_rate": 0.302, "weight": 0.129},
    "30+ Orders": {"threshold_days": 14, "churn_rate": 0.125, "weight": 0.051},
}

# Model performance metrics (actual values from notebooks)
MODEL_METRICS = {
    "xgboost_1order": {
        "recall": 0.7484,
        "precision": 0.923,
        "roc_auc": 0.65,
    },
    "ft_transformer": {
        "overall_recall": 0.9029,
        "weighted_recall": 0.6482,
        "f1": 0.6508,
        "roc_auc": 0.7610,
        "segment_recall": {
            "2-4 Orders": 0.9958,
            "5-10 Orders": 0.8873,
            "11-30 Orders": 0.6880,
            "30+ Orders": 0.5178,
        }
    },
    "combined": {
        "overall_recall": 0.8179,
        "f1": 0.7289,
        "roc_auc": 0.6311,
    }
}

# Feature importance (actual values from SHAP)
FEATURE_IMPORTANCE = [
    {"feature": "recency_tenure_ratio", "importance": 0.215, "display_en": "Recency/Tenure Ratio", "display_fa": "نسبت رسنسی به عمر"},
    {"feature": "recency", "importance": 0.211, "display_en": "Days Since Last Order", "display_fa": "روز از آخرین سفارش"},
    {"feature": "rating_engagement", "importance": 0.175, "display_en": "Rating Engagement", "display_fa": "تعامل امتیازدهی"},
    {"feature": "last_order_rate_to_shop_filled", "importance": 0.061, "display_en": "Last Order Shop Rating", "display_fa": "امتیاز فروشگاه آخرین سفارش"},
    {"feature": "delivered_orders", "importance": 0.044, "display_en": "Delivered Orders", "display_fa": "سفارشات تحویل‌شده"},
    {"feature": "avg_rate_shop", "importance": 0.041, "display_en": "Avg Shop Rating", "display_fa": "میانگین امتیاز فروشگاه"},
    {"feature": "tenure_days", "importance": 0.033, "display_en": "Customer Tenure", "display_fa": "عمر مشتری"},
    {"feature": "cv_order_interval", "importance": 0.029, "display_en": "Order Interval Variance", "display_fa": "واریانس فاصله سفارش"},
    {"feature": "first_order_had_issue", "importance": 0.027, "display_en": "First Order Issue", "display_fa": "مشکل سفارش اول"},
    {"feature": "max_order_interval", "importance": 0.025, "display_en": "Max Order Interval", "display_fa": "حداکثر فاصله سفارش"},
]
