# 📊 Reports Directory

این پوشه شامل گزارش‌های پروژه پیش‌بینی ریزش مشتری است.

## 📁 ساختار فایل‌ها

```
reports/
├── README.md                      # این فایل
├── Gozaresh.html                  # گزارش اصلی (HTML با استایل)
├── Gozaresh.md                    # گزارش (Markdown برای GitHub)
├── er_diagram.svg                 # نمودار ER پایگاه داده
├── generate_plots.py              # 🆕 اسکریپت تولید پلات‌ها
│
└── [پلات‌ها - تولید شده توسط اسکریپت]
    ├── 01_order_distribution.png
    ├── 02_daily_orders_timeseries.png
    ├── 03_cohort_heatmap.png
    ├── 04_conversion_funnel.png
    ├── 05_clv_distribution.png
    ├── 06_survival_curve.png
    ├── 07_segment_distribution.png
    ├── 08_pareto_chart.png
    ├── 09_rf_by_segment.png
    ├── 10_churn_rate_trend.png
    ├── 11_feature_importance.png
    ├── 12_roc_curves.png
    ├── 13_confusion_matrix.png
    ├── 14_shap_summary.png
    └── 15_model_comparison.png
```

## 📄 فایل‌های گزارش

### `Gozaresh.html`
گزارش اصلی با فرمت HTML شامل:
- استایل زیبا و حرفه‌ای
- پشتیبانی کامل از RTL فارسی
- جدول‌ها و کارت‌های آماری
- جای‌گذاری تصاویر

**نحوه مشاهده:** فایل را در مرورگر باز کنید.

### `Gozaresh.md`
نسخه Markdown برای:
- نمایش در GitHub
- سازگاری با سیستم‌های مستندسازی
- ویرایش آسان

### `er_diagram.svg`
نمودار Entity-Relationship شامل:
- ساختار سه جدول اصلی (Orders, CRM, Comments)
- روابط بین جداول
- نوع داده‌ها و کلیدها

## 🖼️ لیست پلات‌های مورد نیاز

برای کامل شدن گزارش، پلات‌های زیر باید در این پوشه قرار بگیرند:

| # | نام فایل | توضیحات | نوت‌بوک منبع |
|---|----------|---------|--------------|
| 1 | `01_order_distribution.png` | هیستوگرام توزیع سفارشات کاربران | `01_comprehensive_eda.ipynb` |
| 2 | `02_daily_orders_timeseries.png` | سری زمانی سفارشات روزانه | `01_comprehensive_eda.ipynb` |
| 3 | `03_cohort_heatmap.png` | Heatmap تحلیل Cohort | `01_comprehensive_eda.ipynb` |
| 4 | `04_conversion_funnel.png` | قیف تبدیل مشتریان | `01_business_metrics_clv_analysis.ipynb` |
| 5 | `05_clv_distribution.png` | توزیع CLV Score | `01_business_metrics_clv_analysis.ipynb` |
| 6 | `06_survival_curve.png` | منحنی Kaplan-Meier | `05_advanced_eda_deep_analysis.ipynb` |
| 7 | `07_segment_distribution.png` | توزیع سگمنت‌ها | `01_segment_based_churn_analysis.ipynb` |
| 8 | `08_pareto_chart.png` | نمودار Pareto (80-20) | `01_comprehensive_eda.ipynb` |
| 9 | `09_rf_by_segment.png` | توزیع RF به تفکیک سگمنت | `01_segment_based_churn_analysis.ipynb` |
| 10 | `10_churn_rate_trend.png` | روند Churn Rate در زمان | `01_business_metrics_clv_analysis.ipynb` |
| 11 | `11_feature_importance.png` | Feature Importance | `03_ml_modeling_experiments.ipynb` |
| 12 | `12_roc_curves.png` | منحنی‌های ROC | `04_neural_network_models_v2.ipynb` |
| 13 | `13_confusion_matrix.png` | Confusion Matrix | `04_neural_network_models_v2.ipynb` |
| 14 | `14_shap_summary.png` | SHAP Summary Plot | `03_ml_modeling_experiments.ipynb` |
| 15 | `15_model_comparison.png` | مقایسه مدل‌ها | `04_neural_network_models_v2.ipynb` |

## 🔧 تولید خودکار پلات‌ها

### روش ساده: اجرای اسکریپت

```bash
cd reports/
python generate_plots.py
```

این اسکریپت تمام ۱۵ پلات را با کیفیت بالا و استایل یکپارچه تولید می‌کند.

### پیش‌نیازها

```bash
pip install pandas numpy matplotlib seaborn
```

### روش دستی: ذخیره از نوت‌بوک

```python
import matplotlib.pyplot as plt

# بعد از ساخت پلات
plt.savefig('../reports/01_order_distribution.png', 
            dpi=150, 
            bbox_inches='tight',
            facecolor='white')
```

## 📋 محتوای گزارش

گزارش شامل بخش‌های زیر است:

1. **تعریف مسئله** - هدف و چالش‌های پروژه
2. **معرفی دیتاست** - ساختار و آمار داده‌ها
3. **EDA** - تحلیل اکتشافی داده‌ها
4. **تحلیل بیزینسی** - CLV، Conversion، Retention
5. **سگمنت‌بندی** - تقسیم‌بندی 5 گروهی
6. **تعریف Churn** - آستانه‌های مختص هر سگمنت
7. **پیش‌پردازش** - مدیریت missing و Rolling Window
8. **Feature Engineering** - 98 ویژگی در 7 دسته
9. **مدل‌سازی** - XGBoost + FT-Transformer
10. **نتایج** - متریک‌ها و تفسیر
11. **پیشنهادات** - کارهای آینده

## 🎯 نکات مهم

- فایل HTML برای ارائه و نمایش بهتر است
- فایل MD برای GitHub و مستندسازی مناسب‌تر است
- همه پلات‌ها باید با فرمت PNG و DPI مناسب (150+) ذخیره شوند
- نام فایل‌ها دقیقاً مطابق جدول بالا باشد

## 📝 ویرایش

برای ویرایش گزارش:
- HTML: مستقیماً فایل را ویرایش کنید
- MD: از هر ویرایشگر Markdown استفاده کنید

---

📊 **پروژه پیش‌بینی ریزش مشتری**