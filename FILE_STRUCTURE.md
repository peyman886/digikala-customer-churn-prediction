# 📁 راهنمای ساختار فایل‌ها (File Structure Guide)

این سند توضیح می‌دهد که هر فایل کجا قرار می‌گیرد و چه کاری انجام می‌دهد.

---

## 🗂️ ساختار کلی پروژه

```
digikala-customer-churn-prediction/
│
├── 📄 requirements.txt          ← نصب پکیج‌ها (GPU)
├── 📄 requirements-cpu.txt      ← نصب پکیج‌ها (CPU)
├── 📄 docker-compose.yml        ← اجرای Docker (GPU)
├── 📄 docker-compose.cpu.yml    ← تنظیمات اضافی برای CPU
├── 📄 Dockerfile.jupyter        ← Jupyter با GPU
├── 📄 Dockerfile.jupyter.cpu    ← Jupyter بدون GPU
├── 📄 Makefile                  ← دستورات سریع
├── 📄 pyproject.toml            ← تنظیمات Python
├── 📄 .env.example              ← نمونه تنظیمات محیطی
├── 📄 .gitignore                ← فایل‌های ignore شده
├── 📄 .dockerignore             ← فایل‌های exclude شده از Docker
├── 📄 README.md                 ← مستندات اصلی
├── 📄 FILE_STRUCTURE.md         ← این فایل!
│
├── 📁 app/                      ← Backend API
│   ├── 📄 Dockerfile            ← Docker برای API (GPU)
│   ├── 📄 Dockerfile.cpu        ← Docker برای API (CPU)
│   ├── 📄 requirements.txt      ← پکیج‌های API
│   ├── 📄 main.py               ← نقاط API
│   ├── 📄 services.py           ← منطق کسب‌وکار
│   ├── 📄 config.py             ← تنظیمات
│   ├── 📄 schemas.py            ← مدل‌های Pydantic
│   └── 📁 models/               ← wrapper های مدل
│
├── 📁 frontend/                 ← Streamlit Dashboard
│   ├── 📄 Dockerfile            ← Docker برای Frontend
│   ├── 📄 requirements.txt      ← پکیج‌های Frontend
│   ├── 📄 Home.py               ← صفحه اصلی
│   └── 📁 pages/                ← صفحات داشبورد
│
├── 📁 data/                     ← فایل‌های داده (CSV)
│   ├── 📄 README.md
│   ├── 📄 orders.csv            ← (باید اضافه کنی)
│   ├── 📄 crm.csv               ← (باید اضافه کنی)
│   └── 📄 order_comments.csv    ← (باید اضافه کنی)
│
├── 📁 db/                       ← پایگاه داده
│   ├── 📄 schema.sql            ← ساختار جداول
│   └── 📄 load_data.py          ← بارگذاری داده‌ها
│
├── 📁 mlops/                    ← MLflow Tracking
│   ├── 📄 tracker.py            ← کلاس tracking
│   ├── 📄 compare.py            ← مقایسه آزمایش‌ها
│   └── 📄 config.py             ← تنظیمات MLOps
│
├── 📁 models_v2/                ← مدل‌های آموزش دیده
│   ├── 📄 xgboost_1order.pkl    ← مدل XGBoost
│   ├── 📄 ft_transformer.pt     ← مدل FT-Transformer
│   └── 📄 scaler.pkl            ← Scaler
│
├── 📁 notebooks/                ← Jupyter Notebooks
│
├── 📁 src/                      ← کد منبع ML
│   ├── 📁 data/                 ← پردازش داده
│   ├── 📁 models/               ← تعریف مدل‌ها
│   ├── 📁 training/             ← آموزش
│   ├── 📁 evaluation/           ← ارزیابی
│   └── 📁 visualization/        ← نمودارها
│
├── 📁 tests/                    ← تست‌ها
│
└── 📁 reports/                  ← گزارش‌های تولید شده
```

---

## 📋 فایل‌های ریشه (Root Files)

| فایل | محل | توضیحات |
|------|-----|---------|
| `requirements.txt` | `/` (ریشه پروژه) | پکیج‌های Python برای GPU |
| `requirements-cpu.txt` | `/` (ریشه پروژه) | پکیج‌های Python برای CPU |
| `docker-compose.yml` | `/` (ریشه پروژه) | تنظیمات Docker با GPU |
| `docker-compose.cpu.yml` | `/` (ریشه پروژه) | override برای CPU |
| `Dockerfile.jupyter` | `/` (ریشه پروژه) | Jupyter با GPU |
| `Dockerfile.jupyter.cpu` | `/` (ریشه پروژه) | Jupyter بدون GPU |
| `Makefile` | `/` (ریشه پروژه) | دستورات make |
| `pyproject.toml` | `/` (ریشه پروژه) | تنظیمات ابزارها |
| `.env.example` | `/` (ریشه پروژه) | نمونه .env |
| `.gitignore` | `/` (ریشه پروژه) | فایل‌های git ignore |
| `.dockerignore` | `/` (ریشه پروژه) | فایل‌های docker ignore |
| `README.md` | `/` (ریشه پروژه) | مستندات اصلی |

---

## 📁 پوشه app/ (Backend API)

| فایل | محل | توضیحات |
|------|-----|---------|
| `Dockerfile` | `/app/` | Docker image برای API با GPU |
| `Dockerfile.cpu` | `/app/` | Docker image برای API بدون GPU |
| `requirements.txt` | `/app/` | پکیج‌های مورد نیاز API |

---

## 📁 پوشه frontend/ (Dashboard)

| فایل | محل | توضیحات |
|------|-----|---------|
| `Dockerfile` | `/frontend/` | Docker image برای Streamlit |
| `requirements.txt` | `/frontend/` | پکیج‌های Streamlit |

---

## 🚀 نحوه استفاده

### ۱. کپی کردن .env

```bash
cp .env.example .env
```

### ۲. اجرا با GPU

```bash
make up
# یا
docker-compose up -d
```

### ۳. اجرا بدون GPU (CPU)

```bash
make up-cpu
# یا
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### ۴. اجرا در حالت توسعه (با Jupyter و PgAdmin)

```bash
# با GPU
make dev

# بدون GPU
make dev-cpu
```

---

## 🔗 آدرس‌های سرویس‌ها

| سرویس | آدرس | توضیحات |
|-------|------|---------|
| Frontend | http://localhost:8501 | داشبورد Streamlit |
| API Docs | http://localhost:9000/docs | مستندات FastAPI |
| MLflow | http://localhost:5000 | ردیابی آزمایش‌ها |
| Jupyter | http://localhost:8888 | نوت‌بوک (token: churn123) |
| PgAdmin | http://localhost:5050 | مدیریت دیتابیس |
| PostgreSQL | localhost:5432 | دیتابیس |

---

## ❓ سوالات متداول

### چرا دو فایل Dockerfile هست؟

- `Dockerfile` = با پشتیبانی GPU (CUDA 12.8)
- `Dockerfile.cpu` = بدون GPU (سبک‌تر و سریع‌تر برای build)

### چرا دو فایل docker-compose هست؟

- `docker-compose.yml` = تنظیمات اصلی با GPU
- `docker-compose.cpu.yml` = override می‌کنه و GPU رو غیرفعال می‌کنه

### چرا دو فایل requirements هست؟

- `requirements.txt` = با `torch==2.9.0+cu128` (نیاز به GPU)
- `requirements-cpu.txt` = با `torch==2.9.0+cpu` (بدون نیاز به GPU)

---

## 🛠️ عیب‌یابی

### خطای GPU

```bash
# چک کردن GPU
nvidia-smi

# اگه GPU نداری از نسخه CPU استفاده کن
make up-cpu
```

### خطای Port in use

```bash
# متوقف کردن همه container ها
make down

# یا تغییر پورت در .env
API_PORT=9001
```

### خطای Permission denied

```bash
# روی Linux/Mac
chmod +x scripts/*.sh
sudo chown -R $USER:$USER .
```
