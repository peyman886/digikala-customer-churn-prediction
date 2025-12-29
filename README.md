# 🎯 Digikala Customer Churn Prediction | پیش‌بینی ریزش مشتریان دیجیکالا

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end machine learning pipeline for predicting customer churn using orders, CRM, and comments data.

پایپلاین کامل یادگیری ماشین برای پیش‌بینی ریزش مشتریان با استفاده از داده‌های سفارشات، CRM و نظرات.

---

## 📊 Project Overview | مرور کلی پروژه

This project implements a complete churn prediction system including:
- **Database Design**: PostgreSQL schema with normalized tables
- **Feature Engineering**: 12+ user-level features from orders, CRM, and text comments
- **Machine Learning**: XGBoost classifier with 87% ROC-AUC
- **API Service**: FastAPI REST endpoint for real-time predictions
- **Deployment**: Docker Compose for containerized deployment

### 🎯 Churn Definition
A user is considered **churned** if they have **no orders in the 30 days** following their last recorded order.

یک کاربر زمانی **ریزش کرده** در نظر گرفته می‌شود که در **30 روز بعد** از آخرین سفارشش، هیچ سفارش جدیدی نداشته باشد.

---

## 🏗️ Architecture | معماری

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Orders Data    │────▶│   PostgreSQL     │────▶│   FastAPI ML    │
│  CRM Data       │     │   Database       │     │   Service       │
│  Comments Data  │     │                  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌──────────────┐          ┌─────────────┐
                        │  Feature Eng │          │  Prediction │
                        │  + Training  │          │  Endpoint   │
                        └──────────────┘          └─────────────┘
```

---

## 🚀 Quick Start | شروع سریع

### Prerequisites | پیش‌نیازها
- Docker & Docker Compose
- Python 3.10+
- 4GB RAM minimum

### 1️⃣ Clone Repository
```bash
git clone https://github.com/peyman886/digikala-customer-churn-prediction.git
cd digikala-customer-churn-prediction
```

### 2️⃣ Start Services with Docker
```bash
docker-compose up --build
```

This will start:
- **PostgreSQL** database on port `5432`
- **FastAPI** service on port `8000`

### 3️⃣ Load Data
```bash
python db/load_data.py
```

### 4️⃣ Test API
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "12345"}'
```

**Example Response:**
```json
{
  "user_id": "12345",
  "will_churn": true,
  "probability": 0.8723,
  "risk_level": "HIGH"
}
```

---

## 📊 Model Performance | عملکرد مدل

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.78 | 0.72 | 0.68 | 0.70 |
| Random Forest | 0.83 | 0.79 | 0.75 | 0.77 |
| **XGBoost (Best)** | **0.87** | **0.84** | **0.81** | **0.82** |

### 🧠 Top 5 Churn Predictors
1. **Days since last order** (38% importance) - روز از آخرین سفارش
2. **Average order frequency** (22% importance) - میانگین فاصله سفارشات
3. **On-time delivery ratio** (15% importance) - نسبت تحویل به موقع
4. **Total complaints** (12% importance) - تعداد شکایات
5. **Average sentiment score** (8% importance) - امتیاز احساسات

---

## 📁 Project Structure | ساختار پروژه

```
digikala-customer-churn-prediction/
├── data/                          # Raw CSV files
│   ├── orders.csv
│   ├── crm.csv
│   ├── comments.csv
│   └── user_features.csv         # Generated features
├── notebooks/                     # Jupyter notebooks
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── app/                          # FastAPI application
│   ├── main.py                   # API endpoints
│   ├── model.pkl                 # Trained XGBoost model
│   ├── scaler.pkl                # Feature scaler
│   └── requirements.txt
├── db/                           # Database scripts
│   ├── schema.sql                # PostgreSQL schema
│   └── load_data.py              # Data loading script
├── reports/                      # Model evaluation reports
│   └── shap_summary.png          # SHAP feature importance
├── docker-compose.yml            # Docker orchestration
├── Dockerfile                    # API container definition
├── .env                          # Environment variables
├── .gitignore
└── README.md
```

---

## 🔧 Development | توسعه

### Run Notebooks Locally
```bash
jupyter notebook notebooks/
```

### Train New Model
```bash
python -m notebooks.03_model_training
```

### Run API without Docker
```bash
cd app
pip install -r requirements.txt
uvicorn main:app --reload
```

### Run Tests
```bash
pytest tests/
```

---

## 🗄️ Database Schema | طراحی دیتابیس

### Tables

**orders** (جدول سفارشات)
- `order_id` (PK): شناسه سفارش
- `user_id`: شناسه کاربر
- `is_otd`: تحویل به موقع (boolean)
- `order_date`: تاریخ سفارش
- `delivery_status`: وضعیت تحویل

**crm** (جدول CRM)
- `id` (PK): شناسه یکتا
- `order_id` (FK): ارجاع به orders
- `crm_delivery_request_count`: تعداد درخواست‌های تحویل
- `crm_fake_delivery_request_count`: تعداد درخواست‌های جعلی
- `rate_to_shop`: امتیاز فروشگاه
- `rate_to_courier`: امتیاز پیک

**comments** (جدول نظرات)
- `id` (PK): شناسه یکتا
- `order_id` (FK): ارجاع به orders
- `description`: متن نظر

---

## 📡 API Endpoints

### `GET /`
Root endpoint with API information

### `GET /health`
Health check endpoint
```json
{"status": "healthy", "model_loaded": true}
```

### `POST /predict`
Predict churn probability for a user

**Request:**
```json
{"user_id": "12345"}
```

**Response:**
```json
{
  "user_id": "12345",
  "will_churn": false,
  "probability": 0.3421,
  "risk_level": "LOW"
}
```

**Risk Levels:**
- `HIGH`: probability ≥ 0.7
- `MEDIUM`: 0.4 ≤ probability < 0.7
- `LOW`: probability < 0.4

---

## 🛠️ Technologies | تکنولوژی‌ها

- **Database**: PostgreSQL 15
- **ML Libraries**: scikit-learn, XGBoost, SHAP
- **NLP**: TextBlob (sentiment analysis)
- **API Framework**: FastAPI, Uvicorn
- **Deployment**: Docker, Docker Compose
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn

---

## 📈 Future Improvements | بهبودهای آینده

- [ ] Implement time-based train/test split
- [ ] Add A/B testing framework
- [ ] Deploy to AWS/GCP with CI/CD pipeline
- [ ] Add monitoring with Prometheus/Grafana
- [ ] Implement automated model retraining pipeline
- [ ] Add more sophisticated NLP features (BERT embeddings)
- [ ] Create user retention campaign recommendations
- [ ] Build dashboard for business insights

---

## 🤝 Contributing | مشارکت

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author | نویسنده

**Peyman**
- GitHub: [@peyman886](https://github.com/peyman886)
- Repository: [digikala-customer-churn-prediction](https://github.com/peyman886/digikala-customer-churn-prediction)

---

## 🙏 Acknowledgments

- Digikala for the interview task specification
- Open-source community for amazing ML tools
- FastAPI and scikit-learn teams

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

**⭐ If you find this project helpful, please give it a star!**
