# 🔮 Churn Prediction Frontend

A bilingual (English/Persian) Streamlit dashboard for customer churn prediction.

## 📁 Structure

```
frontend/
├── Home.py                     # Main dashboard
├── config/
│   ├── __init__.py
│   └── i18n.py                 # Translations & constants
├── utils/
│   ├── __init__.py
│   ├── api_client.py           # Backend API client
│   └── helpers.py              # UI helper functions
├── pages/
│   ├── 1_👤_User_Prediction.py # User churn prediction
│   ├── 2_🚨_High_Risk_Users.py # High risk user list
│   ├── 3_📊_Analytics.py       # Analytics & insights
│   └── 4_📄_Report.py          # Full report viewer
├── requirements.txt
└── Dockerfile
```

## 🚀 Running

### Local Development

```bash
cd frontend
pip install -r requirements.txt
streamlit run Home.py
```

### Docker

```bash
docker build -t churn-frontend .
docker run -p 8501:8501 -e API_URL=http://localhost:8000 churn-frontend
```

### Docker Compose

```bash
docker-compose up frontend
```

## 🌐 Features

### Bilingual Support
- English (🇬🇧)
- Persian/Farsi (🇮🇷)

Switch language using the selector in the sidebar.

### Pages

1. **Home**: Overview dashboard with key metrics and charts
2. **User Prediction**: Predict churn for individual users
3. **High Risk Users**: Filterable list of at-risk users
4. **Analytics**: Feature importance, segment analysis, model performance
5. **Report**: Full HTML/Markdown report viewer

## 📊 Key Information

### Segment-Based Churn Definition

| Segment | Threshold | Churn Rate |
|---------|-----------|------------|
| 1 Order | 45 days | 74.4% |
| 2-4 Orders | 39 days | 54.4% |
| 5-10 Orders | 35 days | 31.6% |
| 11-30 Orders | 17 days | 30.2% |
| 30+ Orders | 14 days | 12.5% |

### Model Performance

| Model | Users | Recall |
|-------|-------|--------|
| XGBoost | 1-Order | 74.8% |
| FT-Transformer | 2+ Orders | 90.3% |
| Combined | All | 81.8% |

## 🔧 Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:8000` | Backend API URL |
| `STREAMLIT_SERVER_PORT` | `8501` | Streamlit port |

## 🎨 Design Principles

- **Clean Code**: OOP where beneficial, not over-engineered
- **Pythonic**: Following Python best practices
- **SOLID**: Practical application of principles
- **Bilingual**: Full Persian and English support
- **Responsive**: Works on different screen sizes
