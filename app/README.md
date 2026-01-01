# 🚀 FastAPI Application

Production-ready inference API with dual-model architecture (XGBoost + Transformer).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                  FastAPI App                    │
│                                                 │
│  ┌──────────────┐         ┌──────────────┐    │
│  │   Schemas    │         │    Config    │    │
│  └──────────────┘         └──────────────┘    │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │         UserDataService                   │  │
│  │  - Load CSV                               │  │
│  │  - Cache predictions                      │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌──────────────────────────────────────────┐  │
│  │        ChurnPredictor (Dual Model)       │  │
│  │  ┌────────────┐      ┌───────────────┐  │  │
│  │  │  XGBoost   │      │  Transformer  │  │  │
│  │  │(1 order)   │      │  (2+ orders)  │  │  │
│  │  └────────────┘      └───────────────┘  │  │
│  └──────────────────────────────────────────┘  │
│                      ↓                          │
│  ┌──────────────────────────────────────────┐  │
│  │           StatsService                    │  │
│  │  - Overview stats                         │  │
│  │  - Risk filtering                         │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## 📁 File Structure

```
app/
├── __init__.py              # Package init
├── main.py                  # FastAPI app + endpoints
├── config.py                # Settings & environment
├── services.py              # Business logic (UserDataService, StatsService)
├── schemas.py               # Pydantic models (request/response)
├── models/
│   ├── xgboost_model.pkl    # XGBoost for single-order users
│   └── transformer.pth      # Transformer for multi-order users
├── user_features.csv        # Precomputed features (98 columns)
├── feature_names.txt        # Feature column names
├── Dockerfile               # Container image
└── requirements.txt         # Python dependencies
```

---

## 🔑 Key Components

### 1. **ChurnPredictor** (`main.py`)
Dual-model prediction engine with intelligent routing.

**Logic:**
```python
if total_orders == 1:
    use XGBoost  # Optimized for cold-start users
else:
    use Transformer  # Leverages sequential patterns
```

**Features:**
- GPU support (auto-detects)
- Batch prediction
- Feature importance extraction
- Risk level assignment

---

### 2. **UserDataService** (`services.py`)
Data loading and caching layer.

**Responsibilities:**
- Load `user_features.csv` on startup
- Validate feature count (98 required)
- Cache predictions for fast lookup
- Handle missing users gracefully

---

### 3. **StatsService** (`services.py`)
Statistics and analytics.

**Provides:**
- Overview (total users, risk distribution)
- Filtered user lists by risk level
- Paginated results

---

## 🌐 API Endpoints

### **GET /** - Root
Returns API information and version.

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "name": "Churn Prediction API",
  "version": "2.0.0",
  "docs": "/docs"
}
```

---

### **GET /health** - Health Check
Checks if models are loaded and service is ready.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "cuda:0",
  "users_loaded": 338101
}
```

---

### **POST /predict** - Single Prediction
Predict churn for one user.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1385028"}'
```

**Response:**
```json
{
  "user_id": "1385028",
  "probability": 0.8234,
  "will_churn": true,
  "risk_level": "HIGH",
  "model_used": "transformer"
}
```

---

### **POST /predict/batch** - Batch Prediction
Predict for multiple users at once.

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"user_ids": ["1385028", "1234567", "9876543"]}'
```

**Response:**
```json
{
  "predictions": [
    {
      "user_id": "1385028",
      "probability": 0.8234,
      "will_churn": true,
      "risk_level": "HIGH",
      "model_used": "transformer"
    }
  ]
}
```

---

### **GET /stats/overview** - Statistics
Get high-level stats.

```bash
curl http://localhost:8000/stats/overview
```

**Response:**
```json
{
  "total_users": 338101,
  "low_risk": 102340,
  "medium_risk": 118456,
  "high_risk": 117305,
  "avg_probability": 0.5423
}
```

---

### **GET /users/{user_id}** - User Details
Get detailed user info with features.

```bash
curl http://localhost:8000/users/1385028
```

**Response:**
```json
{
  "user_id": "1385028",
  "features": {
    "total_orders": 15,
    "days_since_last_order": 8,
    "avg_order_gap_days": 12.4
  },
  "prediction": {
    "probability": 0.8234,
    "risk_level": "HIGH"
  }
}
```

---

## ⚙️ Configuration

Environment variables (`.env`):

```bash
# Model settings
MODEL_PATH=app/models/xgboost_model.pkl
TRANSFORMER_PATH=app/models/transformer.pth
DEVICE=auto  # auto, cpu, cuda

# Data
FEATURES_PATH=app/user_features.csv
FEATURE_COUNT=98

# API settings
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t churn-api app/
```

### Run Container
```bash
docker run -p 8000:8000 \
  -v $(pwd)/app:/app \
  -e DEVICE=cpu \
  churn-api
```

### Using Docker Compose
```bash
docker-compose up -d api
```

---

## 🚀 Local Development

### Install Dependencies
```bash
cd app/
pip install -r requirements.txt
```

### Run with Auto-Reload
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run with Gunicorn (Production)
```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Avg Response Time** | ~50ms |
| **Throughput** | ~2000 req/sec (4 workers) |
| **Memory Usage** | ~1.2GB (with models loaded) |
| **Startup Time** | ~5 seconds |

**GPU Boost:** 3-5x faster inference on CUDA.

---

## 🔒 Security

- ✅ Input validation (Pydantic schemas)
- ✅ CORS configured
- ✅ Rate limiting (optional: add middleware)
- ✅ Health checks for orchestration
- ⚠️ No authentication (add JWT for production)

---

## 📝 Logging

Structured JSON logs:

```python
import logging

logger = logging.getLogger("churn_api")
logger.info("Prediction made", extra={
    "user_id": user_id,
    "probability": prob,
    "model": model_name
})
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/test_api.py

# Integration tests
pytest tests/test_services.py

# API tests with live server
python scripts/test_api.py
```

---

## 🐛 Troubleshooting

### Issue: `FileNotFoundError: user_features.csv`
**Solution:**
```bash
# Generate features
jupyter notebook notebooks/03_preprocessing_feature_engineering_final.ipynb
# Copy to app/
cp data/user_features.csv app/
```

### Issue: `RuntimeError: No CUDA devices available`
**Solution:**
```bash
# Force CPU mode
export DEVICE=cpu
```

### Issue: High memory usage
**Solution:**
```bash
# Reduce batch size or use model quantization
# Consider loading models on-demand
```

---

## 📚 API Documentation

Interactive docs available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🔄 Model Updates

To deploy a new model:

1. Train model in notebooks
2. Copy to `app/models/`
3. Update `config.py` paths
4. Restart API

```bash
docker-compose restart api
```

---

## 👤 Author

**Peyman** - [@peyman886](https://github.com/peyman886)