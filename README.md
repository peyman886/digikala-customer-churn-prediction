# 🔮 Customer Churn Prediction System

A production-ready ML system for predicting customer churn using a **dual-model strategy**: XGBoost for first-time customers and FT-Transformer for repeat customers.

## 📊 Model Performance

| Metric | Score |
|------|-------|
| ROC-AUC | 0.63  |
| F1 Score | 0.73  |
| Recall | 0.82  |
| Weighted Recall | 0.65  |

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│  Streamlit  │───▶│   FastAPI   │───▶│   ML Pipeline   │
│  Frontend   │    │   Backend   │    │  (XGB + FT-T)   │
│  :8501      │    │   :9000     │    │                 │
└─────────────┘    └─────────────┘    └─────────────────┘
                          │                    │
                          ▼                    ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ PostgreSQL  │    │   MLflow    │
                   │   :5432     │    │   :5000     │
                   └─────────────┘    └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose v2+
- **For GPU:** NVIDIA GPU + nvidia-docker2
- **For CPU:** No additional requirements

### 1. Clone & Setup

```bash
git clone <repository-url>
cd churn-prediction

# Copy environment file
cp .env.example .env
```

### 2. Add Data Files

Place these files in the `data/` directory:
- `orders.csv`
- `crm.csv`
- `order_comments.csv`

### 3. Start Services

**With GPU (Recommended):**
```bash
make up
```

**Without GPU:**
```bash
make up-cpu
```

### 4. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8501 | Streamlit Dashboard |
| API Docs | http://localhost:9000/docs | FastAPI Swagger UI |
| MLflow | http://localhost:5000 | Experiment Tracking |

## 📁 Project Structure

```
churn-prediction/
├── app/                    # FastAPI Backend
│   ├── Dockerfile         # GPU Dockerfile
│   ├── Dockerfile.cpu     # CPU Dockerfile
│   ├── main.py            # API endpoints
│   ├── services.py        # Business logic
│   └── models/            # Model wrappers
├── frontend/               # Streamlit Dashboard
│   ├── Dockerfile
│   ├── Home.py
│   └── pages/
├── data/                   # Data files (CSV)
├── db/                     # Database schema & loader
├── mlops/                  # MLflow tracking
├── models_v2/              # Trained models
├── notebooks/              # Jupyter notebooks
├── src/                    # ML source code
│   ├── data/              # Data processing
│   ├── models/            # Model definitions
│   ├── training/          # Training logic
│   ├── evaluation/        # Metrics & evaluation
│   └── visualization/     # Plots & reports
├── tests/                  # Unit tests
├── docker-compose.yml      # GPU configuration
├── docker-compose.cpu.yml  # CPU override
├── requirements.txt        # GPU dependencies
├── requirements-cpu.txt    # CPU dependencies
└── Makefile               # Automation commands
```

## 🐳 Docker Commands

### Production

```bash
# GPU
make up                # Start all services
make down              # Stop all services
make logs              # View logs

# CPU
make up-cpu            # Start without GPU
```

### Development

```bash
# GPU (includes Jupyter + PgAdmin)
make dev

# CPU
make dev-cpu
```

Services in dev mode:
- Jupyter: http://localhost:8888 (token: `churn123`)
- PgAdmin: http://localhost:5050 (admin@local.dev / admin)

## 🔌 API Reference

### Health Check
```bash
curl http://localhost:9000/health
```

### Predict Churn
```bash
curl -X POST http://localhost:9000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"user_id": "1385028"}'
```

Response:
```json
{
  "user_id": "1385028",
  "probability": 0.2931,
  "will_churn": false,
  "risk_level": "LOW",
  "model_used": "ft_transformer"
}
```

### Get High-Risk Users
```bash
curl "http://localhost:9000/api/users/at-risk?risk_level=high&limit=10"

```

## 🧠 Dual-Model Strategy

| Customer Type | Model | Reason |
|---------------|-------|--------|
| 1 order | XGBoost | Limited behavioral data |
| 2+ orders | FT-Transformer | Rich sequential patterns |

## 🛠️ Local Development

### Install Dependencies

```bash
# GPU
make setup

# CPU
make setup-cpu
```

### Run Tests

```bash
make test           # All tests
make test-cov       # With coverage
```

### Code Quality

```bash
make lint           # Check style
make format         # Auto-format
```

## 📈 MLOps

### View Experiments
```bash
make mlflow         # Open MLflow UI
```

### Compare Models
```bash
make compare        # Compare all experiments
make report         # Generate comparison report
```

### Promote Best Model
```bash
make promote        # Promote to production
```

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | Database host |
| `DB_PORT` | 5432 | Database port |
| `DB_NAME` | churn_db | Database name |
| `API_PORT` | 9000 | API port |
| `DEVICE` | auto | cpu, cuda, or auto |
| `LOG_LEVEL` | INFO | Logging level |

## 🐛 Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi
```

### Database Connection Error
```bash
# Ensure DB is running
make db-up

# Check logs
docker-compose logs db
```

### Out of Memory
```bash
# Use CPU version
make up-cpu
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file.
