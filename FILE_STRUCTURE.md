# 📁 File Structure Guide

This document explains where each file is located and what it does.

***

## 🗂️ Overall Project Structure

```
digikala-customer-churn-prediction/
│
├── 📄 requirements.txt          ← Package installation (GPU)
├── 📄 requirements-cpu.txt      ← Package installation (CPU)
├── 📄 docker-compose.yml        ← Docker execution (GPU)
├── 📄 docker-compose.cpu.yml    ← Additional settings for CPU
├── 📄 Dockerfile.jupyter        ← Jupyter with GPU
├── 📄 Dockerfile.jupyter.cpu    ← Jupyter without GPU
├── 📄 Makefile                  ← Quick commands
├── 📄 pyproject.toml            ← Python settings
├── 📄 .env.example              ← Environment settings example
├── 📄 .gitignore                ← Ignored files
├── 📄 .dockerignore             ← Files excluded from Docker
├── 📄 README.md                 ← Main documentation
├── 📄 FILE_STRUCTURE.md         ← This file!
│
├── 📁 app/                      ← Backend API
│   ├── 📄 Dockerfile            ← Docker for API (GPU)
│   ├── 📄 Dockerfile.cpu        ← Docker for API (CPU)
│   ├── 📄 requirements.txt      ← API packages
│   ├── 📄 main.py               ← API endpoints
│   ├── 📄 services.py           ← Business logic
│   ├── 📄 config.py             ← Settings
│   ├── 📄 schemas.py            ← Pydantic models
│   └── 📁 models/               ← Model wrappers
│
├── 📁 frontend/                 ← Streamlit Dashboard
│   ├── 📄 Dockerfile            ← Docker for Frontend
│   ├── 📄 requirements.txt      ← Frontend packages
│   ├── 📄 Home.py               ← Main page
│   └── 📁 pages/                ← Dashboard pages
│
├── 📁 data/                     ← Data files (CSV)
│   ├── 📄 README.md
│   ├── 📄 orders.csv            ← (You need to add)
│   ├── 📄 crm.csv               ← (You need to add)
│   └── 📄 order_comments.csv    ← (You need to add)
│
├── 📁 db/                       ← Database
│   ├── 📄 schema.sql            ← Table structure
│   └── 📄 load_data.py          ← Data loading
│
├── 📁 mlops/                    ← MLflow Tracking
│   ├── 📄 tracker.py            ← Tracking class
│   ├── 📄 compare.py            ← Experiment comparison
│   └── 📄 config.py             ← MLOps settings
│
├── 📁 models_v2/                ← Trained models
│   ├── 📄 xgboost_1order.pkl    ← XGBoost model
│   ├── 📄 ft_transformer.pt     ← FT-Transformer model
│   └── 📄 scaler.pkl            ← Scaler
│
├── 📁 notebooks/                ← Jupyter Notebooks
│
├── 📁 src/                      ← ML source code
│   ├── 📁 data/                 ← Data processing
│   ├── 📁 models/               ← Model definitions
│   ├── 📁 training/             ← Training
│   ├── 📁 evaluation/           ← Evaluation
│   └── 📁 visualization/        ← Charts
│
├── 📁 tests/                    ← Tests
│
└── 📁 reports/                  ← Generated reports
```

***

## 📋 Root Files

| File | Location | Description |
|------|----------|-------------|
| `requirements.txt` | `/` (project root) | Python packages for GPU |
| `requirements-cpu.txt` | `/` (project root) | Python packages for CPU |
| `docker-compose.yml` | `/` (project root) | Docker settings with GPU |
| `docker-compose.cpu.yml` | `/` (project root) | Override for CPU |
| `Dockerfile.jupyter` | `/` (project root) | Jupyter with GPU |
| `Dockerfile.jupyter.cpu` | `/` (project root) | Jupyter without GPU |
| `Makefile` | `/` (project root) | Make commands |
| `pyproject.toml` | `/` (project root) | Tool settings |
| `.env.example` | `/` (project root) | .env example |
| `.gitignore` | `/` (project root) | Git ignored files |
| `.dockerignore` | `/` (project root) | Docker ignored files |
| `README.md` | `/` (project root) | Main documentation |

***

## 📁 app/ Folder (Backend API)

| File | Location | Description |
|------|----------|-------------|
| `Dockerfile` | `/app/` | Docker image for API with GPU |
| `Dockerfile.cpu` | `/app/` | Docker image for API without GPU |
| `requirements.txt` | `/app/` | Required API packages |

***

## 📁 frontend/ Folder (Dashboard)

| File | Location | Description |
|------|----------|-------------|
| `Dockerfile` | `/frontend/` | Docker image for Streamlit |
| `requirements.txt` | `/frontend/` | Streamlit packages |

***

## 🚀 How to Use

### 1. Copy .env

```bash
cp .env.example .env
```

### 2. Run with GPU

```bash
make up
# or
docker-compose up -d
```

### 3. Run without GPU (CPU)

```bash
make up-cpu
# or
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

### 4. Run in development mode (with Jupyter and PgAdmin)

```bash
# With GPU
make dev

# Without GPU
make dev-cpu
```

***

## 🔗 Service URLs

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8501 | Streamlit dashboard |
| API Docs | http://localhost:9000/docs | FastAPI documentation |
| MLflow | http://localhost:5000 | Experiment tracking |
| Jupyter | http://localhost:8888 | Notebook (token: churn123) |
| PgAdmin | http://localhost:5050 | Database management |
| PostgreSQL | localhost:5432 | Database |

***

## ❓ Frequently Asked Questions

### Why are there two Dockerfiles?

- `Dockerfile` = with GPU support (CUDA 12.8)
- `Dockerfile.cpu` = without GPU (lighter and faster to build)

### Why are there two docker-compose files?

- `docker-compose.yml` = main settings with GPU
- `docker-compose.cpu.yml` = overrides and disables GPU

### Why are there two requirements files?

- `requirements.txt` = with `torch==2.9.0+cu128` (requires GPU)
- `requirements-cpu.txt` = with `torch==2.9.0+cpu` (no GPU required)

***

## 🛠️ Troubleshooting

### GPU Error

```bash
# Check GPU
nvidia-smi

# If you don't have GPU, use CPU version
make up-cpu
```

### Port in use Error

```bash
# Stop all containers
make down

# Or change port in .env
API_PORT=9001
```

### Permission denied Error

```bash
# On Linux/Mac
chmod +x scripts/*.sh
sudo chown -R $USER:$USER .
```