# ============================================================
# Makefile - Churn Prediction Project
# ============================================================
# 
# دستورات سریع برای مدیریت پروژه
#
# Usage:
#   make help          # نمایش همه دستورات
#   make setup         # نصب dependencies
#   make train         # آموزش مدل
#   make compare       # مقایسه مدل‌ها
#   make up            # اجرای Docker
#
# ============================================================

.PHONY: help setup install test train compare up down logs clean

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║          Churn Prediction - Available Commands           ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║ SETUP                                                    ║"
	@echo "║   make setup      - Install all dependencies             ║"
	@echo "║   make install    - Alias for setup                      ║"
	@echo "║                                                          ║"
	@echo "║ DATABASE                                                 ║"
	@echo "║   make db-up      - Start PostgreSQL                     ║"
	@echo "║   make db-load    - Load data into database              ║"
	@echo "║                                                          ║"
	@echo "║ MLOPS                                                    ║"
	@echo "║   make train      - Train baseline model                 ║"
	@echo "║   make train-prod - Train and register as production     ║"
	@echo "║   make compare    - Compare all experiments              ║"
	@echo "║   make report     - Generate comparison report           ║"
	@echo "║   make mlflow     - Start MLflow UI                      ║"
	@echo "║                                                          ║"
	@echo "║ DOCKER                                                   ║"
	@echo "║   make up         - Start all services                   ║"
	@echo "║   make down       - Stop all services                    ║"
	@echo "║   make logs       - View logs                            ║"
	@echo "║   make restart    - Restart all services                 ║"
	@echo "║                                                          ║"
	@echo "║ TESTING                                                  ║"
	@echo "║   make test       - Run all tests                        ║"
	@echo "║   make lint       - Check code style                     ║"
	@echo "║                                                          ║"
	@echo "║ CLEANUP                                                  ║"
	@echo "║   make clean      - Remove cache files                   ║"
	@echo "║   make clean-all  - Remove everything (incl. data)       ║"
	@echo "╚══════════════════════════════════════════════════════════╝"

# ============================================================
# Setup
# ============================================================

setup: install
	@echo "✅ Setup complete!"

install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# ============================================================
# Database
# ============================================================

db-up:
	@echo "🐘 Starting PostgreSQL..."
	docker-compose up -d db
	@echo "⏳ Waiting for database..."
	sleep 5
	@echo "✅ Database ready!"

db-load:
	@echo "📥 Loading data into database..."
	python db/load_data.py
	@echo "✅ Data loaded!"

db-shell:
	docker-compose exec db psql -U ds_user -d churn_db

# ============================================================
# MLOps - Training
# ============================================================

train:
	@echo "🏋️ Training baseline model..."
	python mlops/train.py --name baseline --model xgboost
	@echo "✅ Training complete!"

train-prod:
	@echo "🏋️ Training production model..."
	python mlops/train.py --name production --model xgboost --register
	@echo "✅ Production model saved!"

train-rf:
	@echo "🏋️ Training Random Forest model..."
	python mlops/train.py --name rf_experiment --model rf

train-all:
	@echo "🏋️ Training all model types..."
	python mlops/train.py --name logistic_exp --model logistic
	python mlops/train.py --name rf_exp --model rf
	python mlops/train.py --name xgb_exp --model xgboost
	python mlops/train.py --name gbm_exp --model gbm
	@echo "✅ All models trained!"

# ============================================================
# MLOps - Comparison
# ============================================================

compare:
	@echo "📊 Comparing experiments..."
	python mlops/compare.py --top 20

compare-f1:
	@echo "📊 Comparing by F1 score..."
	python mlops/compare.py --metric f1

report:
	@echo "📝 Generating report..."
	python mlops/compare.py --report
	@echo "✅ Report saved to reports/comparison_report.md"

mlflow:
	@echo "🔬 Starting MLflow UI..."
	mlflow ui --port 5000
	@echo "🌐 Open http://localhost:5000"

# ============================================================
# Docker
# ============================================================

up:
	@echo "🚀 Starting all services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   📊 Dashboard: http://localhost:8501"
	@echo "   🔧 API:       http://localhost:8000/docs"
	@echo "   🔬 MLflow:    http://localhost:5000"

down:
	@echo "🛑 Stopping services..."
	docker-compose down
	@echo "✅ Services stopped!"

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-frontend:
	docker-compose logs -f frontend

restart:
	@echo "🔄 Restarting services..."
	docker-compose restart
	@echo "✅ Services restarted!"

rebuild:
	@echo "🔨 Rebuilding containers..."
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Containers rebuilt!"

# ============================================================
# Testing
# ============================================================

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=mlops --cov-report=term-missing
	@echo "✅ Tests complete!"

test-fast:
	@echo "🧪 Running fast tests only..."
	pytest tests/ -v -m "not slow"

lint:
	@echo "🔍 Checking code style..."
	flake8 mlops/ app/ --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "✅ Lint complete!"

# ============================================================
# Jupyter
# ============================================================

notebook:
	@echo "📓 Starting Jupyter..."
	jupyter notebook notebooks/

# ============================================================
# Cleanup
# ============================================================

clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cache cleaned!"

clean-all: clean
	@echo "🧹 Cleaning everything..."
	rm -rf mlruns/ 2>/dev/null || true
	rm -rf models/*.pkl 2>/dev/null || true
	docker-compose down -v 2>/dev/null || true
	@echo "✅ All cleaned!"

# ============================================================
# Quick Workflow
# ============================================================

# Full setup from scratch
full-setup: setup db-up db-load
	@echo "✅ Full setup complete! Now run notebooks to train model."

# Quick demo
demo: db-up
	@echo "🎮 Running demo..."
	python mlops/train.py --name demo_run --model xgboost
	python mlops/compare.py
	@echo "✅ Demo complete!"