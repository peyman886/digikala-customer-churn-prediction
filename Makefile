# ============================================================
# Makefile - Churn Prediction with MLOps
# ============================================================

.PHONY: help setup mlflow compare promote up down clean

# Default
help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════════╗"
	@echo "║          Churn Prediction - Quick Commands               ║"
	@echo "╠══════════════════════════════════════════════════════════╣"
	@echo "║                                                          ║"
	@echo "║  SETUP                                                   ║"
	@echo "║    make setup         Install dependencies               ║"
	@echo "║                                                          ║"
	@echo "║  MLOPS                                                   ║"
	@echo "║    make mlflow        Start MLflow UI (localhost:5000)   ║"
	@echo "║    make compare       Compare all experiments            ║"
	@echo "║    make promote       Promote best model to production   ║"
	@echo "║                                                          ║"
	@echo "║  DOCKER                                                  ║"
	@echo "║    make up            Start all services                 ║"
	@echo "║    make up-dev        Start with Jupyter + PgAdmin       ║"
	@echo "║    make down          Stop all services                  ║"
	@echo "║    make logs          View logs                          ║"
	@echo "║                                                          ║"
	@echo "║  DATABASE                                                ║"
	@echo "║    make db-up         Start PostgreSQL only              ║"
	@echo "║    make db-load       Load data into database            ║"
	@echo "║                                                          ║"
	@echo "║  CLEANUP                                                 ║"
	@echo "║    make clean         Remove cache files                 ║"
	@echo "║                                                          ║"
	@echo "╚══════════════════════════════════════════════════════════╝"
	@echo ""

# ============================================================
# Setup
# ============================================================

setup:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Setup complete!"

# ============================================================
# MLOps
# ============================================================

mlflow:
	@echo "🔬 Starting MLflow UI..."
	@echo "   Open: http://localhost:5000"
	mlflow ui --port 5000 --backend-store-uri ./mlruns

compare:
	@echo "📊 Comparing experiments..."
	python mlops/compare.py

compare-f1:
	@echo "📊 Comparing by F1 score..."
	python mlops/compare.py --metric f1

promote:
	@echo "🚀 Promoting best model to production..."
	python mlops/compare.py --promote best

promote-run:
	@echo "🚀 Promote specific run: make promote-run RUN=run_name"
	python mlops/compare.py --promote $(RUN)

report:
	@echo "📝 Generating comparison report..."
	python mlops/compare.py --report

# ============================================================
# Docker
# ============================================================

up:
	@echo "🚀 Starting services..."
	docker-compose up -d
	@echo ""
	@echo "✅ Services started!"
	@echo "   🔧 API:      http://localhost:8000/docs"
	@echo "   🔬 MLflow:   http://localhost:5000"
	@echo ""

up-dev:
	@echo "🚀 Starting services with dev tools..."
	docker-compose --profile dev up -d
	@echo ""
	@echo "✅ Services started!"
	@echo "   🔧 API:      http://localhost:8000/docs"
	@echo "   🔬 MLflow:   http://localhost:5000"
	@echo "   📓 Jupyter:  http://localhost:8888 (token: churn123)"
	@echo "   🐘 PgAdmin:  http://localhost:5050"
	@echo ""

down:
	@echo "🛑 Stopping services..."
	docker-compose down
	@echo "✅ Services stopped!"

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-mlflow:
	docker-compose logs -f mlflow

restart:
	docker-compose restart

# ============================================================
# Database
# ============================================================

db-up:
	@echo "🐘 Starting PostgreSQL..."
	docker-compose up -d db
	@sleep 5
	@echo "✅ Database ready!"

db-load:
	@echo "📥 Loading data into database..."
	python db/load_data.py
	@echo "✅ Data loaded!"

# ============================================================
# Testing
# ============================================================

test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=mlops --cov-report=term-missing

# ============================================================
# Cleanup
# ============================================================

clean:
	@echo "🧹 Cleaning cache files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned!"

clean-mlflow:
	@echo "🧹 Cleaning MLflow runs..."
	rm -rf mlruns/ 2>/dev/null || true
	@echo "✅ MLflow data cleaned!"

clean-models:
	@echo "🧹 Cleaning saved models..."
	rm -rf models/experiments/*.pkl 2>/dev/null || true
	@echo "✅ Experiment models cleaned!"

# ============================================================
# Quick Workflows
# ============================================================

# Full dev setup
dev-setup: setup db-up up-dev
	@echo "✅ Development environment ready!"

# Quick demo
demo:
	@echo "🎮 Running demo experiment..."
	python -c "from mlops import ExperimentTracker; print('MLOps module OK!')"
	@echo "✅ Demo complete!"