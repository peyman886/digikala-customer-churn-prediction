# =============================================================================
# Makefile - Churn Prediction with MLOps
# =============================================================================
# Usage: make <target>
# Run 'make' or 'make help' for available commands
# =============================================================================

.PHONY: help setup setup-cpu up up-cpu dev dev-cpu down logs build test clean

# Colors for output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
CYAN   := \033[0;36m
RED    := \033[0;31m
NC     := \033[0m

# Default target
help:
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════════╗$(NC)"
	@echo "$(CYAN)║$(NC)       $(GREEN)Churn Prediction System - Command Reference$(NC)          $(CYAN)║$(NC)"
	@echo "$(CYAN)╠══════════════════════════════════════════════════════════════╣$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)SETUP$(NC)                                                       $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make setup          Install GPU dependencies               $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make setup-cpu      Install CPU-only dependencies          $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)DOCKER - PRODUCTION$(NC)                                         $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make up             Start services (GPU)                   $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make up-cpu         Start services (CPU only)              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make down           Stop all services                      $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make restart        Restart all services                   $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make logs           View all logs                          $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make logs-api       View API logs only                     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)DOCKER - DEVELOPMENT$(NC)                                        $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make dev            Start with Jupyter + PgAdmin (GPU)     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make dev-cpu        Start with Jupyter + PgAdmin (CPU)     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)DATABASE$(NC)                                                    $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make db-up          Start PostgreSQL only                  $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make db-load        Load data from CSV files               $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make db-shell       Open psql shell                        $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make db-reset       Reset database (DESTRUCTIVE!)          $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)MLOPS$(NC)                                                       $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make mlflow         Start MLflow UI                        $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make compare        Compare all experiments                $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make promote        Promote best model                     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make report         Generate comparison report             $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)TESTING$(NC)                                                     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make test           Run all tests                          $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make test-cov       Run tests with coverage                $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make test-api       Test API endpoints                     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)CODE QUALITY$(NC)                                                $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make lint           Run flake8 linter                      $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make format         Format code (black + isort)            $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)GPU$(NC)                                                         $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make gpu-check      Check GPU availability                 $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make nvidia-smi     Run nvidia-smi in container            $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)CLEANUP$(NC)                                                     $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make clean          Remove Python cache files              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make clean-docker   Clean Docker resources                 $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make clean-all      Clean everything                       $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)  $(YELLOW)BUILD$(NC)                                                       $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make build          Build Docker images                    $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)    make build-nocache  Build without cache                    $(CYAN)║$(NC)"
	@echo "$(CYAN)║$(NC)                                                              $(CYAN)║$(NC)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════════╝$(NC)"
	@echo ""

# =============================================================================
# Setup
# =============================================================================

setup:
	@echo "$(GREEN)📦 Installing GPU dependencies...$(NC)"
	pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
	@echo "$(GREEN)✅ Setup complete!$(NC)"

setup-cpu:
	@echo "$(GREEN)📦 Installing CPU dependencies...$(NC)"
	pip install -r requirements-cpu.txt
	@echo "$(GREEN)✅ Setup complete!$(NC)"

# =============================================================================
# Docker - Production
# =============================================================================

up:
	@echo "$(GREEN)🚀 Starting services (GPU)...$(NC)"
	docker-compose up -d
	@echo ""
	@echo "$(GREEN)✅ Services started!$(NC)"
	@echo "   $(CYAN)🔧 API:$(NC)       http://localhost:9000/docs"
	@echo "   $(CYAN)🌐 Frontend:$(NC)  http://localhost:8501"
	@echo "   $(CYAN)🔬 MLflow:$(NC)    http://localhost:5000"
	@echo ""

up-cpu:
	@echo "$(GREEN)🚀 Starting services (CPU)...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
	@echo ""
	@echo "$(GREEN)✅ Services started!$(NC)"
	@echo "   $(CYAN)🔧 API:$(NC)       http://localhost:9000/docs"
	@echo "   $(CYAN)🌐 Frontend:$(NC)  http://localhost:8501"
	@echo "   $(CYAN)🔬 MLflow:$(NC)    http://localhost:5000"
	@echo ""

down:
	@echo "$(YELLOW)🛑 Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Services stopped!$(NC)"

restart:
	@echo "$(YELLOW)🔄 Restarting services...$(NC)"
	docker-compose restart
	@echo "$(GREEN)✅ Services restarted!$(NC)"

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-frontend:
	docker-compose logs -f frontend

# =============================================================================
# Docker - Development
# =============================================================================

dev:
	@echo "$(GREEN)🚀 Starting development environment (GPU)...$(NC)"
	docker-compose --profile dev up -d
	@echo ""
	@echo "$(GREEN)✅ Development environment ready!$(NC)"
	@echo "   $(CYAN)🔧 API:$(NC)       http://localhost:9000/docs"
	@echo "   $(CYAN)🌐 Frontend:$(NC)  http://localhost:8501"
	@echo "   $(CYAN)🔬 MLflow:$(NC)    http://localhost:5000"
	@echo "   $(CYAN)📓 Jupyter:$(NC)   http://localhost:8888 (token: churn123)"
	@echo "   $(CYAN)🐘 PgAdmin:$(NC)   http://localhost:5050"
	@echo ""

dev-cpu:
	@echo "$(GREEN)🚀 Starting development environment (CPU)...$(NC)"
	docker-compose -f docker-compose.yml -f docker-compose.cpu.yml --profile dev up -d
	@echo ""
	@echo "$(GREEN)✅ Development environment ready!$(NC)"
	@echo "   $(CYAN)🔧 API:$(NC)       http://localhost:9000/docs"
	@echo "   $(CYAN)🌐 Frontend:$(NC)  http://localhost:8501"
	@echo "   $(CYAN)🔬 MLflow:$(NC)    http://localhost:5000"
	@echo "   $(CYAN)📓 Jupyter:$(NC)   http://localhost:8888 (token: churn123)"
	@echo "   $(CYAN)🐘 PgAdmin:$(NC)   http://localhost:5050"
	@echo ""

# =============================================================================
# Database
# =============================================================================

db-up:
	@echo "$(GREEN)🐘 Starting PostgreSQL...$(NC)"
	docker-compose up -d db
	@sleep 5
	@echo "$(GREEN)✅ Database ready!$(NC)"

db-load:
	@echo "$(GREEN)📥 Loading data into database...$(NC)"
	python db/load_data.py
	@echo "$(GREEN)✅ Data loaded!$(NC)"

db-shell:
	@echo "$(CYAN)🐘 Opening PostgreSQL shell...$(NC)"
	docker-compose exec db psql -U ds_user -d churn_db

db-reset:
	@echo "$(RED)⚠️  WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ]
	docker-compose down -v
	docker-compose up -d db
	@sleep 5
	@echo "$(GREEN)✅ Database reset complete!$(NC)"

# =============================================================================
# MLOps
# =============================================================================

mlflow:
	@echo "$(GREEN)🔬 Starting MLflow UI...$(NC)"
	@echo "   Open: http://localhost:5000"
	mlflow ui --port 5000 --backend-store-uri ./mlruns

compare:
	@echo "$(GREEN)📊 Comparing experiments...$(NC)"
	python mlops/compare.py

compare-f1:
	@echo "$(GREEN)📊 Comparing by F1 score...$(NC)"
	python mlops/compare.py --metric f1

promote:
	@echo "$(GREEN)🚀 Promoting best model to production...$(NC)"
	python mlops/compare.py --promote best

report:
	@echo "$(GREEN)📝 Generating comparison report...$(NC)"
	python mlops/compare.py --report

# =============================================================================
# Testing
# =============================================================================

test:
	@echo "$(GREEN)🧪 Running tests...$(NC)"
	pytest tests/ -v

test-cov:
	@echo "$(GREEN)🧪 Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=app --cov=mlops --cov=src --cov-report=term-missing

test-api:
	@echo "$(GREEN)🧪 Testing API endpoints...$(NC)"
	python scripts/test_api.py

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "$(GREEN)🔍 Running linter...$(NC)"
	flake8 app/ mlops/ src/ tests/ --max-line-length=120

format:
	@echo "$(GREEN)🎨 Formatting code...$(NC)"
	black app/ mlops/ src/ tests/ --line-length=120
	isort app/ mlops/ src/ tests/ --profile=black --line-length=120
	@echo "$(GREEN)✅ Code formatted!$(NC)"

# =============================================================================
# GPU
# =============================================================================

gpu-check:
	@echo "$(GREEN)🔍 Checking GPU availability...$(NC)"
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "$(RED)PyTorch not installed or no GPU available$(NC)"

nvidia-smi:
	@echo "$(GREEN)🔍 Running nvidia-smi in container...$(NC)"
	docker-compose exec api nvidia-smi 2>/dev/null || docker run --rm --gpus all nvidia/cuda:12.8.0-runtime-ubuntu22.04 nvidia-smi

# =============================================================================
# Cleanup
# =============================================================================

clean:
	@echo "$(YELLOW)🧹 Cleaning cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✅ Cleaned!$(NC)"

clean-docker:
	@echo "$(YELLOW)🧹 Cleaning Docker resources...$(NC)"
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "$(GREEN)✅ Docker cleaned!$(NC)"

clean-mlflow:
	@echo "$(YELLOW)🧹 Cleaning MLflow runs...$(NC)"
	rm -rf mlruns/ 2>/dev/null || true
	@echo "$(GREEN)✅ MLflow data cleaned!$(NC)"

clean-all: clean clean-docker clean-mlflow
	@echo "$(GREEN)✅ Everything cleaned!$(NC)"

# =============================================================================
# Build
# =============================================================================

build:
	@echo "$(GREEN)🔨 Building Docker images...$(NC)"
	docker-compose build
	@echo "$(GREEN)✅ Build complete!$(NC)"

build-nocache:
	@echo "$(GREEN)🔨 Building Docker images (no cache)...$(NC)"
	docker-compose build --no-cache
	@echo "$(GREEN)✅ Build complete!$(NC)"

# =============================================================================
# Quick Workflows
# =============================================================================

init: setup db-up db-load
	@echo "$(GREEN)✅ Project initialized!$(NC)"

init-cpu: setup-cpu db-up db-load
	@echo "$(GREEN)✅ Project initialized (CPU)!$(NC)"

demo:
	@echo "$(GREEN)🎮 Running quick demo...$(NC)"
	python -c "import torch; print(f'PyTorch {torch.__version__}')"
	python -c "from mlops import ExperimentTracker; print('MLOps module OK!')"
	@echo "$(GREEN)✅ Demo complete!$(NC)"

deploy: build up
	@echo "$(GREEN)✅ Deployment complete!$(NC)"

ci: lint test
	@echo "$(GREEN)✅ CI checks passed!$(NC)"
