.PHONY: help install setup-db start stop test clean

# Colors for terminal output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
NC=\033[0m # No Color

help:
	@echo "$(GREEN)🛠️  Digikala Churn Prediction - Available Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@echo "  make install      - Install Python dependencies"
	@echo "  make setup-db     - Create database schema"
	@echo "  make load-data    - Load data into database"
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@echo "  make start        - Start all services (Docker Compose)"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View container logs"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make notebook     - Start Jupyter notebook"
	@echo "  make api          - Run API locally (without Docker)"
	@echo "  make test         - Run API tests"
	@echo ""
	@echo "$(YELLOW)Cleanup:$(NC)"
	@echo "  make clean        - Remove cache and generated files"
	@echo "  make clean-all    - Remove everything (including data)"

install:
	@echo "$(GREEN)📦 Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed!$(NC)"

setup-db:
	@echo "$(GREEN)💾 Setting up database...$(NC)"
	docker-compose up -d db
	sleep 5
	psql -h localhost -U ds_user -d churn_db -f db/schema.sql || true
	@echo "$(GREEN)✅ Database setup complete!$(NC)"

load-data:
	@echo "$(GREEN)📤 Loading data into database...$(NC)"
	python db/load_data.py
	@echo "$(GREEN)✅ Data loaded!$(NC)"

start:
	@echo "$(GREEN)🚀 Starting services...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Services started!$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Database: localhost:5432$(NC)"

stop:
	@echo "$(RED)🛑 Stopping services...$(NC)"
	docker-compose down
	@echo "$(GREEN)✅ Services stopped!$(NC)"

restart: stop start

logs:
	@echo "$(GREEN)📜 Showing logs...$(NC)"
	docker-compose logs -f

notebook:
	@echo "$(GREEN)📓 Starting Jupyter notebook...$(NC)"
	jupyter notebook notebooks/

api:
	@echo "$(GREEN)🚀 Starting API locally...$(NC)"
	cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

test:
	@echo "$(GREEN)🧪 Running API tests...$(NC)"
	python scripts/test_api.py

clean:
	@echo "$(YELLOW)🧹 Cleaning cache files...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ Cache cleaned!$(NC)"

clean-all: clean
	@echo "$(RED)⚠️  WARNING: This will delete all data and models!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/*.csv app/*.pkl reports/*.png; \
		docker-compose down -v; \
		echo "$(GREEN)✅ Everything cleaned!$(NC)"; \
	else \
		echo "$(YELLOW)Cancelled.$(NC)"; \
	fi

# Quick setup for first time users
quickstart: install setup-db start
	@echo "$(GREEN)🎉 Quickstart complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Place CSV files in data/ folder"
	@echo "  2. Run: make load-data"
	@echo "  3. Run notebooks to train model"
	@echo "  4. Test API: make test"
