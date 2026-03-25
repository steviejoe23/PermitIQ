# PermitIQ — Boston Zoning Intelligence Platform
# Usage: make <target>

SHELL := /bin/zsh
VENV := zoning-env/bin/activate
PYTHON := source $(VENV) && python3

.PHONY: help run api frontend test retrain rebuild scrape update clean status

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

run: ## Start API + Frontend (two processes)
	@echo "Starting PermitIQ..."
	@$(PYTHON) -c "import fastapi" 2>/dev/null || (echo "Run: make install" && exit 1)
	@trap 'kill %1 %2 2>/dev/null' EXIT; \
	(cd api && source ../$(VENV) && uvicorn main:app --reload --port 8000) & \
	(source $(VENV) && streamlit run frontend/app.py --server.port 8501) & \
	wait

api: ## Start API only
	cd api && source ../$(VENV) && uvicorn main:app --reload --port 8000

frontend: ## Start Frontend only
	source $(VENV) && PERMITIQ_API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py --server.port 8501

test: ## Run all tests (requires API running)
	$(PYTHON) -m pytest tests/ -v --tb=short

retrain: ## Retrain ML model from existing dataset
	$(PYTHON) train_model_v2.py

retrain-clean: ## Clean OCR artifacts → audit → retrain (run after OCR completes)
	$(PYTHON) cleanup_ocr.py
	$(PYTHON) audit_ocr_quality.py
	$(PYTHON) train_model_v2.py

rebuild: ## Full pipeline: OCR → rebuild → retrain (takes hours)
	caffeinate -s $(PYTHON) overnight_rebuild.py --fresh 2>&1 | tee overnight_log.txt

scrape: ## Scrape new ZBA decision PDFs from boston.gov
	$(PYTHON) auto_scrape_decisions.py

update: ## Pull fresh data from data.boston.gov
	$(PYTHON) auto_update_data.py

audit: ## Run OCR quality audit
	$(PYTHON) audit_ocr_quality.py

docker: ## Start with Docker Compose
	docker-compose up --build -d

status: ## Show OCR progress and data freshness
	@echo "=== OCR Pipeline ==="
	@$(PYTHON) -c "import json; d=json.load(open('ocr_checkpoint.json')); print(f'  {len(d.get(\"completed\",[]))}/262 PDFs')" 2>/dev/null || echo "  Not running"
	@echo ""
	@echo "=== Data Files ==="
	@ls -lh zba_cases_cleaned.csv api/zba_model.pkl 2>/dev/null || echo "  No data files"
	@echo ""
	@echo "=== API Health ==="
	@curl -s http://127.0.0.1:8000/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  API not running"

install: ## Install Python dependencies
	source $(VENV) && pip install -r requirements.txt

push: ## Commit and push to GitHub
	git add -A && git status
	@read -p "Commit message: " msg; \
	git commit -m "$$msg" && git push origin main

db-setup: ## Create PostGIS database and import parcels
	PGPASSWORD=permitiq123 /Library/PostgreSQL/18/bin/psql -U postgres -h localhost -c "CREATE DATABASE permitiq;" 2>/dev/null; true
	PGPASSWORD=permitiq123 /Library/PostgreSQL/18/bin/psql -U postgres -h localhost -d permitiq -c "CREATE EXTENSION IF NOT EXISTS postgis;"
	@echo "Database ready. Import parcels with: make db-import"

db-import: ## Import 98K parcels from GeoJSON into PostGIS
	$(PYTHON) -c "exec(open('api/services/database.py').read()); print('PostGIS available:', db_available())"

clean: ## Remove cached/temp files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -f .pytest_cache -rf 2>/dev/null; true
