.PHONY: setup data train ingest up down logs clean

## 1. Install local Python deps (for development outside Docker)
setup:
	pip install -r model-server/requirements.txt
	pip install -r api/requirements.txt
	pip install -r ui/requirements.txt

## 2. Download and preprocess MovieLens 100K
data:
	python data/download.py

## 3. Ingest a custom dataset via the running model-server API
##    Usage: make ingest RATINGS=path/to/ratings.csv ITEMS=path/to/items.csv CONFIG=path/to/config.json
ingest:
	curl -X POST http://localhost:8001/upload \
	  -F "ratings_file=@$(RATINGS)" \
	  -F "items_file=@$(ITEMS)" \
	  -F "config_json=$$(cat $(CONFIG))"

## 4. Train all models (requires preprocessed data)
train:
	cd model-server && python train.py

## 4. Build and start all Docker services
up:
	docker compose up --build -d
	@echo ""
	@echo "Services:"
	@echo "  Model Server → http://localhost:8001/docs"
	@echo "  API Gateway  → http://localhost:8000/docs"
	@echo "  Dashboard    → http://localhost:8501"

## 5. Stop all services
down:
	docker compose down

## 6. Follow logs
logs:
	docker compose logs -f

## 7. Remove containers, volumes, and model artefacts
clean:
	docker compose down -v
	rm -rf data/raw data/processed data/models

## Quick local run (no Docker) – requires `make setup && make data && make train` first
local-api:
	cd api && uvicorn main:app --reload --port 8000

local-model-server:
	cd model-server && uvicorn main:app --reload --port 8001

local-ui:
	cd ui && streamlit run app.py --server.port 8501
