.PHONY: help up down logs ps clean ingest-local features train api dashboard dagster reset-bar-state dbt test lint fmt seed

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

up:  ## Start the whole stack with Docker
	docker compose up -d --build

down:  ## Stop the stack
	docker compose down

logs:  ## Tail logs for all services
	docker compose logs -f --tail=100

ps:  ## Show running services
	docker compose ps

clean:  ## Stop and remove volumes (wipes data)
	docker compose down -v

# ---- Local (no-Docker) developer workflow ----
venv:  ## Create a local virtualenv with all deps
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

ingest-local:  ## Run the ingestion producer locally (needs Redpanda up)
	python ingestion/ws_producer.py

features:  ## Build the feature/serving tables from lake parquet into DuckDB
	python ml/features.py

train:  ## Train the ML model and log to MLflow
	python ml/train.py

api:  ## Run the FastAPI server locally
	uvicorn ml.api.main:app --reload --port 8000

dashboard:  ## Run the Streamlit dashboard locally
	streamlit run dashboard/app.py

dagster:  ## Run the Dagster UI and the 15-minute schedule
	dagster dev -m cryptostream_dagster.definitions

LAKE_BUCKET ?= crypto-lake
FILER = http://localhost:8888/buckets/$(LAKE_BUCKET)
reset-bar-state:  ## Restart Spark's bar queries from scratch (once, after changing how bars are computed)
	docker compose stop spark
	curl -fsS -X DELETE "$(FILER)/_chk/ohlcv_10s/?recursive=true"
	curl -fsS -X DELETE "$(FILER)/_chk/bars_live/?recursive=true"
	curl -fsS -X DELETE "$(FILER)/features/ohlcv_10s/_spark_metadata/?recursive=true"
	docker compose start spark

dbt:  ## Run dbt models + tests
	cd warehouse && dbt build --profiles-dir .

seed:  ## Generate synthetic sample data (no network needed) for a quick demo
	python scripts/generate_sample_data.py

test:  ## Run unit tests
	pytest -q

lint:  ## Lint with ruff
	ruff check .

fmt:  ## Format with ruff
	ruff format .
