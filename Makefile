.PHONY: help up down logs ps clean ingest-local train api dashboard dagster refresh backfill backfill-local reset-bar-state dbt test test-spark smoke lint fmt seed

# Local run targets read .env first, the way Docker Compose does. The shell sources it,
# so it holds plain KEY=value lines.
WITH_ENV = set -a; [ ! -f .env ] || . ./.env; set +a;

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
	$(WITH_ENV) python ingestion/ws_producer.py

train:  ## Train the ML model and log to MLflow
	$(WITH_ENV) python ml/train.py

api:  ## Run the FastAPI server locally
	$(WITH_ENV) uvicorn ml.api.main:app --reload --port 8000

dashboard:  ## Run the Streamlit dashboard locally
	$(WITH_ENV) streamlit run dashboard/app.py

dagster:  ## Run the Dagster UI and the 15-minute schedule
	dagster dev -m cryptostream_dagster.definitions

refresh:  ## Run the Dagster pipeline once, locally: lake -> dbt -> model -> summary
	dagster asset materialize --select '*' -m cryptostream_dagster.definitions

DAYS ?= 7
backfill:  ## Load DAYS (default 7) of real Binance history into the Docker lake
	docker compose exec dagster python -m cryptostream_dagster.backfill --days $(DAYS)

backfill-local:  ## Load DAYS of real Binance history into data/lake (use instead of `make seed`)
	$(WITH_ENV) python -m cryptostream_dagster.backfill --days $(DAYS)

LAKE_BUCKET ?= crypto-lake
FILER = http://localhost:8888/buckets/$(LAKE_BUCKET)
reset-bar-state:  ## Restart Spark's bar queries from scratch (once, after changing how bars are computed)
	docker compose stop spark
	curl -fsS -X DELETE "$(FILER)/_chk/ohlcv_10s/?recursive=true"
	curl -fsS -X DELETE "$(FILER)/features/ohlcv_10s/_spark_metadata/?recursive=true"
	docker compose run --rm --no-deps -T --entrypoint rm spark -rf /checkpoints/bars_live
	-@curl -sS -o /dev/null -X DELETE "$(FILER)/_chk/bars_live/?recursive=true"  # where it lived before
	docker compose start spark

dbt:  ## Run dbt models + tests
	$(WITH_ENV) cd warehouse && dbt build --profiles-dir .

seed:  ## Generate synthetic sample data (no network needed) for a quick demo
	$(WITH_ENV) python scripts/generate_sample_data.py

test:  ## Run unit tests
	pytest -q

smoke:  ## End-to-end check of the running Docker stack (after `make up`)
	./scripts/smoke_test.sh

test-spark:  ## Run the Spark bar tests inside the Spark image (needs Java otherwise)
	tar --no-xattrs -cf - streaming/spark_stream.py tests/test_spark.py pyproject.toml | \
	docker compose run --rm -T --no-deps --entrypoint bash spark -c \
	"mkdir -p /tmp/t && cd /tmp/t && tar -xf - && pip install -q pytest==8.2.2 && python -m pytest -p no:cacheprovider tests/test_spark.py"

lint:  ## Lint with ruff
	ruff check .

fmt:  ## Format with ruff
	ruff format .
