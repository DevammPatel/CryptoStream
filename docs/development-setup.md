# Development setup

Two ways to work on CryptoStream: a Python environment on your machine (fast, no
Docker), and the full Docker stack (live data end to end). Most changes need only the
first; anything touching Kafka, Spark or SeaweedFS needs the second.

## Prerequisites

| Tool | Version | Needed for |
|---|---|---|
| Python | 3.11 | everything outside Docker |
| Docker Desktop, or Docker Engine with Compose v2 | recent | the live stack (it runs in about 4 GB of memory) |
| Internet access | | live trades and the history backfill (Binance's public endpoints, no key) |

Java is only needed to run the Spark tests outside Docker; `make test-spark` runs them
inside the Spark image instead.

## Python environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # also installs the Dagster package in editable mode
```

`requirements.txt` pins every package for local work and CI. Each Docker image has its
own, smaller pinned list next to its Dockerfile.

## Settings (`.env`)

Copy `.env.example` to `.env` to change symbols, topics, the Binance endpoint or the LLM
key. Docker Compose reads it, and so do the local `make` targets (`make api`,
`make train`, ...). Keep it to plain `KEY=value` lines. It is git-ignored; never commit
keys.

## Offline loop (no Docker)

```bash
make seed        # synthetic bars -> data/lake and data/warehouse/crypto.duckdb
make dbt         # dbt models + data tests
make train       # train, compare with the served model, save to ml/models/
make api         # http://localhost:8000/docs
make dashboard   # http://localhost:8501
```

For real prices instead of synthetic ones: `make backfill-local DAYS=2`, then
`make refresh` (the whole Dagster pipeline once).

## Docker stack

```bash
make up      # build and start everything
make ps      # status
make logs    # follow all logs
make smoke   # end-to-end check (what CI runs)
make down    # stop; keeps all data
make clean   # stop and delete all data volumes
```

| Service | URL |
|---|---|
| Dashboard (Streamlit) | http://localhost:8501 |
| API (FastAPI) | http://localhost:8000/docs |
| Dagster | http://localhost:3000 |
| MLflow | http://localhost:5001 (5000 is taken by AirPlay on macOS) |
| Redpanda Console | http://localhost:8085 |
| SeaweedFS file browser | http://localhost:8888 |
| Kafka, from your machine | localhost:9092 |
| S3, from your machine | http://localhost:8333 (key and secret `lakeadmin`) |

A fresh stack streams live trades straight away, but the first useful model needs
history: `make backfill` loads 7 days in a few minutes, and the next Dagster run (every
15 minutes, or "Materialize all" in the Dagster UI) trains on it.

### On a laptop

The stack needs about 4 GB of memory, with Docker Desktop's default of half the RAM on
an 8 GB machine. Spark and Dagster, the heavy services, are limited to 2 cores and 2 GB
each, and only one Dagster run trains at a time. To keep a laptop cool:

- Set `REFRESH_CRON="0 * * * *"` in `.env` to retrain hourly instead of every 15
  minutes.
- Lower the CPUs in Docker Desktop (Settings > Resources) if it still runs hot.
- `make down` when you're done; it keeps all data.

## Tests

```bash
make test        # pytest; never touches data/ or ml/models/
make test-spark  # Spark's bar logic, inside the Spark image
make lint        # ruff
```

CI runs the same tests, plus a job that builds every image, starts the stack and runs
`scripts/smoke_test.sh`.

## When something looks wrong

| Symptom | Cause and fix |
|---|---|
| Ingestion exits with HTTP 451 | Binance blocks that endpoint in your region. Set `BINANCE_WS_BASE` in `.env` to another endpoint. |
| Spark restarts in a loop after you changed how bars are computed | Its saved state no longer matches the code. `make reset-bar-state` clears just the bar queries. |
| Gaps in the charts after the laptop slept | Docker pauses with the machine, so no trades were collected. Windows and labels are time-based, so a gap never joins prices from either side; `make backfill` fills it. |
| `make train` can't reach MLflow | `MLFLOW_TRACKING_URI` in `.env` points at the Docker MLflow server, which isn't running. Start the stack, or comment the line out to log to `./mlruns`. |
| Dashboard says Dagster hasn't saved a summary yet | Normal until the first pipeline run; it shows the offline template meanwhile. |
| Docker's disk keeps growing | Check `docker system df`. Spark copies its jars (about 340 MB) into `/tmp` on every start and deletes them when it stops; a crash leaves the copy until Spark's next start clears it. Old build cache goes with `docker builder prune`. |
