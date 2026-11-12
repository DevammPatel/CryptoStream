# 📈 CryptoStream — Real-Time Crypto Analytics & Prediction Platform

An **end-to-end, real-time data platform** that ingests live crypto market data over
WebSockets, streams it through **Redpanda (Kafka) → Spark → a SeaweedFS data lake**, models
it with **dbt** into a DuckDB warehouse, trains a **LightGBM** price-direction model
tracked in **MLflow**, serves predictions via **FastAPI**, generates an **LLM market
summary**, and visualizes everything in a live **Streamlit** dashboard — all
orchestrated by **Dagster** and fully **Dockerized**. Runs 100% free on a laptop.

> One project that credibly demonstrates **Data Engineering, Data Analytics, Data
> Science, and AI/ML** at once.

[![CI](https://github.com/DevammPatel/CryptoStream/actions/workflows/ci.yml/badge.svg)](https://github.com/DevammPatel/CryptoStream/actions/workflows/ci.yml) ![python](https://img.shields.io/badge/python-3.11-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
Binance WS ─▶ ingestion ─▶ Redpanda ─▶ Spark ─┬─▶ Kafka live bars ──────▶ FastAPI ──▶ Streamlit
                                              │                           ▲   ▲       dashboard
                                              ▼                           │   │           ▲
                                   SeaweedFS Parquet lake                 │   │           │
                                              │                           │   │           │
                       Dagster, every 15 min  ▼                           │   │           │
                         compact, load, dbt ─▶ DuckDB ────────────────────┘   │           │
                                              │      (warehouse fallback)     │           │
                                              ├─▶ train ─▶ promotion gate ────┘           │
                                              │      └─▶ MLflow                           │
                                              └─▶ market summary (template or LLM) ───────┘
```

Full diagram, data contracts and skill-coverage table: [`docs/architecture.md`](docs/architecture.md).
Setup and troubleshooting: [`docs/development-setup.md`](docs/development-setup.md) ·
Ingestion: [`docs/ingestion-notes.md`](docs/ingestion-notes.md) ·
ML layer: [`ml/README.md`](ml/README.md) · Status and known limits: [`docs/roadmap.md`](docs/roadmap.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| Ingestion | Python `asyncio`, `websockets`, `confluent-kafka` |
| Streaming / broker | **Redpanda** (Kafka API), **Spark Structured Streaming** |
| Storage | **SeaweedFS** (S3-compatible) Parquet lake, **DuckDB** serving DB |
| Transformation | **dbt** (staging / intermediate / marts) + data-quality tests |
| ML | **LightGBM**, scikit-learn, walk-forward CV, **MLflow** |
| Serving | **FastAPI**, Pydantic |
| AI | LLM market-summary agent (offline template or OpenAI-compatible) |
| Orchestration | **Dagster** assets + schedule |
| Dashboard | **Streamlit** + Plotly |
| Infra / DevEx | Docker Compose, Makefile, GitHub Actions CI, ruff, pytest |

---

## Quickstart

### Option A — Offline demo (no Docker, ~2 minutes)

Great for a fast local run, screenshots, or CI. Uses a synthetic-data generator that
produces the exact schema the live pipeline emits.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make seed        # generate realistic synthetic OHLCV bars -> DuckDB + Parquet lake
make dbt         # build dbt models + run data-quality tests
make train       # train LightGBM, log to MLflow, save model
make api         # FastAPI at http://localhost:8000  (docs at /docs)
# in a second terminal:
make dashboard   # Streamlit at http://localhost:8501
```

The local `make` targets read `.env` when it exists, the same way Docker Compose does
(copy `.env.example` to start one).

To use real market history instead of synthetic bars (needs internet, no Docker):

```bash
make backfill-local DAYS=2   # real Binance 10s bars -> data/lake
make refresh                 # Dagster pipeline once: lake -> dbt -> model -> summary
```

### Option B — Full live stack (Docker)

Streams **live** Binance trades through the whole pipeline.

```bash
cp .env.example .env
make up           # Redpanda, SeaweedFS, Spark, Dagster, MLflow, API, dashboard, ingestion
make ps           # check services
# Redpanda Console: http://localhost:8085   SeaweedFS files: http://localhost:8888
# MLflow: http://localhost:5001             Dashboard: http://localhost:8501
# Dagster: http://localhost:3000

# Every 15 minutes Dagster builds the marts and retrains on the collected data.
# Early runs fail until there is enough data: the dashboard fills in after the first
# run that finds bars, and predictions start at about 200 rows (15 to 30 minutes).

make backfill     # optional: load 7 days of real history (about 3 minutes) so the
                  # next scheduled run trains on a week of data straight away
```

---

## What each part does

- **`ingestion/ws_producer.py`** — async Binance WebSocket client with auto-reconnect
  and exponential backoff; normalizes trades and produces to Kafka idempotently.
- **`streaming/spark_stream.py`** — Spark Structured Streaming with event-time
  watermarking; computes 10s OHLCV, VWAP, and order-flow imbalance; writes Parquet to
  the SeaweedFS lake with checkpointing, and publishes bars to Kafka every 2 seconds
  for the live view.
- **`warehouse/`** — dbt project: typed staging view, rolling technical indicators
  (SMA, volatility, relative volume) in `intermediate`, and two marts — a model-ready
  feature matrix with a forward-looking label, and a per-symbol snapshot. Includes
  generic + singular data tests.
- **`ml/train.py`** — LightGBM classifier with walk-forward CV, purged so no training
  label reaches into the test period, and probabilities calibrated on held-out data;
  logs params, metrics against simple baselines, feature importance and the model to MLflow.
  A new model replaces the served one only if it scores at least as well on data
  neither has trained on.
- **`ml/api/main.py`** — FastAPI service that hot-reloads the served model and exposes
  `/predict`, `/predict/{symbol}`, `/symbols`, `/bars/{symbol}`, `/model/info`,
  `/model/live`, `/health`.
- **`ml/monitor.py`** — the live track record: every finished live bar gets a call from
  the served model, settled 30 seconds later, counting only bars newer than its
  training data. Served at `/model/live` and shown on the dashboard.
- **`ml/live.py`** — keeps recent bars from Kafka in memory and computes the same
  indicators as dbt, so predictions use data seconds old, not the 15-minute warehouse.
- **`llm/market_agent.py`** — turns the latest snapshot and the model's call on each
  symbol into a plain-English market read; works offline, upgrades to an LLM narrative
  if a key is set. Dagster saves it after each refresh, and the dashboard only reads
  the saved copy, so the LLM runs once per refresh rather than once per page view.
- **`orchestration/`** — Dagster asset DAG (`compact lake → load → dbt → model →
  summary`) with a 15-minute refresh schedule, plus the Binance history backfill.
- **`dashboard/app.py`** — Streamlit: candlesticks + VWAP, indicators, live model call,
  and the market summary. Refreshes itself every 5 seconds.

---

## API example

```bash
curl -s localhost:8000/predict/btcusdt
# {"prob_up": 0.75, "direction": "up", "confidence": 0.51,
#  "source": "live", "as_of": "2026-01-01T12:00:00+00:00"}

curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"ret_1":0.0001,"log_ret_1":0.0001,"px_vs_sma30":0.0004,"volatility_30":0.0003,
       "order_flow_imbalance":0.2,"ofi_6":0.1,"rel_volume":1.1,"trade_count":400}'
```

---

## Testing & CI

```bash
make test        # pytest: ingestion, data, dbt parity, training and promotion, API,
                 # live track record, summary, dashboard, backfill, compaction
make test-spark  # Spark's bar logic, inside the Spark image (needs `make up` once)
make smoke       # end-to-end check of the running Docker stack
make lint        # ruff
```

GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on every push to main and
every pull request:

- **Lint and tests:** ruff, `dbt build` with its tests, pytest, then the Spark tests
  with PySpark and Java.
- **Docker stack smoke test:** builds every image, starts the stack, and runs
  `scripts/smoke_test.sh`: live Binance trades reach the API, six hours of history are
  backfilled, the Dagster pipeline runs, and the API, summary and dashboard respond.

---

## Notes on model performance

On the synthetic random-walk demo data there is nothing to learn, and the model says
so: calibrated probabilities stay within a few points of 50/50 and match the
always-predict-the-base-rate baseline.

On 7 days of real Binance bars (BTC, ETH, SOL, November 2026; about 180,000 rows) the
held-out last 20% scored:

| Metric | Model | Baseline |
|---|---|---|
| AUC | 0.525 | 0.5 |
| Accuracy | 52.7% | 52.5% (always the more common outcome) |
| Brier score (lower is better) | 0.2490 | 0.2502 (always the base rate) |

That is a small, real edge, not a trading strategy. Only 45% of labels are "up"
because a 30-second move of exactly zero counts as "not up". Training splits are purged
around the 30s label so no test-period price leaks into training, and every run logs
these baselines next to the model's numbers.
