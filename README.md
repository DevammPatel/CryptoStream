# 📈 CryptoStream — Real-Time Crypto Analytics & Prediction Platform

An **end-to-end, real-time data platform** that ingests live crypto market data over
WebSockets, streams it through **Redpanda (Kafka) → Spark → a MinIO data lake**, models
it with **dbt** into a DuckDB warehouse, trains a **LightGBM** price-direction model
tracked in **MLflow**, serves predictions via **FastAPI**, generates an **LLM market
summary**, and visualizes everything in a live **Streamlit** dashboard — all
orchestrated by **Dagster** and fully **Dockerized**. Runs 100% free on a laptop.

> One project that credibly demonstrates **Data Engineering, Data Analytics, Data
> Science, and AI/ML** at once.

![CI](https://img.shields.io/badge/tests-passing-brightgreen) ![python](https://img.shields.io/badge/python-3.11-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
Binance WS ─▶ Redpanda ─▶ Spark Structured Streaming ─▶ MinIO (Parquet lake)
                                                              │
                                       Dagster ──────────────┤
                                          │                   ▼
                                          ├──▶ dbt ──▶ DuckDB (staging→marts)
                                          ├──▶ LightGBM trainer ──▶ MLflow
                                          └──▶ LLM summary agent
                                                     │
                              FastAPI /predict ◀─────┤
                                                     ▼
                                            Streamlit dashboard
```

Full diagram, data contracts and skill-coverage table: [`docs/architecture.md`](docs/architecture.md).

---

## Tech stack

| Layer | Tools |
|---|---|
| Ingestion | Python `asyncio`, `websockets`, `confluent-kafka` |
| Streaming / broker | **Redpanda** (Kafka API), **Spark Structured Streaming** |
| Storage | **MinIO** (S3-compatible) Parquet lake, **DuckDB** serving DB |
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

### Option B — Full live stack (Docker)

Streams **live** Binance trades through the whole pipeline.

```bash
cp .env.example .env
make up           # Redpanda, MinIO, Spark, MLflow, API, dashboard, ingestion
make ps           # check services
# Redpanda Console: http://localhost:8085   MinIO: http://localhost:9001
# MLflow: http://localhost:5000             Dashboard: http://localhost:8501

# Build marts + train on the collected data (and schedule retraining):
pip install -r requirements.txt
dagster dev -m cryptostream_dagster.definitions -w orchestration   # http://localhost:3000
```

---

## What each part does

- **`ingestion/ws_producer.py`** — async Binance WebSocket client with auto-reconnect
  and exponential backoff; normalizes trades and produces to Kafka idempotently.
- **`streaming/spark_stream.py`** — Spark Structured Streaming with event-time
  watermarking; computes 10s OHLCV, VWAP, and order-flow imbalance; writes Parquet to
  the MinIO lake with checkpointing.
- **`warehouse/`** — dbt project: typed staging view, rolling technical indicators
  (SMA, volatility, relative volume) in `intermediate`, and two marts — a model-ready
  feature matrix with a forward-looking label, and a per-symbol snapshot. Includes
  generic + singular data tests.
- **`ml/train.py`** — LightGBM classifier with `TimeSeriesSplit` walk-forward CV (no
  look-ahead leakage), logs params/metrics/feature-importance/model to MLflow.
- **`ml/api/main.py`** — FastAPI service that hot-reloads the latest model and exposes
  `/predict`, `/predict/{symbol}`, `/model/info`, `/health`.
- **`llm/market_agent.py`** — turns the latest snapshot + model signal into a plain-
  English market read; works offline, upgrades to an LLM narrative if a key is set.
- **`orchestration/`** — Dagster asset DAG (`lake → dbt → model → summary`) with a
  15-minute refresh schedule.
- **`dashboard/app.py`** — Streamlit: candlesticks + VWAP, indicators, live model call,
  and the market summary.

---

## API example

```bash
curl -s localhost:8000/predict/btcusdt
# {"prob_up": 0.75, "direction": "up", "confidence": 0.51}

curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"ret_1":0.001,"volatility_30":0.002,"rel_volume":1.2}'
```

---

## Testing & CI

```bash
make test    # pytest: ingestion parsing, data gen, feature build, training, inference
make lint    # ruff
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint → `dbt build` (with tests) →
pytest on every push/PR.

---

## Notes on model performance

On the synthetic random-walk demo data, AUC sits near 0.5 by design — there is no real
signal to learn, which the walk-forward CV honestly reflects. Point the live pipeline
at real Binance data and let it collect history to evaluate genuine predictive edge.
This project is engineered to be honest about model quality rather than to overfit a toy
dataset — an intentional signal of ML maturity.

---

## Resume bullet (copy/paste)

> Built an end-to-end real-time data platform ingesting live crypto trades via
> WebSockets through a Kafka (Redpanda) + Spark Structured Streaming pipeline into a
> MinIO data lake and dbt-modeled DuckDB warehouse; trained and served a LightGBM
> price-direction model (walk-forward validated, tracked in MLflow) behind a FastAPI
> API, added an LLM market-summary agent, and shipped a live Streamlit dashboard —
> orchestrated with Dagster, fully Dockerized, with CI and data-quality tests.

## License

MIT
