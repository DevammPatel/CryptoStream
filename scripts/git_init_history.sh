#!/usr/bin/env bash
#
# Build a clean, meaningful git history for CryptoStream.
#
# Replays the project as ~30 logical commits in dependency order (infra -> ingestion
# -> streaming -> warehouse -> ml -> api -> orchestration -> llm -> dashboard -> tests
# -> ci -> docs -> refinements), each with a conventional-commits message.
#
# Usage:
#   cd cryptostream
#   bash scripts/git_init_history.sh                 # commit all-at-once (today)
#   SPREAD_DAYS=21 bash scripts/git_init_history.sh  # spread commits over the last 21 days
#
# Optional env:
#   GIT_NAME / GIT_EMAIL  -> set the author identity for this repo (local, not global)
#   SPREAD_DAYS           -> if set (>0), back-dates commits evenly across that many days
#
set -euo pipefail

# --- must run from the project root ---
if [[ ! -f "docker-compose.yml" || ! -d "warehouse" ]]; then
  echo "ERROR: run this from the cryptostream/ project root." >&2
  exit 1
fi

# --- init repo if needed ---
if [[ ! -d .git ]]; then
  git init -q
  git checkout -q -b main 2>/dev/null || git branch -q -M main
fi

# --- identity (optional, local to this repo) ---
if [[ -n "${GIT_NAME:-}" ]]; then git config user.name  "$GIT_NAME";  fi
if [[ -n "${GIT_EMAIL:-}" ]]; then git config user.email "$GIT_EMAIL"; fi

# --- back-dating support ---
SPREAD_DAYS="${SPREAD_DAYS:-0}"
TOTAL_COMMITS=32
COMMIT_IDX=0

_date_for_commit() {
  # Prints an ISO date for commit N so history is spread across SPREAD_DAYS.
  if [[ "$SPREAD_DAYS" -le 0 ]]; then return 0; fi
  local offset_days
  # oldest commit first: earliest date for idx 0
  offset_days=$(( SPREAD_DAYS - (COMMIT_IDX * SPREAD_DAYS / TOTAL_COMMITS) ))
  # portable-ish: try GNU date, fall back to BSD/macOS date
  if date -d "-${offset_days} days" +%Y-%m-%dT%H:%M:%S 2>/dev/null; then
    return 0
  fi
  date -v-"${offset_days}"d +%Y-%m-%dT%H:%M:%S
}

# commit <message> <path...>
commit() {
  local msg="$1"; shift
  git add -- "$@"
  if git diff --cached --quiet; then
    echo "  (nothing staged for: $msg) — skipping"
    return 0
  fi
  local d; d="$(_date_for_commit || true)"
  if [[ -n "$d" ]]; then
    GIT_AUTHOR_DATE="$d" GIT_COMMITTER_DATE="$d" git commit -q -m "$msg"
  else
    git commit -q -m "$msg"
  fi
  COMMIT_IDX=$(( COMMIT_IDX + 1 ))
  printf '  [%02d] %s\n' "$COMMIT_IDX" "$msg"
}

echo "Creating commit history..."

# 1. project bootstrap
commit "chore: initialize repository with gitignore and MIT license" .gitignore LICENSE
commit "build: add Makefile with dev and docker workflows" Makefile
commit "build: add umbrella requirements and ruff/pytest config" requirements.txt pyproject.toml
commit "build: add docker-compose for local stack (redpanda, minio, mlflow, api, dashboard)" docker-compose.yml
commit "docs: add environment variable template" .env.example

# 2. ingestion
commit "feat(ingestion): add Binance websocket to Kafka trade producer" ingestion/ws_producer.py
commit "build(ingestion): containerize the ingestion service" ingestion/Dockerfile ingestion/requirements.txt

# 3. streaming
commit "feat(streaming): add Spark Structured Streaming OHLCV and order-flow features" streaming/spark_stream.py
commit "build(streaming): add Spark image with Kafka and S3A connectors" streaming/Dockerfile

# 4. warehouse / dbt
commit "feat(warehouse): scaffold dbt project config and DuckDB profile" warehouse/dbt_project.yml warehouse/profiles.yml
commit "feat(warehouse): declare raw streaming source with tests" warehouse/models/staging/_sources.yml
commit "feat(warehouse): add typed OHLCV staging model" warehouse/models/staging/stg_ohlcv.sql
commit "feat(warehouse): add rolling technical-indicator model" warehouse/models/intermediate/int_technical_features.sql
commit "feat(warehouse): add model-ready feature mart with forward label" warehouse/models/marts/mart_ml_features.sql
commit "feat(warehouse): add per-symbol snapshot mart for dashboard and agent" warehouse/models/marts/mart_symbol_snapshot.sql
commit "test(warehouse): add mart schema tests and singular data-quality test" warehouse/models/marts/_marts.yml warehouse/tests/assert_positive_prices.sql
commit "feat(warehouse): use verbatim schema names via generate_schema_name macro" warehouse/macros/generate_schema_name.sql

# 5. sample data
commit "feat(data): add offline synthetic OHLCV generator for demo and CI" scripts/generate_sample_data.py

# 6. ml
commit "feat(ml): add shared feature loading with dbt-mart fallback" ml/features.py ml/__init__.py
commit "feat(ml): add LightGBM trainer with walk-forward CV and MLflow logging" ml/train.py
commit "feat(api): add FastAPI serving with hot model reload" ml/api/main.py ml/api/__init__.py
commit "build(api): containerize the prediction API" ml/Dockerfile ml/requirements.txt

# 7. orchestration
commit "feat(orchestration): add Dagster asset DAG and refresh schedule" orchestration/cryptostream_dagster/definitions.py orchestration/cryptostream_dagster/__init__.py orchestration/pyproject.toml

# 8. llm
commit "feat(llm): add market-summary agent with offline and LLM modes" llm/market_agent.py llm/__init__.py

# 9. dashboard
commit "feat(dashboard): add Streamlit live analytics and prediction dashboard" dashboard/app.py
commit "build(dashboard): containerize the Streamlit dashboard" dashboard/Dockerfile dashboard/requirements.txt

# 10. tests + ci
commit "test: add ingestion, feature, training and inference tests" tests/test_pipeline.py
commit "ci: add GitHub Actions pipeline (lint, dbt build, pytest)" .github/workflows/ci.yml

# 11. docs
commit "docs: add architecture diagram, data contracts and skill matrix" docs/architecture.md
commit "docs: add project README with quickstart and resume bullet" README.md

# 12. anything not yet tracked (safety net) + this script
commit "chore: add git history helper script" scripts/git_init_history.sh
git add -A
if ! git diff --cached --quiet; then
  git commit -q -m "chore: track remaining project files"
  echo "  [+] chore: track remaining project files"
fi

echo
echo "Done. $(git rev-list --count HEAD) commits on branch main."
echo
echo "Next:"
echo "  git log --oneline"
echo "  git remote add origin git@github.com:<you>/cryptostream.git"
echo "  git push -u origin main"
