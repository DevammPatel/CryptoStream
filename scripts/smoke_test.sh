#!/usr/bin/env bash
# End-to-end check of the running Docker stack (`make up`), used by CI.
#
#   1. Live trades flow: Binance -> ingestion -> Kafka -> Spark -> the API's live bars.
#   2. Six hours of real history are backfilled into the lake.
#   3. The Dagster pipeline runs: compaction, load, dbt with its tests, training, summary.
#   4. The API serves a model, predicts from live bars and reports its track record,
#      the summary is saved, and the dashboard answers.
#
# It only adds data (a few hours of backfill and one pipeline run), so it is safe to
# run on a stack you use: `make smoke`.
set -euo pipefail

API=${API_URL:-http://localhost:8000}
DASHBOARD=${DASHBOARD_URL:-http://localhost:8501}

# wait_for <description> <seconds> <command...>: retry every 5s until it succeeds.
wait_for() {
  local what=$1 limit=$2
  shift 2
  local start=$SECONDS
  until "$@" >/dev/null 2>&1; do
    if ((SECONDS - start > limit)); then
      echo "FAIL: $what (gave up after ${limit}s)"
      return 1
    fi
    sleep 5
  done
  echo "ok:   $what ($((SECONDS - start))s)"
}

wait_for "the API answers" 300 curl -fsS "$API/health"
wait_for "live bars reach the API" 600 sh -c "curl -fsS $API/symbols | grep -q btcusdt"

docker compose exec -T dagster python -m cryptostream_dagster.backfill --days 0.25
echo "ok:   backfilled six hours"
docker compose exec -T dagster dagster asset materialize --select '*' -m cryptostream_dagster.definitions
echo "ok:   the pipeline ran"

curl -fsS "$API/model/info" | grep -q '"promotion"'
echo "ok:   a model is served"
# A live call needs 31 finished bars (about 5 minutes of stream).
wait_for "a prediction from live bars" 600 sh -c "curl -fsS $API/predict/btcusdt | grep -q '\"source\":\"live\"'"
curl -fsS "$API/model/live" | grep -q '"last_24h_all_models"'
echo "ok:   the live track record is served"
docker compose exec -T dagster test -s /data/warehouse/market_summary.json
echo "ok:   the market summary is saved"
wait_for "the dashboard answers" 120 curl -fsS "$DASHBOARD/_stcore/health"

echo "smoke test passed"
