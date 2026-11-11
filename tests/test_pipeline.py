"""Unit / integration tests for the CryptoStream pipeline.

Covered:
  - ingestion.normalise maps Binance payloads to the canonical schema
  - synthetic data generation + DuckDB round-trip
  - feature loading + train/label construction
  - model training produces a usable classifier and metadata
  - Dagster definitions load and the refresh schedule is on
  - the warehouse swap keeps views working and never blocks a reader
  - live bars: indicators match dbt, partial and stale bars are ignored, Kafka consumer

Run:  pytest -q
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "ml"))
sys.path.insert(0, os.path.join(REPO, "ingestion"))
sys.path.insert(0, os.path.join(REPO, "orchestration"))


def test_normalise_trade():
    from ingestion.ws_producer import normalise

    raw = {
        "stream": "btcusdt@trade",
        "data": {"e": "trade", "E": 1, "s": "BTCUSDT", "t": 42, "p": "62000.5",
                 "q": "0.01", "T": 1700000000000, "m": False},
    }
    out = normalise(raw)
    assert out["symbol"] == "btcusdt"
    assert out["price"] == pytest.approx(62000.5)
    assert out["trade_id"] == 42
    assert out["is_buyer_maker"] is False


def test_normalise_ignores_non_trade():
    from ingestion.ws_producer import normalise

    assert normalise({"data": {"e": "depthUpdate"}}) is None


@pytest.fixture(scope="session")
def sample_data(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("wh") / "crypto.duckdb")
    os.environ["DUCKDB_PATH"] = db_path
    subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", "generate_sample_data.py"),
         "--minutes", "120", "--symbols", "btcusdt,ethusdt"],
        check=True, cwd=REPO, env=dict(os.environ, DUCKDB_PATH=db_path),
    )
    return db_path


def test_sample_data_created(sample_data):
    import duckdb

    con = duckdb.connect(sample_data, read_only=True)
    n = con.execute("select count(*) from raw.ohlcv_10s").fetchone()[0]
    con.close()
    assert n > 500


def test_feature_frame(sample_data):
    from features import FEATURE_COLUMNS, build_xy, load_training_frame

    df = load_training_frame()
    assert len(df) > 100
    X, y = build_xy(df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert set(y.unique()).issubset({0, 1})
    assert not X.isnull().any().any()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp("models"))


def test_training_runs(sample_data, model_dir, tmp_path):
    env = dict(os.environ, MODEL_DIR=model_dir, MLFLOW_TRACKING_URI=f"file://{tmp_path/'mlruns'}")
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "ml", "train.py")],
        cwd=os.path.join(REPO, "ml"), env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(os.path.join(model_dir, "direction_lgbm.pkl"))
    assert os.path.exists(os.path.join(model_dir, "model_meta.json"))


def test_prediction_from_model(sample_data, model_dir):
    import json

    import joblib
    import numpy as np

    meta = json.load(open(os.path.join(model_dir, "model_meta.json")))
    assert "metrics" in meta and "test_accuracy" in meta["metrics"]

    payload = joblib.load(os.path.join(model_dir, "direction_lgbm.pkl"))
    model, feats = payload["model"], payload["features"]
    proba = model.predict_proba(np.zeros((1, len(feats))))[:, 1][0]
    assert 0.0 <= proba <= 1.0


def test_dagster_definitions_load():
    from cryptostream_dagster.definitions import defs, refresh_schedule
    from dagster import DefaultScheduleStatus

    assert defs.get_job_def("refresh_pipeline") is not None
    assert refresh_schedule.default_status == DefaultScheduleStatus.RUNNING


def test_warehouse_swap_keeps_views_and_readers(sample_data, tmp_path, monkeypatch):
    import duckdb
    from cryptostream_dagster.definitions import dbt_models, lake_to_duckdb
    from dagster import materialize

    served = tmp_path / "crypto.duckdb"
    monkeypatch.setenv("DUCKDB_PATH", str(served))
    assert materialize([lake_to_duckdb, dbt_models]).success

    # A reader holding the old file must not block the next build, and vice versa.
    reader = duckdb.connect(str(served), read_only=True)
    assert materialize([lake_to_duckdb, dbt_models]).success
    reader.close()

    con = duckdb.connect(str(served), read_only=True)
    assert con.execute("select count(*) from intermediate.int_technical_features").fetchone()[0] > 0
    assert con.execute("select count(*) from marts.mart_ml_features").fetchone()[0] > 0
    con.close()


def make_bars(n, end_ms, symbol="btcusdt"):
    """n finished 10s bars, the last one ending at end_ms, with a gently rising price."""
    start_ms = end_ms - n * 10_000
    return [
        {
            "symbol": symbol,
            "window_start": start_ms + i * 10_000,
            "window_end": start_ms + (i + 1) * 10_000,
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 5.0 + i % 3,
            "vwap": 100.2 + i,
            "trade_count": 40,
            "net_signed_volume": 0.5,
            "order_flow_imbalance": 0.1,
        }
        for i in range(n)
    ]


def filled_store(bars):
    from live import LiveBars

    store = LiveBars()
    for bar in bars:
        store.update(bar)
    return store


def test_live_features_match_warehouse(sample_data, tmp_path, monkeypatch):
    import duckdb
    import pandas as pd
    from cryptostream_dagster.definitions import dbt_models, lake_to_duckdb
    from dagster import materialize
    from features import FEATURE_COLUMNS

    served = tmp_path / "crypto.duckdb"
    monkeypatch.setenv("DUCKDB_PATH", str(served))
    assert materialize([lake_to_duckdb, dbt_models]).success

    con = duckdb.connect(str(served), read_only=True)
    raw = con.execute(
        "select * from raw.ohlcv_10s where symbol = 'btcusdt' order by window_start desc limit 300"
    ).df()
    expected = con.execute(
        "select * from intermediate.int_technical_features where symbol = 'btcusdt' "
        "order by window_start desc limit 1"
    ).df().iloc[0]
    con.close()

    bars = raw.to_dict("records")
    for bar in bars:
        bar["window_start"] = pd.Timestamp(bar["window_start"]).value // 10**6
        bar["window_end"] = pd.Timestamp(bar["window_end"]).value // 10**6
    now_ms = max(b["window_end"] for b in bars) + 1

    feats = filled_store(bars).latest_features("btcusdt", now_ms=now_ms)
    for column in FEATURE_COLUMNS:
        assert feats[column] == pytest.approx(float(expected[column]), rel=1e-9, abs=1e-12), column


def test_bar_updates_replace_the_same_window():
    bars = make_bars(2, 1_700_000_000_000)
    store = filled_store(bars + [dict(bars[-1], close=999.0)])
    df = store.frame("btcusdt")
    assert len(df) == 2
    assert df.iloc[-1]["close"] == 999.0


def test_live_features_skip_partial_and_stale_bars():
    now_ms = 1_700_000_000_000
    finished = make_bars(40, now_ms - 5_000)
    forming = dict(make_bars(1, now_ms + 5_000)[0], close=999.0)
    store = filled_store(finished + [forming])

    feats = store.latest_features("btcusdt", now_ms=now_ms)
    assert feats["as_of"].startswith("2023-11-14T22:13")
    assert feats["ret_1"] < 0.5  # the jump to 999 in the forming bar would be far larger

    assert store.latest_features("btcusdt", now_ms=now_ms + 60_000) is None
    assert filled_store(make_bars(10, now_ms)).latest_features("btcusdt", now_ms=now_ms) is None


def test_consumer_reads_bars_from_kafka():
    import json
    import threading
    import time

    from confluent_kafka import Producer
    from live import LiveBars, consume_bars

    # librdkafka's built-in mock cluster: a real broker protocol with no broker to run.
    producer = Producer({"bootstrap.servers": "", "test.mock.num.brokers": 1})
    broker = list(producer.list_topics(timeout=5).brokers.values())[0]
    for bar in make_bars(3, 1_700_000_000_000):
        producer.produce("crypto.bars_10s", key=bar["symbol"], value=json.dumps(bar))
    producer.produce("crypto.bars_10s", key="btcusdt", value="not json")
    assert producer.flush(5) == 0

    store, stop = LiveBars(), threading.Event()
    thread = threading.Thread(
        target=consume_bars, args=(store, f"{broker.host}:{broker.port}", "crypto.bars_10s", stop)
    )
    thread.start()
    deadline = time.time() + 15
    while time.time() < deadline and (store.frame("btcusdt") is None or len(store.frame("btcusdt")) < 3):
        time.sleep(0.1)
    stop.set()
    thread.join(10)

    assert not thread.is_alive()
    assert len(store.frame("btcusdt")) == 3
