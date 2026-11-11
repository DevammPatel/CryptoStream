"""Unit / integration tests for the CryptoStream pipeline.

Covered:
  - ingestion.normalise maps Binance payloads to the canonical schema
  - ingestion stops cleanly on SIGTERM and gives up only on errors retrying can't fix
  - synthetic data generation + DuckDB round-trip, at a realistic and repeatable scale
  - the market summary judges volatility against each symbol's own baseline
  - feature loading + train/label construction
  - model training produces a usable classifier and metadata
  - Dagster definitions load and the refresh schedule is on
  - the warehouse swap keeps views working and never blocks a reader
  - indicators and labels are time-based, identical in dbt and Python, across gaps
  - duplicate copies of a bar collapse to the most complete one
  - live bars: indicators match dbt, partial and stale bars are ignored, Kafka consumer
  - API: predictions from live bars, warehouse fallback, /bars

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


def test_only_refusals_stop_reconnecting():
    from websockets.datastructures import Headers
    from websockets.exceptions import InvalidStatusCode

    from ingestion.ws_producer import is_fatal

    assert is_fatal(InvalidStatusCode(451, Headers()))
    assert not is_fatal(InvalidStatusCode(503, Headers()))
    assert not is_fatal(ConnectionResetError())


def serve_fake_trades():
    """A local WebSocket server that streams Binance-shaped trades. Returns its port."""
    import asyncio
    import json
    import threading

    import websockets

    ready, state = threading.Event(), {}

    async def handler(ws, *args):
        trade_id = 0
        while True:
            trade_id += 1
            data = {"e": "trade", "s": "BTCUSDT", "t": trade_id, "p": "62000.5", "q": "0.01", "T": 1700000000000, "m": False}
            await ws.send(json.dumps({"stream": "btcusdt@trade", "data": data}))
            await asyncio.sleep(0.05)

    async def run():
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            state["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(run()), daemon=True).start()
    assert ready.wait(5)
    return state["port"]


def test_ingestion_stops_cleanly_on_sigterm(tmp_path):
    import signal
    import time

    port = serve_fake_trades()
    log_path = tmp_path / "ingestion.log"
    env = dict(os.environ, BINANCE_WS_BASE=f"ws://127.0.0.1:{port}/stream?streams=", KAFKA_BOOTSTRAP="127.0.0.1:9")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, os.path.join(REPO, "ingestion", "ws_producer.py")], env=env, stdout=log, stderr=log
        )
    deadline = time.time() + 15
    while "connected to" not in log_path.read_text() and time.time() < deadline:
        time.sleep(0.2)
    assert "connected to" in log_path.read_text()

    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=15) == 0
    assert "shutting down after" in log_path.read_text()


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


def test_sample_data_is_realistic_and_repeatable(tmp_path):
    import duckdb
    import numpy as np

    closes = []
    for run in range(2):
        db = str(tmp_path / f"run{run}.duckdb")
        subprocess.run(
            [sys.executable, os.path.join(REPO, "scripts", "generate_sample_data.py"), "--minutes", "60", "--symbols", "btcusdt"],
            check=True, cwd=REPO, env=dict(os.environ, DUCKDB_PATH=db), capture_output=True,
        )
        con = duckdb.connect(db, read_only=True)
        closes.append(np.array([r[0] for r in con.execute("select close from raw.ohlcv_10s order by window_start").fetchall()]))
        con.close()

    assert np.allclose(closes[0], closes[1])  # same prices on every run
    bar_vol = np.diff(np.log(closes[0])).std()
    assert 0.0001 < bar_vol < 0.001  # real BTC moves roughly 0.02-0.03% per 10s bar


def test_summary_judges_volatility_against_its_own_baseline():
    from llm.market_agent import _describe

    row = {"symbol": "btcusdt", "last_return": 0.0001, "px_vs_sma30": 0.001, "order_flow_imbalance": 0.0}
    assert "elevated volatility" in _describe({**row, "volatility_30": 0.0006, "typical_volatility_30": 0.0003})
    assert "subdued volatility" in _describe({**row, "volatility_30": 0.0001, "typical_volatility_30": 0.0003})
    assert "normal volatility" in _describe({**row, "volatility_30": 0.0003, "typical_volatility_30": 0.0003})


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


def lake_with_gaps(tmp_path):
    """The btcusdt sample bars with a 50s outage, every 17th bar missing, and a few bars
    written twice: a partial copy, as Spark leaves when it replays a batch."""
    import glob

    import numpy as np
    import pandas as pd

    files = glob.glob(os.path.join(REPO, "data", "lake", "features", "ohlcv_10s", "symbol=btcusdt", "*.parquet"))
    bars = pd.concat([pd.read_parquet(f) for f in files]).sort_values("window_start").reset_index(drop=True)
    keep = np.ones(len(bars), dtype=bool)
    keep[100:105] = False
    keep[::17] = False
    lake = tmp_path / "lake"
    (lake / "symbol=btcusdt").mkdir(parents=True)
    bars[keep].to_parquet(lake / "symbol=btcusdt" / "part-0.parquet", index=False)
    partial = bars[keep].iloc[200:210].assign(trade_count=lambda d: d["trade_count"] - 5, close=lambda d: d["close"] * 1.01)
    partial.to_parquet(lake / "symbol=btcusdt" / "part-1.parquet", index=False)
    return str(lake)


def test_indicators_and_labels_match_dbt_across_gaps(sample_data, tmp_path, monkeypatch):
    import duckdb
    import numpy as np
    import pandas as pd
    from cryptostream_dagster.definitions import dbt_models, lake_to_duckdb
    from dagster import materialize
    from features import FEATURE_COLUMNS, _build_from_raw, one_bar_per_window, with_indicators

    served = tmp_path / "crypto.duckdb"
    monkeypatch.setenv("LAKE_URI", lake_with_gaps(tmp_path))
    monkeypatch.setenv("DUCKDB_PATH", str(served))
    assert materialize([lake_to_duckdb, dbt_models]).success

    con = duckdb.connect(str(served), read_only=True)
    raw = con.execute("select * from raw.ohlcv_10s").df()
    sql = con.execute("select * from intermediate.int_technical_features order by window_start").df()
    mart = con.execute("select * from marts.mart_ml_features order by window_start").df()
    fallback = _build_from_raw(con).sort_values("window_start").reset_index(drop=True)
    con.close()

    assert raw.duplicated(["symbol", "window_start"]).sum() == 10
    assert not sql.duplicated(["symbol", "window_start"]).any()
    copies = raw[raw.duplicated(["symbol", "window_start"], keep=False)]
    partial = copies.sort_values("trade_count").drop_duplicates("window_start")
    assert not sql["close"].isin(partial["close"]).any()  # the complete copy won
    py = with_indicators(one_bar_per_window(raw))
    assert len(py) == len(sql)
    for column in FEATURE_COLUMNS:
        np.testing.assert_allclose(py[column], sql[column], rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=column)

    # A gap breaks the return instead of comparing prices on either side of it.
    after_gap = sql["window_start"].diff() > pd.Timedelta(seconds=10)
    assert after_gap.sum() > 5
    assert sql.loc[after_gap, "ret_1"].isna().all()

    # Labels only come from the bar exactly 30s later, in dbt and in the Python fallback.
    assert mart["window_start"].add(pd.Timedelta(seconds=30)).isin(set(sql["window_start"])).all()
    assert len(fallback) == len(mart)
    assert (fallback["label_up_3"].to_numpy() == mart["label_up_3"].to_numpy()).all()


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
    from live import GRACE_MS

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
    now_ms = max(b["window_end"] for b in bars) + GRACE_MS

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


@pytest.fixture
def api(sample_data, model_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from live import LiveBars

    from ml.api import main

    monkeypatch.setattr(main, "MODEL_PATH", os.path.join(model_dir, "direction_lgbm.pkl"))
    monkeypatch.setattr(main, "live_bars", LiveBars())
    monkeypatch.setitem(main._state, "mtime", None)
    return main, TestClient(main.app)


def test_api_predicts_from_live_bars(api):
    import time

    main, client = api
    for bar in make_bars(40, int(time.time() * 1000) - 2_000):
        main.live_bars.update(bar)

    body = client.get("/predict/btcusdt").json()
    assert body["source"] == "live"
    assert body["as_of"]

    assert client.get("/symbols").json() == {"symbols": ["btcusdt"]}
    rows = client.get("/bars/btcusdt?limit=5").json()
    assert len(rows) == 5
    assert rows[-1]["rel_volume"] > 0
    assert client.get("/bars/dogeusdt").status_code == 404


def test_api_falls_back_to_warehouse(api):
    main, client = api
    body = client.get("/predict/btcusdt").json()
    assert body["source"] == "warehouse"
    assert client.get("/predict/dogeusdt").status_code == 404
