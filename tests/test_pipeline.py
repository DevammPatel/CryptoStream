"""Unit / integration tests for the CryptoStream pipeline.

Covered:
  - ingestion.normalise maps Binance payloads to the canonical schema
  - ingestion stops cleanly on SIGTERM and gives up only on errors retrying can't fix
  - synthetic data generation + DuckDB round-trip, at a realistic and repeatable scale
  - the market summary judges volatility against each symbol's own baseline
  - the summary carries the model's calls (worded as slight leans), uses the LLM when
    configured, falls back to the template when it fails, and round-trips through its file
  - feature loading + train/label construction
  - model training produces a usable classifier and metadata
  - CV splits never let a training label reach into the test period
  - on data with no signal, calibrated probabilities stay near the base rate
  - a new model replaces the served one only if it scores at least as well on unseen rows
  - Dagster definitions load and the refresh schedule is on
  - the warehouse swap keeps views working and never blocks a reader
  - indicators and labels are time-based, identical in dbt and Python, across gaps
  - duplicate copies of a bar collapse to the most complete one
  - the Binance backfill rolls 1s klines into Spark-shaped 10s bars the pipeline can load
  - bar times stay UTC whatever the time zone of the machine reading them
  - compaction merges finished hours of Spark files, keeps every bar once, expires old trades
  - compaction re-lists S3 instead of trusting a cached listing
  - live bars: indicators match dbt, partial and stale bars are ignored, Kafka consumer
  - API: predictions from live bars, warehouse fallback, /bars, /model/info
  - live track record: only bars newer than the model's training data, settled 30s later

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


SNAPSHOT = [
    {"symbol": "btcusdt", "as_of": "2026-11-11 22:58:00", "last_return": 0.0001, "px_vs_sma30": 0.001,
     "volatility_30": 0.0003, "typical_volatility_30": 0.0003, "order_flow_imbalance": -0.2},
    {"symbol": "ethusdt", "as_of": "2026-11-11 22:58:10", "last_return": -0.0002, "px_vs_sma30": 0.002,
     "volatility_30": 0.0005, "typical_volatility_30": 0.0003, "order_flow_imbalance": -0.3},
]
CALLS = {"btcusdt": {"prob_up": 0.47, "as_of": "2026-11-11T22:58:00+00:00"},
         "ethusdt": {"prob_up": 0.505, "as_of": "2026-11-11T22:58:10+00:00"}}
MODEL_META = {"metrics": {"test_auc": 0.526}, "trained_through": "2026-11-11T22:00:00+00:00"}


@pytest.fixture
def market(monkeypatch):
    from llm import market_agent

    monkeypatch.setattr(market_agent, "load_snapshot", lambda: SNAPSHOT)
    monkeypatch.setattr(market_agent, "load_model_calls", lambda symbols: (CALLS, MODEL_META))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return market_agent


def test_summary_reports_model_calls_without_overstating_them(market):
    record = market.build_market_summary()
    assert record["source"] == "template"
    assert "BTC:" in record["text"] and "a slight lean down (P(up) 0.47" in record["text"]
    assert "ETH:" in record["text"] and "no clear lean (P(up) 0.51" in record["text"]
    assert "held-out AUC 0.526" in record["text"]
    assert "tone is mixed" in record["text"]  # above their averages, but net selling
    assert record["data_as_of"].startswith("2026-11-11T22:58:10")
    assert record["model_trained_through"] == MODEL_META["trained_through"]


def test_llm_summary_uses_the_same_facts_and_falls_back(market, monkeypatch, tmp_path):
    import openai

    prompts = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = self.completions = self

        def create(self, **kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            message = type("Message", (), {"content": " BTC leans slightly down. "})
            return type("Response", (), {"choices": [type("Choice", (), {"message": message})]})

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    record = market.build_market_summary()
    assert (record["source"], record["text"]) == ("llm", "BTC leans slightly down.")
    assert "a slight lean down (P(up) 0.47" in prompts[0] and "AUC 0.526" in prompts[0]

    def quota_exceeded(self, **kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(FakeClient, "create", quota_exceeded)
    record = market.build_market_summary()
    assert record["source"] == "template" and "quota exceeded" in record["llm_error"]

    # saved by Dagster, read back by the dashboard
    path = market.save_market_summary(record, str(tmp_path / "market_summary.json"))
    assert market.load_saved_summary(path) == record


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


def test_purged_splits_never_leak():
    import numpy as np
    from train import HORIZON_NS, purged_splits

    # two symbols sharing one clock of 10s bars
    t = np.repeat(np.arange(0, 2000) * 10 * 10**9, 2)
    folds = list(purged_splits(t))
    assert len(folds) == 5
    for train, test in folds:
        assert t[train].max() + HORIZON_NS < t[test].min()


def test_epoch_ns_is_nanoseconds_whatever_the_input_unit():
    import pandas as pd
    from train import epoch_ns

    for unit in ("us", "ns"):
        times = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-01 00:00:30"]).as_unit(unit))
        assert int(epoch_ns(times)[1] - epoch_ns(times)[0]) == 30 * 10**9


def test_calibrated_model_claims_no_edge_on_noise(sample_data, model_dir):
    import json

    meta = json.load(open(os.path.join(model_dir, "model_meta.json")))
    metrics = meta["metrics"]
    assert metrics["calibrated"] is True
    # the sample data is a random walk: an honest model stays close to 50/50
    assert metrics["test_prob_std"] < 0.1
    assert metrics["test_brier"] < metrics["test_brier_baseline"] + 0.01


def test_first_training_run_is_promoted_and_recorded(sample_data, model_dir):
    import json

    with open(os.path.join(model_dir, "last_training.json")) as fh:
        last = json.load(fh)
    with open(os.path.join(model_dir, "model_meta.json")) as fh:
        served = json.load(fh)
    assert last["promotion"] == {"promoted": True, "reason": "no model served yet"}
    assert served["trained_through"] == last["trained_through"]


def test_a_worse_model_never_replaces_the_served_one(tmp_path):
    import json

    import joblib
    import numpy as np
    import pandas as pd
    from features import FEATURE_COLUMNS
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    from train import judge_against_served

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    y = pd.Series((X["ret_1"] + 0.3 * rng.normal(size=1000) > 0).astype(int))
    t = (1_800_000_000 + np.arange(1000) * 10) * 10**9  # 10s bars
    model_path, meta_path = str(tmp_path / "model.pkl"), str(tmp_path / "meta.json")

    def serve(model, trained_through):
        joblib.dump({"model": model, "features": FEATURE_COLUMNS}, model_path)
        with open(meta_path, "w") as fh:
            json.dump({"trained_through": trained_through}, fh)

    def judge(challenger_proba):
        return judge_against_served(X, y, t, challenger_proba, model_path, meta_path)

    def iso(ns):
        return pd.Timestamp(int(ns), unit="ns", tz="UTC").isoformat()

    skilled = LogisticRegression().fit(X, y)
    guesser = DummyClassifier(strategy="prior").fit(X, y)

    assert judge(np.full(1000, 0.5)) == {"promoted": True, "reason": "no model served yet"}

    serve(skilled, iso(t[0] - 1))
    verdict = judge(guesser.predict_proba(X)[:, 1])
    assert not verdict["promoted"] and verdict["n_compared"] == 1000
    assert verdict["challenger_brier"] > verdict["served_brier"]

    serve(guesser, iso(t[0] - 1))
    assert judge(skilled.predict_proba(X)[:, 1])["promoted"]

    # Rows the served model trained on don't count, so it can't win on its own training data.
    serve(skilled, iso(t[-100]))
    assert judge(np.full(1000, 0.5))["reason"] == "only 99 test rows are new to the served model"


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


def test_summary_gets_calls_from_the_served_model(sample_data, model_dir, monkeypatch):
    from llm import market_agent

    monkeypatch.setenv("MODEL_DIR", model_dir)
    calls, meta = market_agent.load_model_calls(["btcusdt", "dogeusdt"])
    assert list(calls) == ["btcusdt"]
    assert 0 < calls["btcusdt"]["prob_up"] < 1
    assert meta["trained_through"]


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


def kline(open_ms, o, h, low, c, volume, trades, taker_buy):
    """One Binance 1s kline row: the fields the backfill reads, quote volume = volume * close."""
    return [open_ms, str(o), str(h), str(low), str(c), str(volume), open_ms + 999,
            str(volume * c), trades, str(taker_buy), str(taker_buy * c), "0"]


def test_backfill_rolls_klines_into_spark_shaped_bars():
    from cryptostream_dagster.backfill import COLUMNS, bars_from_klines

    t0 = 1_700_000_000_000
    rows = [
        kline(t0, 100, 101, 99, 100.5, 2.0, 10, 1.5),
        kline(t0 + 1000, 100.5, 103, 100, 102, 1.0, 5, 0.0),
        kline(t0 + 2000, 102, 102, 102, 102, 0.0, 0, 0.0),        # no trades: ignored
        kline(t0 + 10_000, 102, 102.5, 101.5, 101.8, 4.0, 8, 3.0),  # next 10s window
    ]
    bars = bars_from_klines(rows)
    assert list(bars.columns) == COLUMNS
    first = bars.iloc[0]
    assert (first["open"], first["high"], first["low"], first["close"]) == (100, 103, 99, 102)
    assert first["volume"] == 3.0 and first["trade_count"] == 15
    assert first["net_signed_volume"] == 2 * 1.5 - 3.0  # taker buys minus the rest
    assert abs(first["vwap"] - (2.0 * 100.5 + 1.0 * 102) / 3.0) < 1e-9
    assert (bars["window_end"] - bars["window_start"]).dt.total_seconds().eq(10).all()
    assert len(bars) == 2

    # The bulk files use microseconds; the result must be the same.
    in_us = [[r[0] * 1000, *r[1:]] for r in rows]
    assert bars_from_klines(in_us).equals(bars)


def test_backfill_files_load_next_to_spark_files(sample_data, tmp_path, monkeypatch):
    import duckdb
    import pandas as pd
    from cryptostream_dagster.backfill import bars_from_klines, write_day
    from cryptostream_dagster.definitions import dbt_models, lake_to_duckdb
    from dagster import materialize

    lake = lake_with_gaps(tmp_path)
    monkeypatch.setenv("LAKE_URI", lake)
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "crypto.duckdb"))
    t0 = 1_600_000_000_000  # far from the sample bars, so nothing overlaps
    rows = [kline(t0 + i * 1000, 100, 101, 99, 100, 1.0, 3, 0.5) for i in range(60)]
    con = duckdb.connect()
    write_day(con, bars_from_klines(rows), "btcusdt", "2020-09-13")
    con.close()

    assert materialize([lake_to_duckdb, dbt_models]).success
    con = duckdb.connect(str(tmp_path / "crypto.duckdb"), read_only=True)
    backfilled = con.execute(
        "select count(*) from intermediate.int_technical_features where window_start < ?", [pd.Timestamp(t0 + 3_600_000, unit="ms")]
    ).fetchone()[0]
    con.close()
    assert backfilled == 6


def test_bar_times_stay_utc_in_any_reader_time_zone(tmp_path, monkeypatch):
    import duckdb
    import pandas as pd
    from cryptostream_dagster.definitions import dbt_models, lake_to_duckdb
    from dagster import materialize

    # A time-zone-aware file next to a plain one, in summer, when London is UTC+1.
    lake = tmp_path / "lake" / "symbol=btcusdt"
    lake.mkdir(parents=True)
    bar = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "vwap": 1.0,
           "trade_count": 1, "net_signed_volume": 0.0, "order_flow_imbalance": 0.0}
    aware = pd.date_range("2026-07-01 12:00", periods=40, freq="10s", tz="UTC")
    pd.DataFrame({"window_start": aware, "window_end": aware + pd.Timedelta(seconds=10), **bar}).to_parquet(lake / "a.parquet")
    plain = aware.tz_localize(None) + pd.Timedelta(minutes=10)
    pd.DataFrame({"window_start": plain, "window_end": plain + pd.Timedelta(seconds=10), **bar}).to_parquet(lake / "b.parquet")

    served = tmp_path / "crypto.duckdb"
    monkeypatch.setenv("LAKE_URI", str(tmp_path / "lake"))
    monkeypatch.setenv("DUCKDB_PATH", str(served))
    assert materialize([lake_to_duckdb, dbt_models]).success

    con = duckdb.connect(str(served), read_only=True)
    con.execute("SET TimeZone = 'Europe/London'")
    first, last = con.execute("select min(window_start), max(window_start) from staging.stg_ohlcv").fetchone()
    con.close()
    assert str(first) == "2026-07-01 12:00:00"
    assert str(last) == "2026-07-01 12:16:30"


def spark_lake(tmp_path, now):
    """Spark-shaped files: a file per minute of bars over the last 3 hours, a duplicate
    partial copy of some bars, and raw trades from 9 days ago and from now."""
    import pandas as pd

    root = tmp_path / "lake"
    feats = root / "features" / "ohlcv_10s" / "symbol=btcusdt"
    trades = root / "raw" / "trades" / "symbol=btcusdt"
    feats.mkdir(parents=True)
    trades.mkdir(parents=True)
    start = pd.Timestamp(now).tz_localize(None).floor("h") - pd.Timedelta(hours=2)
    bars = pd.DataFrame({"window_start": pd.date_range(start, now.replace(tzinfo=None), freq="10s", inclusive="left")})
    bars["window_end"] = bars["window_start"] + pd.Timedelta(seconds=10)
    bars = bars.assign(open=1.0, high=1.0, low=1.0, close=1.0, volume=2.0, vwap=1.0, trade_count=10,
                       net_signed_volume=0.0, order_flow_imbalance=0.0)
    for i, (_, minute) in enumerate(bars.groupby(bars["window_start"].dt.floor("min"))):
        minute.to_parquet(feats / f"part-{i:05d}.parquet", index=False)
    bars.iloc[:30].assign(trade_count=4, close=9.0).to_parquet(feats / "part-replay.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(feats / "backfill-2026-01-01.parquet", index=False)
    for name, when in (("old", pd.Timestamp(now).tz_localize(None) - pd.Timedelta(days=9)), ("new", start)):
        pd.DataFrame({"trade_id": [1, 2], "price": [1.0, 1.0], "quantity": [1.0, 1.0], "trade_time": [0, 0],
                      "is_buyer_maker": [False, True], "ingest_time": [0, 0], "event_ts": [when, when]}
                     ).to_parquet(trades / f"part-{name}.parquet", index=False)
    return root, feats, trades, bars


def test_compaction_merges_finished_hours_and_expires_old_trades(tmp_path):
    from datetime import datetime, timezone

    import duckdb
    from cryptostream_dagster.compaction import compact_lake

    now = datetime(2026, 11, 11, 15, 25, tzinfo=timezone.utc)
    root, feats, trades, bars = spark_lake(tmp_path, now)
    uri = str(root / "features" / "ohlcv_10s")
    con = duckdb.connect()
    summary = compact_lake(con, uri, now=now)

    names = sorted(p.name for p in feats.iterdir())
    assert "compact-2026111113.parquet" in names and "compact-2026111114.parquet" in names
    assert "backfill-2026-01-01.parquet" in names               # not Spark's, left alone
    open_hour = [n for n in names if n.startswith("part-")]
    assert len(open_hour) == 25                                  # 15:00-15:25 is still being written
    assert summary[uri]["written_files"] == 2

    lake = con.execute(  # compact-* and part-* files, not the backfill file
        f"select * from read_parquet('{feats}/[cp][oa]*.parquet', union_by_name=true, hive_partitioning=false)"
    ).df()
    assert len(lake) == len(bars)                                # every bar once, nothing lost
    assert lake["window_start"].is_unique
    assert (lake["trade_count"] == 10).all()                     # the complete copy won over the replay

    # A late file for a finished hour is folded into that hour's compact file.
    bars.iloc[[5]].assign(trade_count=11).to_parquet(feats / "part-late.parquet", index=False)
    compact_lake(con, uri, now=now)
    first_hour = con.execute(f"select trade_count from read_parquet('{feats}/compact-2026111113.parquet') order by window_start").df()
    assert len(first_hour) == 360 and first_hour["trade_count"].iloc[5] == 11
    assert not (feats / "part-late.parquet").exists()

    # Raw trades: the 9-day-old hour is past the 7-day retention and is gone.
    trade_files = sorted(p.name for p in trades.iterdir())
    assert trade_files == ["compact-2026111113.parquet"]
    con.close()


def test_compaction_never_trusts_a_cached_s3_listing(monkeypatch):
    """DuckDB writes the merged file straight to S3, so s3fs must list the bucket again
    to find it. A cached listing made the move fail with FileNotFoundError."""
    from cryptostream_dagster.compaction import filesystem

    monkeypatch.setenv("LAKE_ACCESS_KEY", "test")
    monkeypatch.setenv("LAKE_SECRET_KEY", "test")
    monkeypatch.setenv("LAKE_ENDPOINT", "http://localhost:1")
    uri = "s3://crypto-lake/features/ohlcv_10s"
    fs = filesystem(uri)
    fs.dircache["crypto-lake/features"] = [{"name": "crypto-lake/features/a", "size": 0, "type": "file"}]
    assert "crypto-lake/features" not in fs.dircache
    assert filesystem(uri) is not fs  # no shared instance carrying state between runs


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


class ConstantModel:
    """Always gives the same P(up)."""

    def __init__(self, prob_up):
        self.prob_up = prob_up

    def predict_proba(self, X):
        import numpy as np

        return np.column_stack([np.full(len(X), 1 - self.prob_up), np.full(len(X), self.prob_up)])


def test_track_record_counts_only_unseen_bars_and_settles_them_later():
    import time

    from features import FEATURE_COLUMNS
    from monitor import Scorecard

    now = int(time.time() * 1000)
    bars = make_bars(80, now - 2_000)  # rising prices: every settled call's outcome is "up"
    store = filled_store(bars)
    card = Scorecard()

    # 79 bars are finished; the first 29 lack the history a call needs; the last 3
    # calls wait for their bar 30s later.
    card.step(store, ConstantModel(0.6), FEATURE_COLUMNS, "model-a", None, now)
    assert card.summary("model-a") == {
        "n": 46, "hit_rate": 1.0, "hit_rate_baseline": 1.0, "brier": pytest.approx(0.16),
        "brier_baseline": 0.0, "up_rate": 1.0, "pending": 3,
    }

    # Calls settle as later bars finish, and a new model only scores bars it never saw.
    for bar in make_bars(3, now + 28_000)[-3:]:
        store.update({**bar, "close": 200.0})
    trained_through = bars[60]["window_start"]
    card.step(store, ConstantModel(0.4), FEATURE_COLUMNS, "model-b", trained_through, now + 30_000)
    assert card.summary("model-a")["n"] == 49 and card.summary("model-a")["pending"] == 0
    assert card.summary("model-b") == {**card.summary("model-b"), "n": 0, "pending": 3}
    assert card.summary(symbol="ethusdt") == {"n": 0, "pending": 0}


def test_api_reports_the_live_track_record(api):
    import time

    main, client = api
    main._load_model()
    now, cutoff = int(time.time() * 1000), main._state["trained_through_ms"]
    for bar in make_bars(60, now - 2_000) + make_bars(60, cutoff, symbol="ethusdt"):
        main.live_bars.update(bar)
    main._score_once(now)

    body = client.get("/model/live").json()
    assert body["model_id"] and body["counts_bars_after"]
    assert body["current_model"]["n"] == 26 and body["current_model"]["pending"] == 3
    # bars from before the end of the model's training data never count
    assert body["by_symbol"]["ethusdt"] == {"n": 0, "pending": 0}


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
    monkeypatch.setattr(main, "META_PATH", os.path.join(model_dir, "model_meta.json"))
    monkeypatch.setattr(main, "LAST_RUN_PATH", os.path.join(model_dir, "last_training.json"))
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


def test_api_requires_every_feature(api):
    from features import FEATURE_COLUMNS

    main, client = api
    assert client.post("/predict", json={"ret_1": 0.001}).status_code == 422
    body = client.post("/predict", json={c: 0.1 for c in FEATURE_COLUMNS}).json()
    assert body["source"] == "request"


def test_model_info_reports_the_latest_training_run(api):
    main, client = api
    info = client.get("/model/info").json()
    assert info["promotion"]["promoted"] is True
    assert info["last_training"]["trained_through"] == info["trained_through"]


def test_api_falls_back_to_warehouse(api):
    main, client = api
    body = client.get("/predict/btcusdt").json()
    assert body["source"] == "warehouse"
    assert client.get("/predict/dogeusdt").status_code == 404
