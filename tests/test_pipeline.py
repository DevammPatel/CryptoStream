"""Unit / integration tests for the CryptoStream pipeline.

Covered:
  - ingestion.normalise maps Binance payloads to the canonical schema
  - synthetic data generation + DuckDB round-trip
  - feature loading + train/label construction
  - model training produces a usable classifier and metadata

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


def test_training_runs(sample_data, tmp_path):
    env = dict(os.environ, MLFLOW_TRACKING_URI=f"file://{tmp_path/'mlruns'}")
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "ml", "train.py")],
        cwd=os.path.join(REPO, "ml"), env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(os.path.join(REPO, "ml", "models", "direction_lgbm.pkl"))
    assert os.path.exists(os.path.join(REPO, "ml", "models", "model_meta.json"))


def test_prediction_from_model(sample_data):
    import json

    import joblib
    import numpy as np

    meta = json.load(open(os.path.join(REPO, "ml", "models", "model_meta.json")))
    assert "metrics" in meta and "test_accuracy" in meta["metrics"]

    payload = joblib.load(os.path.join(REPO, "ml", "models", "direction_lgbm.pkl"))
    model, feats = payload["model"], payload["features"]
    proba = model.predict_proba(np.zeros((1, len(feats))))[:, 1][0]
    assert 0.0 <= proba <= 1.0
