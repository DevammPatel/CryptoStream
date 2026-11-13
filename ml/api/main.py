"""FastAPI model-serving app.

Endpoints:
  GET  /health              liveness + whether a model is loaded
  GET  /model/info          metrics + feature importance from the last training run
  POST /predict             predict from an explicit feature vector
  GET  /predict/{symbol}    predict from the newest finished bar for a symbol
  GET  /symbols             symbols with live bars
  GET  /bars/{symbol}       recent live bars with indicators, including the one forming

Loads the model saved by ml/train.py. Reloads lazily so a freshly trained model is
picked up without a restart.

When KAFKA_BOOTSTRAP is set, a background thread keeps live bars in memory (see
live.py). /predict/{symbol} uses them, and falls back to the warehouse while the feed
is warming up or stale. Each response says which one it used in `source`.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Make the sibling `features.py` importable whether run as module or script.
ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from features import FEATURE_COLUMNS, duckdb_path, latest_feature_row  # noqa: E402
from live import MAX_BARS, LiveBars, consume_bars  # noqa: E402

MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ML_DIR, "models"))  # keep in sync with ml/train.py
MODEL_PATH = os.path.join(MODEL_DIR, "direction_lgbm.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

app = FastAPI(title="CryptoStream Prediction API", version="1.0.0")

_state: dict = {"model": None, "features": FEATURE_COLUMNS, "mtime": None}

live_bars = LiveBars()
_stop_consumer = threading.Event()


def _load_model() -> None:
    """(Re)load the model if the file changed on disk."""
    if not os.path.exists(MODEL_PATH):
        _state["model"] = None
        return
    mtime = os.path.getmtime(MODEL_PATH)
    if _state["mtime"] != mtime:
        payload = joblib.load(MODEL_PATH)
        _state["model"] = payload["model"]
        _state["features"] = payload.get("features", FEATURE_COLUMNS)
        _state["mtime"] = mtime


@app.on_event("startup")
def _startup() -> None:
    _load_model()
    bootstrap = os.getenv("KAFKA_BOOTSTRAP")
    if bootstrap:
        _stop_consumer.clear()
        topic = os.getenv("BARS_TOPIC", "crypto.bars_10s")
        threading.Thread(
            target=consume_bars, args=(live_bars, bootstrap, topic, _stop_consumer), daemon=True
        ).start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop_consumer.set()


class Features(BaseModel):
    """Every feature is required: a default would quietly predict from made-up values."""

    ret_1: float
    log_ret_1: float
    px_vs_sma30: float
    volatility_30: float = Field(ge=0)
    order_flow_imbalance: float
    ofi_6: float
    rel_volume: float = Field(ge=0)
    trade_count: float = Field(ge=0)


class Prediction(BaseModel):
    prob_up: float                # calibrated: how often "up" happened at this score
    direction: str
    confidence: float             # distance of prob_up from 50/50, scaled to 0-1
    source: str = "request"       # request, live or warehouse
    as_of: str | None = None      # start of the bar the features came from


def _predict_from_dict(feats: dict, source: str = "request") -> Prediction:
    _load_model()
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run `python ml/train.py` first.")
    order = _state["features"]
    x = pd.DataFrame([{c: float(feats.get(c, 0.0)) for c in order}], columns=order)
    prob_up = float(model.predict_proba(x)[:, 1][0])
    direction = "up" if prob_up >= 0.5 else "down"
    confidence = abs(prob_up - 0.5) * 2
    return Prediction(
        prob_up=round(prob_up, 4),
        direction=direction,
        confidence=round(confidence, 4),
        source=source,
        as_of=feats.get("as_of"),
    )


@app.get("/health")
def health() -> dict:
    _load_model()
    return {"status": "ok", "model_loaded": _state["model"] is not None, "live_symbols": live_bars.symbols()}


@app.get("/model/info")
def model_info() -> dict:
    if not os.path.exists(META_PATH):
        raise HTTPException(status_code=404, detail="No model metadata. Train the model first.")
    with open(META_PATH) as fh:
        return json.load(fh)


@app.post("/predict", response_model=Prediction)
def predict(features: Features) -> Prediction:
    return _predict_from_dict(features.model_dump())


@app.get("/predict/{symbol}", response_model=Prediction)
def predict_symbol(symbol: str) -> Prediction:
    feats, source = live_bars.latest_features(symbol), "live"
    if feats is None:
        if not os.path.exists(duckdb_path()):
            raise HTTPException(status_code=503, detail="Warehouse not built yet.")
        feats, source = latest_feature_row(symbol), "warehouse"
    if feats is None:
        raise HTTPException(status_code=404, detail=f"No live features for '{symbol}'.")
    return _predict_from_dict(feats, source)


@app.get("/symbols")
def symbols() -> dict:
    return {"symbols": live_bars.symbols()}


@app.get("/bars/{symbol}")
def bars(symbol: str, limit: int = Query(300, ge=1, le=MAX_BARS)) -> list[dict]:
    df = live_bars.frame(symbol)
    if df is None:
        raise HTTPException(status_code=404, detail=f"No live bars for '{symbol}'.")
    recent = df.tail(limit)
    return recent.astype(object).where(recent.notna(), None).to_dict("records")
