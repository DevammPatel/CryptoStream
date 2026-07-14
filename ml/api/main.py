"""FastAPI model-serving app.

Endpoints:
  GET  /health              liveness + whether a model is loaded
  GET  /model/info          metrics + feature importance from the last training run
  POST /predict             predict from an explicit feature vector
  GET  /predict/{symbol}    predict from the latest live features for a symbol

Loads the model saved by ml/train.py. Reloads lazily so a freshly trained model is
picked up without a restart.
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make the sibling `features.py` importable whether run as module or script.
ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

from features import FEATURE_COLUMNS, latest_feature_row  # noqa: E402

MODEL_PATH = os.path.join(ML_DIR, "models", "direction_lgbm.pkl")
META_PATH = os.path.join(ML_DIR, "models", "model_meta.json")

app = FastAPI(title="CryptoStream Prediction API", version="1.0.0")

_state: dict = {"model": None, "features": FEATURE_COLUMNS, "mtime": None}


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


class Features(BaseModel):
    ret_1: float = 0.0
    log_ret_1: float = 0.0
    px_vs_sma30: float = 0.0
    volatility_30: float = Field(0.0, ge=0)
    order_flow_imbalance: float = 0.0
    ofi_6: float = 0.0
    rel_volume: float = Field(1.0, ge=0)
    trade_count: float = Field(0.0, ge=0)


class Prediction(BaseModel):
    prob_up: float
    direction: str
    confidence: float


def _predict_from_dict(feats: dict) -> Prediction:
    _load_model()
    model = _state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded. Run `python ml/train.py` first.")
    order = _state["features"]
    x = pd.DataFrame([{c: float(feats.get(c, 0.0)) for c in order}], columns=order)
    prob_up = float(model.predict_proba(x)[:, 1][0])
    direction = "up" if prob_up >= 0.5 else "down"
    confidence = abs(prob_up - 0.5) * 2
    return Prediction(prob_up=round(prob_up, 4), direction=direction, confidence=round(confidence, 4))


@app.get("/health")
def health() -> dict:
    _load_model()
    return {"status": "ok", "model_loaded": _state["model"] is not None}


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
    feats = latest_feature_row(symbol)
    if feats is None:
        raise HTTPException(status_code=404, detail=f"No live features for '{symbol}'.")
    return _predict_from_dict(feats)
