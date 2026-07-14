"""Train a next-bar price-direction classifier and log everything to MLflow.

- Time-series-aware split (no shuffle) to avoid look-ahead leakage.
- LightGBM gradient-boosted trees.
- Logs params, metrics (AUC, accuracy, F1, precision/recall), feature importance,
  and the serialised model to MLflow + a local models/ dir for the API to load.

Run:  python ml/train.py
"""
from __future__ import annotations

import json
import os

import joblib
import lightgbm as lgb
import numpy as np

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # MLflow is optional — training still works without it
    mlflow = None
    _HAS_MLFLOW = False

from features import FEATURE_COLUMNS, build_xy, load_training_frame
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ML_DIR)
MODEL_DIR = os.path.join(ML_DIR, "models")           # keep in sync with ml/api/main.py
MODEL_PATH = os.path.join(MODEL_DIR, "direction_lgbm.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{os.path.join(REPO, 'mlruns')}")
EXPERIMENT = "cryptostream-direction"


def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if _HAS_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)

    df = load_training_frame()
    if len(df) < 200:
        raise SystemExit(
            f"Only {len(df)} rows available. Run `python scripts/generate_sample_data.py` "
            "or let the live pipeline collect more data first."
        )

    X, y = build_xy(df)

    params = {
        "objective": "binary",
        "n_estimators": 400,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }

    # Walk-forward CV for an honest estimate of live performance.
    tscv = TimeSeriesSplit(n_splits=5)
    aucs = []
    for tr, va in tscv.split(X):
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        # roc_auc needs both classes present
        if len(np.unique(y.iloc[va])) > 1:
            aucs.append(roc_auc_score(y.iloc[va], p))
    cv_auc = float(np.mean(aucs)) if aucs else float("nan")

    # Final model trained on all but the last 20% (held-out test).
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = lgb.LGBMClassifier(**params)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "cv_auc": cv_auc,
        "test_auc": float(roc_auc_score(y_te, proba)) if len(np.unique(y_te)) > 1 else float("nan"),
        "test_accuracy": float(accuracy_score(y_te, pred)),
        "test_f1": float(f1_score(y_te, pred, zero_division=0)),
        "test_precision": float(precision_score(y_te, pred, zero_division=0)),
        "test_recall": float(recall_score(y_te, pred, zero_division=0)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "base_rate": float(y.mean()),
    }

    importance = dict(zip(FEATURE_COLUMNS, (int(v) for v in model.feature_importances_)))

    run_id = None
    if _HAS_MLFLOW:
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)})
            mlflow.log_dict(importance, "feature_importance.json")
            try:
                mlflow.lightgbm.log_model(model, artifact_path="model", registered_model_name="cryptostream-direction")
            except Exception:  # registry may be unavailable with file store
                mlflow.lightgbm.log_model(model, artifact_path="model")
            run_id = run.info.run_id
    else:
        print("  (mlflow not installed — skipping experiment logging)")

    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, MODEL_PATH)
    with open(META_PATH, "w") as fh:
        json.dump({"metrics": metrics, "feature_importance": importance, "mlflow_run_id": run_id}, fh, indent=2)

    print("=== Training complete ===")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v}")
    print(f"  model  -> {MODEL_PATH}")
    if _HAS_MLFLOW:
        print(f"  mlflow -> {MLFLOW_TRACKING_URI} (run {run_id})")


if __name__ == "__main__":
    main()
