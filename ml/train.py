"""Train a next-bar price-direction classifier and log everything to MLflow.

- Time-series-aware splits (no shuffle), purged so no training label reaches into
  the test period: each label looks 30s ahead.
- LightGBM gradient-boosted trees, with probabilities calibrated on held-out data.
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
import pandas as pd

try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:  # MLflow is optional, unless a tracking server was requested
    if os.getenv("MLFLOW_TRACKING_URI"):
        raise
    mlflow = None
    _HAS_MLFLOW = False

from features import FEATURE_COLUMNS, build_xy, load_training_frame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

ML_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ML_DIR)
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ML_DIR, "models"))  # keep in sync with ml/api/main.py
MODEL_PATH = os.path.join(MODEL_DIR, "direction_lgbm.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file://{os.path.join(REPO, 'mlruns')}")
EXPERIMENT = "cryptostream-direction"

HORIZON_NS = 30 * 10**9  # how far ahead the label looks (mart_ml_features.sql)


def epoch_ns(times: pd.Series) -> np.ndarray:
    # DuckDB returns microsecond timestamps; pin the unit so HORIZON_NS means 30s.
    return pd.to_datetime(times, utc=True).dt.as_unit("ns").astype("int64").to_numpy()


def purged_splits(t: np.ndarray, n_splits: int = 5):
    """Walk-forward train/test splits over time, with every symbol on one clock.

    A row's label uses the close 30s later, so training rows within 30s of the test
    period would see prices from inside it. They are dropped.
    """
    clock = np.unique(t)
    for _, test_idx in TimeSeriesSplit(n_splits=n_splits).split(clock):
        start, end = clock[test_idx[0]], clock[test_idx[-1]]
        yield np.flatnonzero(t + HORIZON_NS < start), np.flatnonzero((t >= start) & (t <= end))


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
    t = epoch_ns(df["window_start"])

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
    aucs = []
    for tr, va in purged_splits(t):
        # a fold needs enough training rows, and both classes on each side for AUC
        if len(tr) < 50 or y.iloc[tr].nunique() < 2 or y.iloc[va].nunique() < 2:
            continue
        m = lgb.LGBMClassifier(**params)
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    cv_auc = float(np.mean(aucs)) if aucs else float("nan")

    # Fit on the first 60% of the time range, calibrate on the next 20%, test on the
    # last 20%. Each boundary is purged like the CV folds.
    clock = np.unique(t)
    cal_start, test_start = clock[int(len(clock) * 0.6)], clock[int(len(clock) * 0.8)]
    fit = t + HORIZON_NS < cal_start
    cal = (t >= cal_start) & (t + HORIZON_NS < test_start)
    test = t >= test_start

    model = lgb.LGBMClassifier(**params)
    model.fit(X[fit], y[fit])
    # Raw boosted-tree probabilities are overconfident on small, noisy data. Platt
    # scaling on rows the trees never saw maps them to how often "up" really happened.
    served = model
    if y[cal].nunique() > 1:
        served = CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X[cal], y[cal])

    y_te = y[test]
    proba = served.predict_proba(X[test])[:, 1]
    pred = (proba >= 0.5).astype(int)
    base_rate = float(y[fit | cal].mean())
    metrics = {
        "cv_auc": cv_auc,
        "test_auc": float(roc_auc_score(y_te, proba)) if y_te.nunique() > 1 else float("nan"),
        "test_accuracy": float(accuracy_score(y_te, pred)),
        # always guessing the more common outcome
        "test_accuracy_baseline": float(max(y_te.mean(), 1 - y_te.mean())),
        "test_brier": float(brier_score_loss(y_te, proba)),
        # always predicting the training base rate
        "test_brier_baseline": float(brier_score_loss(y_te, np.full(len(y_te), base_rate))),
        "test_prob_std": float(proba.std()),
        "test_f1": float(f1_score(y_te, pred, zero_division=0)),
        "test_precision": float(precision_score(y_te, pred, zero_division=0)),
        "test_recall": float(recall_score(y_te, pred, zero_division=0)),
        "n_train": int(fit.sum()),
        "n_calibration": int(cal.sum()),
        "n_test": int(test.sum()),
        "base_rate": float(y.mean()),
        "calibrated": served is not model,
    }

    importance = dict(zip(FEATURE_COLUMNS, (int(v) for v in model.feature_importances_)))

    run_id = None
    if _HAS_MLFLOW:
        with mlflow.start_run() as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float) and not np.isnan(v)})
            mlflow.log_dict(importance, "feature_importance.json")
            flavor = mlflow.lightgbm if served is model else mlflow.sklearn
            try:
                flavor.log_model(served, artifact_path="model", registered_model_name="cryptostream-direction")
            except Exception:  # registry may be unavailable with file store
                flavor.log_model(served, artifact_path="model")
            run_id = run.info.run_id
    else:
        print("  (mlflow not installed — skipping experiment logging)")

    # Write then rename, so the API can never load a half-written file.
    joblib.dump({"model": served, "features": FEATURE_COLUMNS}, f"{MODEL_PATH}.tmp")
    os.replace(f"{MODEL_PATH}.tmp", MODEL_PATH)
    # NaN isn't valid JSON, and /model/info would fail to serve it.
    stored = {k: None if isinstance(v, float) and np.isnan(v) else v for k, v in metrics.items()}
    with open(f"{META_PATH}.tmp", "w") as fh:
        json.dump({"metrics": stored, "feature_importance": importance, "mlflow_run_id": run_id}, fh, indent=2)
    os.replace(f"{META_PATH}.tmp", META_PATH)

    print("=== Training complete ===")
    for k, v in metrics.items():
        print(f"  {k:16s}: {v}")
    print(f"  model  -> {MODEL_PATH}")
    if _HAS_MLFLOW:
        print(f"  mlflow -> {MLFLOW_TRACKING_URI} (run {run_id})")


if __name__ == "__main__":
    main()
