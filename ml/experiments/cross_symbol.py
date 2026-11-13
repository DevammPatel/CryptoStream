"""Do cross-symbol features improve the direction model?

Trains the model twice on the warehouse, with ml/train.py's splits, settings and
calibration: once on the 8 served features, and once with 5 more taken from the other
symbols' bars that finish at the same moment (BTC's last two moves, the other symbols'
average move and order flow, and the symbol's move relative to all of them). Prints
walk-forward CV and held-out results for both.

Not part of the pipeline: the features are not used (see ml/README.md for the result).

    python ml/experiments/cross_symbol.py [path/to/crypto.duckdb]
"""
from __future__ import annotations

import os
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

ML_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ML_DIR)
from features import FEATURE_COLUMNS, LABEL_COLUMN, duckdb_path  # noqa: E402
from train import HORIZON_NS, epoch_ns, purged_splits  # noqa: E402

CROSS = ["btc_ret_1", "btc_ret_prev", "others_ret_1", "others_ofi", "rel_ret_1"]
PARAMS = {"objective": "binary", "n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31,
          "max_depth": -1, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
          "random_state": 42, "n_jobs": 2, "verbose": -1}


def with_cross_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """Add CROSS, from bars that finish at the same moment as each row's own bar."""
    wide = df.pivot_table(index="window_start", columns="symbol", values=["ret_1", "order_flow_imbalance"])
    totals = pd.DataFrame({
        "n_sym": wide["ret_1"].notna().sum(axis=1),
        "sum_ret": wide["ret_1"].sum(axis=1),
        "sum_ofi": wide["order_flow_imbalance"].sum(axis=1),
    })
    df = df.join(totals, on="window_start")
    others = (df["n_sym"] - 1).where(lambda n: n > 0)
    btc = df[df["symbol"] == "btcusdt"].set_index("window_start")["ret_1"]
    df["btc_ret_1"] = df["window_start"].map(btc)
    df["btc_ret_prev"] = df["window_start"].map(btc.shift(1))  # BTC one bar earlier, if it leads
    df["others_ret_1"] = (df["sum_ret"] - df["ret_1"]) / others
    df["others_ofi"] = (df["sum_ofi"] - df["order_flow_imbalance"]) / others
    df["rel_ret_1"] = df["ret_1"] - df["sum_ret"] / df["n_sym"]
    return df


def evaluate(df: pd.DataFrame, cols: list[str]) -> dict:
    X, y, t = df[cols].astype(float), df[LABEL_COLUMN].astype(int), epoch_ns(df["window_start"])
    cv = []
    for tr, va in purged_splits(t):
        model = lgb.LGBMClassifier(**PARAMS).fit(X.iloc[tr], y.iloc[tr])
        cv.append(roc_auc_score(y.iloc[va], model.predict_proba(X.iloc[va])[:, 1]))
    clock = np.unique(t)
    cal_start, test_start = clock[int(len(clock) * 0.6)], clock[int(len(clock) * 0.8)]
    fit, cal, test = t + HORIZON_NS < cal_start, (t >= cal_start) & (t + HORIZON_NS < test_start), t >= test_start
    model = lgb.LGBMClassifier(**PARAMS).fit(X[fit], y[fit])
    served = CalibratedClassifierCV(model, method="sigmoid", cv="prefit").fit(X[cal], y[cal])
    proba = served.predict_proba(X[test])[:, 1]
    return {"cv": cv, "test_auc": roc_auc_score(y[test], proba), "test_brier": brier_score_loss(y[test], proba)}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else duckdb_path()
    con = duckdb.connect(path, read_only=True)
    df = con.execute("select * from marts.mart_ml_features order by window_start, symbol").df()
    con.close()
    df = with_cross_symbol(df)
    print(f"{len(df)} rows, {df['window_start'].min()} to {df['window_start'].max()}")
    base, cross = evaluate(df, FEATURE_COLUMNS), evaluate(df, FEATURE_COLUMNS + CROSS)
    for name, r in (("served features", base), ("with cross-symbol", cross)):
        folds = " ".join(f"{a:.4f}" for a in r["cv"])
        print(f"{name:18} CV AUC {np.mean(r['cv']):.4f} ({folds})  test AUC {r['test_auc']:.4f}  "
              f"Brier {r['test_brier']:.5f}")
    gains = np.array(cross["cv"]) - np.array(base["cv"])
    print(f"gain: CV {gains.mean():+.4f} (better in {int((gains > 0).sum())} of {len(gains)} folds, "
          f"latest {gains[-1]:+.4f}), test AUC {cross['test_auc'] - base['test_auc']:+.4f}")


if __name__ == "__main__":
    main()
