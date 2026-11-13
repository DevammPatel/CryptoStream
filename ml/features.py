"""Shared feature definitions + loading helpers for the ML layer.

Reads the dbt-built `marts.mart_ml_features` table from DuckDB. If dbt hasn't run
yet, falls back to building an equivalent frame directly from `raw.ohlcv_10s` so the
ML pipeline is runnable stand-alone (useful in CI).
"""
from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DUCKDB = os.path.join(HERE, "data", "warehouse", "crypto.duckdb")

FEATURE_COLUMNS = [
    "ret_1",
    "log_ret_1",
    "px_vs_sma30",
    "volatility_30",
    "order_flow_imbalance",
    "ofi_6",
    "rel_volume",
    "trade_count",
]
LABEL_COLUMN = "label_up_3"


def duckdb_path() -> str:
    return os.getenv("DUCKDB_PATH", DEFAULT_DUCKDB)


def _table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> bool:
    q = """
        select count(*) from information_schema.tables
        where table_schema = ? and table_name = ?
    """
    return con.execute(q, [schema, table]).fetchone()[0] > 0


def one_bar_per_window(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop degenerate bars and duplicate copies, keeping the most complete one.

    Keep in sync with stg_ohlcv.sql.
    """
    bars = bars[(bars["close"] > 0) & (bars["volume"] > 0)]
    bars = bars.sort_values(["trade_count", "volume"], ascending=False)
    return bars.drop_duplicates(["symbol", "window_start"]).sort_values(["symbol", "window_start"])


def with_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Indicators for one symbol's bars. Keep in sync with int_technical_features.sql.

    Windows are measured in time, not rows: a 300s window holds the last 30 bars when
    none are missing, and a return only exists when the previous bar is exactly 10
    seconds earlier. window_start may be a timestamp or epoch milliseconds.
    """
    df = bars[(bars["close"] > 0) & (bars["volume"] > 0)].sort_values("window_start").reset_index(drop=True)
    t = df["window_start"]
    df["_t"] = pd.to_datetime(t, unit="ms") if pd.api.types.is_integer_dtype(t) else pd.to_datetime(t)
    follows = (df["_t"] - df["_t"].shift(1)) == pd.Timedelta(seconds=10)
    previous = df["close"].shift(1).where(follows)
    df["ret_1"] = df["close"] / previous - 1
    df["log_ret_1"] = np.log(df["close"] / previous)
    w30, w6 = df.rolling("300s", on="_t"), df.rolling("60s", on="_t")
    df["px_vs_sma30"] = df["close"] / w30["close"].mean() - 1
    df["volatility_30"] = df.rolling("300s", on="_t", min_periods=2)["log_ret_1"].std(ddof=1)
    df["ofi_6"] = w6["order_flow_imbalance"].mean()
    df["rel_volume"] = df["volume"] / w30["volume"].mean()
    return df.drop(columns="_t")


def _build_from_raw(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Recreate the ML mart in Python if dbt hasn't run. A test checks it matches dbt."""
    raw = one_bar_per_window(con.execute("select * from raw.ohlcv_10s").df())
    frames = []
    for _, bars in raw.groupby("symbol"):
        df = with_indicators(bars)
        # Label from the bar exactly 30s later, as in mart_ml_features.sql.
        future = df[["window_start", "close"]].rename(columns={"close": "future_close_3"})
        future["window_start"] = future["window_start"] - pd.Timedelta(seconds=30)
        frames.append(df.merge(future, on="window_start", how="left"))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["future_close_3"].notna() & df["volatility_30"].notna()].copy()
    df["label_up_3"] = (df["future_close_3"] > df["close"]).astype(int)
    return df


def load_training_frame() -> pd.DataFrame:
    con = duckdb.connect(duckdb_path(), read_only=True)
    try:
        if _table_exists(con, "marts", "mart_ml_features"):
            df = con.execute("select * from marts.mart_ml_features").df()
        else:
            df = _build_from_raw(con)
    finally:
        con.close()

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])
    return df.sort_values("window_start").reset_index(drop=True)


def latest_feature_row(symbol: str) -> dict | None:
    """Most recent feature vector for a symbol in the warehouse.

    Reads the intermediate view because the ML mart drops the newest 3 bars, which
    have no label yet. Falls back to the training frame if dbt hasn't run.
    """
    symbol = symbol.lower()
    df = None
    con = duckdb.connect(duckdb_path(), read_only=True)
    try:
        if _table_exists(con, "intermediate", "int_technical_features"):
            df = con.execute(
                f"select window_start, {', '.join(FEATURE_COLUMNS)} "
                "from intermediate.int_technical_features "
                "where symbol = ? and volatility_30 is not null "
                "order by window_start desc limit 1",
                [symbol],
            ).df()
    finally:
        con.close()
    if df is None:
        df = load_training_frame()
        df = df[df["symbol"] == symbol]
    if df.empty:
        return None
    row = df.iloc[-1]
    as_of = pd.Timestamp(row["window_start"])
    as_of = as_of.tz_localize("UTC") if as_of.tzinfo is None else as_of.tz_convert("UTC")
    return {**{c: float(row[c]) for c in FEATURE_COLUMNS}, "as_of": as_of.isoformat()}


def build_xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS].astype(float), df[LABEL_COLUMN].astype(int)
