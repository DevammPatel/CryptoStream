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


def _build_from_raw(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Recreate the mart logic in SQL if dbt hasn't been run."""
    return con.execute(
        """
        with base as (
            select * from raw.ohlcv_10s where close > 0 and volume > 0
        ),
        -- layer 1: simple returns (single window level)
        rets as (
            select
                symbol, window_start, close, order_flow_imbalance, trade_count, volume,
                close / nullif(lag(close) over w, 0) - 1  as ret_1,
                ln(close / nullif(lag(close) over w, 0))  as log_ret_1
            from base
            window w as (partition by symbol order by window_start)
        ),
        -- layer 2: rolling aggregates over the layer-1 outputs (no nested windows)
        rolled as (
            select
                symbol, window_start, close, ret_1, log_ret_1,
                order_flow_imbalance, trade_count,
                avg(close) over w30                          as sma_30,
                stddev_samp(log_ret_1) over w30              as volatility_30,
                avg(order_flow_imbalance) over w6            as ofi_6,
                volume / nullif(avg(volume) over w30, 0)     as rel_volume,
                lead(close, 3) over w_order                  as future_close_3
            from rets
            window
                w_order as (partition by symbol order by window_start),
                w6  as (partition by symbol order by window_start rows between 5 preceding and current row),
                w30 as (partition by symbol order by window_start rows between 29 preceding and current row)
        )
        select
            symbol, window_start, close,
            ret_1, log_ret_1,
            close / nullif(sma_30, 0) - 1 as px_vs_sma30,
            volatility_30, order_flow_imbalance, ofi_6, rel_volume, trade_count,
            case when future_close_3 > close then 1 else 0 end as label_up_3
        from rolled
        where future_close_3 is not null and volatility_30 is not null
        """
    ).df()


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
    """Most recent feature vector for a symbol, for live inference."""
    df = load_training_frame()
    sub = df[df["symbol"] == symbol.lower()]
    if sub.empty:
        return None
    row = sub.iloc[-1]
    return {c: float(row[c]) for c in FEATURE_COLUMNS}


def build_xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS].astype(float), df[LABEL_COLUMN].astype(int)
