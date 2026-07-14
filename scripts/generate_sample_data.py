"""Generate realistic synthetic OHLCV feature data — no network or Docker needed.

Produces the same schema the Spark job writes (features/ohlcv_10s) so that dbt, the
ML pipeline, the API, and the dashboard all work end-to-end for a quick local demo
or in CI. Writes both Parquet (the "lake") and a DuckDB serving DB.

Usage:  python scripts/generate_sample_data.py [--minutes 600] [--symbols btcusdt,ethusdt]
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
LAKE_DIR = os.path.join(DATA_DIR, "lake", "features", "ohlcv_10s")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", os.path.join(DATA_DIR, "warehouse", "crypto.duckdb"))

START_PRICES = {"btcusdt": 62000.0, "ethusdt": 3400.0, "solusdt": 145.0}


def simulate_symbol(symbol: str, start_price: float, n: int, start_ts: datetime) -> pd.DataFrame:
    """Geometric-brownian-motion-ish 10s OHLCV bars with plausible micro-structure."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    dt = 10.0 / (60 * 60 * 24)  # 10 seconds in days
    mu, sigma = 0.05, 0.9       # annualised drift / vol
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n)
    close = start_price * np.exp(np.cumsum(shocks))
    open_ = np.concatenate([[start_price], close[:-1]])

    intrabar = np.abs(rng.normal(0, sigma * np.sqrt(dt) * 0.6, n)) * close
    high = np.maximum(open_, close) + intrabar
    low = np.minimum(open_, close) - intrabar

    volume = np.abs(rng.normal(12, 5, n)) + 0.5
    trade_count = rng.poisson(40, n) + 1
    net_signed = rng.normal(0, 1, n) * volume * 0.3
    ofi = np.clip(net_signed / volume, -1, 1)
    vwap = (high + low + close) / 3

    windows = [start_ts + timedelta(seconds=10 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "window_start": windows,
            "window_end": [w + timedelta(seconds=10) for w in windows],
            "symbol": symbol,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "vwap": vwap,
            "trade_count": trade_count,
            "net_signed_volume": net_signed,
            "order_flow_imbalance": ofi,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, default=600, help="minutes of 10s bars to generate")
    ap.add_argument("--symbols", type=str, default="btcusdt,ethusdt,solusdt")
    args = ap.parse_args()

    symbols = [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
    n = args.minutes * 6  # 6 ten-second bars per minute
    start_ts = datetime.now(timezone.utc) - timedelta(minutes=args.minutes)

    os.makedirs(LAKE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)

    frames = []
    for sym in symbols:
        df = simulate_symbol(sym, START_PRICES.get(sym, 100.0), n, start_ts)
        part_dir = os.path.join(LAKE_DIR, f"symbol={sym}")
        os.makedirs(part_dir, exist_ok=True)
        df.drop(columns=["symbol"]).to_parquet(os.path.join(part_dir, "part-000.parquet"), index=False)
        frames.append(df)
        print(f"  {sym}: {len(df):,} bars")

    allframes = pd.concat(frames, ignore_index=True)

    con = duckdb.connect(DUCKDB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    con.execute("DROP TABLE IF EXISTS raw.ohlcv_10s;")
    con.register("df", allframes)
    con.execute("CREATE TABLE raw.ohlcv_10s AS SELECT * FROM df;")
    con.close()

    print(f"\nWrote {len(allframes):,} rows")
    print(f"  Parquet lake -> {LAKE_DIR}")
    print(f"  DuckDB       -> {DUCKDB_PATH} (raw.ohlcv_10s)")


if __name__ == "__main__":
    main()
