"""Backfill the feature lake with real history from Binance.

Downloads 1-second klines from Binance (no key needed) and rolls them up into the same
10s bars Spark writes, so the model can train on days of real data instead of whatever
the live pipeline has collected so far. Finished days come from Binance's bulk files
(one checksummed download per day); today, which has no file yet, comes from the REST
API, several pages at a time.

Each kline carries its taker-buy volume, which gives order flow exactly: buys hit by
aggressive buyers count +, the rest count -, as in streaming/spark_stream.py.

Writes one file per symbol per UTC day, symbol=<s>/backfill-<day>.parquet, into
LAKE_URI (a local folder or s3://). Rerunning overwrites those files, and dbt keeps
the most complete copy where a backfilled bar overlaps one Spark wrote.

Run:   python -m cryptostream_dagster.backfill --days 7
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import duckdb
import pandas as pd

from cryptostream_dagster.definitions import connect_lake_storage, lake_uri

REST_BASE = os.getenv("BINANCE_REST_BASE", "https://data-api.binance.vision")
BULK_BASE = os.getenv("BINANCE_BULK_BASE", "https://data.binance.vision")
BAR_MS = 10_000
PAGE_MS = 1000 * 1000      # 1000 one-second klines per request, the API's maximum
WEIGHT_LIMIT = 5000        # Binance allows 6000 request weight per minute; stay under it
PARALLEL_PAGES = 8         # each page takes seconds, so fetch several at once
COLUMNS = [
    "window_start", "window_end", "open", "high", "low", "close", "volume", "vwap",
    "trade_count", "net_signed_volume", "order_flow_imbalance",
]


def _get(url: str) -> tuple[bytes, dict]:
    """GET with retries on rate limits, server errors and dropped connections."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 418:  # IP banned for ignoring 429s: stop, don't make it worse
                raise SystemExit("Binance has temporarily banned this IP (HTTP 418). Try again later.") from exc
            if exc.code != 429 and exc.code < 500:
                raise
            time.sleep(int(exc.headers.get("Retry-After") or 2**attempt))
        except urllib.error.URLError:
            time.sleep(2**attempt)
    raise SystemExit(f"Binance kept failing for {url}; try again later.")


def _rest_page(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    body, headers = _get(
        f"{REST_BASE}/api/v3/klines?symbol={symbol.upper()}&interval=1s"
        f"&startTime={start_ms}&endTime={end_ms - 1}&limit=1000"
    )
    if int(headers.get("x-mbx-used-weight-1m") or headers.get("X-Mbx-Used-Weight-1m") or 0) > WEIGHT_LIMIT:
        time.sleep(10)
    return json.loads(body)


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """1s klines with open time in [start_ms, end_ms), from the REST API.

    Each page covers 1000 seconds, so pages are independent and fetched in parallel.
    Seconds with no trades have no kline, so a page can hold fewer than 1000 rows.
    """
    pages = [(p, min(p + PAGE_MS, end_ms)) for p in range(start_ms, end_ms, PAGE_MS)]
    with ThreadPoolExecutor(PARALLEL_PAGES) as pool:
        results = pool.map(lambda page: _rest_page(symbol, *page), pages)
        return [row for page in results for row in page]


def fetch_day_file(symbol: str, day: str) -> list[list] | None:
    """A whole UTC day of 1s klines from Binance's bulk files, or None if not published yet."""
    url = f"{BULK_BASE}/data/spot/daily/klines/{symbol.upper()}/1s/{symbol.upper()}-1s-{day}.zip"
    try:
        body, _ = _get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # a day's file appears the following day
            return None
        raise
    expected = _get(f"{url}.CHECKSUM")[0].decode().split()[0]
    if hashlib.sha256(body).hexdigest() != expected:
        raise SystemExit(f"Checksum mismatch for {url}")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        frame = pd.read_csv(z.open(z.namelist()[0]), header=None)
    return frame.values.tolist()


def bars_from_klines(rows: list[list]) -> pd.DataFrame:
    """Roll 1s klines up into 10s bars with the same columns Spark writes."""
    k = pd.DataFrame(
        [r[:11] for r in rows],
        columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote"],
    )
    for c in ("open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume"):
        k[c] = k[c].astype(float)
    k["open_time"] = k["open_time"].astype("int64")
    # The bulk files switched to microsecond timestamps in 2025; the REST API uses ms.
    if k["open_time"].max() > 10**14:
        k["open_time"] //= 1000
    k = k[k["volume"] > 0].sort_values("open_time")
    k["window"] = k["open_time"] // BAR_MS * BAR_MS
    bars = k.groupby("window").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        notional=("quote_volume", "sum"),
        trade_count=("trades", "sum"),
        taker_buy=("taker_buy_volume", "sum"),
    ).reset_index()
    bars["window_start"] = pd.to_datetime(bars["window"], unit="ms")
    bars["window_end"] = bars["window_start"] + pd.Timedelta(milliseconds=BAR_MS)
    bars["vwap"] = bars["notional"] / bars["volume"]
    bars["net_signed_volume"] = 2 * bars["taker_buy"] - bars["volume"]
    bars["order_flow_imbalance"] = bars["net_signed_volume"] / bars["volume"]
    bars["trade_count"] = bars["trade_count"].astype("int64")
    return bars[COLUMNS]


def write_day(con: duckdb.DuckDBPyConnection, bars: pd.DataFrame, symbol: str, day: str) -> str:
    """Write one day's bars, keeping bars an earlier run wrote for the rest of that day.

    A shorter rerun (say the last 6 hours) would otherwise replace a whole day. Where
    both runs have a window, the new bar wins.
    """
    target = f"{lake_uri()}/symbol={symbol}/backfill-{day}.parquet"
    if not target.startswith("s3://"):
        os.makedirs(os.path.dirname(target), exist_ok=True)
    con.register("new_bars", bars)
    try:
        # Read the old file fully before the COPY below overwrites it.
        con.execute(f"CREATE OR REPLACE TEMP TABLE old_bars AS SELECT * FROM read_parquet('{target}')")
        source = """
            SELECT * FROM new_bars
            UNION ALL BY NAME
            SELECT * FROM old_bars WHERE window_start NOT IN (SELECT window_start FROM new_bars)
            ORDER BY window_start
        """
    except duckdb.IOException:  # no earlier file for this day
        source = "SELECT * FROM new_bars"
    con.execute(f"COPY ({source}) TO '{target}' (FORMAT PARQUET)")
    con.execute("DROP TABLE IF EXISTS old_bars")
    con.unregister("new_bars")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=float, default=7, help="how many days back from now")
    ap.add_argument("--symbols", default=os.getenv("SYMBOLS", "btcusdt,ethusdt,solusdt"))
    args = ap.parse_args()

    symbols = [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
    now = datetime.now(timezone.utc)
    end = now.replace(microsecond=0) - timedelta(seconds=now.second % 10)  # last finished bar
    start = end - timedelta(days=args.days)

    con = duckdb.connect()
    connect_lake_storage(con)
    for symbol in symbols:
        day = start
        while day < end:
            day_end = min(datetime.combine(day.date() + timedelta(days=1), datetime.min.time(), timezone.utc), end)
            start_ms, end_ms = int(day.timestamp() * 1000), int(day_end.timestamp() * 1000)
            rows = fetch_day_file(symbol, f"{day:%Y-%m-%d}") if day.date() < now.date() else None
            if rows is None:
                rows = fetch_klines(symbol, start_ms, end_ms)
            open_ms = [r[0] // 1000 if r[0] > 10**14 else r[0] for r in rows]
            rows = [r for r, t in zip(rows, open_ms) if start_ms <= t < end_ms]  # the window may cut a day
            if rows:
                bars = bars_from_klines(rows)
                target = write_day(con, bars, symbol, day.strftime("%Y-%m-%d"))
                print(f"{symbol} {day:%Y-%m-%d %H:%M} -> {day_end:%H:%M}: {len(bars):,} bars -> {target}", flush=True)
            day = day_end
    con.close()


if __name__ == "__main__":
    main()
