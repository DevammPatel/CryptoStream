"""In-memory live bars for the fast path: Kafka -> API -> dashboard.

Spark publishes every 10s bar to Kafka while the bar is still forming, so one bar
arrives many times and the newest copy wins. The warehouse only refreshes every 15
minutes, which is too slow for a 30-second prediction, so the API keeps the recent
bars here and computes the same indicators dbt computes.

On startup the consumer replays the whole topic (one hour of retention), which
rebuilds the rolling windows after a restart.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid

import numpy as np
import pandas as pd
from features import FEATURE_COLUMNS

log = logging.getLogger("uvicorn.error")

MAX_BARS = 400       # per symbol; a little over the hour of bars the topic retains
MIN_BARS = 31        # the 30-bar window, plus one previous close for the first return
MAX_AGE_MS = 30_000  # older than the model's 30-second horizon means the feed is stale
GRACE_MS = 5_000     # Spark updates every 2 seconds, so a bar's last trades land after it ends


def with_indicators(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep in sync with warehouse/models/intermediate/int_technical_features.sql."""
    df = bars[(bars["close"] > 0) & (bars["volume"] > 0)].sort_values("window_start").reset_index(drop=True)
    previous = df["close"].shift(1)
    df["ret_1"] = df["close"] / previous - 1
    df["log_ret_1"] = np.log(df["close"] / previous)
    sma_30 = df["close"].rolling(30, min_periods=1).mean()
    df["px_vs_sma30"] = df["close"] / sma_30 - 1
    df["volatility_30"] = df["log_ret_1"].rolling(30, min_periods=2).std(ddof=1)
    df["ofi_6"] = df["order_flow_imbalance"].rolling(6, min_periods=1).mean()
    df["rel_volume"] = df["volume"] / df["volume"].rolling(30, min_periods=1).mean()
    return df


class LiveBars:
    """Thread-safe store of recent bars: the Kafka thread writes, requests read."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bars: dict[str, dict[int, dict]] = {}

    def update(self, bar: dict) -> None:
        with self._lock:
            bars = self._bars.setdefault(bar["symbol"], {})
            bars[int(bar["window_start"])] = bar
            if len(bars) > MAX_BARS:
                del bars[min(bars)]

    def symbols(self) -> list[str]:
        with self._lock:
            return sorted(self._bars)

    def frame(self, symbol: str) -> pd.DataFrame | None:
        """All bars for a symbol with indicators, including the one still forming."""
        with self._lock:
            rows = list(self._bars.get(symbol.lower(), {}).values())
        if not rows:
            return None
        return with_indicators(pd.DataFrame(rows))

    def latest_features(self, symbol: str, now_ms: int | None = None) -> dict | None:
        """Features from the newest finished bar, or None if the feed can't supply them.

        The bar still forming is skipped: its volume and trade count are partial, and
        the model only ever trained on finished bars.
        """
        df = self.frame(symbol)
        if df is None:
            return None
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        finished = df[df["window_end"] + GRACE_MS <= now_ms]
        if len(finished) < MIN_BARS:
            return None
        row = finished.iloc[-1]
        if now_ms - row["window_end"] > MAX_AGE_MS:
            return None
        as_of = pd.Timestamp(int(row["window_start"]), unit="ms", tz="UTC").isoformat()
        return {**{c: float(row[c]) for c in FEATURE_COLUMNS}, "as_of": as_of}


def consume_bars(store: LiveBars, bootstrap: str, topic: str, stop: threading.Event) -> None:
    """Feed the store from Kafka until `stop` is set. Runs on its own thread."""
    from confluent_kafka import Consumer

    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"cryptostream-api-{uuid.uuid4()}",  # a new group replays the topic
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([topic])
    log.info("consuming live bars from %s on %s", topic, bootstrap)
    try:
        while not stop.is_set():
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.warning("live bars: %s", msg.error())
                continue
            try:
                store.update(json.loads(msg.value()))
            except (ValueError, KeyError) as exc:
                log.warning("live bars: skipped bad message (%s)", exc)
    finally:
        consumer.close()
