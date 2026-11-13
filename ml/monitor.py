"""Live track record: how the served model does on bars it has never seen.

Every finished live bar gets one prediction from the model being served. Thirty
seconds later the bar at +30s settles it: did the close go up? If that bar never
comes (a gap in the feed), the call is void, just as training gives such a row no
label. Only bars newer than everything the model trained on count (its
trained_through), so the record is out of sample by construction. The API serves the
totals at /model/live and the dashboard shows them next to the model's call.

Records are kept for 24 hours. The API saves them to a file every minute and loads it
on startup, so a restart or a new image keeps the history.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
import pandas as pd
from live import GRACE_MS, MIN_BARS, LiveBars

HORIZON_MS = 30_000  # the label looks one bar 30s ahead (mart_ml_features.sql)
KEEP_MS = 24 * 3600 * 1000


def settle_stats(prob_up: np.ndarray, outcome: np.ndarray) -> dict:
    """Hit rate and Brier score, next to the best constant guess in hindsight."""
    n = len(outcome)
    if n == 0:
        return {"n": 0}
    up_rate = float(outcome.mean())
    return {
        "n": n,
        "hit_rate": float(((prob_up >= 0.5) == outcome).mean()),
        "hit_rate_baseline": max(up_rate, 1 - up_rate),  # always calling the more common side
        "brier": float(((prob_up - outcome) ** 2).mean()),
        "brier_baseline": up_rate * (1 - up_rate),  # always predicting the observed up rate
        "up_rate": up_rate,
    }


class Scorecard:
    """Predictions on live bars, settled once the bar 30s later finishes. Thread-safe."""

    def __init__(self, keep_ms: int = KEEP_MS) -> None:
        self.keep_ms = keep_ms
        self._lock = threading.Lock()
        # symbol -> window_start (ms) -> {prob_up, close, outcome, void, model_id}
        self._records: dict[str, dict[int, dict]] = {}

    def step(
        self,
        store: LiveBars,
        model,
        features: list[str],
        model_id: str | None,
        trained_through_ms: int | None,
        now_ms: int,
    ) -> None:
        """Predict newly finished bars and settle earlier predictions."""
        for symbol in store.symbols():
            df = store.frame(symbol)
            if df is None:
                continue
            finished = df[df["window_end"] + GRACE_MS <= now_ms].reset_index(drop=True)
            closes = dict(zip(finished["window_start"].astype("int64"), finished["close"]))
            newest = max(closes, default=None)
            with self._lock:
                records = self._records.setdefault(symbol, {})
                for start, record in records.items():
                    if record["outcome"] is not None or record["void"]:
                        continue
                    if start + HORIZON_MS in closes:
                        record["outcome"] = int(closes[start + HORIZON_MS] > record["close"])
                    elif newest is not None and newest > start + HORIZON_MS:
                        record["void"] = True  # later bars finished, but not the one at +30s
                seen = set(records)
            if model is None:
                continue
            # The same history the API needs before it serves a call for a symbol.
            new = finished.iloc[MIN_BARS - 1 :]
            new = new[~new["window_start"].isin(seen)].dropna(subset=features)
            if trained_through_ms is not None:
                new = new[new["window_start"] > trained_through_ms]
            if new.empty:
                continue
            prob_up = model.predict_proba(new[features].astype(float))[:, 1]
            with self._lock:
                for start, close, p in zip(new["window_start"].astype("int64"), new["close"], prob_up):
                    later = closes.get(start + HORIZON_MS)
                    records[int(start)] = {
                        "prob_up": float(p),
                        "close": float(close),
                        "outcome": None if later is None else int(later > close),
                        "void": later is None and newest is not None and newest > start + HORIZON_MS,
                        "model_id": model_id,
                    }
        self._prune(now_ms)

    def save(self, path: str) -> None:
        """Write every record to a JSON file (write then rename)."""
        with self._lock:
            data = {sym: {str(start): dict(r) for start, r in by_start.items()} for sym, by_start in self._records.items()}
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)

    def load(self, path: str, now_ms: int) -> int:
        """Add the records an earlier run saved, except those past keep_ms. Returns how many."""
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):  # no file yet, or a damaged one: start afresh
            return 0
        oldest = now_ms - self.keep_ms
        added = 0
        with self._lock:
            for symbol, by_start in data.items():
                records = self._records.setdefault(symbol, {})
                for start, record in by_start.items():
                    if int(start) >= oldest and int(start) not in records:
                        records[int(start)] = record
                        added += 1
        return added

    def _prune(self, now_ms: int) -> None:
        with self._lock:
            for records in self._records.values():
                for start in [s for s in records if s < now_ms - self.keep_ms]:
                    del records[start]

    def summary(self, model_id: str | None = None, symbol: str | None = None) -> dict:
        """Settled predictions, optionally for one model or one symbol only."""
        with self._lock:
            records = [
                r
                for sym, by_start in self._records.items()
                if symbol is None or sym == symbol
                for r in by_start.values()
                if model_id is None or r["model_id"] == model_id
            ]
        settled = [r for r in records if r["outcome"] is not None]
        pending = sum(1 for r in records if r["outcome"] is None and not r["void"])
        stats = settle_stats(
            np.array([r["prob_up"] for r in settled]), np.array([r["outcome"] for r in settled])
        )
        return {**stats, "pending": pending}


def epoch_ms(iso: str | None) -> int | None:
    return None if iso is None else int(pd.Timestamp(iso).value // 1_000_000)
