"""Spark's bar logic, run on a small batch of trades.

Needs pyspark and Java, which the main test environment doesn't install; CI runs this
file in its own step. Locally: `make test-spark` runs it inside the Spark image.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

pytest.importorskip("pyspark")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streaming"))

T0 = 1_700_000_000_000  # ms, the start of a 10-second window


def trade(trade_id, price, quantity, offset_ms, buyer_is_maker, symbol="btcusdt"):
    return json.dumps(
        {
            "symbol": symbol,
            "trade_id": trade_id,
            "price": price,
            "quantity": quantity,
            "trade_time": T0 + offset_ms,
            "is_buyer_maker": buyer_is_maker,
            "ingest_time": T0 + offset_ms + 50,
        }
    )


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    # Spark's Python workers must run the same Python as this test.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("cryptostream-tests")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_bars_take_open_and_close_by_trade_time_not_arrival(spark):
    import spark_stream

    arrivals = [
        trade(4, 100.5, 1.0, 9_000, False),  # the close: latest time, and the higher id of a tie
        trade(3, 101.0, 1.0, 9_000, True),
        trade(2, 105.0, 1.0, 5_000, False),
        trade(1, 100.0, 2.0, 1_000, True),  # the open: earliest time, though it arrives late
        trade(5, 102.0, 1.0, 10_000, False),  # the next window
        trade(1, 50.0, 4.0, 2_000, False, symbol="solusdt"),
    ]
    raw = spark.createDataFrame([(value,) for value in arrivals], "value string")
    bars = spark_stream.compute_ohlcv(spark_stream.parse_trades(raw))
    records = [json.loads(r.value) for r in spark_stream.bars_to_kafka(bars).collect()]
    by_key = {(r["symbol"], r["window_start"]): r for r in records}

    first = by_key[("btcusdt", T0)]
    assert (first["open"], first["high"], first["low"], first["close"]) == (100.0, 105.0, 100.0, 100.5)
    assert first["window_end"] == T0 + 10_000
    assert first["volume"] == 5.0 and first["trade_count"] == 4
    assert first["vwap"] == pytest.approx((100.5 + 101.0 + 105.0 + 2 * 100.0) / 5)
    # buyer-is-maker trades are sells: +1 (id 4) -1 (id 3) +1 (id 2) -2 (id 1)
    assert first["net_signed_volume"] == -1.0
    assert first["order_flow_imbalance"] == pytest.approx(-0.2)

    second = by_key[("btcusdt", T0 + 10_000)]
    assert second["open"] == second["close"] == 102.0 and second["trade_count"] == 1
    assert by_key[("solusdt", T0)]["order_flow_imbalance"] == 1.0
