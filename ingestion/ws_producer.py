"""Binance WebSocket -> Redpanda/Kafka producer.

Connects to Binance's free public combined trade stream (no API key required) and
publishes normalised trade events to a Kafka topic. Designed to run forever with
automatic reconnection and exponential backoff.

Run locally:   python ingestion/ws_producer.py
Env vars:      KAFKA_BOOTSTRAP, SYMBOLS, TRADES_TOPIC, BINANCE_WS_BASE
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from typing import Any

# NOTE: `websockets` and `confluent_kafka` are imported lazily inside the functions
# that need them, so pure helpers like `normalise` stay importable (and unit-testable)
# without the networking/Kafka dependencies installed.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [ingestion] %(message)s",
)
log = logging.getLogger("ingestion")

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
SYMBOLS = [s.strip().lower() for s in os.getenv("SYMBOLS", "btcusdt,ethusdt,solusdt").split(",") if s.strip()]
TRADES_TOPIC = os.getenv("TRADES_TOPIC", "crypto.trades")

# Binance's market-data-only endpoint. wss://stream.binance.com:9443/stream?streams=
# sends the same data but answers HTTP 451 in some regions, such as the US.
BINANCE_WS_BASE = os.getenv("BINANCE_WS_BASE", "wss://data-stream.binance.vision/stream?streams=")

# HTTP statuses that mean Binance refuses this client, so reconnecting cannot help.
FATAL_STATUSES = {401, 403, 451}


def build_stream_url(symbols: list[str]) -> str:
    """Binance combined stream URL, e.g. btcusdt@trade/ethusdt@trade."""
    streams = "/".join(f"{s}@trade" for s in symbols)
    return f"{BINANCE_WS_BASE}{streams}"


def normalise(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Map a Binance trade payload to our canonical schema.

    Binance @trade fields: e(event), E(event time), s(symbol), t(trade id),
    p(price), q(qty), T(trade time), m(buyer is maker).
    """
    data = raw.get("data", raw)
    if data.get("e") != "trade":
        return None
    return {
        "symbol": data["s"].lower(),
        "trade_id": int(data["t"]),
        "price": float(data["p"]),
        "quantity": float(data["q"]),
        "trade_time": int(data["T"]),          # ms epoch
        "is_buyer_maker": bool(data["m"]),     # True => sell-side aggressor
        "ingest_time": int(time.time() * 1000),
    }


def make_producer():
    from confluent_kafka import Producer

    conf = {
        "bootstrap.servers": KAFKA_BOOTSTRAP,
        "client.id": "cryptostream-ingestion",
        "linger.ms": 50,
        "compression.type": "lz4",
        "enable.idempotence": True,
    }
    return Producer(conf)


def _delivery_report(err, msg) -> None:
    if err is not None:
        log.warning("delivery failed: %s", err)


def is_fatal(exc: Exception) -> bool:
    """True when reconnecting cannot help, e.g. HTTP 451 from a restricted region."""
    return getattr(exc, "status_code", None) in FATAL_STATUSES


def publish(producer, event: dict[str, Any]) -> None:
    kwargs = {"key": event["symbol"].encode(), "value": json.dumps(event).encode(), "callback": _delivery_report}
    try:
        producer.produce(TRADES_TOPIC, **kwargs)
    except BufferError:
        # The local queue is full because the broker is slow or down: let deliveries
        # drain, then retry once.
        producer.poll(1)
        producer.produce(TRADES_TOPIC, **kwargs)
    producer.poll(0)


async def stream_forever() -> None:
    import websockets

    url = build_stream_url(SYMBOLS)
    producer = make_producer()
    log.info("bootstrap=%s topic=%s symbols=%s", KAFKA_BOOTSTRAP, TRADES_TOPIC, SYMBOLS)

    backoff = 1
    count = 0
    try:
        while True:
            try:
                # Closing waits up to close_timeout at each of up to five steps, and
                # Binance usually leaves the goodbye unanswered. With 1 s, closing plus the
                # 5 s flush below always fits in the 20 s stop_grace_period in
                # docker-compose.yml.
                async with websockets.connect(url, ping_interval=20, ping_timeout=20, close_timeout=1) as ws:
                    log.info("connected to %s", url.split("/stream")[0])
                    backoff = 1  # reset after a successful connect
                    async for message in ws:
                        try:
                            event = normalise(json.loads(message))
                        except (ValueError, KeyError) as exc:
                            log.warning("skipped a malformed message (%s)", exc)
                            continue
                        if event is None:
                            continue
                        publish(producer, event)
                        count += 1
                        if count % 500 == 0:
                            log.info("produced %d trades (last %s @ %.2f)", count, event["symbol"], event["price"])
            except (websockets.WebSocketException, OSError) as exc:
                if is_fatal(exc):
                    raise SystemExit(
                        f"Binance refused the connection ({exc}). If your region is blocked, "
                        "set BINANCE_WS_BASE to another endpoint."
                    ) from exc
                log.warning("connection dropped (%s); reconnecting in %ss", exc, backoff)
                producer.flush(5)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
    finally:
        producer.flush(5)
        log.info("shutting down after %d trades, producer flushed", count)


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(stream_forever())
    # Cancelling the task unwinds it at its current await, which runs the flush in
    # stream_forever's finally block. Raising from a signal handler instead would only
    # be logged by the event loop, and the process would keep running.
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, task.cancel)
        except NotImplementedError:  # pragma: no cover (Windows)
            pass
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
