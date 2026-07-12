"""Binance WebSocket -> Redpanda/Kafka producer.

Connects to Binance's free public combined trade stream (no API key required) and
publishes normalised trade events to a Kafka topic. Designed to run forever with
automatic reconnection and exponential backoff.

Run locally:   python ingestion/ws_producer.py
Env vars:      KAFKA_BOOTSTRAP, SYMBOLS, TRADES_TOPIC
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

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="


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


class GracefulExit(Exception):
    pass


async def stream_forever() -> None:
    import websockets

    url = build_stream_url(SYMBOLS)
    producer = make_producer()
    log.info("bootstrap=%s topic=%s symbols=%s", KAFKA_BOOTSTRAP, TRADES_TOPIC, SYMBOLS)

    backoff = 1
    count = 0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                log.info("connected to Binance stream")
                backoff = 1  # reset after a successful connect
                async for message in ws:
                    raw = json.loads(message)
                    event = normalise(raw)
                    if event is None:
                        continue
                    producer.produce(
                        TRADES_TOPIC,
                        key=event["symbol"].encode(),
                        value=json.dumps(event).encode(),
                        callback=_delivery_report,
                    )
                    count += 1
                    if count % 500 == 0:
                        producer.poll(0)
                        log.info("produced %d trades (last %s @ %.2f)", count, event["symbol"], event["price"])
                    else:
                        producer.poll(0)
        except (websockets.ConnectionClosed, OSError) as exc:
            log.warning("connection dropped (%s); reconnecting in %ss", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
        except GracefulExit:
            break
        finally:
            producer.flush(5)
    log.info("shutting down, flushed producer")


def _install_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    def _raise(*_):
        raise GracefulExit()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _raise)
        except NotImplementedError:  # pragma: no cover (Windows)
            pass


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    _install_signal_handlers(loop)
    try:
        loop.run_until_complete(stream_forever())
    except GracefulExit:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
