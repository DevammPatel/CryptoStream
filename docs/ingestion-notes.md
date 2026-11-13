# Ingestion notes

`ingestion/ws_producer.py` streams live trades from Binance into Kafka (Redpanda). It is
the only part of the stack that talks to the exchange in real time.

## What it does

1. Opens Binance's combined trade stream for `SYMBOLS` (one `<symbol>@trade` stream
   each). The default endpoint is `wss://data-stream.binance.vision`, Binance's
   market-data-only host; set `BINANCE_WS_BASE` to use another.
2. Maps each trade to the canonical schema (see "Data contracts" in
   [architecture.md](architecture.md)) and ignores other event types.
3. Publishes it to `crypto.trades`, keyed by symbol, so each symbol's trades stay in
   order within one partition.

The topic has 3 partitions and keeps one day of trades. Spark reads it from there.

## Producer settings

| Setting | Value | Why |
|---|---|---|
| `enable.idempotence` | true | a retried send never writes a trade twice |
| `compression.type` | lz4 | cheap compression for many small messages |
| `linger.ms` | 50 | batches trades for up to 50 ms instead of one request each |

If the local send queue fills up (the broker is slow or down), the producer waits up
to a second for deliveries to drain and retries once.

## Failures

| What happens | What the producer does |
|---|---|
| The connection drops or times out | Flushes, waits, reconnects. The wait doubles from 1 s up to 30 s and resets after a good connection. |
| Binance refuses the client (HTTP 401, 403 or 451) | Exits with a message naming `BINANCE_WS_BASE`, because retrying can't help. |
| A message isn't valid JSON or lacks a field | Logs a warning and skips it. |
| SIGTERM or Ctrl-C | Closes the WebSocket (at most 1 s per step), flushes Kafka (at most 5 s), logs how many trades it sent, and exits 0. Docker allows it 20 s; it takes 2 to 4 s. |

Binance does not replay trades missed while disconnected. The 10-second bars for any
gap can be filled afterwards with `make backfill`, which rebuilds them from Binance's
1-second klines; raw trades for the gap are not recovered.

## Running it outside Docker

```bash
docker compose stop ingestion   # otherwise every trade is sent twice
make ingest-local               # reads KAFKA_BOOTSTRAP (localhost:9092) from .env
```

Spark does not de-duplicate trades by id, so two producers on the same topic would
double every bar's volume and trade count.
