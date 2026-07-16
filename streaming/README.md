# Streaming pipeline

The streaming layer turns incoming trade events into useful OHLCV and feature signals.

The workflow includes:
- ingesting messages from the broker feed
- grouping events into windows
- producing derived features for later modeling
