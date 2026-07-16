# Ingestion service

This service collects market data from the configured exchange feed and passes it into the platform.

It is responsible for:
- connecting to the provider stream
- normalizing incoming payloads
- publishing the data for downstream processing
