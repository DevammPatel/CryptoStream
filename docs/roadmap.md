# CryptoStream roadmap

## Done

1. **The stack starts.** `make up` brings every service up from a fresh clone: Redpanda
   with its topics, SeaweedFS as the S3 lake, Spark, Dagster on a 15-minute schedule,
   MLflow, the API and the dashboard.
2. **The pipeline is connected.** Live trades flow from Binance through Kafka and Spark
   into the lake; Dagster loads it, builds the dbt marts, trains, and writes a summary.
3. **A real-time fast path.** Spark also publishes bars to Kafka as they form, and the
   API predicts from bars seconds old instead of the 15-minute warehouse.
4. **The promised features.** A promotion gate keeps a worse model from replacing the
   served one; the market summary includes the model's calls and is written once per
   refresh; the API keeps a live track record on bars the model never trained on.
5. **Trustworthy results.** Time-based windows and labels, one bar per window, plain
   UTC, open and close by trade time, purged and calibrated training with baselines, a
   real-history backfill, and lake compaction with retention.
6. **Tests, CI and docs.** Tests cover ingestion, dbt parity, training and promotion,
   the API, the summary, the dashboard, Spark's bar logic, the backfill and compaction.
   CI also builds and smoke-tests the whole Docker stack.

## Known limits

- **The live track record is kept in memory.** An API restart keeps only what the
  one-hour bars topic can replay. Saving it to the lake would keep a longer history.
- **Spark's live query saves progress to S3 every 2 seconds.** Each 2-second batch takes
  about 3 seconds, so live bars arrive about a second later than planned. A local
  checkpoint volume would remove most of that.
- **Spark does not de-duplicate trades by id.** Two producers on one topic double the
  volume (see [ingestion-notes.md](ingestion-notes.md)).
- **One machine, no authentication.** The stack is for development and demos: every
  service listens without a password, and SeaweedFS uses a fixed key.
- **The model's edge is small.** About 0.525 AUC on a week of real data. Richer
  features (cross-symbol moves, order-book data) are the obvious next step.
