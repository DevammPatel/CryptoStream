"""Spark Structured Streaming: Kafka trades -> features -> Parquet lake (SeaweedFS).

Reads the raw trade stream from Redpanda, computes windowed OHLCV + micro-structure
features per symbol, and writes them as Parquet to the S3-compatible SeaweedFS lake.

Three sinks:
  1. Raw trades          -> s3a://<bucket>/raw/trades/            (append, exactly-once via checkpoint)
  2. 10s OHLCV features  -> s3a://<bucket>/features/ohlcv_10s/    (append)
  3. Live bars           -> Kafka topic crypto.bars_10s           (update, every 2 seconds)

Append only writes a bar once the watermark passes it, so the lake is complete but
about a minute behind. Update publishes each bar while it is still forming, which is
what the API and dashboard read for the live view.

Run inside the `spark` container (see streaming/Dockerfile), which supplies the
Kafka + Hadoop-AWS jars. Configure via env vars (see docker-compose.yml).
"""
from __future__ import annotations

import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "redpanda:29092")
TRADES_TOPIC = os.getenv("TRADES_TOPIC", "crypto.trades")
BARS_TOPIC = os.getenv("BARS_TOPIC", "crypto.bars_10s")
LAKE_ENDPOINT = os.getenv("LAKE_ENDPOINT", "http://seaweedfs:8333")
LAKE_ACCESS_KEY = os.getenv("LAKE_ACCESS_KEY", "lakeadmin")
LAKE_SECRET_KEY = os.getenv("LAKE_SECRET_KEY", "lakeadmin")
LAKE_BUCKET = os.getenv("LAKE_BUCKET", "crypto-lake")

TRADE_SCHEMA = StructType(
    [
        StructField("symbol", StringType()),
        StructField("trade_id", LongType()),
        StructField("price", DoubleType()),
        StructField("quantity", DoubleType()),
        StructField("trade_time", LongType()),
        StructField("is_buyer_maker", BooleanType()),
        StructField("ingest_time", LongType()),
    ]
)


def build_spark() -> SparkSession:
    spark = (
        SparkSession.builder.appName("cryptostream-features")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.hadoop.fs.s3a.endpoint", LAKE_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", LAKE_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", LAKE_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def read_trades(spark: SparkSession):
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TRADES_TOPIC)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 20000)
        # If the stack was off longer than Kafka's retention, carry on from what is
        # left. A gap in a live feed is better than a job that can never restart.
        .option("failOnDataLoss", "false")
        .load()
    )
    parsed = (
        raw.select(F.from_json(F.col("value").cast("string"), TRADE_SCHEMA).alias("t"))
        .select("t.*")
        .withColumn("event_ts", (F.col("trade_time") / 1000).cast("timestamp"))
        .withWatermark("event_ts", "30 seconds")
    )
    return parsed


def compute_ohlcv(trades):
    """10-second OHLCV + order-flow features per symbol."""
    signed_qty = F.when(F.col("is_buyer_maker"), -F.col("quantity")).otherwise(F.col("quantity"))
    notional = F.col("price") * F.col("quantity")
    # Trades in a window can arrive in any order across partitions and batches, so open
    # and close come from the earliest and latest trade time, with the trade id as a
    # tie-break for trades in the same millisecond. first()/last() only see arrival order.
    trade_order = F.struct("trade_time", "trade_id")

    agg = (
        trades.withColumn("signed_qty", signed_qty)
        .withColumn("notional", notional)
        .groupBy(F.window("event_ts", "10 seconds"), F.col("symbol"))
        .agg(
            F.min_by("price", trade_order).alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.max_by("price", trade_order).alias("close"),
            F.sum("quantity").alias("volume"),
            F.sum("notional").alias("notional"),
            F.count("*").alias("trade_count"),
            F.sum("signed_qty").alias("net_signed_volume"),
        )
        .withColumn("vwap", F.col("notional") / F.col("volume"))
        .withColumn(
            "order_flow_imbalance",
            F.col("net_signed_volume") / F.when(F.col("volume") == 0, F.lit(None)).otherwise(F.col("volume")),
        )
        .select(
            F.col("window.start").alias("window_start"),
            F.col("window.end").alias("window_end"),
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "trade_count",
            "net_signed_volume",
            "order_flow_imbalance",
        )
    )
    return agg


def bars_to_kafka(features):
    """Kafka records for the live view: keyed by symbol, times in epoch milliseconds."""
    return features.select(
        F.col("symbol").alias("key"),
        F.to_json(
            F.struct(
                "symbol",
                (F.col("window_start").cast("long") * 1000).alias("window_start"),
                (F.col("window_end").cast("long") * 1000).alias("window_end"),
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "trade_count",
                "net_signed_volume",
                "order_flow_imbalance",
            )
        ).alias("value"),
    )


def main() -> None:
    spark = build_spark()
    trades = read_trades(spark)

    (
        trades.writeStream.format("parquet")
        .option("path", f"s3a://{LAKE_BUCKET}/raw/trades/")
        .option("checkpointLocation", f"s3a://{LAKE_BUCKET}/_chk/raw_trades/")
        .partitionBy("symbol")
        .outputMode("append")
        .trigger(processingTime="10 seconds")
        .start()
    )

    features = compute_ohlcv(trades)
    (
        features.writeStream.format("parquet")
        .option("path", f"s3a://{LAKE_BUCKET}/features/ohlcv_10s/")
        .option("checkpointLocation", f"s3a://{LAKE_BUCKET}/_chk/ohlcv_10s/")
        .partitionBy("symbol")
        .outputMode("append")
        .trigger(processingTime="10 seconds")
        .start()
    )

    (
        bars_to_kafka(features)
        .writeStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("topic", BARS_TOPIC)
        .option("checkpointLocation", f"s3a://{LAKE_BUCKET}/_chk/bars_live/")
        .outputMode("update")
        .trigger(processingTime="2 seconds")
        .start()
    )

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
