"""Spark Structured Streaming: Kafka trades -> features -> Parquet lake (MinIO).

Reads the raw trade stream from Redpanda, computes windowed OHLCV + micro-structure
features per symbol, and writes them as Parquet to the S3-compatible MinIO lake.

Two sinks:
  1. Raw trades          -> s3a://<bucket>/raw/trades/            (append, exactly-once via checkpoint)
  2. 10s OHLCV features   -> s3a://<bucket>/features/ohlcv_10s/     (append)

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
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
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
        .config("spark.hadoop.fs.s3a.endpoint", MINIO_ENDPOINT)
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
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
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", 20000)
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

    agg = (
        trades.withColumn("signed_qty", signed_qty)
        .withColumn("notional", notional)
        .groupBy(F.window("event_ts", "10 seconds"), F.col("symbol"))
        .agg(
            F.first("price", ignorenulls=True).alias("open"),
            F.max("price").alias("high"),
            F.min("price").alias("low"),
            F.last("price", ignorenulls=True).alias("close"),
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

    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
