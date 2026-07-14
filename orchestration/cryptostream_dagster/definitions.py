"""Dagster orchestration for the CryptoStream batch layer.

Assets (DAG):
    lake_to_duckdb  ->  dbt_models  ->  ml_model  ->  market_summary

`lake_to_duckdb` loads the Parquet feature lake (written by Spark, or by the sample
generator) into DuckDB's raw schema. dbt then builds staging/marts, the ML asset
trains + logs to MLflow, and the summary asset produces the LLM market read.

Run the UI:   dagster dev -m cryptostream_dagster.definitions
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys

import duckdb
from dagster import (
    AssetExecutionContext,
    Definitions,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LAKE_DIR = os.path.join(REPO, "data", "lake", "features", "ohlcv_10s")
DUCKDB_PATH = os.path.join(REPO, "data", "warehouse", "crypto.duckdb")
WAREHOUSE_DIR = os.path.join(REPO, "warehouse")


@asset(description="Load Parquet feature lake into DuckDB raw.ohlcv_10s.")
def lake_to_duckdb(context: AssetExecutionContext) -> MaterializeResult:
    os.makedirs(os.path.dirname(DUCKDB_PATH), exist_ok=True)
    parts = glob.glob(os.path.join(LAKE_DIR, "**", "*.parquet"), recursive=True)
    if not parts:
        raise Exception(
            f"No parquet found under {LAKE_DIR}. Run the sample generator or start the Spark job."
        )
    con = duckdb.connect(DUCKDB_PATH)
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    con.execute("DROP TABLE IF EXISTS raw.ohlcv_10s;")
    # hive_partitioning recovers the symbol=... partition column from the path.
    con.execute(
        f"""
        CREATE TABLE raw.ohlcv_10s AS
        SELECT * FROM read_parquet('{LAKE_DIR}/**/*.parquet', hive_partitioning=true)
        """
    )
    n = con.execute("SELECT count(*) FROM raw.ohlcv_10s").fetchone()[0]
    con.close()
    context.log.info("loaded %d rows into raw.ohlcv_10s", n)
    return MaterializeResult(metadata={"rows": MetadataValue.int(int(n)), "files": len(parts)})


@asset(deps=[lake_to_duckdb], description="Run dbt build (models + tests).")
def dbt_models(context: AssetExecutionContext) -> MaterializeResult:
    result = subprocess.run(
        ["dbt", "build", "--profiles-dir", "."],
        cwd=WAREHOUSE_DIR,
        capture_output=True,
        text=True,
    )
    context.log.info(result.stdout[-3000:])
    if result.returncode != 0:
        context.log.error(result.stderr[-3000:])
        raise Exception("dbt build failed")
    return MaterializeResult(metadata={"stdout_tail": MetadataValue.md(f"```\n{result.stdout[-1500:]}\n```")})


@asset(deps=[dbt_models], description="Train the direction classifier and log to MLflow.")
def ml_model(context: AssetExecutionContext) -> MaterializeResult:
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "ml", "train.py")],
        cwd=os.path.join(REPO, "ml"),
        capture_output=True,
        text=True,
    )
    context.log.info(result.stdout[-3000:])
    if result.returncode != 0:
        context.log.error(result.stderr[-3000:])
        raise Exception("model training failed")
    return MaterializeResult(metadata={"log_tail": MetadataValue.md(f"```\n{result.stdout[-1500:]}\n```")})


@asset(deps=[ml_model], description="Generate the LLM market summary.")
def market_summary(context: AssetExecutionContext) -> MaterializeResult:
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from llm.market_agent import generate_market_summary

    summary = generate_market_summary()
    context.log.info(summary)
    return MaterializeResult(metadata={"summary": MetadataValue.md(summary)})


refresh_job = define_asset_job("refresh_pipeline", selection="*")

# Retrain / refresh every 15 minutes (adjust cadence to taste).
refresh_schedule = ScheduleDefinition(job=refresh_job, cron_schedule="*/15 * * * *")

defs = Definitions(
    assets=[lake_to_duckdb, dbt_models, ml_model, market_summary],
    jobs=[refresh_job],
    schedules=[refresh_schedule],
)
