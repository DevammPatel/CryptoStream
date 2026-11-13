"""Dagster orchestration for the CryptoStream batch layer.

Assets (DAG):
    compact_lake  ->  lake_to_duckdb  ->  dbt_models  ->  ml_model  ->  market_summary

`compact_lake` merges each finished hour of Spark's small files and expires old raw
trades (see compaction.py).

`lake_to_duckdb` loads the Parquet feature lake (written by Spark, or by the sample
generator) into a fresh build copy of the warehouse. dbt builds staging/marts in that
copy and, if every test passes, swaps it in as the served warehouse. The ML asset then
trains + logs to MLflow, and the summary asset produces the LLM market read.

The swap is why the API and dashboard never hit DuckDB's file lock: they only open the
served file, and this pipeline only writes the build copy.

Env vars:
    LAKE_URI        local folder or s3://bucket/path of the feature lake
    LAKE_ENDPOINT   S3 endpoint, needed when LAKE_URI is s3://
    LAKE_ACCESS_KEY, LAKE_SECRET_KEY
    DUCKDB_PATH     the served warehouse file

Run the UI:   dagster dev -m cryptostream_dagster.definitions

Dagster inspects the `context` annotation at runtime, so this module must not use
`from __future__ import annotations`.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

import duckdb
from dagster import (
    AssetExecutionContext,
    DefaultScheduleStatus,
    Definitions,
    MaterializeResult,
    MetadataValue,
    ScheduleDefinition,
    asset,
    define_asset_job,
)

from cryptostream_dagster import compaction

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAREHOUSE_DIR = os.path.join(REPO, "warehouse")


def lake_uri() -> str:
    return os.getenv("LAKE_URI", os.path.join(REPO, "data", "lake", "features", "ohlcv_10s")).rstrip("/")


def served_path() -> str:
    return os.getenv("DUCKDB_PATH", os.path.join(REPO, "data", "warehouse", "crypto.duckdb"))


BUILD_KEEP = 24 * 3600  # seconds; build folders left by failed runs are cleared after this


def build_path(run_id: str) -> str:
    """Each run builds its own copy, so a scheduled run and a manual one never fight
    over one file. The copy keeps the served file's name: dbt writes it into its views."""
    served = served_path()
    return os.path.join(os.path.dirname(served), ".build", run_id, os.path.basename(served))


def clear_old_builds() -> None:
    """Remove what failed runs left in .build/ (a published build removes its own folder)."""
    root = os.path.join(os.path.dirname(served_path()), ".build")
    for entry in glob.glob(os.path.join(root, "*")):
        if time.time() - os.path.getmtime(entry) > BUILD_KEEP:
            shutil.rmtree(entry) if os.path.isdir(entry) else os.remove(entry)


def connect_lake_storage(con: duckdb.DuckDBPyConnection) -> None:
    """Give DuckDB credentials for an S3 lake. A local lake needs nothing."""
    if not lake_uri().startswith("s3://"):
        return
    endpoint = urlparse(os.environ["LAKE_ENDPOINT"])
    con.execute("LOAD httpfs;")
    con.execute(
        f"""
        CREATE SECRET lake (
            TYPE S3,
            KEY_ID '{os.environ["LAKE_ACCESS_KEY"]}',
            SECRET '{os.environ["LAKE_SECRET_KEY"]}',
            ENDPOINT '{endpoint.netloc}',
            URL_STYLE 'path',
            USE_SSL {str(endpoint.scheme == "https").lower()}
        );
        """
    )


def publish_warehouse(run_id: str) -> None:
    """Swap the run's finished build copy in as the served warehouse."""
    build = build_path(run_id)
    con = duckdb.connect(build)
    con.execute("CHECKPOINT;")
    con.close()
    os.replace(build, served_path())
    shutil.rmtree(os.path.dirname(build), ignore_errors=True)


@asset(description="Merge each finished hour of Spark's small lake files, and expire old raw trades.")
def compact_lake(context: AssetExecutionContext) -> MaterializeResult:
    con = duckdb.connect()
    try:
        connect_lake_storage(con)
        summary = compaction.compact_lake(con, lake_uri())
    finally:
        con.close()
    context.log.info("compaction: %s", summary)
    return MaterializeResult(metadata={"summary": MetadataValue.json(summary)})


@asset(deps=[compact_lake], description="Load the Parquet feature lake into raw.ohlcv_10s in a fresh build copy.")
def lake_to_duckdb(context: AssetExecutionContext) -> MaterializeResult:
    clear_old_builds()
    build = build_path(context.run_id)
    os.makedirs(os.path.dirname(build), exist_ok=True)

    parquet = f"{lake_uri()}/**/*.parquet"
    con = duckdb.connect(build)
    try:
        connect_lake_storage(con)
        files = con.execute("SELECT count(*) FROM glob(?)", [parquet]).fetchone()[0]
        if not files:
            raise Exception(
                f"No parquet found under {lake_uri()}. Run the sample generator or start the Spark job."
            )
        con.execute("CREATE SCHEMA raw;")
        # Store every bar time as plain UTC. A time-zone-aware file would otherwise be
        # converted later, in whichever session reads the views, using that machine's
        # zone: on a Mac in BST every bar would shift by an hour.
        con.execute("SET TimeZone = 'UTC';")
        # hive_partitioning recovers the symbol=... partition column from the path, and
        # union_by_name lines up columns by name across Spark, backfill and sample files.
        con.execute(
            f"""
            CREATE TABLE raw.ohlcv_10s AS
            SELECT * REPLACE (
                CAST(window_start AS TIMESTAMP) AS window_start,
                CAST(window_end AS TIMESTAMP) AS window_end
            )
            FROM read_parquet('{parquet}', hive_partitioning=true, union_by_name=true)
            """
        )
        n = con.execute("SELECT count(*) FROM raw.ohlcv_10s").fetchone()[0]
    finally:
        con.close()
    context.log.info("loaded %d rows into raw.ohlcv_10s", n)
    return MaterializeResult(metadata={"rows": MetadataValue.int(int(n)), "files": int(files)})


@asset(deps=[lake_to_duckdb], description="Run dbt build (models + tests), then publish the warehouse.")
def dbt_models(context: AssetExecutionContext) -> MaterializeResult:
    build = build_path(context.run_id)
    run_dir = os.path.dirname(build)
    # dbt's own files go in the run's folder too, so overlapping runs don't share them.
    result = subprocess.run(
        ["dbt", "build", "--profiles-dir", ".",
         "--target-path", os.path.join(run_dir, "dbt-target"), "--log-path", os.path.join(run_dir, "dbt-logs")],
        cwd=WAREHOUSE_DIR,
        env=dict(os.environ, DBT_DUCKDB_PATH=build),
        capture_output=True,
        text=True,
    )
    context.log.info(result.stdout[-3000:])
    if result.returncode != 0:
        context.log.error(result.stderr[-3000:])
        raise Exception("dbt build failed")
    publish_warehouse(context.run_id)
    return MaterializeResult(metadata={"stdout_tail": MetadataValue.md(f"```\n{result.stdout[-1500:]}\n```")})


def model_dir() -> str:
    return os.getenv("MODEL_DIR", os.path.join(REPO, "ml", "models"))


def training_lock():
    """One training at a time per model folder.

    Training is the heaviest step. A scheduled run and a manual one training together
    needed more memory than a laptop's Docker VM has, and Docker killed Spark. A run
    that finds the lock taken skips training: the other run is about to produce a
    model anyway, and the served one stays in place until then.
    """
    folder = os.path.abspath(model_dir())
    return compaction.only_one(f"training-{hashlib.sha1(folder.encode()).hexdigest()[:12]}")


@asset(
    deps=[dbt_models],
    description="Train the direction classifier, log it to MLflow, and serve it if it beats the current one.",
)
def ml_model(context: AssetExecutionContext) -> MaterializeResult:
    with training_lock() as free:
        if not free:
            context.log.info("Another run is training, so this run keeps the served model.")
            return MaterializeResult(metadata={"skipped": MetadataValue.text("another run is training")})
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
    metadata = {"log_tail": MetadataValue.md(f"```\n{result.stdout[-1500:]}\n```")}
    last_run = os.path.join(model_dir(), "last_training.json")
    if os.path.exists(last_run):
        with open(last_run) as fh:
            record = json.load(fh)
        metadata["promoted"] = MetadataValue.bool(record["promotion"]["promoted"])
        metadata["promotion_reason"] = MetadataValue.text(record["promotion"]["reason"])
        metadata["test_auc"] = MetadataValue.float(record["metrics"]["test_auc"] or float("nan"))
    return MaterializeResult(metadata=metadata)


@asset(deps=[ml_model], description="Write the market summary, with the model's calls, for the dashboard.")
def market_summary(context: AssetExecutionContext) -> MaterializeResult:
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from llm.market_agent import build_market_summary, save_market_summary

    record = build_market_summary()
    path = save_market_summary(record)
    context.log.info(record["text"])
    if "llm_error" in record:
        context.log.warning("LLM unavailable, saved the offline summary: %s", record["llm_error"])
    return MaterializeResult(
        metadata={
            "summary": MetadataValue.md(record["text"]),
            "source": MetadataValue.text(record["source"]),
            "path": MetadataValue.path(path),
        }
    )


refresh_job = define_asset_job("refresh_pipeline", selection="*")

# Refresh and retrain every 15 minutes by default. REFRESH_CRON changes it: hourly
# ("0 * * * *") is much lighter on a laptop. The dashboard reads the same setting.
refresh_schedule = ScheduleDefinition(
    job=refresh_job,
    cron_schedule=os.getenv("REFRESH_CRON", "*/15 * * * *"),
    default_status=DefaultScheduleStatus.RUNNING,
)

defs = Definitions(
    assets=[compact_lake, lake_to_duckdb, dbt_models, ml_model, market_summary],
    jobs=[refresh_job],
    schedules=[refresh_schedule],
)
