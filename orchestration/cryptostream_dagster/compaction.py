"""Compact Spark's small lake files and expire old raw trades.

Spark appends a new Parquet file per symbol every minute, about 1,440 a day for each
dataset and symbol. Reading thousands of tiny files over S3 makes every refresh slow,
so once an hour is over, its files are merged into one compact-<hour>.parquet per
symbol and the originals deleted. Duplicate rows (Spark replays a batch after a
restart) are dropped on the way.

The merged file is written before any original is deleted. If a run stops in between,
the next run merges the compact file with the leftovers again, so nothing is lost or
double counted.

Only Spark's part-* files and earlier compact-* files are touched. Backfill and sample
files keep their names and are left alone.

Only one compaction of a lake runs at a time on a machine: a run that finds another
one busy skips it, since the next run will catch up.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import duckdb

# An hour is only compacted once Spark can no longer write to it: its watermark is 30
# seconds and it writes the lake once a minute, so 10 minutes is a generous margin.
SETTLE = timedelta(minutes=10)
RAW_RETENTION_DAYS = float(os.getenv("RAW_RETENTION_DAYS", "7"))


@dataclass(frozen=True)
class Dataset:
    path: str        # folder holding symbol=<s>/ partitions
    time_col: str    # what decides a row's hour
    key: str         # what makes a row unique
    keep_first: str  # among duplicates, the row that sorts first is kept


def datasets(features_uri: str) -> list[Dataset]:
    """The feature bars, plus the raw trades when the lake has Spark's usual layout."""
    found = [Dataset(features_uri, "window_start", "window_start", "trade_count DESC, volume DESC")]
    if features_uri.endswith("/features/ohlcv_10s"):
        root = features_uri[: -len("/features/ohlcv_10s")]
        found.append(Dataset(f"{root}/raw/trades", "event_ts", "trade_id", "ingest_time"))
    return found


def filesystem(uri: str):
    """An fsspec filesystem for the lake: local folders, or S3 with the LAKE_* settings.

    s3fs normally caches folder listings, but DuckDB writes the merged files behind its
    back, so a cached listing would say a file just written does not exist. The cache is
    off, and so is fsspec's reuse of filesystem objects, which would carry it between runs.
    """
    import fsspec

    if uri.startswith("s3://"):
        return fsspec.filesystem(
            "s3",
            key=os.environ["LAKE_ACCESS_KEY"],
            secret=os.environ["LAKE_SECRET_KEY"],
            client_kwargs={"endpoint_url": os.environ["LAKE_ENDPOINT"]},
            use_listings_cache=False,
            skip_instance_cache=True,
        )
    return fsspec.filesystem("file")


def _full(uri: str, path: str) -> str:
    """fsspec lists S3 paths without the scheme; DuckDB needs it."""
    return f"s3://{path}" if uri.startswith("s3://") and not path.startswith("s3://") else path


def _sql_list(paths: list[str]) -> str:
    return "[" + ", ".join("'" + p.replace("'", "''") + "'" for p in paths) + "]"


def compact(con: duckdb.DuckDBPyConnection, fs, ds: Dataset, now: datetime) -> dict:
    """Merge each settled hour's Spark files into one file per symbol."""
    stats = {"merged_files": 0, "written_files": 0}
    cutoff = (now - SETTLE).replace(tzinfo=None)
    for partition in fs.glob(f"{ds.path}/symbol=*"):
        files = [_full(ds.path, f) for f in fs.glob(f"{partition}/part-*.parquet")]
        if not files:
            continue
        hours = con.execute(
            f"""
            SELECT filename, date_trunc('hour', min({ds.time_col})::TIMESTAMP) AS hour
            FROM read_parquet({_sql_list(files)}, filename=true, hive_partitioning=false)
            GROUP BY filename
            """
        ).fetchall()
        by_hour: dict[datetime, list[str]] = {}
        for filename, hour in hours:
            if hour + timedelta(hours=1) <= cutoff:
                by_hour.setdefault(hour, []).append(filename)
        for hour, sources in sorted(by_hour.items()):
            target = _full(ds.path, f"{partition}/compact-{hour:%Y%m%d%H}.parquet")
            if fs.exists(target):
                sources = [target, *sources]  # late or replayed files for an hour already done
            tmp = f"{target}.tmp"
            con.execute(
                f"""
                COPY (
                    SELECT * FROM read_parquet({_sql_list(sources)}, union_by_name=true, hive_partitioning=false)
                    QUALIFY row_number() OVER (PARTITION BY {ds.key} ORDER BY {ds.keep_first}) = 1
                ) TO '{tmp}' (FORMAT PARQUET)
                """
            )
            fs.mv(tmp, target)
            parts = [s for s in sources if s != target]
            fs.rm(parts)
            stats["merged_files"] += len(parts)
            stats["written_files"] += 1
    return stats


def expire(fs, ds: Dataset, now: datetime, days: float) -> int:
    """Delete compacted hours older than `days`. Returns how many files were removed."""
    oldest = (now - timedelta(days=days)).replace(tzinfo=None)
    old = []
    for f in fs.glob(f"{ds.path}/symbol=*/compact-*.parquet"):
        match = re.search(r"compact-(\d{10})\.parquet$", f)
        if match and datetime.strptime(match.group(1), "%Y%m%d%H") + timedelta(hours=1) <= oldest:
            old.append(f)
    if old:
        fs.rm(old)
    return len(old)


@contextmanager
def only_one(name: str):
    """Yield True while holding a lock no other process on this machine holds, else False."""
    import fcntl

    with open(os.path.join(tempfile.gettempdir(), f"cryptostream-{name}.lock"), "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def compaction_lock(features_uri: str):
    """The lock for compacting one lake; other lakes compact independently."""
    return only_one(f"compaction-{hashlib.sha1(features_uri.encode()).hexdigest()[:12]}")


def compact_lake(con: duckdb.DuckDBPyConnection, features_uri: str, now: datetime | None = None) -> dict:
    """Compact every dataset, then expire raw trades past RAW_RETENTION_DAYS."""
    with compaction_lock(features_uri) as free:
        if not free:
            return {"skipped": "another run is compacting the lake"}
        now = now or datetime.now(timezone.utc)
        fs = filesystem(features_uri)
        con.execute("SET TimeZone = 'UTC';")
        summary = {}
        for ds in datasets(features_uri):
            if not fs.exists(ds.path):
                continue
            summary[ds.path] = compact(con, fs, ds, now)
            if ds.time_col == "event_ts":
                summary[ds.path]["expired_files"] = expire(fs, ds, now, RAW_RETENTION_DAYS)
        return summary
