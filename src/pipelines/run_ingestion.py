import argparse
import functools
import logging
import os
from datetime import timedelta

import pandas as pd
import requests

from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HISTORY_DAYS,
    WARMUP_HOURS,
)
from src.pipelines import fetch_and_engineer as fe

log = logging.getLogger(__name__)

FEATURE_GROUP_DESCRIPTION = "Hourly fused weather, air quality, and engineered features for Lahore (UTC)."


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.pipelines.run_ingestion")
    parser.add_argument("--mode", choices=["backfill", "hourly"], required=True)
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def connect_feature_group():
    import hopsworks

    project = hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"],
    )
    return project.get_feature_store().get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["timestamp_utc"],
        event_time="timestamp_utc",
        online_enabled=False,
        description=FEATURE_GROUP_DESCRIPTION,
    )


def read_history(feature_group, current_hour):
    since = current_hour - pd.Timedelta(days=HISTORY_DAYS)
    return feature_group.filter(feature_group.timestamp_utc >= since).read()


def build_backfill(session, days, current_hour):
    end = current_hour.date() - timedelta(days=1)
    start = end - timedelta(days=days + WARMUP_HOURS // 24 - 1)
    sources = fe.fetch_backfill_sources(session, start, end)
    return fe.assemble_features(sources, warmup_hours=WARMUP_HOURS)


def build_hourly(session, current_hour, get_feature_group):
    sources = fe.fetch_hourly_sources(session, current_hour)
    history = None
    if fe.fully_null_columns(sources):
        history = read_history(get_feature_group(), current_hour)
    features = fe.assemble_features(sources, warmup_hours=0, history=history)
    return fe.latest_closed_hours(features, current_hour)


def summarize(df):
    start, end = df["timestamp_utc"].min(), df["timestamp_utc"].max()
    print(f"rows: {len(df)}")
    print(f"range_utc: {start} -> {end}")
    print(f"imputation_rate: {df['is_imputed'].mean():.4f}")
    print("source_used:")
    for name, count in df["source_used"].value_counts().items():
        print(f"  {name}: {count}")


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    current_hour = pd.Timestamp.now(tz="UTC").floor("h")
    session = requests.Session()
    get_feature_group = functools.cache(connect_feature_group)

    if args.mode == "backfill":
        frame = build_backfill(session, args.days, current_hour)
    else:
        frame = build_hourly(session, current_hour, get_feature_group)

    fe.validate_frame(frame)
    summarize(frame)
    if args.dry_run:
        return

    get_feature_group().insert(frame, write_options={"wait_for_job": True})
    log.info("inserted %d rows into %s v%d", len(frame), FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)


if __name__ == "__main__":
    main()
