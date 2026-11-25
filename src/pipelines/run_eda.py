import json
import logging
from pathlib import Path

import numpy as np

from src.config import CORE_METRICS
from src.models import dataset

ARTIFACTS_DIR = Path("artifacts")
RAW_TARGETS = [
    "us_aqi",
    "temperature_2m",
    "precipitation_mm",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m",
]
CONTEXT_COLS = ["pm2_5", "pm10", "ozone", "surface_pressure", "cloud_cover"]
HAZARDS = {
    "us_aqi_gt_150": lambda d: d["us_aqi"] > 150,
    "us_aqi_gt_300": lambda d: d["us_aqi"] > 300,
    "temperature_2m_gt_40": lambda d: d["temperature_2m"] > 40,
    "uv_index_gte_8": lambda d: d["uv_index"] >= 8,
    "wind_speed_10m_gt_40": lambda d: d["wind_speed_10m"] > 40,
}

log = logging.getLogger(__name__)


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else 0.0


def _describe(series):
    return {
        "mean": _finite(series.mean()),
        "std": _finite(series.std()),
        "min": _finite(series.min()),
        "p25": _finite(series.quantile(0.25)),
        "median": _finite(series.median()),
        "p75": _finite(series.quantile(0.75)),
        "p95": _finite(series.quantile(0.95)),
        "max": _finite(series.max()),
        "skewness": _finite(series.skew()),
    }


def _profile(df, key, keys):
    grouped = df.groupby(key)[CORE_METRICS].mean()
    return {m: {str(k): _finite(grouped[m].get(k, np.nan)) for k in keys} for m in CORE_METRICS}


def compute_eda_summary(df):
    stamps = df["timestamp_utc"]
    corr_cols = [*RAW_TARGETS, *CONTEXT_COLS]
    corr = df[corr_cols].corr(method="pearson")
    return {
        "dataset": {
            "start_utc": stamps.min().isoformat(),
            "end_utc": stamps.max().isoformat(),
            "rows": len(df),
            "imputation_rate": _finite(df["is_imputed"].mean()),
            "source_used_counts": {str(k): int(v) for k, v in df["source_used"].value_counts().items()},
        },
        "descriptive_stats": {c: _describe(df[c]) for c in RAW_TARGETS},
        "diurnal_profile": _profile(df, stamps.dt.hour, range(24)),
        "seasonal_profile": _profile(df, stamps.dt.month, range(1, 13)),
        "correlation": {r: {c: _finite(corr.loc[r, c]) for c in corr_cols} for r in corr_cols},
        "extreme_event_counts": {name: int(rule(df).sum()) for name, rule in HAZARDS.items()},
    }


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = dataset.read_features(dataset.connect_project())
    summary = compute_eda_summary(df)
    write_json(ARTIFACTS_DIR / "eda_summary.json", summary)
    log.info("wrote %s (%d rows)", ARTIFACTS_DIR / "eda_summary.json", summary["dataset"]["rows"])


if __name__ == "__main__":
    main()
