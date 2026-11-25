import os

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    HORIZON_HOURS,
    MODEL_FEATURE_COLS,
    SEQ_LEN,
    TARGET_COLS,
    VAL_HOURS,
)

AUX_TARGETS = ["wind_direction_10m", "is_day"]


def normalize(df):
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True)
    out = out.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")
    return out.reset_index(drop=True)


def build_paired(df):
    columns = [*TARGET_COLS, *AUX_TARGETS]
    target_df = df[["timestamp_utc", *columns]].copy()
    target_df["timestamp_utc"] = target_df["timestamp_utc"] - pd.Timedelta(hours=HORIZON_HOURS)
    target_df = target_df.rename(columns={c: f"target_{c}" for c in columns})
    paired = df.merge(target_df, on="timestamp_utc", how="inner")
    return paired.sort_values("timestamp_utc").reset_index(drop=True)


def split_embargo(paired_df):
    val_start = paired_df["timestamp_utc"].max() - pd.Timedelta(hours=VAL_HOURS - 1)
    embargo_end = val_start - pd.Timedelta(hours=HORIZON_HOURS)
    train = paired_df[paired_df["timestamp_utc"] <= embargo_end].reset_index(drop=True)
    val = paired_df[paired_df["timestamp_utc"] >= val_start].reset_index(drop=True)
    return train, val


def target_frame(df):
    return df[[f"target_{c}" for c in TARGET_COLS]].rename(columns=lambda c: c.removeprefix("target_")).reset_index(drop=True)


def clip_predictions(preds_df, is_day_target):
    out = preds_df.copy()
    for col in ("us_aqi", "precipitation_mm", "uv_index", "wind_speed_10m"):
        out[col] = out[col].clip(lower=0.0)
    for col in ("precipitation_probability", "relative_humidity_2m"):
        out[col] = out[col].clip(lower=0.0, upper=100.0)
    for col in ("wind_dir_sin", "wind_dir_cos"):
        out[col] = out[col].clip(lower=-1.0, upper=1.0)
    night = np.asarray(is_day_target) == 0
    out.loc[night, "uv_index"] = 0.0
    return out


def lstm_sequences(df, timestamps, seq_len=SEQ_LEN):
    indexed = df.set_index("timestamp_utc")[MODEL_FEATURE_COLS].astype(float)
    grid = indexed.reindex(pd.date_range(indexed.index.min(), indexed.index.max(), freq="h"))
    values = grid.to_numpy()
    windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0).transpose(0, 2, 1)
    complete = ~np.isnan(windows).any(axis=(1, 2))
    ends = pd.Series(np.arange(seq_len - 1, len(grid)), index=grid.index[seq_len - 1 :])
    wanted = ends.reindex(pd.DatetimeIndex(timestamps))
    usable = wanted.notna().to_numpy()
    positions = wanted.to_numpy()[usable].astype(int) - (seq_len - 1)
    keep = complete[positions]
    kept = pd.DatetimeIndex(timestamps)[usable][keep]
    return windows[positions[keep]].copy(), kept


def connect_project():
    import hopsworks

    return hopsworks.login(
        api_key_value=os.environ["HOPSWORKS_API_KEY"],
        project=os.environ["HOPSWORKS_PROJECT"],
    )


def read_features(project):
    feature_group = project.get_feature_store().get_feature_group(FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION)
    return normalize(feature_group.read())
