import logging
import time
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd
import requests

from src.config import (
    AIR_HOURLY,
    CHUNK_DAYS,
    CORE_METRICS,
    EXPECTED_COLUMNS,
    LAT,
    LON,
    MAX_RETRIES,
    OPEN_METEO_AIR_URL,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_HISTORICAL_URL,
    OWM_AIR_URL,
    OWM_CURRENT_URL,
    RAW_COLUMNS,
    REQUEST_TIMEOUT,
    WEATHER_COLUMNS,
    WEATHER_HOURLY,
    openweather_key,
)

log = logging.getLogger(__name__)

WINDOW_HOURS = 48
OWM_AIR_COLUMNS = ["owm_pm2_5", "owm_pm10"]
AIR_OM_COLUMNS = ["us_aqi", "om_pm2_5", "om_pm10", "ozone", "uv_index"]
ZERO_FILL_COLUMNS = ["precipitation_mm", "precipitation_probability", "uv_index"]
UNIT_PARAMS = {
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",
}


@dataclass
class Sources:
    weather: pd.DataFrame
    air_om: pd.DataFrame
    air_owm: pd.DataFrame


class RetryableStatusError(requests.RequestException):
    pass


def get_json(session, url, params):
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 429 or response.status_code >= 500:
                raise RetryableStatusError(f"HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except (requests.ConnectionError, requests.Timeout, RetryableStatusError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2**attempt)


def encode_wind_direction(degrees):
    radians = np.deg2rad(degrees)
    return np.sin(radians), np.cos(radians)


def reconstruct_wind_direction(sin_val, cos_val):
    return (np.rad2deg(np.arctan2(sin_val, cos_val)) + 360.0) % 360.0


def date_chunks(start, end, size_days=CHUNK_DAYS):
    chunks = []
    cursor = start
    while cursor <= end:
        stop = min(cursor + timedelta(days=size_days - 1), end)
        chunks.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return chunks


def chunk_unix_bounds(start, end):
    first = int(pd.Timestamp(start, tz="UTC").timestamp())
    last = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp()) - 1
    return first, last


def cutoff_future_hours(df, current_hour):
    return df[df["timestamp_utc"] <= current_hour].reset_index(drop=True)


def _hourly_frame(payload):
    df = pd.DataFrame(payload["hourly"])
    df["timestamp_utc"] = pd.to_datetime(df.pop("time"), utc=True).dt.floor("h")
    return df


def _empty_owm_air():
    return pd.DataFrame(
        {
            "timestamp_utc": pd.DatetimeIndex([], tz="UTC"),
            "owm_pm2_5": pd.Series(dtype="float64"),
            "owm_pm10": pd.Series(dtype="float64"),
        }
    )


def _weather_params(extra):
    return {
        "latitude": LAT,
        "longitude": LON,
        "timezone": "UTC",
        "hourly": ",".join(WEATHER_HOURLY),
        **UNIT_PARAMS,
        **extra,
    }


def _fetch_weather(session, url, extra):
    payload = get_json(session, url, _weather_params(extra))
    return _hourly_frame(payload).rename(columns={"precipitation": "precipitation_mm"})


def _fetch_air_om(session, extra):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "timezone": "UTC",
        "hourly": ",".join(AIR_HOURLY),
        **extra,
    }
    frame = _hourly_frame(get_json(session, OPEN_METEO_AIR_URL, params))
    return frame.rename(columns={"pm2_5": "om_pm2_5", "pm10": "om_pm10"})


def _fetch_owm_air(session, first, last):
    params = {"lat": LAT, "lon": LON, "start": first, "end": last, "appid": openweather_key()}
    try:
        payload = get_json(session, OWM_AIR_URL, params)
    except requests.RequestException as exc:
        log.warning("OpenWeather air pollution unavailable (%s); using Open-Meteo only", type(exc).__name__)
        return _empty_owm_air()
    rows = [
        {
            "timestamp_utc": pd.to_datetime(item["dt"], unit="s", utc=True).floor("h"),
            "owm_pm2_5": item["components"].get("pm2_5"),
            "owm_pm10": item["components"].get("pm10"),
        }
        for item in payload.get("list", [])
    ]
    if not rows:
        return _empty_owm_air()
    return pd.DataFrame(rows)


def _concat_unique(frames, empty):
    frames = [f for f in frames if not f.empty]
    if not frames:
        return empty
    merged = pd.concat(frames, ignore_index=True)
    return merged.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def fetch_backfill_sources(session, start, end):
    weather, air_om, air_owm = [], [], []
    for chunk_start, chunk_end in date_chunks(start, end):
        span = {"start_date": chunk_start.isoformat(), "end_date": chunk_end.isoformat()}
        weather.append(_fetch_weather(session, OPEN_METEO_HISTORICAL_URL, span))
        air_om.append(_fetch_air_om(session, span))
        air_owm.append(_fetch_owm_air(session, *chunk_unix_bounds(chunk_start, chunk_end)))
    return Sources(
        weather=_concat_unique(weather, pd.DataFrame(columns=["timestamp_utc"])),
        air_om=_concat_unique(air_om, pd.DataFrame(columns=["timestamp_utc"])),
        air_owm=_concat_unique(air_owm, _empty_owm_air()),
    )


def _needs_weather_fallback(weather, current_hour):
    row = weather[weather["timestamp_utc"] == current_hour]
    return row.empty or bool(row[WEATHER_COLUMNS].isna().all(axis=1).iloc[0])


def _fetch_current_weather_row(session, current_hour):
    params = {"lat": LAT, "lon": LON, "units": "metric", "appid": openweather_key()}
    payload = get_json(session, OWM_CURRENT_URL, params)
    sys_info = payload["sys"]
    wind = payload.get("wind", {})
    return pd.DataFrame(
        [
            {
                "timestamp_utc": current_hour,
                "temperature_2m": payload["main"]["temp"],
                "relative_humidity_2m": payload["main"]["humidity"],
                "precipitation_mm": payload.get("rain", {}).get("1h", 0.0),
                "precipitation_probability": np.nan,
                "surface_pressure": payload["main"]["pressure"],
                "cloud_cover": payload.get("clouds", {}).get("all", np.nan),
                "wind_speed_10m": wind.get("speed", 0.0) * 3.6,
                "wind_direction_10m": wind.get("deg", np.nan),
                "is_day": int(sys_info["sunrise"] <= payload["dt"] <= sys_info["sunset"]),
                "owm_fallback": True,
            }
        ]
    )


def _with_weather_fallback(session, weather, current_hour):
    if not _needs_weather_fallback(weather, current_hour):
        return weather
    try:
        fallback = _fetch_current_weather_row(session, current_hour)
    except requests.RequestException as exc:
        log.warning("OpenWeather current weather unavailable (%s)", type(exc).__name__)
        return weather
    kept = weather[weather["timestamp_utc"] != current_hour]
    if kept.empty:
        return fallback
    return pd.concat([kept, fallback], ignore_index=True)


def fetch_hourly_sources(session, current_hour):
    window = {"past_days": 2, "forecast_days": 1}
    try:
        weather = _fetch_weather(session, OPEN_METEO_FORECAST_URL, window)
    except requests.RequestException as exc:
        log.warning("Open-Meteo weather unavailable (%s)", type(exc).__name__)
        weather = pd.DataFrame(columns=["timestamp_utc", *WEATHER_COLUMNS])
        weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
    air_om = _fetch_air_om(session, window)
    first = int((current_hour - pd.Timedelta(hours=WINDOW_HOURS)).timestamp())
    air_owm = _fetch_owm_air(session, first, int(current_hour.timestamp()) + 3599)

    weather = _with_weather_fallback(session, cutoff_future_hours(weather, current_hour), current_hour)
    return Sources(
        weather=weather,
        air_om=cutoff_future_hours(air_om, current_hour),
        air_owm=cutoff_future_hours(air_owm, current_hour),
    )


def _hourly_unique(df):
    out = df.copy()
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], utc=True).dt.floor("h")
    return out.drop_duplicates("timestamp_utc", keep="last").set_index("timestamp_utc")


def build_grid(sources):
    frames = [_hourly_unique(f) for f in (sources.weather, sources.air_om, sources.air_owm)]
    populated = [f.index for f in frames if len(f)]
    if not populated:
        raise ValueError("no source returned any rows")
    index = pd.date_range(
        start=min(i.min() for i in populated),
        end=max(i.max() for i in populated),
        freq="1h",
        tz="UTC",
        name="timestamp_utc",
    )
    grid = pd.concat([f.reindex(index) for f in frames], axis=1)
    required = WEATHER_COLUMNS + AIR_OM_COLUMNS + OWM_AIR_COLUMNS + ["owm_fallback"]
    grid = grid.reindex(columns=required)
    grid["owm_fallback"] = grid["owm_fallback"].eq(True)
    return grid.reset_index()


def fuse_sources(grid):
    out = grid.copy()
    for metric in ("pm2_5", "pm10"):
        out[metric] = out[[f"om_{metric}", f"owm_{metric}"]].mean(axis=1)
    owm_present = out[OWM_AIR_COLUMNS].notna().any(axis=1)
    out["source_used"] = np.where(owm_present, "fused", "open_meteo_only")
    out.loc[out["owm_fallback"], "source_used"] = "owm_fallback"
    return out.drop(columns=["om_pm2_5", "om_pm10", "owm_pm2_5", "owm_pm10", "owm_fallback"])


def _same_hour_stat(frame, stat):
    # Previous 7 values at the same UTC hour, strictly before t.
    def trailing(series):
        window = series.shift(1).rolling(7, min_periods=1)
        return getattr(window, stat)()

    return frame.groupby(frame.index.hour).transform(trailing)


def _lookup_frame(observed, history):
    if history is None:
        return observed
    past = history.set_index("timestamp_utc")[RAW_COLUMNS]
    past = past[past.index < observed.index[0]]
    combined = pd.concat([past, observed])
    return combined.reindex(pd.date_range(combined.index[0], combined.index[-1], freq="1h", tz="UTC"))


def _fill_warmup_prefix(filled, warmup_hours):
    prefix = filled.index[:warmup_hours]
    for col in RAW_COLUMNS:
        if col in ZERO_FILL_COLUMNS:
            value = 0.0
        else:
            first = filled[col].first_valid_index()
            if first is None:
                continue
            value = filled.at[first, col]
        filled.loc[prefix, col] = filled.loc[prefix, col].fillna(value)
    return filled


def impute_causal(df, history=None, warmup_hours=0):
    # Assumes a continuous hourly index so shift(24) is exactly t - 24h.
    out = df.reset_index(drop=True).copy()
    out["is_imputed"] = out[RAW_COLUMNS].isna().any(axis=1).astype("int32")
    observed = out.set_index("timestamp_utc")[RAW_COLUMNS].astype("float64")
    filled = observed.copy()

    # Sunrise/sunset drift ~1 min/day, so t-24h beats ffill across the day boundary.
    filled["is_day"] = observed["is_day"].fillna(observed["is_day"].shift(24)).ffill()

    numeric = [c for c in RAW_COLUMNS if c != "is_day"]
    filled[numeric] = filled[numeric].ffill(limit=3)
    filled[numeric] = filled[numeric].fillna(observed[numeric].shift(24))

    if filled[numeric].isna().any().any():
        lookup = _lookup_frame(observed, history)
        scalars = [c for c in numeric if c != "wind_direction_10m"]
        medians = _same_hour_stat(lookup[scalars], "median").reindex(observed.index)
        filled[scalars] = filled[scalars].fillna(medians)

        # Circular mean: an arithmetic median would turn 355/5 degrees into 180.
        sin_val, cos_val = encode_wind_direction(lookup["wind_direction_10m"])
        means = _same_hour_stat(pd.DataFrame({"sin": sin_val, "cos": cos_val}), "mean")
        direction = reconstruct_wind_direction(means["sin"], means["cos"]).where(means["sin"].notna())
        filled["wind_direction_10m"] = filled["wind_direction_10m"].fillna(direction.reindex(observed.index))

    filled = _fill_warmup_prefix(filled, warmup_hours)
    out[RAW_COLUMNS] = filled.reset_index(drop=True)
    return out


def enforce_bounds(df):
    out = df.copy()
    for col in ("us_aqi", "precipitation_mm", "uv_index", "wind_speed_10m"):
        out[col] = out[col].clip(lower=0.0)
    for col in ("precipitation_probability", "relative_humidity_2m"):
        out[col] = out[col].clip(lower=0.0, upper=100.0)
    out.loc[out["is_day"] == 0, "uv_index"] = 0.0
    out["wind_direction_10m"] = np.mod(out["wind_direction_10m"], 360.0)
    return out


def add_features(df):
    out = df.copy()
    stamp = out["timestamp_utc"]
    for name, values, period in (
        ("hour", stamp.dt.hour, 24),
        ("dow", stamp.dt.dayofweek, 7),
        ("month", stamp.dt.month, 12),
    ):
        angle = 2 * np.pi * values / period
        out[f"{name}_sin"] = np.sin(angle)
        out[f"{name}_cos"] = np.cos(angle)
    out["wind_dir_sin"], out["wind_dir_cos"] = encode_wind_direction(out["wind_direction_10m"])
    for metric in CORE_METRICS:
        series = out[metric]
        out[f"{metric}_lag_24h"] = series.shift(24)
        out[f"{metric}_diff_3h"] = series - series.shift(3)
        out[f"{metric}_roll_mean_24h"] = series.rolling(window=24, min_periods=1).mean()
    return out


def cast_is_day(df):
    out = df.copy()
    if out["is_day"].notna().all():
        out["is_day"] = out["is_day"].astype("int32")
    return out


def fully_null_columns(sources):
    fused = fuse_sources(build_grid(sources))
    return [c for c in RAW_COLUMNS if fused[c].isna().all()]


def assemble_features(sources, warmup_hours, history=None):
    fused = fuse_sources(build_grid(sources))
    imputed = impute_causal(fused, history, warmup_hours)
    featured = add_features(enforce_bounds(imputed))
    trimmed = featured.iloc[warmup_hours:][EXPECTED_COLUMNS].reset_index(drop=True)
    return cast_is_day(trimmed)


def validate_frame(df):
    assert df.isna().sum().sum() == 0, "frame contains nulls"
    assert df["timestamp_utc"].is_unique, "duplicate timestamps"
    assert list(df.columns) == EXPECTED_COLUMNS, "column mismatch"


def latest_closed_hours(df, current_hour, hours=3):
    cutoff = current_hour - pd.Timedelta(hours=hours - 1)
    return cast_is_day(df[df["timestamp_utc"] >= cutoff].reset_index(drop=True))
