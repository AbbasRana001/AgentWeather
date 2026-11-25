import itertools

import numpy as np
import pandas as pd
import pytest
import requests

from src.config import EXPECTED_COLUMNS, RAW_COLUMNS
from src.pipelines import fetch_and_engineer as fe

START = pd.Timestamp("2025-03-01 00:00", tz="UTC")
WEATHER_FIELDS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "precipitation_probability",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "is_day",
]
AIR_FIELDS = ["us_aqi", "pm2_5", "pm10", "ozone", "uv_index"]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "test-key")
    sleeps = []
    monkeypatch.setattr(fe.time, "sleep", sleeps.append)
    return sleeps


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        for fragment, handler in self.routes.items():
            if fragment in url:
                result = handler(params) if callable(handler) else handler
                if isinstance(result, Exception):
                    raise result
                if isinstance(result, FakeResponse):
                    return result
                return FakeResponse(result)
        raise AssertionError(f"unrouted url {url}")


def signal(index, scale=1.0):
    hours = index.hour.to_numpy()
    days = (index - START).days.to_numpy()
    return 20 + 5 * np.sin(2 * np.pi * hours / 24) + 0.5 * days * scale


def om_times(index):
    return [ts.strftime("%Y-%m-%dT%H:%M") for ts in index]


def weather_payload(index, null_at=None):
    values = {
        "temperature_2m": signal(index),
        "relative_humidity_2m": 50 + signal(index) / 2,
        "precipitation": np.zeros(len(index)),
        "precipitation_probability": np.full(len(index), 10.0),
        "surface_pressure": np.full(len(index), 1000.0),
        "cloud_cover": np.full(len(index), 30.0),
        "wind_speed_10m": np.full(len(index), 12.0),
        "wind_direction_10m": np.full(len(index), 90.0),
        "is_day": ((index.hour >= 6) & (index.hour < 18)).astype(int),
    }
    hourly = {"time": om_times(index)}
    for name, arr in values.items():
        col = [float(v) for v in arr]
        if null_at is not None:
            col[index.get_loc(null_at)] = None
        hourly[name] = col
    return {"hourly": hourly}


def air_payload(index):
    hourly = {
        "time": om_times(index),
        "us_aqi": [float(v) for v in 80 + signal(index)],
        "pm2_5": [40.0] * len(index),
        "pm10": [60.0] * len(index),
        "ozone": [30.0] * len(index),
        "uv_index": [float(v) for v in np.where((index.hour >= 6) & (index.hour < 18), 5.0, 0.0)],
    }
    return {"hourly": hourly}


def owm_payload(index, pm2_5=20.0, pm10=30.0):
    return {
        "list": [
            {"dt": int(ts.timestamp()), "components": {"pm2_5": pm2_5, "pm10": pm10}}
            for ts in index
        ]
    }


def make_sources(n_hours=24 * 10, owm_gap=None):
    index = pd.date_range(START, periods=n_hours, freq="1h", tz="UTC")
    weather = pd.DataFrame(
        {
            "timestamp_utc": index,
            "temperature_2m": signal(index),
            "relative_humidity_2m": 50 + signal(index) / 2,
            "precipitation_mm": 0.0,
            "precipitation_probability": 10.0,
            "surface_pressure": 1000.0,
            "cloud_cover": 30.0,
            "wind_speed_10m": 12.0,
            "wind_direction_10m": 90.0,
            "is_day": ((index.hour >= 6) & (index.hour < 18)).astype(int),
        }
    )
    air_om = pd.DataFrame(
        {
            "timestamp_utc": index,
            "us_aqi": 80 + signal(index),
            "om_pm2_5": 40.0,
            "om_pm10": 60.0,
            "ozone": 30.0,
            "uv_index": np.where((index.hour >= 6) & (index.hour < 18), 5.0, 0.0),
        }
    )
    air_owm = pd.DataFrame({"timestamp_utc": index, "owm_pm2_5": 20.0, "owm_pm10": 30.0})
    if owm_gap is not None:
        air_owm = air_owm.drop(index=owm_gap)
    weather = weather.drop(index=range(100, 103))
    weather.loc[weather.index[150:160], "temperature_2m"] = np.nan
    return fe.Sources(weather=weather, air_om=air_om, air_owm=air_owm)


def raw_frame(days=14):
    index = pd.date_range(START, periods=24 * days, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": index})
    base = signal(index)
    for i, col in enumerate(RAW_COLUMNS):
        df[col] = base + i
    df["is_day"] = ((index.hour >= 6) & (index.hour < 18)).astype(float)
    df["wind_direction_10m"] = 90.0
    return df


def blank(df, col, start, length):
    out = df.copy()
    out.loc[out.index[start : start + length], col] = np.nan
    return out


def test_output_schema_and_no_nulls():
    sources = make_sources(owm_gap=range(200, 230))
    out = fe.assemble_features(sources, warmup_hours=48)
    assert list(out.columns) == EXPECTED_COLUMNS
    assert len(EXPECTED_COLUMNS) == 43
    assert out.isna().sum().sum() == 0
    assert out["timestamp_utc"].is_unique
    assert str(out["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
    assert out["is_imputed"].dtype == np.int32
    assert out["is_day"].dtype == np.int32
    assert len(out) == 24 * 10 - 48


def test_engineered_features_ignore_future_rows():
    sources = make_sources()
    baseline = fe.assemble_features(sources, warmup_hours=48)
    t = START + pd.Timedelta(hours=200)
    mutated = fe.Sources(
        weather=sources.weather.copy(),
        air_om=sources.air_om.copy(),
        air_owm=sources.air_owm.copy(),
    )
    for frame in (mutated.weather, mutated.air_om, mutated.air_owm):
        mask = frame["timestamp_utc"] == t + pd.Timedelta(hours=1)
        numeric = frame.select_dtypes("number").columns.drop("is_day", errors="ignore")
        frame.loc[mask, numeric] = frame.loc[mask, numeric] * 1.7 + 3
    changed = fe.assemble_features(mutated, warmup_hours=48)
    before = baseline[baseline["timestamp_utc"] <= t].reset_index(drop=True)
    after = changed[changed["timestamp_utc"] <= t].reset_index(drop=True)
    pd.testing.assert_frame_equal(before, after)
    assert not baseline.equals(changed)


def test_cutoff_drops_future_hours():
    index = pd.date_range(START, periods=6, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp_utc": index, "x": range(6)})
    out = fe.cutoff_future_hours(df, index[3])
    assert out["timestamp_utc"].max() == index[3]
    assert len(out) == 4


def test_hourly_fetch_never_returns_future_hours():
    now = pd.Timestamp("2025-06-15 12:00", tz="UTC")
    index = pd.date_range("2025-06-13", "2025-06-15 23:00", freq="1h", tz="UTC")
    session = FakeSession(
        {
            "api.open-meteo.com/v1/forecast": weather_payload(index),
            "air-quality-api": air_payload(index),
            "air_pollution": owm_payload(index[index <= now]),
        }
    )
    sources = fe.fetch_hourly_sources(session, now)
    assert sources.weather["timestamp_utc"].max() == now
    assert sources.air_om["timestamp_utc"].max() == now
    assert sources.air_owm["timestamp_utc"].max() <= now


@pytest.mark.parametrize("degrees", [0, 90, 180, 270, 359])
def test_wind_direction_round_trip(degrees):
    sin_val, cos_val = fe.encode_wind_direction(np.array([degrees], dtype=float))
    back = fe.reconstruct_wind_direction(sin_val, cos_val)[0]
    assert abs(back - degrees) < 1e-6 or abs(abs(back - degrees) - 360) < 1e-6


def test_long_gap_wind_direction_uses_circular_mean():
    df = raw_frame(days=6)
    hour0 = df.index[df["timestamp_utc"].dt.hour == 0]
    df.loc[hour0[:4], "wind_direction_10m"] = [355.0, 358.0, 2.0, 5.0]
    df = blank(df, "wind_direction_10m", 96, 36)
    out = fe.impute_causal(df)
    value = out.loc[120, "wind_direction_10m"]
    assert min(value, 360 - value) < 2.0


def test_short_gap_forward_fills():
    df = blank(raw_frame(), "temperature_2m", 150, 2)
    out = fe.impute_causal(df)
    assert out.loc[150, "temperature_2m"] == df.loc[149, "temperature_2m"]
    assert out.loc[151, "temperature_2m"] == df.loc[149, "temperature_2m"]
    assert out.loc[150, "is_imputed"] == 1
    assert out.loc[152, "is_imputed"] == 0


def test_medium_gap_uses_previous_day():
    df = blank(raw_frame(), "temperature_2m", 150, 10)
    out = fe.impute_causal(df)
    for k in (3, 6, 9):
        assert out.loc[150 + k, "temperature_2m"] == df.loc[150 + k - 24, "temperature_2m"]
    assert out.loc[150:159, "is_imputed"].eq(1).all()


def test_extended_gap_uses_trailing_same_hour_median():
    df = blank(raw_frame(), "temperature_2m", 200, 30)
    out = fe.impute_causal(df)
    for k in (24, 27, 29):
        row = 200 + k
        t = df.loc[row, "timestamp_utc"]
        prior = df[
            (df["timestamp_utc"] < t)
            & (df["timestamp_utc"] >= t - pd.Timedelta(hours=168))
            & (df["timestamp_utc"].dt.hour == t.hour)
        ]["temperature_2m"].dropna()
        assert out.loc[row, "temperature_2m"] == pytest.approx(prior.median())


def test_is_day_uses_previous_day_before_forward_fill():
    df = blank(raw_frame(), "is_day", 149, 4)
    out = fe.impute_causal(df)
    expected = df.loc[149 - 24 : 152 - 24, "is_day"].to_numpy()
    assert out.loc[149:152, "is_day"].to_numpy().tolist() == expected.tolist()
    assert out.loc[149:152, "is_imputed"].eq(1).all()


def test_fully_missing_column_falls_back_to_history():
    df = raw_frame(days=3)
    df["temperature_2m"] = np.nan
    history = raw_frame(days=7)
    history["timestamp_utc"] = history["timestamp_utc"] - pd.Timedelta(days=7)
    out = fe.impute_causal(df, history=history)
    row = out.index[out["timestamp_utc"].dt.hour == 5][0]
    expected = history.loc[history["timestamp_utc"].dt.hour == 5, "temperature_2m"].median()
    assert out.loc[row, "temperature_2m"] == pytest.approx(expected)


def test_particulates_are_averaged_when_both_sources_present():
    out = fe.assemble_features(make_sources(), warmup_hours=48)
    assert (out["pm2_5"] == 30.0).all()
    assert (out["pm10"] == 45.0).all()
    assert (out["source_used"] == "fused").all()


def test_missing_openweather_data_degrades_to_open_meteo_only():
    sources = make_sources(owm_gap=range(200, 230))
    out = fe.assemble_features(sources, warmup_hours=48)
    gap = out[(out["timestamp_utc"] >= START + pd.Timedelta(hours=200))
              & (out["timestamp_utc"] < START + pd.Timedelta(hours=230))]
    assert (gap["source_used"] == "open_meteo_only").all()
    assert (gap["pm2_5"] == 40.0).all()
    assert (out["source_used"] == "fused").sum() == len(out) - 30


def test_openweather_air_pollution_failure_does_not_crash():
    now = pd.Timestamp("2025-06-15 12:00", tz="UTC")
    index = pd.date_range("2025-06-13", "2025-06-15 23:00", freq="1h", tz="UTC")
    session = FakeSession(
        {
            "api.open-meteo.com/v1/forecast": weather_payload(index),
            "air-quality-api": air_payload(index),
            "air_pollution": FakeResponse({}, status=500),
        }
    )
    sources = fe.fetch_hourly_sources(session, now)
    assert sources.air_owm.empty
    out = fe.assemble_features(sources, warmup_hours=0)
    assert (out["source_used"] == "open_meteo_only").all()


@pytest.mark.parametrize("failure", ["error", "null"])
def test_weather_fallback_uses_openweather_current(failure):
    now = pd.Timestamp("2025-06-15 12:00", tz="UTC")
    index = pd.date_range("2025-06-13", "2025-06-15 23:00", freq="1h", tz="UTC")
    forecast = FakeResponse({}, status=503) if failure == "error" else weather_payload(index, null_at=now)
    current = {
        "dt": int(now.timestamp()),
        "main": {"temp": 35.0, "humidity": 40, "pressure": 998},
        "clouds": {"all": 20},
        "wind": {"speed": 5.0, "deg": 270},
        "sys": {
            "sunrise": int((now - pd.Timedelta(hours=7)).timestamp()),
            "sunset": int((now + pd.Timedelta(hours=6)).timestamp()),
        },
    }
    session = FakeSession(
        {
            "api.open-meteo.com/v1/forecast": forecast,
            "air-quality-api": air_payload(index),
            "air_pollution": owm_payload(index[index <= now]),
            "weather": current,
        }
    )
    sources = fe.fetch_hourly_sources(session, now)
    row = sources.weather[sources.weather["timestamp_utc"] == now].iloc[0]
    assert row["wind_speed_10m"] == pytest.approx(18.0)
    assert row["wind_direction_10m"] == 270
    assert row["is_day"] == 1
    assert row["precipitation_mm"] == 0.0
    assert pd.isna(row["precipitation_probability"])

    out = fe.assemble_features(sources, warmup_hours=0)
    last = out.iloc[-1]
    assert last["timestamp_utc"] == now
    assert last["source_used"] == "owm_fallback"
    assert last["is_imputed"] == 1


def test_get_json_retries_with_exponential_backoff(no_sleep):
    session = FakeSession(
        {"x": lambda params: FakeResponse({}, 500) if len(session.calls) < 3 else {"ok": 1}}
    )
    assert fe.get_json(session, "http://x", {}) == {"ok": 1}
    assert no_sleep == [1, 2]


def test_get_json_raises_after_retries_exhausted(no_sleep):
    session = FakeSession({"x": requests.ConnectionError("down")})
    with pytest.raises(requests.RequestException):
        fe.get_json(session, "http://x", {})
    assert no_sleep == [1, 2, 4]
    assert len(session.calls) == 4


def test_date_chunks_cover_range_without_gaps():
    start, end = pd.Timestamp("2024-01-01").date(), pd.Timestamp("2025-01-02").date()
    chunks = fe.date_chunks(start, end, size_days=90)
    assert chunks[0][0] == start and chunks[-1][1] == end
    for (_, prev_end), (next_start, _) in itertools.pairwise(chunks):
        assert next_start - prev_end == pd.Timedelta(days=1)
    assert all((b - a).days < 90 for a, b in chunks)


def test_openweather_chunk_bounds_span_full_days():
    day = pd.Timestamp("2025-01-01").date()
    lo, hi = fe.chunk_unix_bounds(day, day)
    assert hi - lo == 86399
    assert pd.Timestamp(lo, unit="s", tz="UTC") == pd.Timestamp("2025-01-01", tz="UTC")


def test_backfill_fetch_chunks_and_deduplicates():
    def om_handler(params):
        idx = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="1h", tz="UTC")
        return weather_payload(idx)

    def aq_handler(params):
        idx = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="1h", tz="UTC")
        return air_payload(idx)

    def owm_handler(params):
        idx = pd.date_range(
            pd.Timestamp(params["start"], unit="s", tz="UTC"),
            pd.Timestamp(params["end"], unit="s", tz="UTC"),
            freq="1h",
        )
        return owm_payload(idx)

    session = FakeSession(
        {
            "historical-forecast-api": om_handler,
            "air-quality-api": aq_handler,
            "air_pollution": owm_handler,
        }
    )
    start, end = pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-07-09").date()
    sources = fe.fetch_backfill_sources(session, start, end)
    om_calls = [c for c in session.calls if "historical-forecast-api" in c[0]]
    assert len(om_calls) == 3
    expected = (pd.Timestamp(end) - pd.Timestamp(start)).days * 24 + 24
    for frame in (sources.weather, sources.air_om, sources.air_owm):
        assert frame["timestamp_utc"].is_unique
        assert len(frame) == expected
