import os

from dotenv import load_dotenv

load_dotenv()

LAT = 31.5204
LON = 74.3587

FEATURE_GROUP_NAME = "lahore_weather_features"
FEATURE_GROUP_VERSION = 1

REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
CHUNK_DAYS = 90
WARMUP_HOURS = 48
HISTORY_DAYS = 7

OPEN_METEO_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OWM_AIR_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
OWM_CURRENT_URL = "https://pro.openweathermap.org/data/2.5/weather"

WEATHER_HOURLY = [
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
AIR_HOURLY = ["us_aqi", "pm2_5", "pm10", "ozone", "uv_index"]

CORE_METRICS = [
    "us_aqi",
    "temperature_2m",
    "precipitation_mm",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_mm",
    "precipitation_probability",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "is_day",
]

RAW_COLUMNS = [
    "us_aqi",
    "temperature_2m",
    "precipitation_mm",
    "precipitation_probability",
    "relative_humidity_2m",
    "uv_index",
    "wind_speed_10m",
    "wind_direction_10m",
    "pm2_5",
    "pm10",
    "ozone",
    "surface_pressure",
    "cloud_cover",
    "is_day",
]

EXPECTED_COLUMNS = (
    ["timestamp_utc", "source_used", "is_imputed"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_day"]
    + [
        "us_aqi",
        "temperature_2m",
        "precipitation_mm",
        "precipitation_probability",
        "relative_humidity_2m",
        "uv_index",
        "wind_speed_10m",
        "wind_direction_10m",
    ]
    + ["wind_dir_sin", "wind_dir_cos"]
    + ["pm2_5", "pm10", "ozone", "surface_pressure", "cloud_cover"]
    + [f"{m}_lag_24h" for m in CORE_METRICS]
    + [f"{m}_diff_3h" for m in CORE_METRICS]
    + [f"{m}_roll_mean_24h" for m in CORE_METRICS]
)


def openweather_key():
    return os.environ["OPENWEATHER_API_KEY"]
