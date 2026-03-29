from __future__ import annotations

"""
Live data layer.
- OpenAQ supplies the latest available PM2.5 sensor readings and PM history
- Open-Meteo supplies real-time weather, PM10, and gaseous pollutants
"""

from datetime import datetime
from typing import TypedDict
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

from config import HCMC_LAT, HCMC_LON, HOURLY_VARIABLES, SENSOR_NAME
from src.aqi import VN_AQIResult, calculate_vn_aqi_hourly
from src.services.openmeteo_client import OpenMeteoClient
from src.services.openaq_client import OpenAQClient


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
UTC_TZ = ZoneInfo("UTC")
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

AIR_QUALITY_VARS = {
    "pm10": "pm10",
    "co": "carbon_monoxide",
    "no2": "nitrogen_dioxide",
    "so2": "sulphur_dioxide",
    "o3": "ozone",
}

DISPLAY_NAMES = {
    "pm25": "PM2.5",
    "pm10": "PM10",
    "co": "CO",
    "no2": "NO2",
    "so2": "SO2",
    "o3": "O3",
}


class PollutantData(TypedDict):
    code: str
    label: str
    value_1h: float | None
    unit: str
    hourly_24h: list[float | None]
    updated_at: str


class CurrentData(TypedDict):
    station: str
    updated_at: str
    pm25_updated_at: str
    weather_updated_at: str
    air_quality_updated_at: str
    aligned_model_at: str
    pollutants: dict[str, PollutantData]
    pm25: float | None
    pm10: float | None
    o3: float | None
    no2: float | None
    so2: float | None
    co: float | None
    pm25_3h: float | None
    pm25_24h: float | None
    temp: float
    humidity: int
    wind: float
    wind_dir: float
    precipitation: float
    pressure: float
    boundary_layer_height: float
    aqi: int | None
    aqi_label: str
    aqi_color: str
    aqi_text_color: str
    primary_pollutant: str | None


class HistoryPoint(TypedDict):
    time: str
    pm25: float | None


def _average(values: list[float | None]) -> float | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 1)


def _window_average(pm_history: pd.DataFrame, hours: int) -> float | None:
    if pm_history.empty:
        return None
    latest_timestamp = pd.to_datetime(pm_history["datetime"]).max()
    cutoff = latest_timestamp - pd.Timedelta(hours=hours - 1)
    window = pm_history[pd.to_datetime(pm_history["datetime"]) >= cutoff]
    if window.empty:
        return None
    return round(float(window["pm25_avg"].mean()), 1)


@st.cache_data(ttl=300)
def _fetch_openaq_pm25_history(hours: int = 36) -> pd.DataFrame:
    return OpenAQClient().fetch_pm25_history(lookback_hours=hours)


@st.cache_data(ttl=300)
def _fetch_weather_history(hours: int = 36) -> pd.DataFrame:
    return OpenMeteoClient().fetch_weather_history(lookback_hours=hours)


@st.cache_data(ttl=300)
def _fetch_current_weather() -> dict:
    current_vars = [value for value in HOURLY_VARIABLES if value != "boundary_layer_height"]
    response = requests.get(
        OPENMETEO_FORECAST_URL,
        params={
            "latitude": HCMC_LAT,
            "longitude": HCMC_LON,
            "current": ",".join(current_vars),
            "hourly": "boundary_layer_height",
            "timezone": "Asia/Ho_Chi_Minh",
            "forecast_days": 1,
            "wind_speed_unit": "ms",
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    current = payload["current"]

    hourly_times = payload["hourly"]["time"]
    hourly_blh = payload["hourly"]["boundary_layer_height"]
    current_time = current["time"]
    if current_time in hourly_times:
        current["boundary_layer_height"] = hourly_blh[hourly_times.index(current_time)] or 0.0
    else:
        current["boundary_layer_height"] = hourly_blh[0] if hourly_blh else 0.0
    return current


@st.cache_data(ttl=300)
def _fetch_air_quality() -> dict:
    current_vars = list(AIR_QUALITY_VARS.values())
    hourly_vars = list(AIR_QUALITY_VARS.values())
    response = requests.get(
        OPENMETEO_AIR_URL,
        params={
            "latitude": HCMC_LAT,
            "longitude": HCMC_LON,
            "current": ",".join(current_vars),
            "hourly": ",".join(hourly_vars),
            "timezone": "Asia/Ho_Chi_Minh",
            "past_hours": 24,
            "forecast_hours": 0,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def _build_current_pollutants(data: dict) -> tuple[dict[str, PollutantData], str]:
    current = data.get("current", {})
    current_units = data.get("current_units", {})
    hourly = data.get("hourly", {})
    updated_at = current.get("time", "")

    pollutants: dict[str, PollutantData] = {}
    for code, api_name in AIR_QUALITY_VARS.items():
        series = hourly.get(api_name, []) or []
        pollutants[code] = {
            "code": code,
            "label": DISPLAY_NAMES[code],
            "value_1h": None if current.get(api_name) is None else round(float(current[api_name]), 1),
            "unit": current_units.get(api_name, "µg/m³"),
            "hourly_24h": [None if value is None else round(float(value), 1) for value in series[-24:]],
            "updated_at": updated_at,
        }
    return pollutants, updated_at


def _inject_openaq_pm25(
    pollutants: dict[str, PollutantData],
    pm_history: pd.DataFrame,
) -> tuple[dict[str, PollutantData], str]:
    updated = dict(pollutants)
    latest_timestamp = pd.to_datetime(pm_history["datetime"]).max()
    cutoff = latest_timestamp - pd.Timedelta(hours=23)
    pm_values = pm_history[pd.to_datetime(pm_history["datetime"]) >= cutoff]["pm25_avg"].tolist()
    updated_at = latest_timestamp.replace(tzinfo=UTC_TZ).astimezone(VN_TZ).strftime("%H:%M")
    updated["pm25"] = {
        "code": "pm25",
        "label": DISPLAY_NAMES["pm25"],
        "value_1h": None if pm_history.empty else round(float(pm_history.iloc[-1]["pm25_avg"]), 1),
        "unit": "µg/m³",
        "hourly_24h": [round(float(value), 1) for value in pm_values],
        "updated_at": updated_at,
    }
    return updated, updated_at


def _calculate_aqi(pollutants: dict[str, PollutantData]) -> VN_AQIResult:
    payload = {
        "pm25": {
            "hourly_12h": list(reversed(pollutants["pm25"]["hourly_24h"][-12:])),
            "value_1h": pollutants["pm25"]["value_1h"],
        },
        "pm10": {
            "hourly_12h": list(reversed(pollutants["pm10"]["hourly_24h"][-12:])),
            "value_1h": pollutants["pm10"]["value_1h"],
        },
        "o3": {"value_1h": pollutants["o3"]["value_1h"]},
        "no2": {"value_1h": pollutants["no2"]["value_1h"]},
        "so2": {"value_1h": pollutants["so2"]["value_1h"]},
        "co": {"value_1h": pollutants["co"]["value_1h"]},
    }
    return calculate_vn_aqi_hourly(payload)


@st.cache_data(ttl=300)
def get_current_data() -> CurrentData:
    fetched_at = datetime.now(VN_TZ).strftime("%H:%M")
    pm_history = _fetch_openaq_pm25_history()
    weather_history = _fetch_weather_history()
    current_weather = _fetch_current_weather()
    air_quality = _fetch_air_quality()

    pollutants, air_quality_updated_iso = _build_current_pollutants(air_quality)
    pollutants, pm25_updated_at = _inject_openaq_pm25(pollutants, pm_history)
    aqi = _calculate_aqi(pollutants)

    aligned_model_utc = min(
        pd.to_datetime(pm_history["datetime"]).max().to_pydatetime(),
        pd.to_datetime(weather_history["datetime"]).max().to_pydatetime(),
    )
    aligned_model_at = aligned_model_utc.replace(tzinfo=UTC_TZ).astimezone(VN_TZ).strftime("%H:%M")
    weather_updated_at = datetime.fromisoformat(current_weather["time"]).strftime("%H:%M")
    air_quality_updated_at = datetime.fromisoformat(air_quality_updated_iso).strftime("%H:%M") if air_quality_updated_iso else fetched_at

    pm25_3h = _window_average(pm_history, 3)
    pm25_24h = _window_average(pm_history, 24)

    return CurrentData(
        station=f"{SENSOR_NAME} Station - TP.HCM",
        updated_at=fetched_at,
        pm25_updated_at=pm25_updated_at,
        weather_updated_at=weather_updated_at,
        air_quality_updated_at=air_quality_updated_at,
        aligned_model_at=aligned_model_at,
        pollutants=pollutants,
        pm25=pollutants["pm25"]["value_1h"],
        pm10=pollutants["pm10"]["value_1h"],
        o3=pollutants["o3"]["value_1h"],
        no2=pollutants["no2"]["value_1h"],
        so2=pollutants["so2"]["value_1h"],
        co=pollutants["co"]["value_1h"],
        pm25_3h=pm25_3h,
        pm25_24h=pm25_24h,
        temp=float(current_weather["temperature_2m"]),
        humidity=int(current_weather["relative_humidity_2m"]),
        wind=float(current_weather["wind_speed_10m"]),
        wind_dir=float(current_weather["wind_direction_10m"]),
        precipitation=float(current_weather["precipitation"]),
        pressure=float(current_weather["surface_pressure"]),
        boundary_layer_height=float(current_weather["boundary_layer_height"]),
        aqi=aqi.aqi,
        aqi_label=aqi.label,
        aqi_color=aqi.bg_color,
        aqi_text_color=aqi.text_color,
        primary_pollutant=aqi.primary_pollutant,
    )


@st.cache_data(ttl=300)
def get_history_24h() -> list[HistoryPoint]:
    pm_history = _fetch_openaq_pm25_history()
    now_utc_hour = datetime.now(UTC_TZ).replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
    start_utc_hour = now_utc_hour - pd.Timedelta(hours=23)

    frame = pd.DataFrame({"datetime": pd.date_range(start=start_utc_hour, end=now_utc_hour, freq="h")})
    pm_history = pm_history.copy()
    pm_history["datetime"] = pd.to_datetime(pm_history["datetime"], utc=False)
    merged = frame.merge(pm_history, on="datetime", how="left")

    history: list[HistoryPoint] = []
    for _, row in merged.iterrows():
        history.append(
            HistoryPoint(
                time=row["datetime"].replace(tzinfo=UTC_TZ).astimezone(VN_TZ).strftime("%H:%M"),
                pm25=None if pd.isna(row["pm25_avg"]) else round(float(row["pm25_avg"]), 1),
            )
        )
    return history
