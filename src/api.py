"""
Data fetching layer.
- get_current_data : PM2.5 từ OpenAQ v3 + meteorology từ Open-Meteo
- get_history_24h  : TODO – thay bằng OpenAQ /hours data khi cần chart thực
- get_forecast_6h  : TODO – thay bằng model ML khi sẵn sàng
"""

import random
from datetime import datetime, timedelta
from typing import TypedDict

import requests
import streamlit as st
from dotenv import load_dotenv

from config import (
    BASE_URL_OPENAQ,
    HEADERS,
    HCMC_LAT,
    HCMC_LON,
    HOURLY_VARIABLES,
    LOCATION_ID,
    SENSOR_ID,
    SENSOR_NAME,
)

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# TYPES
# ══════════════════════════════════════════════════════════════════════════════

class CurrentData(TypedDict):
    pm25: float
    pm25_3h: float
    pm25_24h: float
    temp: float
    humidity: int
    wind: float
    wind_dir: float
    precipitation: float
    pressure: float
    boundary_layer_height: float
    updated_at: str
    station: str

class HistoryPoint(TypedDict):
    time: str
    pm25: float


# ══════════════════════════════════════════════════════════════════════════════
# MOCK HELPERS  (vẫn dùng cho get_history_24h / get_forecast_6h)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_pm25_series(n: int, base: float = 38.0, noise: float = 8.0) -> list[float]:
    values = []
    val = base
    for _ in range(n):
        val += random.gauss(0, noise * 0.3)
        val = max(5.0, min(200.0, val))
        values.append(round(val, 1))
    return values


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – PM2.5  (OpenAQ v3)
# Docs: https://docs.openaq.org
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def _fetch_pm25() -> tuple[float, list[float]]:
    """
    Trả về (pm25_latest, pm25_values_24h).

    Bước 1 – GET /v3/locations/{LOCATION_ID}/latest
        Lấy giá trị PM2.5 mới nhất từ SENSOR_ID đã biết trong config.

    Bước 2 – GET /v3/sensors/{SENSOR_ID}/hours
        Lấy 24 bản ghi hourly gần nhất (sort desc) để tính avg 3h / 24h.
    """
    # ── Bước 1: latest PM2.5 ─────────────────────────────────────────────────
    r = requests.get(
        f"{BASE_URL_OPENAQ}/locations/{LOCATION_ID}/latest",
        headers=HEADERS,
        timeout=10,
    )
    r.raise_for_status()

    pm25_entries = [
        row for row in r.json().get("results", [])
        if row["sensorsId"] == SENSOR_ID
    ]
    if not pm25_entries:
        raise ValueError(f"Không tìm thấy dữ liệu cho sensor {SENSOR_ID} ({SENSOR_NAME})")

    pm25_latest = pm25_entries[0]["value"]

    # ── Bước 2: 24h hourly history ───────────────────────────────────────────
    r = requests.get(
        f"{BASE_URL_OPENAQ}/sensors/{SENSOR_ID}/hours",
        params={"limit": 24, "order_by": "datetime", "sort_order": "desc"},
        headers=HEADERS,
        timeout=10,
    )
    values_24h: list[float] = []
    if r.ok:
        values_24h = [
            row["value"]
            for row in r.json().get("results", [])
            if row.get("value") is not None
        ]
    if not values_24h:
        values_24h = [pm25_latest]  # fallback

    return pm25_latest, values_24h


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – METEOROLOGY  (Open-Meteo)
# Docs: https://open-meteo.com/en/docs
# Dùng HCMC_LAT, HCMC_LON, HOURLY_VARIABLES từ config
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def _fetch_meteo() -> dict:
    """
    Một request duy nhất lấy:
      - current : tất cả biến trong HOURLY_VARIABLES trừ boundary_layer_height
      - hourly  : boundary_layer_height
                  (không có trong current → lấy từ hourly, match theo current["time"])

    Lưu ý: BASE_URL_OPENMETEO trong config trỏ đến archive API (dùng để train).
    Ở đây dùng forecast endpoint để lấy current weather.
    """
    current_vars = [v for v in HOURLY_VARIABLES if v != "boundary_layer_height"]

    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude":      HCMC_LAT,
            "longitude":     HCMC_LON,
            "current":       ",".join(current_vars),
            "hourly":        "boundary_layer_height",
            "timezone":      "Asia/Ho_Chi_Minh",
            "forecast_days": 1,
        },
        timeout=10,
    )
    r.raise_for_status()
    data    = r.json()
    current = data["current"]

    # current["time"] format: "2025-03-29T14:00" — khớp với hourly["time"]
    hourly_times = data["hourly"]["time"]
    hourly_blh   = data["hourly"]["boundary_layer_height"]
    current_time = current["time"]

    blh = 0.0
    if current_time in hourly_times:
        blh = hourly_blh[hourly_times.index(current_time)] or 0.0
    elif hourly_blh:
        blh = hourly_blh[0] or 0.0

    current["boundary_layer_height"] = blh
    return current


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def get_current_data() -> CurrentData:
    meteo = _fetch_meteo()

    pm25_latest = 0.0
    pm25_3h     = 0.0
    pm25_24h    = 0.0
    station     = f"{SENSOR_NAME} Station - TP.HCM"

    try:
        pm25_latest, values_24h = _fetch_pm25()
        pm25_3h  = round(sum(values_24h[:3]) / min(3, len(values_24h)), 1)
        pm25_24h = round(sum(values_24h) / len(values_24h), 1)
    except Exception as e:
        station = f"{SENSOR_NAME} [OpenAQ lỗi: {e}]"

    return CurrentData(
        pm25                  = round(pm25_latest, 1),
        pm25_3h               = pm25_3h,
        pm25_24h              = pm25_24h,
        temp                  = meteo["temperature_2m"],
        humidity              = int(meteo["relative_humidity_2m"]),
        wind                  = meteo["wind_speed_10m"],
        wind_dir              = meteo["wind_direction_10m"],
        precipitation         = meteo["precipitation"],
        pressure              = meteo["surface_pressure"],
        boundary_layer_height = meteo["boundary_layer_height"],
        updated_at            = datetime.now().strftime("%H:%M"),
        station               = station,
    )


@st.cache_data(ttl=300)
def get_history_24h() -> list[HistoryPoint]:
    try:
        _, values_24h = _fetch_pm25()

        now = datetime.now()

        # đảo lại: oldest → latest
        values_24h = list(reversed(values_24h))

        return [
            HistoryPoint(
                time=(now - timedelta(hours=23 - i)).strftime("%H:%M"),
                pm25=round(values_24h[i], 1),
            )
            for i in range(len(values_24h))
        ]

    except Exception as e:
        print("get_history_24h error:", e)
        return []


def get_forecast_6h() -> list[HistoryPoint]:
    """
    Dự báo PM2.5 6h tới.
    TODO: thay bằng model ML khi sẵn sàng.
    """
    now    = datetime.now()
    series = _mock_pm25_series(6, base=get_current_data()["pm25"], noise=6.0)
    return [
        HistoryPoint(
            time=(now + timedelta(hours=i + 1)).strftime("%H:%M"),
            pm25=series[i],
        )
        for i in range(6)
    ]