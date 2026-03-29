from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import requests

try:
    from config import HCMC_LAT as DEFAULT_LAT
    from config import HCMC_LON as DEFAULT_LON
except Exception:
    DEFAULT_LAT = 10.8231
    DEFAULT_LON = 106.6297


OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
    "boundary_layer_height",
]


@dataclass(slots=True)
class OpenMeteoClient:
    latitude: float = DEFAULT_LAT
    longitude: float = DEFAULT_LON
    base_url: str = OPENMETEO_FORECAST_URL
    timeout: int = 30

    def __post_init__(self) -> None:
        self.latitude = float(os.getenv("OPENMETEO_LAT", self.latitude))
        self.longitude = float(os.getenv("OPENMETEO_LON", self.longitude))

    def fetch_weather_history(self, lookback_hours: int = 72) -> pd.DataFrame:
        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")

        now_utc = datetime.now(timezone.utc)
        response = requests.get(
            self.base_url,
            params={
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ",".join(HOURLY_VARIABLES),
                "past_hours": lookback_hours + 4,
                "forecast_hours": 0,
                "timezone": "GMT",
                "wind_speed_unit": "ms",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly", {})

        if not hourly or "time" not in hourly:
            raise RuntimeError("Open-Meteo did not return hourly weather history.")

        dataframe = pd.DataFrame({"datetime": pd.to_datetime(hourly["time"], utc=True).tz_localize(None)})
        for column in HOURLY_VARIABLES:
            dataframe[column] = hourly.get(column)

        dataframe = dataframe.sort_values("datetime").reset_index(drop=True)
        lower_bound = now_utc.replace(tzinfo=None) - pd.Timedelta(hours=lookback_hours)
        upper_bound = now_utc.replace(tzinfo=None)
        dataframe = dataframe[(dataframe["datetime"] >= lower_bound) & (dataframe["datetime"] <= upper_bound)].copy()
        return dataframe.reset_index(drop=True)
