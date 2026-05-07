from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

_DEFAULT_CONNECT_TIMEOUT = 10   # seconds to establish connection
_DEFAULT_READ_TIMEOUT    = 60   # seconds to wait for response
_DEFAULT_MAX_RETRIES     = 3
_DEFAULT_BACKOFF_FACTOR  = 2    # waits 2s → 4s → 8s between retries


def _build_session(max_retries: int, backoff_factor: float) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass(slots=True)
class OpenMeteoClient:
    latitude: float = DEFAULT_LAT
    longitude: float = DEFAULT_LON
    base_url: str = OPENMETEO_FORECAST_URL
    connect_timeout: int = _DEFAULT_CONNECT_TIMEOUT
    read_timeout: int = _DEFAULT_READ_TIMEOUT
    max_retries: int = _DEFAULT_MAX_RETRIES
    backoff_factor: float = _DEFAULT_BACKOFF_FACTOR
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.latitude = float(os.getenv("OPENMETEO_LAT", self.latitude))
        self.longitude = float(os.getenv("OPENMETEO_LON", self.longitude))
        self._session = _build_session(self.max_retries, self.backoff_factor)

    @property
    def _timeout(self) -> tuple[int, int]:
        return (self.connect_timeout, self.read_timeout)

    def fetch_weather_history(self, lookback_hours: int = 72) -> pd.DataFrame:
        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")

        now_utc = datetime.now(timezone.utc)

        try:
            response = self._session.get(
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
                timeout=self._timeout,
            )
            response.raise_for_status()

        except requests.exceptions.ConnectTimeout:
            raise RuntimeError(
                f"Could not connect to Open-Meteo within {self.connect_timeout}s. "
                "Check your network connection."
            )
        except requests.exceptions.ReadTimeout:
            raise RuntimeError(
                f"Open-Meteo did not respond within {self.read_timeout}s after "
                f"{self.max_retries} retries. The API may be temporarily slow."
            )
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(f"Network error reaching Open-Meteo: {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Open-Meteo returned HTTP {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc

        payload = response.json()
        hourly = payload.get("hourly", {})

        if not hourly or "time" not in hourly:
            raise RuntimeError("Open-Meteo did not return hourly weather history.")

        dataframe = pd.DataFrame({
            "datetime": pd.to_datetime(hourly["time"], utc=True).tz_localize(None)
        })
        for column in HOURLY_VARIABLES:
            dataframe[column] = hourly.get(column)

        dataframe = dataframe.sort_values("datetime").reset_index(drop=True)
        lower_bound = now_utc.replace(tzinfo=None) - pd.Timedelta(hours=lookback_hours)
        upper_bound = now_utc.replace(tzinfo=None)
        dataframe = dataframe[
            (dataframe["datetime"] >= lower_bound) &
            (dataframe["datetime"] <= upper_bound)
        ].copy()

        return dataframe.reset_index(drop=True)