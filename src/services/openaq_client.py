from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

try:
    from config import SENSOR_ID as DEFAULT_SENSOR_ID
except Exception:
    DEFAULT_SENSOR_ID = 11357424


OPENAQ_BASE_URL = "https://api.openaq.org/v3"


@dataclass(slots=True)
class OpenAQClient:
    api_key: str | None = None
    sensor_id: int = DEFAULT_SENSOR_ID
    base_url: str = OPENAQ_BASE_URL
    timeout: int = 30

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("OPENAQ_API_KEY")
        sensor_override = os.getenv("OPENAQ_SENSOR_ID")
        if sensor_override:
            self.sensor_id = int(sensor_override)

    @property
    def headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError(
                "Missing OPENAQ_API_KEY. Add it to your environment or .env file to fetch live PM2.5 data."
            )
        return {"X-API-Key": self.api_key}

    def fetch_pm25_history(self, lookback_hours: int = 72) -> pd.DataFrame:
        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")

        now_utc = datetime.now(timezone.utc)
        start_utc = now_utc - timedelta(hours=lookback_hours + 4)
        page = 1
        results: list[dict[str, Any]] = []
        page_limit = min(max(lookback_hours * 2, 200), 1000)

        while True:
            response = requests.get(
                f"{self.base_url}/sensors/{self.sensor_id}/hours",
                params={
                    "datetime_from": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "datetime_to": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "limit": page_limit,
                    "page": page,
                },
                headers=self.headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("results", [])
            if not batch:
                break
            results.extend(batch)
            if len(batch) < page_limit:
                break
            page += 1

        history = self._parse_history(results)
        if history.empty:
            raise RuntimeError("OpenAQ returned no hourly PM2.5 history for the configured sensor.")
        lower_bound = now_utc.replace(tzinfo=None) - timedelta(hours=lookback_hours)
        history = history[(history["datetime"] >= lower_bound) & (history["datetime"] <= now_utc.replace(tzinfo=None))].copy()
        if history.empty:
            raise RuntimeError("OpenAQ returned no PM2.5 rows inside the requested real-time window.")
        return history.reset_index(drop=True)

    def _parse_history(self, results: list[dict[str, Any]]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for item in results:
            dt_utc = item.get("period", {}).get("datetimeFrom", {}).get("utc")
            if not dt_utc:
                continue

            summary = item.get("summary", {})
            coverage = item.get("coverage", {})
            avg = summary.get("avg")
            if avg is None or (isinstance(avg, float) and math.isnan(avg)):
                continue

            rows.append(
                {
                    "datetime": pd.to_datetime(dt_utc, utc=True).tz_localize(None),
                    "pm25_avg": float(avg),
                    "coverage_pct": float(coverage.get("percentComplete", 100.0) or 100.0),
                }
            )

        dataframe = pd.DataFrame(rows)
        if dataframe.empty:
            return dataframe

        return (
            dataframe.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .reset_index(drop=True)
        )
