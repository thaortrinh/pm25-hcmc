from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from src.inference.artifact import LoadedArtifact, load_or_create_artifact
from src.inference.feature_builder import merge_pm_and_weather, prepare_live_feature_row
from src.services.openaq_client import OpenAQClient
from src.services.openmeteo_client import OpenMeteoClient


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
UTC_TZ = ZoneInfo("UTC")


@dataclass(slots=True)
class ForecastBundle:
    generated_at_utc: datetime
    generated_at_vn: datetime
    predictions: list[float]
    prediction_times_utc: list[datetime]
    prediction_times_vn: list[datetime]
    merged_history: pd.DataFrame
    feature_row: pd.DataFrame
    artifact: LoadedArtifact
    warnings: list[str]


def fetch_recent_inputs(lookback_hours: int = 72) -> tuple[pd.DataFrame, pd.DataFrame]:
    pm_history = OpenAQClient().fetch_pm25_history(lookback_hours=lookback_hours)
    weather_history = OpenMeteoClient().fetch_weather_history(lookback_hours=lookback_hours)
    return pm_history, weather_history


def summarize_alignment(pm_history: pd.DataFrame, weather_history: pd.DataFrame) -> dict[str, datetime | int | None]:
    pm = pm_history.copy()
    weather = weather_history.copy()
    pm["datetime"] = pd.to_datetime(pm["datetime"], utc=False)
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=False)
    common_timestamps = sorted(set(pm["datetime"]).intersection(set(weather["datetime"])))
    latest_common = common_timestamps[-1] if common_timestamps else None
    return {
        "latest_pm": None if pm.empty else pm["datetime"].max().to_pydatetime(),
        "latest_weather": None if weather.empty else weather["datetime"].max().to_pydatetime(),
        "latest_common": None if latest_common is None else pd.Timestamp(latest_common).to_pydatetime(),
        "common_count": len(common_timestamps),
    }


def apply_latest_overrides(
    pm_history: pd.DataFrame,
    weather_history: pd.DataFrame,
    overrides: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not overrides:
        return pm_history, weather_history

    pm = pm_history.copy()
    weather = weather_history.copy()
    pm["datetime"] = pd.to_datetime(pm["datetime"], utc=False)
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=False)

    alignment = summarize_alignment(pm, weather)
    target_timestamp = alignment["latest_common"]

    if not pm.empty:
        if target_timestamp is not None and (pm["datetime"] == target_timestamp).any():
            pm_target_index = pm.index[pm["datetime"] == target_timestamp][-1]
        else:
            pm_target_index = pm.index[-1]
        pm.loc[pm_target_index, "pm25_avg"] = float(overrides.get("pm25_avg", pm.loc[pm_target_index, "pm25_avg"]))
        pm.loc[pm_target_index, "coverage_pct"] = float(
            overrides.get("coverage_pct", pm.loc[pm_target_index].get("coverage_pct", 100.0))
        )

    if not weather.empty:
        if target_timestamp is not None and (weather["datetime"] == target_timestamp).any():
            weather_target_index = weather.index[weather["datetime"] == target_timestamp][-1]
        else:
            weather_target_index = weather.index[-1]
        for column in [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "surface_pressure",
            "boundary_layer_height",
        ]:
            if column in overrides:
                weather.loc[weather_target_index, column] = float(overrides[column])

    return pm, weather


def validate_recent_history(merged_history: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if merged_history.empty:
        raise RuntimeError("No aligned PM2.5 and weather rows are available for inference.")
    if len(merged_history) < 30:
        raise RuntimeError("The aligned history window is too short. At least 30 hourly rows are recommended.")

    latest_timestamp = pd.to_datetime(merged_history["datetime"]).max()
    age_hours = (datetime.utcnow() - latest_timestamp.to_pydatetime()).total_seconds() / 3600
    if age_hours > 3:
        warnings.append("Dữ liệu PM2.5 mới nhất đã cũ hơn 3 giờ; kết quả có thể kém ổn định.")
    if merged_history["pm25_avg"].isna().any():
        warnings.append("Có giá trị PM2.5 bị thiếu sau khi ghép dữ liệu; các hàng thiếu sẽ bị loại trước suy luận.")
    return warnings


def run_forecast(
    *,
    lookback_hours: int = 72,
    pm_history: pd.DataFrame | None = None,
    weather_history: pd.DataFrame | None = None,
    overrides: dict[str, float] | None = None,
    force_rebuild_model: bool = False,
) -> ForecastBundle:
    if pm_history is None or weather_history is None:
        pm_history, weather_history = fetch_recent_inputs(lookback_hours=lookback_hours)

    pm_history, weather_history = apply_latest_overrides(pm_history, weather_history, overrides)
    merged_history = merge_pm_and_weather(pm_history, weather_history)
    warnings = validate_recent_history(merged_history)

    artifact = load_or_create_artifact(force_rebuild=force_rebuild_model)
    feature_row, enriched_history = prepare_live_feature_row(merged_history, artifact.metadata)
    raw_prediction = artifact.model.predict(feature_row)[0]
    predictions = [round(max(float(value), 0.0), 1) for value in raw_prediction]

    latest_timestamp = pd.to_datetime(merged_history["datetime"]).max().to_pydatetime()
    prediction_times_utc = [latest_timestamp + timedelta(hours=offset) for offset in range(1, 7)]
    prediction_times_vn = [
        timestamp.replace(tzinfo=UTC_TZ).astimezone(VN_TZ).replace(tzinfo=None)
        for timestamp in prediction_times_utc
    ]
    generated_at_vn = latest_timestamp.replace(tzinfo=UTC_TZ).astimezone(VN_TZ).replace(tzinfo=None)

    return ForecastBundle(
        generated_at_utc=latest_timestamp,
        generated_at_vn=generated_at_vn,
        predictions=predictions,
        prediction_times_utc=prediction_times_utc,
        prediction_times_vn=prediction_times_vn,
        merged_history=enriched_history,
        feature_row=feature_row,
        artifact=artifact,
        warnings=warnings,
    )
