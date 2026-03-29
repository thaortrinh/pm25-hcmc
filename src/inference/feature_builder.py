from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import holidays
except Exception:  # pragma: no cover
    holidays = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PM_PATH = PROJECT_ROOT / "data" / "raw" / "pm25_sensor_11357424.csv"
RAW_WEATHER_PATH = PROJECT_ROOT / "data" / "raw" / "weather_openmeteo.csv"
PROCESSED_REFERENCE_PATH = PROJECT_ROOT / "data" / "processed" / "pm25_processed_data.csv"

SOURCE_TARGET_COL = "pm25_avg"
HORIZONS = [1, 2, 3, 4, 5, 6]
CATEGORICAL_COLS = ["Season", "time_of_the_day"]
PM_LAGS_TO_ADD = [2, 3, 4, 5, 12, 18]
PM_ROLL_STD_WINDOWS = [3, 6]
PM_TREND_WINDOWS = [3, 6]
WEATHER_LAG_SPECS = {
    "temperature_2m": [1, 3, 6],
    "relative_humidity_2m": [1, 3, 6],
    "precipitation": [1, 3, 6],
    "wind_speed_10m": [1, 3, 6],
}
WEATHER_ROLL_MEAN_SPECS = {
    "temperature_2m": [3],
    "relative_humidity_2m": [3],
    "wind_speed_10m": [3],
}
BASE_BINARY_COLUMNS = [
    "is_weekend",
    "rush_hour_weekday",
    "Monday_start",
    "special_holidays",
    "is_stagnant_humid",
]
TRAINING_ONLY_COLUMNS = ["target_next_hour"]
MANUAL_SPECIAL_DATES = {
    "2025-02-12",
    "2025-04-30",
    "2025-06-01",
    "2025-08-08",
    "2025-09-02",
    "2026-02-16",
    "2026-02-17",
    "2026-02-18",
    "2026-02-19",
    "2026-03-03",
}


@dataclass(slots=True)
class PreprocessorMetadata:
    feature_columns: list[str]
    categorical_columns: list[str]
    scale_columns: list[str]
    base_reference_columns: list[str]
    scaler_mean: dict[str, float]
    scaler_scale: dict[str, float]
    target_columns: list[str]
    special_holiday_dates: list[str]
    excluded_training_columns: list[str]
    reconstruction_mae: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return {
            "feature_columns": self.feature_columns,
            "categorical_columns": self.categorical_columns,
            "scale_columns": self.scale_columns,
            "base_reference_columns": self.base_reference_columns,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "target_columns": self.target_columns,
            "special_holiday_dates": self.special_holiday_dates,
            "excluded_training_columns": self.excluded_training_columns,
            "reconstruction_mae": self.reconstruction_mae,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PreprocessorMetadata":
        return cls(
            feature_columns=list(payload["feature_columns"]),
            categorical_columns=list(payload["categorical_columns"]),
            scale_columns=list(payload["scale_columns"]),
            base_reference_columns=list(payload["base_reference_columns"]),
            scaler_mean={key: float(value) for key, value in payload["scaler_mean"].items()},
            scaler_scale={key: float(value) for key, value in payload["scaler_scale"].items()},
            target_columns=list(payload["target_columns"]),
            special_holiday_dates=list(payload.get("special_holiday_dates", [])),
            excluded_training_columns=list(payload.get("excluded_training_columns", [])),
            reconstruction_mae=payload.get("reconstruction_mae"),
        )


def load_local_training_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    pm_history = pd.read_csv(RAW_PM_PATH, parse_dates=["datetime"])
    weather_history = pd.read_csv(RAW_WEATHER_PATH, parse_dates=["datetime"])
    return pm_history, weather_history


def merge_pm_and_weather(pm_history: pd.DataFrame, weather_history: pd.DataFrame) -> pd.DataFrame:
    pm = pm_history.copy()
    weather = weather_history.copy()
    pm["datetime"] = pd.to_datetime(pm["datetime"], utc=False)
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=False)
    return (
        pd.merge(
            pm.drop(columns=["pm25_sd"], errors="ignore"),
            weather,
            on="datetime",
            how="inner",
        )
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )


def infer_special_holiday_dates() -> list[str]:
    inferred = set(MANUAL_SPECIAL_DATES)
    if holidays is not None:
        vn_holidays = holidays.country_holidays("VN", years=[2024, 2025, 2026, 2027])
        inferred.update(str(day) for day in vn_holidays.keys())
    return sorted(inferred)


def _build_special_holiday_flag(index: pd.Series, special_dates: Iterable[str]) -> pd.Series:
    normalized = {pd.Timestamp(day).date() for day in special_dates}
    return index.dt.date.map(lambda day: 1 if day in normalized else 0).astype(int)


def _derive_time_of_day(hours: pd.Series) -> pd.Series:
    conditions = [
        hours.between(5, 11),
        hours.between(12, 16),
        hours.between(17, 20),
    ]
    choices = ["Morning", "Noon", "Evening"]
    return pd.Series(np.select(conditions, choices, default="Night"), index=hours.index)


def _ensure_wind_components(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    if "wind_u" not in df.columns or "wind_v" not in df.columns:
        wind_rad = np.radians(df["wind_direction_10m"].fillna(0.0))
        df["wind_u"] = -df["wind_speed_10m"].fillna(0.0) * np.sin(wind_rad)
        df["wind_v"] = -df["wind_speed_10m"].fillna(0.0) * np.cos(wind_rad)
    return df


def build_base_feature_frame(
    merged: pd.DataFrame,
    *,
    include_training_only_columns: bool,
    special_holiday_dates: Iterable[str] | None = None,
) -> pd.DataFrame:
    df = _ensure_wind_components(merged)
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=False)
    df = df.sort_values("datetime").reset_index(drop=True)

    df["coverage_pct"] = df.get("coverage_pct", 100.0)
    df["month"] = df["datetime"].dt.month
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }
    df["Season"] = df["month"].map(season_map)
    df["time_of_the_day"] = _derive_time_of_day(df["hour"])
    df["rush_hour_weekday"] = (
        (df["weekday"] < 5) & (df["hour"].isin([6, 7, 8, 9, 16, 17, 18, 19]))
    ).astype(int)
    df["Monday_start"] = ((df["weekday"] == 0) & (df["hour"] == 0)).astype(int)

    df["pm25_avg_lag1"] = df["pm25_avg"].shift(1)
    df["pm25_avg_lag6"] = df["pm25_avg"].shift(6)
    df["pm25_avg_lag24"] = df["pm25_avg"].shift(24)
    df["temperature_2m_lag1"] = df["temperature_2m"].shift(1)
    df["pm25_avg_rolling_mean_3h"] = df["pm25_avg"].rolling(window=3, min_periods=3).mean()
    df["pm25_avg_rolling_mean_6h"] = df["pm25_avg"].rolling(window=6, min_periods=6).mean()
    df["pm25_avg_rolling_mean_12h"] = df["pm25_avg"].rolling(window=12, min_periods=12).mean()
    df["temperature_2m_rolling_mean_3h"] = (
        df["temperature_2m"].rolling(window=3, min_periods=3).mean()
    )

    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 366)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 366)
    holiday_dates = list(special_holiday_dates or infer_special_holiday_dates())
    df["special_holidays"] = _build_special_holiday_flag(df["datetime"], holiday_dates)

    df["pm25_avg_diff_1h"] = df["pm25_avg"].diff(1)
    df["ventilation_coeff"] = df["wind_speed_10m"] * df["boundary_layer_height"]
    df["wind_magnitude"] = np.sqrt(df["wind_u"] ** 2 + df["wind_v"] ** 2)
    df["wind_stagnation_index"] = 1.0 / (df["wind_speed_10m"] + 0.1)
    df["is_stagnant_humid"] = (
        (df["relative_humidity_2m"] > 80) & (df["wind_speed_10m"] < 2)
    ).astype(int)
    df["dew_point"] = df["temperature_2m"] - ((100 - df["relative_humidity_2m"]) / 5.0)
    df["pm25_delta"] = df["pm25_avg"].shift(1) - df["pm25_avg"].shift(2)
    df["pm25_acceleration"] = df["pm25_avg"] - (2 * df["pm25_avg"].shift(1)) + df["pm25_avg"].shift(2)

    if include_training_only_columns:
        df["target_next_hour"] = df["pm25_avg"].shift(-1)

    ordered_columns = [
        "datetime",
        "pm25_avg",
        "coverage_pct",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "wind_direction_10m",
        "surface_pressure",
        "wind_u",
        "wind_v",
        "is_weekend",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "Season",
        "time_of_the_day",
        "rush_hour_weekday",
        "Monday_start",
        "pm25_avg_lag1",
        "pm25_avg_lag6",
        "pm25_avg_lag24",
        "temperature_2m_lag1",
        "pm25_avg_rolling_mean_3h",
        "pm25_avg_rolling_mean_6h",
        "pm25_avg_rolling_mean_12h",
        "temperature_2m_rolling_mean_3h",
        "day_of_year_sin",
        "day_of_year_cos",
        "special_holidays",
        "pm25_avg_diff_1h",
        "ventilation_coeff",
        "wind_magnitude",
        "wind_stagnation_index",
        "is_stagnant_humid",
        "dew_point",
        "pm25_delta",
        "pm25_acceleration",
    ]
    if include_training_only_columns:
        ordered_columns.append("target_next_hour")

    for column in ordered_columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[ordered_columns].copy()


def apply_saved_scaling(dataframe: pd.DataFrame, metadata: PreprocessorMetadata) -> pd.DataFrame:
    df = dataframe.copy()
    for column in metadata.scale_columns:
        mean = metadata.scaler_mean[column]
        scale = metadata.scaler_scale[column] or 1.0
        df[column] = (df[column] - mean) / scale
    return df


def add_multi_horizon_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    for lag in PM_LAGS_TO_ADD:
        column = f"pm25_avg_lag{lag}"
        if column not in enriched.columns:
            enriched[column] = enriched[SOURCE_TARGET_COL].shift(lag)

    for window in PM_ROLL_STD_WINDOWS:
        column = f"pm25_avg_rolling_std_{window}h"
        if column not in enriched.columns:
            enriched[column] = enriched[SOURCE_TARGET_COL].rolling(window=window, min_periods=window).std()

    for window in PM_TREND_WINDOWS:
        column = f"pm25_avg_trend_{window}h"
        if column not in enriched.columns:
            enriched[column] = enriched[SOURCE_TARGET_COL] - enriched[SOURCE_TARGET_COL].shift(window)

    for base_column, lags in WEATHER_LAG_SPECS.items():
        if base_column in enriched.columns:
            for lag in lags:
                column = f"{base_column}_lag{lag}"
                if column not in enriched.columns:
                    enriched[column] = enriched[base_column].shift(lag)

    for base_column, windows in WEATHER_ROLL_MEAN_SPECS.items():
        if base_column in enriched.columns:
            for window in windows:
                column = f"{base_column}_rolling_mean_{window}h"
                if column not in enriched.columns:
                    enriched[column] = enriched[base_column].rolling(window=window, min_periods=window).mean()

    return enriched


def build_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dataset = df.copy()
    target_columns: list[str] = []
    for horizon in HORIZONS:
        column = f"target_tplus{horizon}"
        dataset[column] = dataset[SOURCE_TARGET_COL].shift(-horizon)
        target_columns.append(column)
    return dataset, target_columns


def infer_base_reference_columns() -> list[str]:
    if PROCESSED_REFERENCE_PATH.exists():
        reference = pd.read_csv(PROCESSED_REFERENCE_PATH, nrows=2)
        return reference.columns.tolist()
    return []


def fit_preprocessor_metadata(
    merged_history: pd.DataFrame,
    *,
    special_holiday_dates: Iterable[str] | None = None,
    reconstruction_reference: pd.DataFrame | None = None,
) -> PreprocessorMetadata:
    base = build_base_feature_frame(
        merged_history,
        include_training_only_columns=True,
        special_holiday_dates=special_holiday_dates,
    )
    base = base.dropna().reset_index(drop=True)

    base_reference_columns = infer_base_reference_columns() or base.columns.tolist()
    scale_columns = [
        column
        for column in base_reference_columns
        if column
        not in {
            "pm25_avg",
            "target_next_hour",
            *CATEGORICAL_COLS,
            *BASE_BINARY_COLUMNS,
        }
    ]

    scaler_mean: dict[str, float] = {}
    scaler_scale: dict[str, float] = {}
    scaled = base.copy()
    for column in scale_columns:
        mean = float(base[column].mean())
        scale = float(base[column].std(ddof=0))
        if scale == 0:
            scale = 1.0
        scaler_mean[column] = mean
        scaler_scale[column] = scale
        scaled[column] = (base[column] - mean) / scale

    enriched = add_multi_horizon_features(scaled)
    training_frame, target_columns = build_targets(enriched)
    training_frame = training_frame.dropna().reset_index(drop=True)
    feature_columns = [
        column
        for column in training_frame.columns
        if column not in {"datetime", *target_columns, "target_next_hour"}
    ]

    reconstruction_mae: dict[str, float] | None = None
    if reconstruction_reference is not None and not reconstruction_reference.empty:
        comparable = scaled[base_reference_columns].iloc[: len(reconstruction_reference)].copy()
        reference = reconstruction_reference.iloc[: len(comparable)].copy()
        reconstruction_mae = {}
        for column in comparable.columns:
            if column in CATEGORICAL_COLS:
                reconstruction_mae[column] = float(
                    1.0 - (comparable[column].astype(str) == reference[column].astype(str)).mean()
                )
            else:
                reconstruction_mae[column] = float(
                    np.mean(np.abs(comparable[column].astype(float) - reference[column].astype(float)))
                )

    return PreprocessorMetadata(
        feature_columns=feature_columns,
        categorical_columns=CATEGORICAL_COLS.copy(),
        scale_columns=scale_columns,
        base_reference_columns=base_reference_columns,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        target_columns=target_columns,
        special_holiday_dates=list(special_holiday_dates or infer_special_holiday_dates()),
        excluded_training_columns=TRAINING_ONLY_COLUMNS.copy(),
        reconstruction_mae=reconstruction_mae,
    )


def build_training_frame(merged_history: pd.DataFrame, metadata: PreprocessorMetadata) -> pd.DataFrame:
    base = build_base_feature_frame(
        merged_history,
        include_training_only_columns=True,
        special_holiday_dates=metadata.special_holiday_dates,
    )
    base = base.dropna().reset_index(drop=True)
    scaled = apply_saved_scaling(base, metadata)
    enriched = add_multi_horizon_features(scaled)
    training_frame, _ = build_targets(enriched)
    return training_frame.dropna().reset_index(drop=True)


def prepare_live_feature_row(
    merged_history: pd.DataFrame,
    metadata: PreprocessorMetadata,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = build_base_feature_frame(
        merged_history,
        include_training_only_columns=False,
        special_holiday_dates=metadata.special_holiday_dates,
    )
    scaled = apply_saved_scaling(base, metadata)
    enriched = add_multi_horizon_features(scaled)
    candidate = enriched.dropna(subset=metadata.feature_columns).reset_index(drop=True)
    if candidate.empty:
        raise RuntimeError(
            "Not enough recent history to build a complete inference row. Fetch more hours or check missing API fields."
        )

    latest_row = candidate.iloc[[-1]].copy()
    for column in metadata.categorical_columns:
        latest_row[column] = latest_row[column].astype(str)
    return latest_row[metadata.feature_columns].copy(), enriched


def save_metadata(metadata: PreprocessorMetadata, path: Path) -> None:
    path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")


def load_metadata(path: Path) -> PreprocessorMetadata:
    return PreprocessorMetadata.from_dict(json.loads(path.read_text(encoding="utf-8")))

