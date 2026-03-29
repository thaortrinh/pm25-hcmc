from __future__ import annotations

from dataclasses import dataclass


AQI_LEVELS = [
    (0, 50, "Tốt", "#00e400", "#000000"),
    (51, 100, "Trung bình", "#ffff00", "#000000"),
    (101, 150, "Kém", "#ff7e00", "#000000"),
    (151, 200, "Xấu", "#ff0000", "#ffffff"),
    (201, 300, "Rất xấu", "#8f3f97", "#ffffff"),
    (301, 500, "Nguy hại", "#7e0023", "#ffffff"),
]


AQI_BREAKPOINTS = {
    "o3_1h": [0, 160, 200, 300, 400, 800, 1000, 1200],
    "o3_8h": [0, 100, 120, 170, 210, 400],
    "co": [0, 10000, 30000, 45000, 60000, 90000, 120000, 150000],
    "so2": [0, 125, 350, 550, 800, 1600, 2100, 2630],
    "no2": [0, 100, 200, 700, 1200, 2350, 3100, 3850],
    "pm10": [0, 50, 150, 250, 350, 420, 500, 600],
    "pm25": [0, 25, 50, 80, 150, 250, 350, 500],
}


GAS_MOLECULAR_WEIGHTS = {
    "o3": 48.0,
    "no2": 46.0,
    "so2": 64.066,
    "co": 28.01,
}


@dataclass(frozen=True)
class AQIResult:
    concentration: float
    aqi: int
    label: str
    bg_color: str
    text_color: str


@dataclass(frozen=True)
class PollutantAQI:
    pollutant: str
    concentration: float
    aqi: int
    label: str
    bg_color: str
    text_color: str
    basis: str


@dataclass(frozen=True)
class VN_AQIResult:
    aqi: int | None
    label: str
    bg_color: str
    text_color: str
    primary_pollutant: str | None
    sub_indices: dict[str, PollutantAQI]


def _category_for_aqi(aqi: int) -> tuple[str, str, str]:
    for lower, upper, label, bg, text in AQI_LEVELS:
        if lower <= aqi <= upper:
            return label, bg, text
    label, bg, text = AQI_LEVELS[-1][2:]
    return label, bg, text


def _interpolate_index(value: float, breakpoints: list[float]) -> int:
    if value <= breakpoints[0]:
        return 0

    for idx in range(len(breakpoints) - 1):
        bp_low = breakpoints[idx]
        bp_high = breakpoints[idx + 1]
        i_low = 0 if idx == 0 else AQI_LEVELS[idx - 1][1]
        i_high = AQI_LEVELS[idx][1]

        if value <= bp_high:
            aqi = round((i_high - i_low) / (bp_high - bp_low) * (value - bp_low) + i_low)
            return max(0, min(aqi, 500))

    return 500


def _make_result(value: float, aqi: int) -> AQIResult:
    label, bg, text = _category_for_aqi(aqi)
    return AQIResult(
        concentration=round(value, 1),
        aqi=aqi,
        label=label,
        bg_color=bg,
        text_color=text,
    )


def compute_nowcast(hourly_values: list[float | None]) -> float | None:
    """
    Vietnam AQI requires PM2.5 and PM10 hourly AQI to use a 12-hour Nowcast.
    The weighting follows the standard Cmin/Cmax approach, bounded to 0.5.
    """
    if len(hourly_values) < 3:
        return None

    if sum(value is not None for value in hourly_values[:3]) < 2:
        return None

    valid = [value for value in hourly_values[:12] if value is not None]
    if not valid:
        return None

    c_min = min(valid)
    c_max = max(valid)
    if c_max <= 0:
        return 0.0

    weight_star = c_min / c_max
    weight = max(weight_star, 0.5)

    numerator = 0.0
    denominator = 0.0
    for idx, value in enumerate(hourly_values[:12]):
        if value is None:
            continue
        factor = weight ** idx
        numerator += value * factor
        denominator += factor

    if denominator == 0:
        return None
    return numerator / denominator


def pollutant_to_aqi(
    pollutant: str,
    concentration: float,
    *,
    basis: str | None = None,
) -> PollutantAQI:
    metric = basis or pollutant
    aqi = _interpolate_index(concentration, AQI_BREAKPOINTS[metric])
    label, bg, text = _category_for_aqi(aqi)
    return PollutantAQI(
        pollutant=pollutant,
        concentration=round(concentration, 1),
        aqi=aqi,
        label=label,
        bg_color=bg,
        text_color=text,
        basis=metric,
    )


def calculate_vn_aqi_hourly(pollutants: dict[str, dict]) -> VN_AQIResult:
    sub_indices: dict[str, PollutantAQI] = {}

    pm_available = False

    for pollutant in ("pm25", "pm10"):
        payload = pollutants.get(pollutant, {})
        hourly_values = payload.get("hourly_12h", [])
        nowcast = compute_nowcast(hourly_values)
        if nowcast is None:
            continue
        sub_indices[pollutant] = pollutant_to_aqi(pollutant, nowcast)
        pm_available = True

    o3_payload = pollutants.get("o3", {})
    o3_value = o3_payload.get("value_1h")
    if o3_value is not None:
        sub_indices["o3"] = pollutant_to_aqi("o3", o3_value, basis="o3_1h")

    for pollutant in ("no2", "so2", "co"):
        payload = pollutants.get(pollutant, {})
        value = payload.get("value_1h")
        if value is None:
            continue
        sub_indices[pollutant] = pollutant_to_aqi(pollutant, value)

    if not pm_available or not sub_indices:
        return VN_AQIResult(
            aqi=None,
            label="Không có dữ liệu",
            bg_color="#6b7280",
            text_color="#ffffff",
            primary_pollutant=None,
            sub_indices=sub_indices,
        )

    primary = max(sub_indices.values(), key=lambda item: item.aqi)
    return VN_AQIResult(
        aqi=primary.aqi,
        label=primary.label,
        bg_color=primary.bg_color,
        text_color=primary.text_color,
        primary_pollutant=primary.pollutant,
        sub_indices=sub_indices,
    )


def pm25_to_aqi(pm25: float) -> AQIResult:
    pm25 = max(0.0, pm25)
    aqi = _interpolate_index(pm25, AQI_BREAKPOINTS["pm25"])
    return _make_result(pm25, aqi)


def aqi_scale_segments() -> list[tuple[int, str, str, str]]:
    return [(upper, bg, text, label) for _, upper, label, bg, text in AQI_LEVELS]
