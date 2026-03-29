from dataclasses import dataclass

# ── Breakpoints US EPA (PM2.5, 24h average) ──────────────────────────────────
# (C_low, C_high, I_low, I_high, label, hex_color, text_color)
_BREAKPOINTS = [
    (0.0,   12.0,   0,   50,  "Good",            "#00e400", "#000000"),
    (12.1,  35.4,  51,  100,  "Moderate",         "#ffff00", "#000000"),
    (35.5,  55.4, 101,  150,  "Unhealthy for SG", "#ff7e00", "#000000"),
    (55.5, 150.4, 151,  200,  "Unhealthy",        "#ff0000", "#ffffff"),
    (150.5, 250.4, 201, 300,  "Very Unhealthy",   "#8f3f97", "#ffffff"),
    (250.5, 500.4, 301, 500,  "Hazardous",        "#7e0023", "#ffffff"),
]


@dataclass
class AQIResult:
    pm25: float
    aqi: int
    label: str
    bg_color: str       # hex background
    text_color: str     # hex text (đảm bảo contrast)


def pm25_to_aqi(pm25: float) -> AQIResult:
    """
    Chuyển PM2.5 (μg/m³) sang AQI và thông tin hiển thị.
    Clamp về 0 nếu âm, clamp về 500 nếu vượt ngưỡng.
    """
    pm25 = max(0.0, pm25)

    for c_low, c_high, i_low, i_high, label, bg, text in _BREAKPOINTS:
        if pm25 <= c_high:
            # Linear interpolation
            aqi = round(
                (i_high - i_low) / (c_high - c_low) * (pm25 - c_low) + i_low
            )
            return AQIResult(pm25=pm25, aqi=aqi, label=label,
                             bg_color=bg, text_color=text)

    # Vượt 500 → Hazardous max
    *_, label, bg, text = _BREAKPOINTS[-1]
    return AQIResult(pm25=pm25, aqi=500, label=label,
                     bg_color=bg, text_color=text)


def aqi_badge_html(result: AQIResult, size: str = "normal") -> str:
    """
    Trả về HTML string cho badge AQI màu.
    size = "normal" | "large"
    """
    font_size = "1.1rem" if size == "large" else "0.85rem"
    padding   = "6px 14px" if size == "large" else "3px 10px"
    return (
        f'<span style="'
        f'background:{result.bg_color};'
        f'color:{result.text_color};'
        f'border-radius:999px;'
        f'padding:{padding};'
        f'font-size:{font_size};'
        f'font-weight:600;'
        f'">{result.label} (AQI {result.aqi})</span>'
    )


def aqi_scale_html(current_aqi: int) -> str:
    """
    Render thanh AQI color scale dạng HTML,
    với marker "▼ You are here" tại vị trí current_aqi.
    """
    segments = [
        (50,  "#00e400", "#000", "Good"),
        (100, "#ffff00", "#000", "Moderate"),
        (150, "#ff7e00", "#000", "USG"),
        (200, "#ff0000", "#fff", "Unhealthy"),
        (300, "#8f3f97", "#fff", "Very"),
        (500, "#7e0023", "#fff", "Hazardous"),
    ]

    total = 500
    bars = ""
    marker_left = min(max(current_aqi / total * 100, 0), 100)

    for upper, bg, fg, lbl in segments:
        width = (upper / total) * 100  # width % so với 500 range
        bars += (
            f'<div style="flex:{upper};background:{bg};'
            f'color:{fg};font-size:0.65rem;'
            f'display:flex;align-items:center;justify-content:center;">'
            f'{lbl}</div>'
        )

    html = f"""
    <div style="position:relative;margin-top:8px;">
      <div style="display:flex;height:28px;border-radius:6px;overflow:hidden;">
        {bars}
      </div>
      <div style="
        position:absolute;
        left:{marker_left:.1f}%;
        top:-18px;
        transform:translateX(-50%);
        font-size:0.75rem;
        white-space:nowrap;
        font-weight:700;
      ">▼ {current_aqi}</div>
    </div>
    """
    return html


# ── Helper nhanh ─────────────────────────────────────────────────────────────

def get_color(pm25: float) -> str:
    """Trả về hex color ứng với mức PM2.5."""
    return pm25_to_aqi(pm25).bg_color


def get_label(pm25: float) -> str:
    """Trả về label text ứng với mức PM2.5."""
    return pm25_to_aqi(pm25).label