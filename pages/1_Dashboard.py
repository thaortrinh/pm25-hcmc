# pages/1_Dashboard.py
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.aqi import aqi_scale_segments
from src.api import get_current_data, get_history_24h
from src.ui import get_pm25_chart_color, get_theme_mode, inject_base_css


def inject_css():
    glossy_expander_css = ""
    if get_theme_mode() == "dark":
        glossy_expander_css = """
        div[data-testid="stExpander"] details {
          background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(15,23,42,0.92) 100%);
        }

        div[data-testid="stExpander"] details summary {
          border-radius: 14px;
          padding: 0.72rem 1rem;
          background: linear-gradient(135deg, #194b9b 0%, #0b2d68 55%, #163c82 100%);
          border: 1px solid rgba(96, 165, 250, 0.28);
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.14), 0 16px 30px rgba(2,6,23,0.32);
          transition: background 0.2s ease, border-color 0.2s ease, transform 0.2s ease;
        }

        div[data-testid="stExpander"] details summary:hover {
          background: linear-gradient(135deg, #05070b 0%, #000000 100%);
          border-color: rgba(15, 23, 42, 0.95);
          transform: translateY(-1px);
        }

        div[data-testid="stExpander"] details[open] summary {
          margin-bottom: 12px;
        }

        div[data-testid="stExpander"] details summary p {
          color: #eff6ff !important;
        }
        """

    inject_base_css(
        """
        .page-header {
          display:flex;
          align-items:flex-start;
          justify-content:space-between;
          gap:24px;
          margin-bottom:14px;
        }

        .page-title {
          font-size:1.55rem;
          font-weight:700;
          color:var(--text-primary);
          margin:0;
        }

        .page-meta {
          font-size:0.72rem;
          color:var(--text-faint);
          white-space:nowrap;
        }

        .station-tag {
          display:inline-flex;
          align-items:center;
          gap:8px;
          background:var(--card-bg);
          border:1px solid var(--border-color);
          border-radius:999px;
          padding:5px 14px;
          font-size:0.78rem;
          color:var(--text-muted);
          margin-top:10px;
        }

        .live-dot {
          width:8px;
          height:8px;
          border-radius:50%;
          background:#22c55e;
        }

        .metric-card,
        .chart-container,
        .aqi-scale-container {
          background:linear-gradient(135deg, var(--card-bg) 0%, var(--card-bg-alt) 100%);
          border:1px solid var(--border-color);
          border-radius:18px;
          box-shadow:0 10px 30px var(--shadow-color);
        }

        .metric-card {
          padding:18px 20px;
          position:relative;
          overflow:hidden;
          height:100%;
        }

        .metric-card::before {
          content:'';
          position:absolute;
          top:0;
          left:0;
          right:0;
          height:3px;
          background:var(--accent, var(--accent-color));
        }

        .metric-label,
        .block-label,
        .section-label {
          font-size:0.72rem;
          font-weight:700;
          letter-spacing:0.12em;
          text-transform:uppercase;
          color:var(--text-faint);
        }

        .metric-value {
          font-family:'Space Mono', monospace;
          font-size:1.85rem;
          font-weight:700;
          line-height:1.1;
          color:var(--text-primary);
          margin-top:10px;
        }

        .metric-unit {
          font-size:0.8rem;
          color:var(--text-muted);
          margin-left:4px;
        }

        .metric-note {
          font-size:0.74rem;
          color:var(--text-faint);
          margin-top:10px;
        }

        .chart-container,
        .aqi-scale-container {
          padding:20px;
          height:100%;
        }

        .section-title {
          font-size:0.72rem;
          font-weight:700;
          letter-spacing:0.15em;
          text-transform:uppercase;
          color:var(--text-faint);
          margin:28px 0 12px 0;
          display:flex;
          align-items:center;
          gap:8px;
        }

        .section-title::after {
          content:'';
          flex:1;
          height:1px;
          background:var(--border-color);
        }
        """
        + glossy_expander_css
    )


def _hex_to_rgba(hex_color: str, alpha: float = 0.1) -> str:
    color = hex_color.lstrip("#")
    red, green, blue = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


def _format_value(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    if decimals == 0:
        return f"{value:.0f}"
    return f"{value:.{decimals}f}"


def render_metric_card(
    label: str,
    value: float | None,
    unit: str,
    *,
    accent: str,
    decimals: int = 1,
    note: str = "",
):
    formatted = _format_value(value, decimals)
    suffix = f'<span class="metric-unit">{unit}</span>' if formatted != "N/A" else ""
    note_html = f'<div class="metric-note">{note}</div>' if note else ""

    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{formatted}{suffix}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_aqi_scale(current_aqi: int | None):
    segments = aqi_scale_segments()
    bars = "".join(
        f'<div style="flex:{upper};background:{bg};display:flex;align-items:center;justify-content:center;color:{text};font-size:0.66rem;font-weight:700;"></div>'
        for upper, bg, text, _ in segments
    )
    labels = "".join(f"<span>{label}</span>" for _, _, _, label in segments)

    marker = ""
    if current_aqi is not None:
        marker_pct = min(max(current_aqi / 500 * 100, 0), 99.2)
        marker = (
            f'<div style="position:absolute;left:{marker_pct:.1f}%;top:-18px;transform:translateX(-50%);'
            f'font-family:\'Space Mono\', monospace;font-size:0.74rem;font-weight:700;'
            f'background:var(--app-bg);padding:2px 7px;border-radius:6px;border:1px solid var(--border-color);'
            f'white-space:nowrap;">▼ {current_aqi}</div>'
        )

    st.markdown(
        f"""
        <div class="aqi-scale-container">
          <div class="block-label">AQI Color Scale</div>
          <div style="position:relative;margin-top:28px;">
            {marker}
            <div style="display:flex;height:26px;border-radius:10px;overflow:hidden;">{bars}</div>
            <div style="display:flex;justify-content:space-between;margin-top:10px;font-size:0.66rem;color:var(--text-faint);">{labels}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_chart(history: list[dict], line_color: str, latest_pm_label: str):
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown('<div class="block-label">PM2.5 · 24H</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-size:0.68rem;color:var(--text-faint);margin-top:6px;margin-bottom:8px;">Today: {datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y")}</div>',
        unsafe_allow_html=True,
    )

    if not history:
        st.info("Không có dữ liệu lịch sử PM2.5 để hiển thị.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    st.caption(
        f"Dữ liệu PM2.5 từ OpenAQ hiện có đến {latest_pm_label}. Nếu vài giờ gần hiện tại chưa có giá trị, đó là do nguồn chưa cập nhật thêm bản ghi mới."
    )

    dataframe = pd.DataFrame(history)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dataframe["time"],
            y=dataframe["pm25"],
            mode="lines",
            line=dict(color=line_color, width=2.8, shape="spline"),
            fill="tozeroy",
            fillcolor=_hex_to_rgba(line_color, 0.1),
            hovertemplate="<b>%{x}</b><br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    figure.add_hline(
        y=15,
        line_dash="dot",
        line_color="rgba(148,163,184,0.6)",
        annotation_text="WHO 15",
        annotation_font_size=10,
        annotation_font_color="#94a3b8",
    )
    figure.add_hline(
        y=35,
        line_dash="dot",
        line_color="rgba(148,163,184,0.6)",
        annotation_text="QCVN 35",
        annotation_font_size=10,
        annotation_font_color="#94a3b8",
        annotation_position="bottom right",
    )
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=0, r=10),
        height=260,
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=dataframe["time"].iloc[::4].tolist(),
            tickfont=dict(color="#94a3b8", size=10, family="Space Mono"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148,163,184,0.14)",
            zeroline=False,
            tickfont=dict(color="#94a3b8", size=10, family="Space Mono"),
            title=dict(text="µg/m³", font=dict(color="#94a3b8", size=10)),
        ),
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="AQI Dashboard", page_icon="🌫️", layout="wide")
    inject_css()

    current = get_current_data()
    chart_color = get_pm25_chart_color()
    current_local_dt = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))

    st.markdown(
        f"""
        <div class="page-header">
          <div>
            <h1 class="page-title">Chỉ số Chất lượng Không khí (AQI) ở Quận 1 - TP.HCM</h1>
            <div class="station-tag"><span class="live-dot"></span>{current["station"]}</div>
          </div>
          <div class="page-meta">Today: {current_local_dt.strftime("%d/%m/%Y")} · Đồng bộ lúc {current["updated_at"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"PM2.5 OpenAQ mới nhất: {current['pm25_updated_at']} · Thời tiết Open-Meteo mới nhất: {current['weather_updated_at']} · Mốc đồng bộ dùng cho mô hình: {current['aligned_model_at']}"
    )

    st.markdown('<div class="section-title">Chỉ số hiện tại</div>', unsafe_allow_html=True)

    with st.expander("Particulate Pollutants", expanded=True):
        cols = st.columns(2)
        with cols[0]:
            render_metric_card(
                "PM2.5",
                current["pm25"],
                "µg/m³",
                accent=chart_color,
                note=f"OpenAQ · {current['pm25_updated_at']}",
            )
        with cols[1]:
            render_metric_card(
                "PM10",
                current["pm10"],
                "µg/m³",
                accent="#9ca3af",
                note=f"Open-Meteo AQ · {current['air_quality_updated_at']}",
            )

    with st.expander("Gaseous Pollutants", expanded=True):
        cols = st.columns(4)
        cards = [
            ("O3", current["o3"], "#38bdf8"),
            ("NO2", current["no2"], "#f97316"),
            ("SO2", current["so2"], "#e11d48"),
            ("CO", current["co"], "#10b981"),
        ]
        for column, (label, value, accent) in zip(cols, cards):
            with column:
                render_metric_card(
                    label,
                    value,
                    "µg/m³",
                    accent=accent,
                    note=f"Open-Meteo AQ · {current['air_quality_updated_at']}",
                )

    with st.expander("Surface Weather Conditions", expanded=True):
        cols = st.columns(6)
        weather_cards = [
            ("Nhiệt độ", current["temp"], "°C", "#f97316", 1),
            ("Độ ẩm", current["humidity"], "%", "#38bdf8", 0),
            ("Tốc độ gió", current["wind"], "m/s", "#a78bfa", 1),
            ("Hướng gió", current["wind_dir"], "°", "#34d399", 0),
            ("Lượng mưa", current["precipitation"], "mm", "#60a5fa", 1),
            ("Áp suất", current["pressure"], "hPa", "#fb923c", 1),
        ]
        for column, (label, value, unit, accent, decimals) in zip(cols, weather_cards):
            with column:
                render_metric_card(
                    label,
                    value,
                    unit,
                    accent=accent,
                    decimals=decimals,
                    note=f"Open-Meteo · {current['weather_updated_at']}",
                )

    with st.expander("Vertical Atmospheric Structure", expanded=True):
        render_metric_card(
            "Lớp biên (BLH)",
            current["boundary_layer_height"],
            "m",
            accent="#c084fc",
            decimals=0,
            note=f"Open-Meteo · {current['weather_updated_at']}",
        )

    st.markdown('<div class="section-title">Lịch sử 24 giờ qua</div>', unsafe_allow_html=True)
    col_chart, col_scale = st.columns([3, 2], gap="large")
    with col_chart:
        render_history_chart(get_history_24h(), chart_color, current["pm25_updated_at"])
    with col_scale:
        render_aqi_scale(current["aqi"])


if __name__ == "__main__":
    main()
