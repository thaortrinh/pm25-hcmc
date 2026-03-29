# pages/1_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.aqi import pm25_to_aqi
from src.api import get_current_data, get_history_24h


# ─── CSS ────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      #MainMenu, footer, header, [data-testid="stSidebar"], [data-testid="stSidebarNav"] { visibility: hidden; display: none; }

      .stApp { background: #f5f6fa; color: #111827; }

      .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fb 100%);
        border: 1px solid #e2e5ea;
        border-radius: 16px;
        padding: 20px 24px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
      }
      .metric-card:hover { border-color: #c8ccd4; box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
      .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: var(--accent, #4fc3f7);
        border-radius: 16px 16px 0 0;
      }
      .metric-label {
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #9ca3af; margin-bottom: 8px;
      }
      .metric-value {
        font-family: 'Space Mono', monospace;
        font-size: 2rem; font-weight: 700;
        color: #111827; line-height: 1.1;
      }
      .metric-unit  { font-size: 0.8rem; color: #6b7280; margin-left: 4px; }
      .metric-badge {
        display: inline-block; margin-top: 8px;
        padding: 3px 10px; border-radius: 999px;
        font-size: 0.75rem; font-weight: 700;
      }

      .section-title {
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.15em; text-transform: uppercase;
        color: #9ca3af; margin: 28px 0 12px 0;
        display: flex; align-items: center; gap: 8px;
      }
      .section-title::after { content: ''; flex: 1; height: 1px; background: #e5e7eb; }

      .forecast-card {
        background: #ffffff; border: 1px solid #e2e5ea;
        border-radius: 12px; padding: 14px 10px; text-align: center;
        transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s;
      }
      .forecast-card:hover {
        transform: translateY(-2px); border-color: #c8ccd4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      }
      .forecast-hour  { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #9ca3af; margin-bottom: 6px; }
      .forecast-pm25  { font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #111827; }
      .forecast-unit  { font-size: 0.6rem; color: #9ca3af; }
      .forecast-badge { display: inline-block; margin-top: 6px; padding: 2px 8px; border-radius: 999px; font-size: 0.65rem; font-weight: 700; }

      .aqi-scale-container { background: #ffffff; border: 1px solid #e2e5ea; border-radius: 16px; padding: 20px; height: 100%; }
      .scale-label-row     { display: flex; justify-content: space-between; font-size: 0.6rem; color: #9ca3af; margin-top: 4px; }

      .chart-container { background: #ffffff; border: 1px solid #e2e5ea; border-radius: 16px; padding: 20px; }

      .station-tag {
        display: inline-flex; align-items: center; gap: 6px;
        background: #ffffff; border: 1px solid #e2e5ea;
        border-radius: 999px; padding: 4px 14px;
        font-size: 0.75rem; color: #6b7280; margin-bottom: 8px;
      }
      .live-dot { width: 7px; height: 7px; background: #22c55e; border-radius: 50%; animation: pulse 2s infinite; }
      @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
    </style>
    """, unsafe_allow_html=True)


# ─── Render helpers ──────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 0.08) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_metric_card(label: str, value: str, unit: str,
                       accent: str, badge_html: str = ""):
    st.markdown(f"""
    <div class="metric-card" style="--accent:{accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}<span class="metric-unit">{unit}</span></div>
      {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_aqi_badge(label: str, bg: str, text: str,
                     css_class: str = "metric-badge") -> str:
    return (f'<div class="{css_class}" '
            f'style="background:{bg};color:{text}">{label}</div>')


def render_aqi_scale(current_aqi: int):
    segments = [
        (50,  "#00e400", "Good"),
        (100, "#ffff00", "Mod."),
        (150, "#ff7e00", "USG"),
        (200, "#ff0000", "Unhl."),
        (300, "#8f3f97", "Very"),
        (500, "#7e0023", "Haz."),
    ]
    bars   = "".join(f'<div style="flex:{u};background:{bg}"></div>' for u, bg, _ in segments)
    labels = "".join(f'<span>{lbl}</span>' for _, _, lbl in segments)
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'<div style="width:10px;height:10px;border-radius:3px;background:{bg};flex-shrink:0;"></div>'
        f'<span style="font-size:0.72rem;color:#6b7280;">{lbl}</span></div>'
        for _, bg, lbl in segments
    )
    marker_pct = min(current_aqi / 500 * 100, 99)

    st.markdown(f"""
    <div class="aqi-scale-container">
      <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
                  text-transform:uppercase;color:#6b7280;">AQI Color Scale</div>
      <div style="position:relative;margin-top:24px;">
        <div style="position:absolute;left:{marker_pct:.1f}%;top:-20px;
                    transform:translateX(-50%);font-family:'Space Mono',monospace;
                    font-size:0.7rem;font-weight:700;color:#111827;white-space:nowrap;
                    background:#f5f6fa;padding:1px 6px;border-radius:4px;
                    border:1px solid #c8ccd4;">▼ {current_aqi}</div>
        <div style="display:flex;height:24px;border-radius:8px;overflow:hidden;margin:20px 0 8px 0;">{bars}</div>
        <div class="scale-label-row">{labels}</div>
      </div>
      <div style="margin-top:20px;">{legend}</div>
    </div>
    """, unsafe_allow_html=True)


def render_history_chart(history: list):
    if not history:
        st.info("Không có dữ liệu lịch sử 24h.")
        return

    df = pd.DataFrame(history)

    aqi_info = pm25_to_aqi(df["pm25"].iloc[-1])
    color    = aqi_info.bg_color

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["pm25"],
        mode="lines",
        line=dict(color=color, width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.08),
        hovertemplate="<b>%{x}</b><br>PM2.5: %{y} μg/m³<extra></extra>",
    ))
    fig.add_hline(y=15, line_dash="dot", line_color="#d1d5db",
                  annotation_text="WHO 15", annotation_font_size=10,
                  annotation_font_color="#6b7280")
    fig.add_hline(y=35, line_dash="dot", line_color="#d1d5db",
                  annotation_text="QCVN 35", annotation_font_size=10,
                  annotation_font_color="#6b7280",
                  annotation_position="bottom right")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=10, b=10, l=0, r=10), height=220,
        xaxis=dict(showgrid=False, zeroline=False,
                   tickfont=dict(color="#9ca3af", size=10, family="Space Mono"),
                   tickmode="array", tickvals=df["time"].iloc[::4].tolist()),
        yaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=False,
                   tickfont=dict(color="#9ca3af", size=10, family="Space Mono"),
                   title=dict(text="μg/m³", font=dict(color="#9ca3af", size=10))),
        showlegend=False, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_forecast_cards(forecasts: list):
    cols = st.columns(6)
    for i, (col, fc) in enumerate(zip(cols, forecasts)):
        info = pm25_to_aqi(fc["pm25"])
        with col:
            st.markdown(f"""
            <div class="forecast-card">
              <div class="forecast-hour">t+{i+1}h<br>{fc["time"]}</div>
              <div class="forecast-pm25">{fc["pm25"]}<span class="forecast-unit"> μg/m³</span></div>
              <div class="forecast-badge" style="background:{info.bg_color};color:{info.text_color}">
                {info.label}
              </div>
            </div>
            """, unsafe_allow_html=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="PM2.5 Dashboard – CMT8", page_icon="🌫️", layout="wide")
    inject_css()

    current  = get_current_data()
    aqi_info = pm25_to_aqi(current["pm25"])

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
      <h1 style="font-size:1.5rem;font-weight:700;color:#111827;margin:0;">Chỉ số Chất lượng Không khí (AQI) ở Quận 1 - TP.HCM</h1>
      <span style="font-size:0.72rem;color:#9ca3af;">Today: {datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y")} · Cập nhật lúc {current['updated_at']}</span>
    </div>
    <div class="station-tag">
      <div class="live-dot"></div>{current['station']}
    </div>
    """, unsafe_allow_html=True)

    # Row 1 – Metrics
    st.markdown('<div class="section-title">Chỉ số hiện tại</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        badge = render_aqi_badge(f"{aqi_info.label}  AQI {aqi_info.aqi}",
                                 aqi_info.bg_color, aqi_info.text_color)
        render_metric_card("PM2.5", str(current["pm25"]), " μg/m³",
                           accent=aqi_info.bg_color, badge_html=badge)
    with c2:
        render_metric_card("Nhiệt độ", str(current["temp"]), " °C", accent="#f97316")
    with c3:
        render_metric_card("Độ ẩm", str(current["humidity"]), " %", accent="#38bdf8")
    with c4:
        render_metric_card("Tốc độ gió", str(current["wind"]), " m/s", accent="#a78bfa")

    # Row 2
    # st.markdown('<div class="section-title">Điều kiện khí tượng</div>', unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        render_metric_card("Hướng gió", str(current["wind_dir"]), " °", accent="#34d399")
    with c6:
        render_metric_card("Lượng mưa", str(current["precipitation"]), " mm", accent="#60a5fa")
    with c7:
        render_metric_card("Áp suất", str(current["pressure"]), " hPa", accent="#fb923c")
    with c8:
        render_metric_card("Lớp biên (BLH)", str(round(current["boundary_layer_height"])), " m", accent="#c084fc")

    # Row 3 – Chart + Scale
    st.markdown('<div class="section-title">Lịch sử 24 giờ qua</div>', unsafe_allow_html=True)
    col_chart, col_scale = st.columns([3, 2])
    with col_chart:
        today_str = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y")
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;'
                    f'text-transform:uppercase;color:#9ca3af;margin-bottom:4px;">PM2.5 · 24h</div>'
                    f'<div style="font-size:0.68rem;color:#b0b5bf;margin-bottom:8px;">Today: {today_str}</div>',
                    unsafe_allow_html=True)
        render_history_chart(get_history_24h())
        st.markdown('</div>', unsafe_allow_html=True)
    with col_scale:
        render_aqi_scale(aqi_info.aqi)

    # # Row 3 – Forecast
    # st.markdown('<div class="section-title">Dự báo 6 giờ tới</div>', unsafe_allow_html=True)
    # render_forecast_cards(get_forecast_6h())
    # st.markdown('<div style="margin-top:12px;font-size:0.7rem;color:#9ca3af;text-align:right;">'
    #             '* Dự báo được tạo tự động bởi model ML mỗi khi tải trang. Kết quả mang tính tham khảo.'
    #             '</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()