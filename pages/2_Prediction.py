from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.api import get_current_data
from src.aqi import pm25_to_aqi
from src.inference.feature_builder import merge_pm_and_weather
from src.inference.predict import ForecastBundle, fetch_recent_inputs, run_forecast, summarize_alignment
from src.ui import inject_base_css


VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


def inject_css():
    inject_base_css(
        """
        .page-hero {
          background: linear-gradient(135deg, rgba(17,24,39,0.06), rgba(59,130,246,0.10));
          border: 1px solid var(--border-color);
          border-radius: 22px;
          padding: 22px 24px;
          margin-bottom: 20px;
          box-shadow: 0 12px 30px var(--shadow-color);
        }

        .page-hero h1 {
          font-size: 1.6rem;
          font-weight: 700;
          margin: 0;
        }

        .page-hero p {
          margin: 8px 0 0 0;
          color: var(--text-muted);
          max-width: 760px;
        }

        .block-title {
          font-size: 0.75rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--text-faint);
          margin-bottom: 12px;
        }

        .input-panel,
        .result-card,
        .result-card-large,
        .suggestion-box,
        .status-chip {
          background: linear-gradient(135deg, var(--card-bg) 0%, var(--card-bg-alt) 100%);
          border: 1px solid var(--border-color);
          box-shadow: 0 10px 30px var(--shadow-color);
        }

        .input-panel {
          border-radius: 18px;
          padding: 18px 18px 8px 18px;
          margin-bottom: 18px;
        }

        .result-card {
          border-radius: 14px;
          padding: 18px 12px;
          text-align: center;
        }

        .result-card-large {
          border-radius: 18px;
          padding: 40px 24px;
          text-align: center;
        }

        .hour,
        .unit {
          color: var(--text-faint);
        }

        .val {
          font-family: 'Space Mono', monospace;
          font-weight: 800;
          color: var(--text-primary);
        }

        .result-card .val {
          font-size: 1.55rem;
        }

        .result-card-large .val {
          font-size: 3rem;
        }

        .aqi-badge,
        .aqi-badge-large {
          display: inline-block;
          border-radius: 999px;
          font-weight: 700;
        }

        .aqi-badge {
          padding: 4px 10px;
          font-size: 0.72rem;
        }

        .aqi-badge-large {
          padding: 6px 18px;
          font-size: 0.95rem;
          margin-top: 12px;
        }

        .suggestion-box {
          border-left: 4px solid var(--accent-color);
          border-radius: 10px;
          padding: 14px 18px;
          margin-top: 16px;
          color: var(--text-primary);
        }

        .status-chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          border-radius: 999px;
          padding: 8px 12px;
          font-size: 0.78rem;
          color: var(--text-muted);
          margin-top: 14px;
        }

        .status-dot {
          width: 8px;
          height: 8px;
          border-radius: 999px;
          background: #ef4444;
        }
        """
    )


def render_aqi_badge(label: str, bg: str, text: str, css_class: str = "aqi-badge") -> str:
    return f'<span class="{css_class}" style="background:{bg};color:{text}">{label}</span>'


def render_result_cards(predictions: list[float | None], future_hours: list[str]):
        display_predictions = predictions[:6] + [None] * max(0, 6 - len(predictions))
        display_hours = future_hours[:6] + ["--:--"] * max(0, 6 - len(future_hours))

        cols = st.columns(6)
        for index, (column, pm25_value, hour_str) in enumerate(zip(cols, display_predictions, display_hours), start=1):
                with column:
                        if pm25_value is None:
                                st.markdown(
                                        f"""
                                        <div class="result-card">
                                            <div class="hour" style="font-family:'Space Mono', monospace;font-size:0.75rem;margin-bottom:4px;">
                                                t+{index}h · {hour_str}
                                            </div>
                                            <div class="val" style="color:var(--text-faint);">--</div>
                                            <div class="unit">µg/m³</div>
                                            <div style="margin-top:8px;">
                                                <span class="aqi-badge" style="background:var(--border-color);color:var(--text-muted);">Chưa có dự báo</span>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                )
                                continue

                        info = pm25_to_aqi(pm25_value)
                        badge = render_aqi_badge(info.label, info.bg_color, info.text_color)
                        st.markdown(
                                f"""
                                <div class="result-card">
                                    <div class="hour" style="font-family:'Space Mono', monospace;font-size:0.75rem;margin-bottom:4px;">
                                        t+{index}h · {hour_str}
                                    </div>
                                    <div class="val" style="color:{info.bg_color};">{pm25_value:.1f}</div>
                                    <div class="unit">µg/m³</div>
                                    <div style="margin-top:8px;">{badge}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                        )


def render_suggestion(predictions: list[float], hours: list[str]):
    max_val = max(predictions)
    min_val = min(predictions)
    max_hour = hours[predictions.index(max_val)]
    min_hour = hours[predictions.index(min_val)]
    trend = predictions[-1] - predictions[0]

    if max_val > 150:
        level = f"PM2.5 có thể đạt <strong>{max_val:.1f} µg/m³</strong> lúc {max_hour}, mức ô nhiễm cao."
    elif max_val > 80:
        level = f"PM2.5 có thể tăng lên <strong>{max_val:.1f} µg/m³</strong> lúc {max_hour}, nên theo dõi kỹ."
    elif max_val > 35:
        level = f"Không khí ở mức trung bình, đỉnh dự báo là <strong>{max_val:.1f} µg/m³</strong> lúc {max_hour}."
    else:
        level = f"Không khí tương đối tốt, PM2.5 giữ dưới <strong>{max_val:.1f} µg/m³</strong> trong {len(predictions)} giờ tới."

    if trend > 10:
        trend_msg = f" Xu hướng đang <strong>tăng dần</strong>; nếu cần ra ngoài, nên ưu tiên trước {hours[0]}."
    elif trend < -10:
        trend_msg = f" Dự báo <strong>cải thiện</strong> rõ sau {min_hour}."
    else:
        trend_msg = " Mức ô nhiễm nhìn chung <strong>ổn định</strong> trong khung giờ này."

    st.markdown(f'<div class="suggestion-box">{level}{trend_msg}</div>', unsafe_allow_html=True)


def render_live_chart(bundle: ForecastBundle, horizon: int):
    history = bundle.merged_history[["datetime", "pm25_avg"]].dropna().copy()
    history["datetime"] = pd.to_datetime(history["datetime"])
    history["datetime_vn"] = history["datetime"].dt.tz_localize("UTC").dt.tz_convert(VN_TZ).dt.tz_localize(None)

    latest_utc = history["datetime"].max().to_pydatetime()
    latest_vn = latest_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(VN_TZ).replace(tzinfo=None)
    forecast_df = pd.DataFrame(
        {
            "datetime_vn": bundle.prediction_times_vn[:horizon],
            "pm25_avg": bundle.predictions[:horizon],
        }
    )

    chart_end = forecast_df["datetime_vn"].max()
    chart_start = chart_end - timedelta(hours=12)
    history_window = history[history["datetime_vn"] >= chart_start].copy()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=history_window["datetime_vn"],
            y=history_window["pm25_avg"],
            mode="lines+markers",
            name="PM2.5 lịch sử",
            line=dict(color="#2563eb", width=3),
            marker=dict(color="#2563eb", size=7),
            hovertemplate="%{x|%H:%M %d/%m}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[latest_vn] + forecast_df["datetime_vn"].tolist(),
            y=[history_window.iloc[-1]["pm25_avg"]] + forecast_df["pm25_avg"].tolist(),
            mode="lines+markers",
            name="PM2.5 dự báo",
            line=dict(color="#dc2626", width=3),
            marker=dict(color="#dc2626", size=8),
            hovertemplate="%{x|%H:%M %d/%m}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        )
    )
    figure.add_vline(
        x=latest_vn,
        line_width=2,
        line_dash="dash",
        line_color="#94a3b8",
    )
    figure.add_vrect(
        x0=latest_vn,
        x1=chart_end,
        fillcolor="rgba(220, 38, 38, 0.08)",
        line_width=0,
    )
    figure.add_annotation(
        x=latest_vn,
        y=1.05,
        xref="x",
        yref="paper",
        text="Hiện tại",
        showarrow=False,
        font=dict(size=11, color="#94a3b8"),
    )
    figure.add_annotation(
        x=forecast_df["datetime_vn"].iloc[-1],
        y=1.05,
        xref="x",
        yref="paper",
        text="Vùng dự báo",
        xanchor="right",
        showarrow=False,
        font=dict(size=11, color="#dc2626"),
    )
    figure.update_layout(
        height=380,
        margin=dict(l=12, r=12, t=20, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Thời gian", showgrid=False, range=[chart_start, chart_end]),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False),
        hovermode="x unified",
    )
    st.plotly_chart(figure, use_container_width=True)


def _store_autofill_payload():
    pm_history, weather_history = fetch_recent_inputs(lookback_hours=72)
    aligned_history = merge_pm_and_weather(pm_history, weather_history)
    current_snapshot = get_current_data()
    override_defaults = {
        "pm25_avg": float(current_snapshot["pm25"] or 0.0),
        "coverage_pct": 100.0,
        "temperature_2m": float(current_snapshot["temp"]),
        "relative_humidity_2m": float(current_snapshot["humidity"]),
        "precipitation": float(current_snapshot["precipitation"]),
        "wind_speed_10m": float(current_snapshot["wind"]),
        "wind_direction_10m": float(current_snapshot["wind_dir"]),
        "surface_pressure": float(current_snapshot["pressure"]),
        "boundary_layer_height": float(current_snapshot["boundary_layer_height"]),
    }
    st.session_state["prediction_payload"] = {
        "pm_history": pm_history.reset_index(drop=True),
        "weather_history": weather_history.reset_index(drop=True),
        "aligned_history": aligned_history.reset_index(drop=True),
        "current_snapshot": current_snapshot,
        "override_defaults": override_defaults,
    }
    st.session_state["latest_forecast_bundle"] = None


def render_input_summary():
    payload = st.session_state.get("prediction_payload")
    if not payload:
        st.info("Nhấn “Tự động điền từ API” để nạp dữ liệu PM2.5 và thời tiết mới nhất.")
        return

    pm_history = payload["pm_history"].copy()
    weather_history = payload["weather_history"].copy()
    aligned_history = payload["aligned_history"].copy()
    current_snapshot = payload["current_snapshot"]
    alignment = summarize_alignment(pm_history, weather_history)
    latest_aligned = aligned_history.iloc[-1]
    latest_label = pd.to_datetime(latest_aligned["datetime"]).tz_localize("UTC").tz_convert(VN_TZ).strftime("%H:%M %d/%m/%Y")
    st.caption(
        f"PM2.5 OpenAQ mới nhất: {current_snapshot['pm25_updated_at']} · Thời tiết Open-Meteo mới nhất: {current_snapshot['weather_updated_at']} · Mốc đồng bộ dùng cho mô hình: {latest_label}"
    )
    if alignment["latest_pm"] and alignment["latest_weather"] and alignment["latest_common"]:
        common_local = pd.Timestamp(alignment["latest_common"]).tz_localize("UTC").tz_convert(VN_TZ).strftime("%H:%M")
        st.caption(
            f"Nguồn hiện tại hiển thị theo thời gian thực của từng API. Nếu bạn không chỉnh tay gì, lần dự báo sẽ dùng lịch sử đồng bộ đến {common_local}."
        )

    metric_cols = st.columns(4)
    metric_items = [
        ("PM2.5 hiện tại", f"{float(current_snapshot['pm25'] or 0.0):.1f} µg/m³"),
        ("Nhiệt độ", f"{float(current_snapshot['temp']):.1f} °C"),
        ("Độ ẩm", f"{float(current_snapshot['humidity']):.0f} %"),
        ("Tốc độ gió", f"{float(current_snapshot['wind']):.1f} m/s"),
    ]
    for column, (label, value) in zip(metric_cols, metric_items):
        with column:
            st.metric(label, value)

    st.caption("12 giờ lịch sử đồng bộ dưới đây là phần nền mà mô hình dùng để dựng đặc trưng trước khi suy luận.")
    preview = (
        aligned_history.tail(12)
        .assign(
            datetime=lambda df: pd.to_datetime(df["datetime"]).dt.tz_localize("UTC").dt.tz_convert(VN_TZ).dt.strftime("%H:%M")
        )
        .rename(columns={"datetime": "Giờ", "pm25_avg": "PM2.5", "coverage_pct": "Độ phủ (%)"})
    )
    st.dataframe(preview, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.number_input(
            "PM2.5 hiện tại (µg/m³)",
            min_value=0.0,
            max_value=500.0,
            step=0.1,
            key="override_pm25_avg",
            value=float(current_snapshot["pm25"] or 0.0),
        )
        st.number_input(
            "Độ phủ cảm biến (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="override_coverage_pct",
            value=float(payload["override_defaults"]["coverage_pct"]),
        )
        st.number_input(
            "Nhiệt độ (°C)",
            min_value=-10.0,
            max_value=50.0,
            step=0.1,
            key="override_temperature_2m",
            value=float(current_snapshot["temp"]),
        )
    with col2:
        st.number_input(
            "Độ ẩm (%)",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key="override_relative_humidity_2m",
            value=float(current_snapshot["humidity"]),
        )
        st.number_input(
            "Lượng mưa (mm)",
            min_value=0.0,
            max_value=200.0,
            step=0.1,
            key="override_precipitation",
            value=float(current_snapshot["precipitation"]),
        )
        st.number_input(
            "Tốc độ gió (m/s)",
            min_value=0.0,
            max_value=50.0,
            step=0.1,
            key="override_wind_speed_10m",
            value=float(current_snapshot["wind"]),
        )
    with col3:
        st.number_input(
            "Hướng gió (°)",
            min_value=0.0,
            max_value=360.0,
            step=1.0,
            key="override_wind_direction_10m",
            value=float(current_snapshot["wind_dir"]),
        )
        st.number_input(
            "Áp suất bề mặt (hPa)",
            min_value=900.0,
            max_value=1100.0,
            step=0.1,
            key="override_surface_pressure",
            value=float(current_snapshot["pressure"]),
        )
        st.number_input(
            "Lớp biên khí quyển (m)",
            min_value=0.0,
            max_value=5000.0,
            step=10.0,
            key="override_boundary_layer_height",
            value=float(current_snapshot["boundary_layer_height"]),
        )


def collect_overrides() -> dict[str, float]:
    defaults = st.session_state["prediction_payload"]["override_defaults"]
    current_values = {
        "pm25_avg": float(st.session_state["override_pm25_avg"]),
        "coverage_pct": float(st.session_state["override_coverage_pct"]),
        "temperature_2m": float(st.session_state["override_temperature_2m"]),
        "relative_humidity_2m": float(st.session_state["override_relative_humidity_2m"]),
        "precipitation": float(st.session_state["override_precipitation"]),
        "wind_speed_10m": float(st.session_state["override_wind_speed_10m"]),
        "wind_direction_10m": float(st.session_state["override_wind_direction_10m"]),
        "surface_pressure": float(st.session_state["override_surface_pressure"]),
        "boundary_layer_height": float(st.session_state["override_boundary_layer_height"]),
    }
    changed: dict[str, float] = {}
    for key, value in current_values.items():
        if abs(value - float(defaults[key])) > 1e-9:
            changed[key] = value
    return changed


def main():
    st.set_page_config(page_title="Dự báo PM2.5", page_icon="🔬", layout="wide")
    inject_css()

    st.markdown(
        """
        <div class="page-hero">
          <h1>Dự báo PM2.5 trong 1 đến 6 giờ tới</h1>
          <p>
            Hệ thống tự động lấy lịch sử PM2.5 từ OpenAQ, thời tiết từ Open-Meteo, tái tạo đầy đủ đặc trưng của mô hình CatBoost
            và hiển thị phần dự báo tách biệt rõ với quan trắc hiện tại.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Các nhãn giao diện được giữ bằng tiếng Việt, còn pipeline suy luận bám theo mô hình CatBoost đa đầu ra.")

    col_left, col_right = st.columns([1.1, 1], gap="large")
    with col_left:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown('<div class="block-title">Lịch sử PM2.5</div>', unsafe_allow_html=True)
        if st.button("Tự động điền từ API", type="primary", use_container_width=True):
            with st.spinner("Đang lấy dữ liệu PM2.5 và thời tiết mới nhất..."):
                try:
                    _store_autofill_payload()
                    st.success("Đã nạp dữ liệu thật từ API. Bạn có thể chỉnh tay ở các ô bên dưới nếu muốn.")
                except Exception as exc:
                    st.error(f"Không thể lấy dữ liệu tự động: {exc}")
        render_input_summary()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="input-panel">', unsafe_allow_html=True)
        st.markdown('<div class="block-title">Tùy chọn dự báo</div>', unsafe_allow_html=True)
        st.markdown("Mô hình luôn sinh đủ 6 giá trị từ `t+1` đến `t+6` và hiển thị đồng thời trong 6 khung kết quả.")
        st.markdown(
            """
            <div class="status-chip">
              <span class="status-dot"></span>
              Đường màu đỏ là vùng dự báo, đường màu xanh là dữ liệu lịch sử.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if "latest_forecast_bundle" not in st.session_state:
        st.session_state["latest_forecast_bundle"] = None

    run_clicked = st.button("Chạy dự báo", type="primary", use_container_width=True)
    if run_clicked:
        if "prediction_payload" not in st.session_state:
            st.warning("Hãy nhấn “Tự động điền từ API” trước khi chạy dự báo.")

        else:
            overrides = collect_overrides()
            payload = st.session_state["prediction_payload"]
            with st.spinner("Đang dựng đặc trưng và chạy mô hình CatBoost..."):
                try:
                    bundle = run_forecast(
                        pm_history=payload["pm_history"],
                        weather_history=payload["weather_history"],
                        overrides=overrides,
                    )
                    st.session_state["latest_forecast_bundle"] = bundle
                except Exception as exc:
                    st.error(f"Lỗi dự báo: {exc}")

    bundle = st.session_state.get("latest_forecast_bundle")
    predictions: list[float | None]
    future_hours: list[str]
    if bundle is None:
        predictions = [None] * 6
        future_hours = ["--:--"] * 6
    else:
        predictions = bundle.predictions[:6]
        future_hours = [timestamp.strftime("%H:%M") for timestamp in bundle.prediction_times_vn[:6]]

    st.markdown("### Kết quả dự báo")
    render_result_cards(predictions, future_hours)

    if bundle is None:
        st.caption("Nhấn “Chạy dự báo” để cập nhật 6 giá trị PM2.5 cho các mốc t+1 đến t+6.")
        return

    for warning in bundle.warnings:
        st.warning(warning)

    artifact_name = bundle.artifact.model_path.name
    generated_text = "được tạo tự động cho deployment" if bundle.artifact.generated else "được nạp từ thư mục model"
    st.caption(
        f"Artifact đang dùng: `{artifact_name}` ({generated_text}). Thời điểm dự báo: {bundle.generated_at_vn.strftime('%H:%M %d/%m/%Y')}."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_live_chart(bundle, 6)
    render_suggestion(bundle.predictions[:6], future_hours)


if __name__ == "__main__":
    main()
