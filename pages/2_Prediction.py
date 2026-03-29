# pages/2_Prediction.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.aqi import pm25_to_aqi
from src.api import get_current_data
from src.model import predict_multi_horizon


# ─── CSS ─────────────────────────────────────────────────────────────────────

def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

      html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
      #MainMenu, footer, header  { visibility: hidden; }

      .stApp { background: #f5f6fa; color: #111827; }

      .block-title {
        font-size: 0.72rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #9ca3af; margin-bottom: 12px;
      }
      .result-card {
        background: #ffffff; border: 1px solid #e2e5ea;
        border-radius: 12px; padding: 16px 12px; text-align: center;
        transition: box-shadow 0.15s;
      }
      .result-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.06); }
      .result-card .hour  { font-size: 0.75rem; color: #9ca3af; margin-bottom: 4px; font-family: 'Space Mono', monospace; }
      .result-card .val   { font-size: 1.5rem; font-weight: 700; color: #111827; margin: 4px 0; font-family: 'Space Mono', monospace; }
      .result-card .unit  { font-size: 0.65rem; color: #9ca3af; }

      .result-card-large {
        background: #ffffff; border: 2px solid #e2e5ea;
        border-radius: 16px; padding: 40px 24px; text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
      }
      .result-card-large .val { font-size: 3rem; font-weight: 800; color: #111827; font-family: 'Space Mono', monospace; }

      .aqi-badge {
        display: inline-block; padding: 3px 12px;
        border-radius: 999px; font-size: 0.75rem; font-weight: 700;
      }
      .aqi-badge-large {
        display: inline-block; padding: 6px 18px;
        border-radius: 999px; font-size: 1rem; font-weight: 700;
        margin-top: 12px;
      }
      .suggestion-box {
        background: #ffffff; border-left: 4px solid #7c6af7;
        border-radius: 8px; padding: 14px 18px; margin-top: 16px;
        font-size: 0.95rem; color: #374151;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
      }
    </style>
    """, unsafe_allow_html=True)


# ─── Render helpers ──────────────────────────────────────────────────────────

def render_aqi_badge(label: str, bg: str, text: str,
                     css_class: str = "aqi-badge") -> str:
    return f'<span class="{css_class}" style="background:{bg};color:{text}">{label}</span>'


def render_result_cards(predictions: list[float], future_hours: list[str]):
    """Render 1 ô lớn hoặc nhiều ô nhỏ tuỳ horizon."""
    horizon = len(predictions)

    if horizon == 1:
        info = pm25_to_aqi(predictions[0])
        badge = render_aqi_badge(f"{info['label']} · AQI {info['aqi']}",
                                 info["bg"], info["text"], "aqi-badge-large")
        st.markdown(f"""
        <div class="result-card-large">
          <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:8px;">
            Dự báo lúc {future_hours[0]}
          </div>
          <div class="val" style="color:{info['bg']};">
            {predictions[0]:.1f} <span style="font-size:1.4rem;font-weight:400;color:#6b7280;">μg/m³</span>
          </div>
          {badge}
        </div>
        """, unsafe_allow_html=True)

    else:
        cols = st.columns(horizon)
        for i, (col, pm25_val, hour_str) in enumerate(zip(cols, predictions, future_hours)):
            info  = pm25_to_aqi(pm25_val)
            badge = render_aqi_badge(info["label"], info["bg"], info["text"])
            with col:
                st.markdown(f"""
                <div class="result-card">
                  <div class="hour">t+{i+1}h · {hour_str}</div>
                  <div class="val" style="color:{info['bg']};">{pm25_val:.1f}</div>
                  <div class="unit">μg/m³</div>
                  <div style="margin-top:6px;">{badge}</div>
                </div>
                """, unsafe_allow_html=True)


def render_forecast_chart(predictions: list[float], future_hours: list[str]):
    chart_df = pd.DataFrame({
        "Giờ": future_hours,
        "PM2.5 (μg/m³)": predictions,
    }).set_index("Giờ")
    st.line_chart(chart_df, height=200)


def render_suggestion(predictions: list[float], hours: list[str]):
    max_val  = max(predictions)
    min_val  = min(predictions)
    max_hour = hours[predictions.index(max_val)]
    min_hour = hours[predictions.index(min_val)]
    trend    = predictions[-1] - predictions[0]

    if max_val > 150:
        level = f"⚠️ PM2.5 dự kiến đạt **{max_val:.1f} μg/m³** lúc {max_hour} – mức **Không lành mạnh**, hạn chế ra ngoài."
    elif max_val > 55:
        level = f"🟠 PM2.5 có thể lên **{max_val:.1f} μg/m³** lúc {max_hour} – nhóm nhạy cảm nên hạn chế hoạt động ngoài trời."
    elif max_val > 35:
        level = f"🟡 Chất lượng không khí trung bình, PM2.5 cao nhất **{max_val:.1f} μg/m³** lúc {max_hour}."
    else:
        level = f"🟢 Chất lượng không khí tốt, PM2.5 dưới **{max_val:.1f} μg/m³** trong {len(predictions)} giờ tới."

    if trend > 10:
        trend_msg = f" Xu hướng **tăng dần** – nên ra ngoài trước {hours[0]}."
    elif trend < -10:
        trend_msg = f" Dự kiến **cải thiện** sau {min_hour}."
    else:
        trend_msg = " Mức độ ô nhiễm **ổn định** trong khung giờ này."

    st.markdown(f'<div class="suggestion-box">💬 {level}{trend_msg}</div>',
                unsafe_allow_html=True)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Dự báo PM2.5", page_icon="🔬", layout="wide")
    inject_css()

    st.markdown('<h1 style="font-size:1.5rem;font-weight:700;color:#111827;"> Predict PM2.5 within 6 hours</h1>',
                unsafe_allow_html=True)
    st.caption("Nhập các thông số môi trường để xem mô hình dự báo nồng độ bụi mịn.")
    st.divider()

    # ── Inputs ──
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="block-title">📊 PM2.5 lịch sử</div>', unsafe_allow_html=True)
        if st.button("⚡ Auto-fill từ API", use_container_width=True):
            with st.spinner("Đang lấy dữ liệu..."):
                try:
                    data = get_current_data()
                    st.session_state["lag_1h"]  = float(data.get("pm25", 35.0))
                    st.session_state["lag_3h"]  = float(data.get("pm25_3h", 30.0))
                    st.session_state["lag_24h"] = float(data.get("pm25_24h", 28.0))
                    st.success("Đã điền giá trị thực tế ✓")
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        lag_1h  = st.number_input("PM2.5 1 giờ trước (μg/m³)",  0.0, 500.0, step=0.5, key="lag_1h",  value=st.session_state.get("lag_1h",  35.0))
        lag_3h  = st.number_input("PM2.5 3 giờ trước (μg/m³)",  0.0, 500.0, step=0.5, key="lag_3h",  value=st.session_state.get("lag_3h",  30.0))
        lag_24h = st.number_input("PM2.5 24 giờ trước (μg/m³)", 0.0, 500.0, step=0.5, key="lag_24h", value=st.session_state.get("lag_24h", 28.0))

    with col_right:
        st.markdown('<div class="block-title">🌤️ Thông số khí tượng</div>', unsafe_allow_html=True)
        temperature = st.slider("Nhiệt độ (°C)",    15.0, 45.0, 30.0, step=0.5, format="%.1f °C")
        humidity    = st.slider("Độ ẩm (%)",         10,  100,   70,  step=1,   format="%d %%")
        wind_speed  = st.slider("Tốc độ gió (m/s)",  0.0, 20.0,  2.5, step=0.1, format="%.1f m/s")

    st.divider()

    # ── Horizon ──
    st.markdown("#### Dự báo bao nhiêu giờ tới?")
    horizon = st.radio("horizon", [1, 2, 3, 4, 5, 6],
                       format_func=lambda x: f"{x} giờ",
                       horizontal=True, label_visibility="collapsed")
    st.divider()

    # ── Predict ──
    if st.button("Dự báo", type="primary", use_container_width=True):
        features = {
            "pm25_lag1":   lag_1h,
            "pm25_lag3":   lag_3h,
            "pm25_lag24":  lag_24h,
            "temperature": temperature,
            "humidity":    humidity,
            "wind_speed":  wind_speed,
        }

        with st.spinner("Đang chạy mô hình..."):
            try:
                predictions = predict_multi_horizon(features, horizon)
            except Exception as e:
                st.error(f"Lỗi predict: {e}")
                st.stop()

        now          = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
        future_hours = [(now + timedelta(hours=i + 1)).strftime("%H:%M") for i in range(horizon)]

        st.markdown("### Kết quả dự báo")
        render_result_cards(predictions, future_hours)

        if horizon >= 3:
            st.markdown("<br>", unsafe_allow_html=True)
            render_forecast_chart(predictions, future_hours)

        render_suggestion(predictions, future_hours)


if __name__ == "__main__":
    main()