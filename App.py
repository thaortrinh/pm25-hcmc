import streamlit as st

from src.ui import inject_base_css


def main():
    st.set_page_config(page_title="PM2.5 TP.HCM", page_icon="🌫️", layout="wide")
    inject_base_css(
        """
        .home-card {
          background: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 18px;
          padding: 24px;
          box-shadow: 0 10px 30px var(--shadow-color);
          margin-top: 18px;
        }
        """
    )

    st.title("PM2.5 TP.HCM")
    st.markdown("Dùng thanh điều hướng bên trái để mở trang Dashboard, Prediction hoặc Settings.")
    st.markdown(
        """
        <div class="home-card">
          <b>Các trang hiện có</b><br>
          Dashboard: quan trắc hiện tại, thời tiết và VN_AQI<br>
          Prediction: dự báo PM2.5 ngắn hạn 1 đến 6 giờ<br>
          Settings: giao diện và màu đường biểu đồ PM2.5
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
