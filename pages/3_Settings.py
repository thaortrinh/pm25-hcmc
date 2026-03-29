from __future__ import annotations

import streamlit as st

from src.ui import GRAPH_COLORS, THEME_OPTIONS, apply_settings, inject_base_css, init_session_settings


def inject_css():
    inject_base_css(
        """
        .settings-card {
          background: var(--card-bg);
          border: 1px solid var(--border-color);
          border-radius: 18px;
          padding: 22px;
          box-shadow: 0 10px 30px var(--shadow-color);
          margin-bottom: 18px;
        }

        .settings-title {
          font-size: 0.74rem;
          font-weight: 700;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--text-faint);
          margin-bottom: 10px;
        }

        .color-picker-row [data-testid="stButton"] > button {
          min-height: 42px;
          border-radius: 999px;
          border: 1px solid rgba(71, 85, 105, 0.85);
          background: linear-gradient(180deg, #253146 0%, #1a2436 100%);
          color: #f8fafc;
          font-weight: 600;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
        }

        .color-picker-row [data-testid="stButton"] > button:hover {
          border-color: rgba(96, 165, 250, 0.45);
          background: linear-gradient(180deg, #2d3a52 0%, #1d2940 100%);
          color: #ffffff;
        }

        .color-picker-row [data-testid="stButton"] > button:focus {
          box-shadow: 0 0 0 0.18rem rgba(96, 165, 250, 0.25);
        }

        .color-picker-row [data-testid="column"] {
          display: flex;
          align-items: stretch;
        }

        .selected-color-chip {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          padding: 6px 12px;
          border-radius: 999px;
          background: linear-gradient(180deg, #253146 0%, #1a2436 100%);
          border: 1px solid rgba(71, 85, 105, 0.85);
          color: #f8fafc;
          font-size: 0.82rem;
          margin-top: 4px;
        }
        """
    )


def _set_draft_color(color_name: str) -> None:
    st.session_state.draft_pm25_chart_color = color_name


COLOR_BUTTON_LABELS = {
    "Red": "🔴 Red",
    "Black": "⚫ Black",
    "Yellow": "🟡 Yellow",
    "Green": "🟢 Green",
    "Blue": "🔵 Blue",
}


def main():
    st.set_page_config(page_title="Cài đặt", page_icon="⚙️", layout="wide")
    init_session_settings()
    inject_css()

    st.title("Cài đặt")
    st.caption("Thay đổi giao diện và áp dụng khi bấm “Lưu cài đặt”.")

    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">Chế độ giao diện</div>', unsafe_allow_html=True)
    st.radio(
        "Chế độ giao diện",
        THEME_OPTIONS,
        index=THEME_OPTIONS.index(st.session_state.draft_theme_mode),
        format_func=lambda value: "Sáng" if value == "light" else "Tối",
        key="draft_theme_mode",
        horizontal=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">Màu đường PM2.5 trên Dashboard</div>', unsafe_allow_html=True)
    selected_color = st.session_state.draft_pm25_chart_color
    st.markdown(
        f'<div class="selected-color-chip">Đang chọn: {COLOR_BUTTON_LABELS.get(selected_color, selected_color)}</div>',
        unsafe_allow_html=True,
    )

    color_items = list(GRAPH_COLORS.items())
    st.markdown('<div class="color-picker-row">', unsafe_allow_html=True)
    color_columns = st.columns(len(color_items))
    for column, (label, hex_color) in zip(color_columns, color_items):
        with column:
            button_label = COLOR_BUTTON_LABELS.get(label, label)
            if st.button(button_label, key=f"color_button_{label}", use_container_width=True):
                _set_draft_color(label)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Lưu cài đặt", type="primary"):
        apply_settings()
        st.success("Đã áp dụng cài đặt cho toàn bộ ứng dụng.")


if __name__ == "__main__":
    main()
