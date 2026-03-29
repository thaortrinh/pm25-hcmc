from __future__ import annotations

import streamlit as st


THEME_OPTIONS = ("light", "dark")

GRAPH_COLORS = {
    "Red": "#ef4444",
    "Black": "#111827",
    "Yellow": "#eab308",
    "Green": "#22c55e",
    "Blue": "#3b82f6",
}


THEME_PALETTES = {
    "light": {
        "app_bg": "#f5f6fa",
        "card_bg": "#ffffff",
        "card_bg_alt": "#f8f9fb",
        "text": "#111827",
        "text_muted": "#6b7280",
        "text_faint": "#9ca3af",
        "border": "#e2e5ea",
        "border_hover": "#c8ccd4",
        "shadow": "rgba(15, 23, 42, 0.08)",
        "input_bg": "#ffffff",
        "accent": "#3b82f6",
    },
    "dark": {
        "app_bg": "#0f172a",
        "card_bg": "#111827",
        "card_bg_alt": "#1f2937",
        "text": "#f3f4f6",
        "text_muted": "#cbd5e1",
        "text_faint": "#94a3b8",
        "border": "#334155",
        "border_hover": "#475569",
        "shadow": "rgba(2, 6, 23, 0.35)",
        "input_bg": "#0b1220",
        "accent": "#60a5fa",
    },
}


def init_session_settings() -> None:
    if "applied_theme_mode" not in st.session_state:
        st.session_state.applied_theme_mode = "light"
    if "applied_pm25_chart_color" not in st.session_state:
        st.session_state.applied_pm25_chart_color = "Blue"
    if "draft_theme_mode" not in st.session_state:
        st.session_state.draft_theme_mode = st.session_state.applied_theme_mode
    if "draft_pm25_chart_color" not in st.session_state:
        st.session_state.draft_pm25_chart_color = st.session_state.applied_pm25_chart_color


def get_theme_mode() -> str:
    init_session_settings()
    mode = st.session_state.get("applied_theme_mode", "light")
    return mode if mode in THEME_PALETTES else "light"


def get_theme_palette() -> dict[str, str]:
    return THEME_PALETTES[get_theme_mode()]


def get_pm25_chart_color() -> str:
    init_session_settings()
    label = st.session_state.get("applied_pm25_chart_color", "Blue")
    return GRAPH_COLORS.get(label, GRAPH_COLORS["Blue"])


def apply_settings() -> None:
    init_session_settings()
    draft_theme = st.session_state.get("draft_theme_mode", "light")
    draft_color = st.session_state.get("draft_pm25_chart_color", "Blue")
    if draft_theme in THEME_PALETTES:
        st.session_state.applied_theme_mode = draft_theme
    if draft_color in GRAPH_COLORS:
        st.session_state.applied_pm25_chart_color = draft_color


def inject_base_css(extra_css: str = "") -> dict[str, str]:
    init_session_settings()
    palette = get_theme_palette()

    st.markdown(
        f"""
        <style>
          :root {{
            --app-bg: {palette["app_bg"]};
            --card-bg: {palette["card_bg"]};
            --card-bg-alt: {palette["card_bg_alt"]};
            --text-primary: {palette["text"]};
            --text-muted: {palette["text_muted"]};
            --text-faint: {palette["text_faint"]};
            --border-color: {palette["border"]};
            --border-hover: {palette["border_hover"]};
            --shadow-color: {palette["shadow"]};
            --input-bg: {palette["input_bg"]};
            --accent-color: {palette["accent"]};
          }}

          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

          html, body, [class*="css"] {{
            font-family: 'DM Sans', sans-serif;
          }}

          #MainMenu, footer, header {{
            visibility: hidden;
          }}

          .stApp {{
            background: var(--app-bg);
            color: var(--text-primary);
          }}

          [data-testid="stSidebar"] {{
            background: var(--card-bg-alt);
            border-right: 1px solid var(--border-color);
          }}

          [data-testid="stSidebar"] * {{
            color: var(--text-primary);
          }}

          [data-testid="stSidebarNav"] {{
            background: transparent;
          }}

          [data-testid="stSidebarNav"] a {{
            border-radius: 10px;
            color: var(--text-primary);
          }}

          [data-testid="stSidebarNav"] a:hover {{
            background: rgba(148, 163, 184, 0.16);
          }}

          [data-testid="stSidebarNav"] a[aria-current="page"] {{
            background: rgba(148, 163, 184, 0.22);
            font-weight: 700;
          }}

          h1, h2, h3, h4, h5, h6, p, span, label {{
            color: var(--text-primary);
          }}

          [data-testid="stMarkdownContainer"] p {{
            color: var(--text-primary);
          }}

          div[data-baseweb="select"] > div,
          div[data-baseweb="input"] > div,
          .stNumberInput input,
          .stTextInput input,
          .stTextArea textarea {{
            background: var(--input-bg);
            color: var(--text-primary);
          }}

          div[data-testid="stRadio"] label,
          div[data-testid="stSelectbox"] label,
          div[data-testid="stSlider"] label,
          div[data-testid="stNumberInput"] label,
          div[data-testid="stExpander"] details summary p {{
            color: var(--text-primary);
          }}

          div[data-testid="stExpander"] details {{
            border: 1px solid var(--border-color);
            border-radius: 16px;
            background: var(--card-bg);
          }}

          div[data-testid="stExpander"] details summary {{
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
          }}

          div[data-testid="stExpander"] details summary:hover {{
            background: transparent;
          }}

          div[data-testid="stCaptionContainer"] {{
            color: var(--text-faint);
          }}

          {extra_css}
        </style>
        """,
        unsafe_allow_html=True,
    )
    return palette
