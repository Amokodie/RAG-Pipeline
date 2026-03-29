"""
Dark / Light appearance: one CSS template for both (same selectors as dark), different palettes.
Streamlit Cloud [theme] is dark-first; Light overrides those tokens so widgets match the shell.
"""

from __future__ import annotations

import streamlit as st

# Must match .streamlit/config.toml for default (dark) deploy.
_BG_DARK = "#0c1118"
_SIDEBAR_DARK = "#0f1419"


def _theme_tokens(light: bool) -> dict[str, str]:
    """Parallel palettes: light mirrors dark structure to avoid token conflict with config.toml."""
    if not light:
        return {
            "scheme": "dark",
            "bg": _BG_DARK,
            "bg_sidebar": _SIDEBAR_DARK,
            "fg": "#e5e7eb",
            "fg_muted": "#94a3b8",
            "fg_caption": "#94a3b8",
            "fg_h1": "#f0f9ff",
            "fg_h23": "#bae6fd",
            "link": "#38bdf8",
            "header_bg": "rgba(12, 17, 24, 0.95)",
            "header_border": "rgba(56, 189, 248, 0.12)",
            "sidebar_border": "rgba(56, 189, 248, 0.12)",
            "metric_bg": "rgba(15, 23, 42, 0.55)",
            "metric_border": "rgba(56, 189, 248, 0.15)",
            "tabs_bg": "rgba(15, 23, 42, 0.55)",
            "tabs_border": "rgba(56, 189, 248, 0.12)",
            "tab_inactive": "#94a3b8",
            "tab_active": "#f0f9ff",
            "expander_bg": "rgba(15, 23, 42, 0.4)",
            "expander_border": "rgba(56, 189, 248, 0.12)",
            "input_bg": "rgba(15, 23, 42, 0.35)",
            "code_block_fg": "#e5e7eb",
            "primary_var": "#f8fafc",
            "secondary_var": "#94a3b8",
        }
    return {
        "scheme": "light",
        "bg": "#ffffff",
        "bg_sidebar": "#f1f5f9",
        "fg": "#0f172a",
        "fg_muted": "#475569",
        "fg_caption": "#64748b",
        "fg_h1": "#0f172a",
        "fg_h23": "#1e293b",
        "link": "#0369a1",
        "header_bg": "rgba(255, 255, 255, 0.98)",
        "header_border": "rgba(15, 23, 42, 0.1)",
        "sidebar_border": "rgba(15, 23, 42, 0.12)",
        "metric_bg": "#ffffff",
        "metric_border": "rgba(15, 23, 42, 0.1)",
        "tabs_bg": "#f1f5f9",
        "tabs_border": "rgba(15, 23, 42, 0.12)",
        "tab_inactive": "#64748b",
        "tab_active": "#0f172a",
        "expander_bg": "#ffffff",
        "expander_border": "rgba(15, 23, 42, 0.12)",
        "input_bg": "#ffffff",
        "code_block_fg": "#0f172a",
        "primary_var": "#0f172a",
        "secondary_var": "#475569",
    }


def _shell_css(t: dict[str, str]) -> str:
    """Same rules for light and dark; only token values change."""
    bg = t["bg"]
    return f"""
  html {{
    color-scheme: {t["scheme"]} !important;
  }}
  html, body {{
    background-color: {bg} !important;
    color: {t["fg"]} !important;
  }}
  .stApp {{
    background: {bg} !important;
    background-color: {bg} !important;
    background-image: none !important;
    min-height: 100vh;
    color: {t["fg"]} !important;
    --text-color: {t["fg"]} !important;
    --primary-text-color: {t["primary_var"]} !important;
    --secondary-text-color: {t["secondary_var"]} !important;
    --background-color: {bg} !important;
    --secondary-background-color: {t["bg_sidebar"]} !important;
  }}
  [data-testid="stAppViewContainer"] {{
    background: transparent !important;
    color: {t["fg"]} !important;
  }}
  section[data-testid="stMain"] > div {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] {{
    background: transparent !important;
    color: {t["fg"]} !important;
  }}
  .stApp .main {{
    background: transparent !important;
    color: {t["fg"]} !important;
  }}
  .main > div {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] .block-container {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] [class*="block-container"] {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent !important;
  }}
  section[data-testid="stMain"] [data-testid="stVerticalBlock"] {{
    background: transparent !important;
  }}
  [data-testid="stHeader"] {{
    background: {t["header_bg"]} !important;
    backdrop-filter: blur(10px);
    border-bottom: 1px solid {t["header_border"]};
    color: {t["fg"]} !important;
  }}
  section[data-testid="stSidebar"] {{
    background: {t["bg_sidebar"]} !important;
    border-right: 1px solid {t["sidebar_border"]};
    color: {t["fg"]} !important;
    --text-color: {t["fg"]} !important;
    --primary-text-color: {t["primary_var"]} !important;
    --secondary-text-color: {t["secondary_var"]} !important;
    --background-color: {t["bg_sidebar"]} !important;
    --secondary-background-color: {t["input_bg"]} !important;
  }}
  section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.1rem;
  }}
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] li {{
    color: {t["fg"]} !important;
  }}
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea {{
    color: {t["fg"]} !important;
    -webkit-text-fill-color: {t["fg"]} !important;
    background-color: {t["input_bg"]} !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="radio"] label,
  section[data-testid="stSidebar"] [data-baseweb="radio"] div {{
    color: {t["fg"]} !important;
  }}
  section[data-testid="stSidebar"] [data-baseweb="select"] div[class] {{
    color: {t["fg"]} !important;
  }}
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stMarkdownContainer"] li,
  [data-testid="stMarkdownContainer"] ol,
  [data-testid="stMarkdownContainer"] ul,
  [data-testid="stMarkdownContainer"] strong,
  [data-testid="stMarkdownContainer"] em {{
    color: {t["fg"]} !important;
  }}
  [data-testid="stMarkdownContainer"] pre code {{
    color: {t["code_block_fg"]} !important;
  }}
  [data-testid="stCaption"] p,
  [data-testid="stCaption"] {{
    color: {t["fg_caption"]} !important;
  }}
  .stApp a {{
    color: {t["link"]} !important;
  }}
  h1 {{
    color: {t["fg_h1"]} !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
  }}
  h2, h3 {{
    color: {t["fg_h23"]} !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stMetric"] {{
    background: {t["metric_bg"]} !important;
    border: 1px solid {t["metric_border"]};
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    color: {t["fg"]} !important;
  }}
  div[data-testid="stMetric"] p,
  div[data-testid="stMetric"] span {{
    color: {t["fg"]} !important;
  }}
  div[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p {{
    color: {t["fg_muted"]} !important;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: {t["tabs_bg"]} !important;
    padding: 6px 8px;
    border-radius: 10px;
    border: 1px solid {t["tabs_border"]};
  }}
  [data-baseweb="tab"] {{
    color: {t["tab_inactive"]} !important;
  }}
  [data-baseweb="tab"][aria-selected="true"] {{
    color: {t["tab_active"]} !important;
    font-weight: 600 !important;
  }}
  div[data-testid="stExpander"] details {{
    border: 1px solid {t["expander_border"]};
    border-radius: 10px;
    background: {t["expander_bg"]} !important;
    color: {t["fg"]} !important;
  }}
  div[data-testid="stExpander"] summary {{
    color: {t["fg"]} !important;
  }}
"""


def _light_interactive_widgets_css() -> str:
    """
    Streamlit [theme] is dark-first; Base Web can still paint dark surfaces on tabs / buttons / selects.
    Match Streamlit's *secondary* button look: #f1f5f9 bg, #0f172a text, #cbd5e1 border (same as a good download button).
    """
    # Shared "secondary control" look (user-requested: like Download analysis brief)
    sec_bg = "#f1f5f9"
    sec_fg = "#0f172a"
    sec_border = "#cbd5e1"
    return f"""
  .stApp {{
    --primary-color: #0284c7 !important;
    --text-color: {sec_fg} !important;
  }}
  /* ---- Tabs: pill style = secondary button (all labels visible, no black fill) ---- */
  .stApp .stTabs [data-baseweb="tab-list"],
  .stApp [data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding: 8px !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
  }}
  .stApp .stTabs [data-baseweb="tab"],
  .stApp [data-testid="stTabs"] [data-baseweb="tab"],
  .stApp .stTabs [role="tab"] {{
    background-color: {sec_bg} !important;
    color: {sec_fg} !important;
    border: 1px solid {sec_border} !important;
    border-radius: 8px !important;
    opacity: 1 !important;
    box-shadow: none !important;
  }}
  .stApp .stTabs [data-baseweb="tab"]:hover,
  .stApp [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
    background-color: #e2e8f0 !important;
    color: {sec_fg} !important;
  }}
  .stApp .stTabs [data-baseweb="tab"][aria-selected="true"],
  .stApp [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {{
    background-color: #ffffff !important;
    color: {sec_fg} !important;
    border: 2px solid #0284c7 !important;
    font-weight: 600 !important;
  }}
  .stApp .stTabs [data-baseweb="tab"] *,
  .stApp [data-testid="stTabs"] [data-baseweb="tab"] *,
  .stApp .stTabs [role="tab"] * {{
    color: {sec_fg} !important;
    opacity: 1 !important;
  }}
  .stApp .stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
  }}
  /* ---- Download buttons: uniform secondary (gray) — not primary blue ---- */
  .stApp [data-testid="stDownloadButton"] button,
  .stApp [data-testid="stDownloadButton"] [data-baseweb="button"] {{
    background-color: {sec_bg} !important;
    color: {sec_fg} !important;
    border: 1px solid {sec_border} !important;
    border-radius: 0.5rem !important;
    opacity: 1 !important;
  }}
  .stApp [data-testid="stDownloadButton"] button *,
  .stApp [data-testid="stDownloadButton"] [data-baseweb="button"] * {{
    color: {sec_fg} !important;
    opacity: 1 !important;
  }}
  .stApp [data-testid="stDownloadButton"] svg {{
    fill: {sec_fg} !important;
  }}
  /* ---- Select (sidebar + main): light surface ---- */
  .stApp [data-baseweb="select"] > div,
  .stApp [data-baseweb="select"] [class*="control"] {{
    background-color: #ffffff !important;
    color: {sec_fg} !important;
    border: 1px solid {sec_border} !important;
  }}
  .stApp [data-baseweb="select"] svg {{
    fill: {sec_fg} !important;
  }}
  /* Sidebar code / install hints */
  section[data-testid="stSidebar"] [data-testid="stCode"],
  section[data-testid="stSidebar"] pre {{
    background-color: #f8fafc !important;
    color: {sec_fg} !important;
    border: 1px solid #e2e8f0 !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stCode"] code,
  section[data-testid="stSidebar"] pre code {{
    color: {sec_fg} !important;
    background: transparent !important;
  }}
  /* ---- Dataframes: avoid dark glass blocks on white page ---- */
  .stApp [data-testid="stDataFrame"],
  .stApp [data-testid="stDataFrame"] > div,
  .stApp [data-testid="stDataFrame"] [class*="glide"] {{
    background-color: rgba(248, 250, 252, 0.92) !important;
    color: {sec_fg} !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
  }}
  .stApp [data-testid="stDataFrame"] [role="gridcell"],
  .stApp [data-testid="stDataFrame"] [role="columnheader"] {{
    color: {sec_fg} !important;
  }}
"""


def inject_engineering_theme(mode: str = "dark") -> None:
    """Apply unified shell + widget colors (light uses same structure as dark)."""
    t = _theme_tokens(mode == "light")
    app_css = _shell_css(t)
    if t["scheme"] == "light":
        app_css += _light_interactive_widgets_css()
    style_block = f"<style>{app_css}</style>"
    if hasattr(st, "html"):
        st.html(style_block)
    else:
        st.markdown(style_block, unsafe_allow_html=True)


def hero_engineering_ribbon(mode: str = "dark") -> None:
    """Thin accent under title."""
    if mode == "light":
        grad = "linear-gradient(90deg, transparent 0%, #64748b 38%, #94a3b8 50%, #64748b 62%, transparent 100%)"
        op = "0.45"
    else:
        grad = "linear-gradient(90deg, transparent 0%, #0ea5e9 25%, #38bdf8 50%, #22d3ee 75%, transparent 100%)"
        op = "0.75"
    st.markdown(
        f"""
<div style="
  height:3px;
  border-radius:2px;
  margin: 0 0 0.85rem 0;
  background: {grad};
  opacity: {op};
"></div>
        """,
        unsafe_allow_html=True,
    )


def plotly_template(mode: str) -> str:
    return "plotly_white" if mode == "light" else "plotly_dark"
