"""
Institution logos + author banner (same pattern as Assignment 2).
Place PNG files under ./assets/ — copied from Ass2 when available.
"""

from __future__ import annotations

import base64
import io
import os

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_BEIHANG = os.path.join(_ASSET_DIR, "beihang_university_logo.png")
LOGO_RCSSTEAP = os.path.join(_ASSET_DIR, "rcssteap_logo.png")


def _logo_on_white_tile(path: str, max_height_px: int = 140) -> None:
    """Show a PNG on a white rounded tile (same treatment for both institution logos)."""
    if not os.path.isfile(path):
        return
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            composed = Image.alpha_composite(bg, im).convert("RGB")
            buf = io.BytesIO()
            composed.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    components.html(
        f"""
<div style="background:#ffffff;border-radius:10px;padding:10px 14px;text-align:center;box-sizing:border-box;">
  <img src="data:image/png;base64,{b64}"
       style="max-height:{max_height_px}px;width:100%;object-fit:contain;display:block;margin:0 auto;" alt="" />
</div>
        """,
        height=max_height_px + 36,
    )


def render_authors_banner() -> None:
    """Institution logos + team (compact header)."""
    with st.container(border=True):
        c_logo1, c_logo2, c_text = st.columns([1.15, 1.15, 2.5])
        with c_logo1:
            if os.path.isfile(LOGO_BEIHANG):
                _logo_on_white_tile(LOGO_BEIHANG, max_height_px=150)
            else:
                st.caption("Add `assets/beihang_university_logo.png`")
        with c_logo2:
            if os.path.isfile(LOGO_RCSSTEAP):
                _logo_on_white_tile(LOGO_RCSSTEAP, max_height_px=150)
            else:
                st.caption("Add `assets/rcssteap_logo.png`")
        with c_text:
            st.markdown(
                """
**Beihang University** · *Beijing University of Aeronautics and Astronautics*  
**Regional Centre for Space Science and Technology Education in Asia and the Pacific (RCSSTEAP), China**

**Team:** Kodie Amo Kwame (LS2525226) · Sumara Alfred Salifu (LS2525245) · Peta Mimi Precious (LS2525255)
                """
            )
