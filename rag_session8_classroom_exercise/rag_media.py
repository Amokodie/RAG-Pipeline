"""
Shared explainer video resolution + Streamlit embedding (local MP4 or remote URL).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st


EXPLAINER_NAMES = ("rag_explainer.mp4", "RAG_Stops_AI_Hallucinations.mp4")


def resolve_explainer_video(*search_bases: Path) -> Path | None:
    """First matching file wins. Each base is tried with optional `assets/` subfolder."""
    for base in search_bases:
        for name in EXPLAINER_NAMES:
            for folder in (base, base / "assets"):
                p = folder / name
                if p.is_file():
                    return p
    return None


def _explainer_url() -> str | None:
    u = (os.environ.get("RAG_EXPLAINER_VIDEO_URL") or "").strip()
    if u:
        return u
    try:
        u = (st.secrets.get("RAG_EXPLAINER_VIDEO_URL") or "").strip()
    except Exception:
        u = ""
    return u or None


@st.cache_data(show_spinner="Loading explainer video…")
def _cached_explainer_bytes(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


_DEFAULT_MISSING_CAPTION = (
    "No explainer MP4 in this deployment. Add **`rag_session8_classroom_exercise/assets/rag_explainer.mp4`** "
    "to the repository, or set secret **`RAG_EXPLAINER_VIDEO_URL`** to a direct MP4 link."
)


def render_rag_explainer_block(
    *search_bases: Path,
    caption: str,
    missing_hint: str | None = None,
) -> None:
    """
    Embed explainer: optional remote URL, else local MP4 bytes (reliable on Windows paths with spaces).
    Bundled path checked first: ``rag_session8_classroom_exercise/assets/rag_explainer.mp4``.
    """
    st.header("How this RAG Pipeline Prevents Hallucinations")

    url = _explainer_url()
    if url:
        st.video(url)
        st.caption(caption)
        st.divider()
        return

    hint = missing_hint if missing_hint is not None else _DEFAULT_MISSING_CAPTION

    vid = resolve_explainer_video(*search_bases)
    if vid is not None:
        try:
            data = _cached_explainer_bytes(str(vid.resolve()))
            st.video(data, format="video/mp4")
            st.caption(caption)
            st.caption(f"Loaded: `{vid.name}` ({len(data) / (1024 * 1024):.1f} MB).")
        except OSError as e:
            st.error(f"Could not read video file: {e}")
            st.caption(hint)
    else:
        st.caption(hint)
    st.divider()
