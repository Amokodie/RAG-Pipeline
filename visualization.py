"""
Plotly 3D + heatmaps + retrieval analytics for TF‑IDF / LSA visualization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_theme import apply_plotly_theme


def _themed(fig: go.Figure, template: str | None) -> go.Figure:
    if template:
        return apply_plotly_theme(fig, template)
    return fig


def category_color_map(categories: list[str]) -> dict[str, str]:
    palette = {
        "Helpfulness": "#636EFA",
        "Safety": "#EF553B",
        "Honesty": "#00CC96",
        "Bias": "#AB63FA",
    }
    out: dict[str, str] = {}
    for c in categories:
        out[c] = palette.get(c, "#888888")
    return out


def figure_3d_chunks_and_query(
    df: pd.DataFrame,
    chunk_xyz: np.ndarray,
    query_xyz: np.ndarray | None,
    top_k_indices: list[int],
    title: str = "LSA 3D projection of TF‑IDF chunks",
    template: str | None = None,
) -> go.Figure:
    """
    3D scatter of chunks (by category) + optional query point + dashed lines to top‑k chunks.
    """
    case_ids = df["case_id"].astype(str).tolist()
    cats = df["category"].astype(str).tolist()
    cmap = category_color_map(list(pd.Series(cats).unique()))

    fig = go.Figure()

    for cat in sorted(set(cats)):
        idx = [i for i, c in enumerate(cats) if c == cat]
        fig.add_trace(
            go.Scatter3d(
                x=chunk_xyz[idx, 0],
                y=chunk_xyz[idx, 1],
                z=chunk_xyz[idx, 2],
                mode="markers+text",
                marker=dict(size=9, color=cmap.get(cat, "#888"), opacity=0.92, line=dict(width=0.5, color="#222")),
                text=[case_ids[i] for i in idx],
                textposition="top center",
                textfont=dict(size=10),
                name=cat,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    + cat
                    + "<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
                ),
            )
        )

    if query_xyz is not None and query_xyz.size:
        q = np.asarray(query_xyz).reshape(1, -1)[:, :3]
        fig.add_trace(
            go.Scatter3d(
                x=[float(q[0, 0])],
                y=[float(q[0, 1])],
                z=[float(q[0, 2])],
                mode="markers+text",
                marker=dict(size=14, color="#FFAA00", symbol="diamond", line=dict(width=1, color="#333")),
                text=["query"],
                textposition="bottom center",
                name="Query (projected)",
                hovertemplate="query vector<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )
        qx, qy, qz = float(q[0, 0]), float(q[0, 1]), float(q[0, 2])
        lx, ly, lz = [], [], []
        for idx in top_k_indices:
            if 0 <= idx < len(chunk_xyz):
                lx.extend([qx, float(chunk_xyz[idx, 0]), None])
                ly.extend([qy, float(chunk_xyz[idx, 1]), None])
                lz.extend([qz, float(chunk_xyz[idx, 2]), None])
        if lx:
            fig.add_trace(
                go.Scatter3d(
                    x=lx,
                    y=ly,
                    z=lz,
                    mode="lines",
                    line=dict(color="rgba(255,170,0,0.55)", width=4),
                    name="Query → top‑k",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    fig.update_layout(
        title=dict(text=title, x=0.02),
        scene=dict(
            xaxis_title="LSA dim 1",
            yaxis_title="LSA dim 2",
            zaxis_title="LSA dim 3",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=640,
    )
    return _themed(fig, template)


def figure_similarity_heatmap(
    sim_matrix: np.ndarray, labels: list[str], title: str = "Chunk–chunk cosine similarity", template: str | None = None
) -> go.Figure:
    """2D heatmap (paired with 3D view for complementary analysis)."""
    fig = go.Figure(
        data=go.Heatmap(
            z=np.asarray(sim_matrix),
            x=labels,
            y=labels,
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            colorbar=dict(title="cosine"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(autorange="reversed"),
        height=max(420, min(80 + 28 * len(labels), 900)),
        margin=dict(l=80, r=40, t=50, b=120),
    )
    return _themed(fig, template)


def figure_margin_bar(case_ids: list[str], margins: list[float], title: str, template: str | None = None) -> go.Figure:
    """Bar chart: retrieval margin (top1 − top2) per probe query."""
    dfp = pd.DataFrame({"case_id": case_ids, "margin": margins}).sort_values("margin", ascending=True)
    fig = go.Figure(
        data=go.Bar(
            x=dfp["margin"],
            y=dfp["case_id"],
            orientation="h",
            marker_color="#00A08A",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="cosine margin (top1 − top2)",
        yaxis_title="",
        height=max(400, 24 * len(case_ids)),
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return _themed(fig, template)


def figure_lsa_variance(ev: np.ndarray, template: str | None = None) -> go.Figure:
    """Bar + cumulative line for TruncatedSVD explained variance ratios."""
    ev = np.asarray(ev).ravel()
    comps = [f"Comp {i + 1}" for i in range(len(ev))]
    cum = np.cumsum(ev)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=comps, y=ev, name="Explained ratio", marker_color="#636EFA"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=comps, y=cum, name="Cumulative", mode="lines+markers", line=dict(color="#EF553B", width=2)),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="ratio", secondary_y=False, range=[0, max(0.15, float(np.max(ev)) * 1.15)])
    fig.update_yaxes(title_text="cumulative", secondary_y=True, range=[0, 1.05])
    fig.update_layout(
        title="LSA (TruncatedSVD) variance explained",
        height=400,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=1.12),
    )
    return _themed(fig, template)


def figure_softmax_mass(
    case_ids: list[str],
    probs: np.ndarray,
    *,
    title: str = "Softmax mass over chunks (full retrieval distribution)",
    template: str | None = None,
) -> go.Figure:
    """Horizontal bar: softmax probability per chunk (sums to 1)."""
    dfp = (
        pd.DataFrame({"case_id": case_ids, "p": np.asarray(probs).ravel()})
        .sort_values("p", ascending=True)
        .tail(16)
    )
    fig = go.Figure(
        go.Bar(
            x=dfp["p"],
            y=dfp["case_id"],
            orientation="h",
            marker=dict(color=dfp["p"], colorscale="Blues", showscale=False),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="softmax probability",
        height=max(380, 22 * len(dfp)),
        margin=dict(l=56, r=20, t=50, b=36),
    )
    return _themed(fig, template)


def figure_category_radar(
    category_to_score: dict[str, float],
    *,
    title: str = "Best cosine per alignment category (single query)",
    template: str | None = None,
) -> go.Figure:
    """Polar chart: max similarity to any chunk in each HHH/Bias category."""
    cats = sorted(category_to_score.keys())
    vals = [category_to_score[c] for c in cats]
    cats_loop = cats + [cats[0]]
    vals_loop = vals + [vals[0]]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals_loop,
            theta=cats_loop,
            fill="toself",
            name="max cosine",
            line=dict(color="#636EFA"),
            fillcolor="rgba(99, 110, 250, 0.35)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickformat=".2f")),
        title=title,
        height=480,
        showlegend=False,
        margin=dict(t=50, b=40),
    )
    return _themed(fig, template)


def figure_sankey_retrieval(
    query_short: str,
    target_case_ids: list[str],
    weights: list[float],
    *,
    template: str | None = None,
) -> go.Figure:
    """Sankey: query node → retrieved chunk nodes (weights sum to 1 if inputs normalized)."""
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        fig = go.Figure()
        fig.update_layout(title="Sankey (no links)", height=320)
        return _themed(fig, template)
    w = np.clip(w, 1e-9, None)
    w = w / w.sum()
    labels = [query_short[:42] + ("…" if len(query_short) > 42 else "")] + [str(c) for c in target_case_ids]
    n = len(target_case_ids)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(label=labels, pad=18, thickness=14, line=dict(color="rgba(100,100,120,0.35)", width=0.5)),
            link=dict(
                source=[0] * n,
                target=list(range(1, n + 1)),
                value=w.tolist(),
            ),
        )
    )
    fig.update_layout(title="Retrieval flow: query → top‑k chunks (normalized weights)", height=420, margin=dict(t=50, b=20))
    return _themed(fig, template)


def figure_chunk_length_bars(stats_df: pd.DataFrame, template: str | None = None) -> go.Figure:
    """Grouped bars: prompt vs response character counts per case."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stats_df["case_id"],
            y=stats_df["prompt_chars"],
            name="user_prompt chars",
            marker_color="#38bdf8",
        )
    )
    fig.add_trace(
        go.Bar(
            x=stats_df["case_id"],
            y=stats_df["response_chars"],
            name="model_response chars",
            marker_color="#a78bfa",
        )
    )
    fig.update_layout(
        title="Chunk text size (character counts from CSV)",
        barmode="group",
        xaxis=dict(tickangle=-45),
        yaxis_title="characters",
        height=max(400, 280),
        legend=dict(orientation="h", y=1.12),
        margin=dict(b=120, t=50),
    )
    return _themed(fig, template)
