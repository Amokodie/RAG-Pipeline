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


# ── Animated figures (Areas 3A / 3B / 3C / 4A) ───────────────────────────


def figure_chunking_animation(chunks: list[str]) -> go.Figure:
    """
    3A — Animated 4-stage chunking pipeline.
    Stage 0: Raw document → Stage 1: Split chunks → Stage 2: Embedded →
    Stage 3: Indexed in vector store.
    Uses go.Frame so Plotly's Play button steps through stages.
    """
    N = min(len(chunks), 12)
    rng = np.random.default_rng(42)

    # Simulated embedding positions (reproducible random 3-D coords)
    emb_x = rng.uniform(-1.5, 1.5, N).tolist()
    emb_y = rng.uniform(-1.5, 1.5, N).tolist()
    emb_z = rng.uniform(-1.0, 1.0, N).tolist()

    # Chunk labels and preview text
    labels = [f"Chunk {i + 1}" for i in range(N)]
    previews = [c[:45] + "…" if len(c) > 45 else c for c in chunks[:N]]
    color_vals = list(range(N))

    stage_titles = [
        "Stage 1 — Raw Document: full text as one blob",
        "Stage 2 — Splitting: text divided into chunks",
        "Stage 3 — Embedding: chunks projected to vector space",
        "Stage 4 — Indexed: vector store ready for retrieval",
    ]

    def _dots(x, y, z, text, colors, sizes, hover, symbol="circle") -> go.Scatter3d:
        return go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, colorscale="Viridis",
                        opacity=0.92, symbol=symbol,
                        line=dict(width=0.5, color="#333")),
            text=text,
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
            customdata=hover,
            showlegend=False,
        )

    def _lines(lx, ly, lz) -> go.Scatter3d:
        return go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode="lines",
            line=dict(color="rgba(99,110,250,0.45)", width=2),
            hoverinfo="skip",
            showlegend=False,
        )

    empty_lines = _lines([], [], [])

    # Stage 0: 1 big grey blob
    f0_dots = _dots([0], [0], [0], ["Full Document"], [10], [24],
                    ["Click ▶ Play to begin chunking"])
    f0_dots.marker.color = "#888888"

    # Stage 1: chunks in a horizontal row
    row_x = np.linspace(-1.5, 1.5, N).tolist()
    f1_dots = _dots(row_x, [0] * N, [0] * N, labels, color_vals, [10] * N, previews)

    # Stage 2: chunks at embedding positions
    f2_dots = _dots(emb_x, emb_y, emb_z, labels, color_vals, [10] * N, previews)

    # Stage 3: embedding positions + Vector Store node + index lines
    vs_x = emb_x + [0.0]
    vs_y = emb_y + [0.0]
    vs_z = emb_z + [0.0]
    vs_labels = labels + ["Vector Store"]
    vs_colors = color_vals + [N // 2]
    vs_sizes = [10] * N + [20]
    vs_previews = previews + ["Central index — all chunks stored here"]

    lx: list = []
    ly: list = []
    lz: list = []
    for i in range(N):
        lx += [0.0, emb_x[i], None]
        ly += [0.0, emb_y[i], None]
        lz += [0.0, emb_z[i], None]

    f3_dots = _dots(vs_x, vs_y, vs_z, vs_labels, vs_colors, vs_sizes, vs_previews)
    f3_lines = _lines(lx, ly, lz)

    frames = [
        go.Frame(data=[f0_dots, empty_lines], name="0",
                 layout=go.Layout(title_text=stage_titles[0])),
        go.Frame(data=[f1_dots, empty_lines], name="1",
                 layout=go.Layout(title_text=stage_titles[1])),
        go.Frame(data=[f2_dots, empty_lines], name="2",
                 layout=go.Layout(title_text=stage_titles[2])),
        go.Frame(data=[f3_dots, f3_lines], name="3",
                 layout=go.Layout(title_text=stage_titles[3])),
    ]

    _play_btn = dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 1400, "redraw": True},
                                  "fromcurrent": True, "mode": "immediate"}])
    _pause_btn = dict(label="⏸ Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])

    fig = go.Figure(
        data=[f0_dots, empty_lines],
        frames=frames,
        layout=go.Layout(
            title=dict(text=stage_titles[0], x=0.02),
            scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2",
                       zaxis_title="Dim 3", aspectmode="cube"),
            height=540,
            margin=dict(l=0, r=0, t=60, b=60),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=-0.05, x=0.5, xanchor="center", yanchor="top",
                buttons=[_play_btn, _pause_btn],
            )],
            sliders=[dict(
                currentvalue={"prefix": "Stage: ", "font": {"size": 13}},
                x=0.1, len=0.8, y=0.0,
                steps=[
                    dict(label=f"Stage {i + 1}", method="animate",
                         args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate"}])
                    for i in range(4)
                ],
            )],
        ),
    )
    return fig


def figure_retrieval_animation(
    chunk_xyz: np.ndarray,
    all_sims: np.ndarray,
    query_xyz: np.ndarray | None,
    top_k_indices: list[int],
    labels: list[str],
    categories: list[str],
) -> go.Figure:
    """
    3B — Animated similarity-search: query dot fires arcs to chunks;
    top-k light up gold. Four animation frames using go.Frame.
    Caller passes pre-computed LSA coords and cosine similarities.
    """
    n = len(labels)
    xyz = np.asarray(chunk_xyz)
    sims = np.asarray(all_sims).ravel() if all_sims is not None and len(all_sims) else np.zeros(n)
    cmap = {"Helpfulness": "#636EFA", "Safety": "#EF553B",
            "Honesty": "#00CC96", "Bias": "#AB63FA"}
    dot_colors = [cmap.get(c, "#888888") for c in categories]

    q_pos = ([0.0, 0.0, 0.0] if query_xyz is None or not np.asarray(query_xyz).size
              else np.asarray(query_xyz).reshape(-1)[:3].tolist())

    top_set = set(top_k_indices)

    def _chunk_scatter(colors, sizes, opacity=0.9) -> go.Scatter3d:
        return go.Scatter3d(
            x=xyz[:, 0].tolist(), y=xyz[:, 1].tolist(), z=xyz[:, 2].tolist(),
            mode="markers+text",
            marker=dict(size=sizes, color=colors, opacity=opacity,
                        line=dict(width=0.5, color="#333")),
            text=labels, textposition="top center", textfont=dict(size=9),
            hovertemplate="<b>%{text}</b><br>cosine=%{customdata:.3f}<extra></extra>",
            customdata=sims.tolist(),
            name="chunks", showlegend=False,
        )

    def _query_scatter(visible=True) -> go.Scatter3d:
        return go.Scatter3d(
            x=[q_pos[0]] if visible else [],
            y=[q_pos[1]] if visible else [],
            z=[q_pos[2]] if visible else [],
            mode="markers+text",
            marker=dict(size=16, color="#FF4444", symbol="diamond",
                        line=dict(width=1.5, color="#fff")),
            text=["Query"] if visible else [],
            textposition="bottom center",
            name="Query", showlegend=False,
        )

    def _arc_lines(indices, opacities) -> go.Scatter3d:
        lx: list = []; ly: list = []; lz: list = []
        lo: list = []
        for idx, op in zip(indices, opacities):
            lx += [q_pos[0], float(xyz[idx, 0]), None]
            ly += [q_pos[1], float(xyz[idx, 1]), None]
            lz += [q_pos[2], float(xyz[idx, 2]), None]
        color = f"rgba(255,200,0,0.5)" if indices else "rgba(0,0,0,0)"
        return go.Scatter3d(x=lx, y=ly, z=lz, mode="lines",
                            line=dict(color=color, width=3),
                            hoverinfo="skip", showlegend=False)

    # Frame 0: all grey chunks, no query, no lines
    f0 = [_chunk_scatter(["#888"] * n, [8] * n), _query_scatter(False),
          _arc_lines([], [])]

    # Frame 1: coloured chunks, red query dot appears
    f1 = [_chunk_scatter(dot_colors, [8] * n), _query_scatter(True),
          _arc_lines([], [])]

    # Frame 2: lines from query to ALL chunks (opacity = sim value)
    all_idx = list(range(n))
    ops = [max(0.1, float(sims[i])) for i in all_idx]
    f2 = [_chunk_scatter(dot_colors, [8] * n, opacity=0.5),
          _query_scatter(True), _arc_lines(all_idx, ops)]

    # Frame 3: top-k gold, others faded; lines only to top-k
    f3_colors = ["#FFD700" if i in top_set else "#444" for i in range(n)]
    f3_sizes = [14 if i in top_set else 6 for i in range(n)]
    f3_ops = [float(sims[i]) if i in top_set else 0.05 for i in range(n)]
    top_ops = [float(sims[i]) for i in top_k_indices]
    f3 = [_chunk_scatter(f3_colors, f3_sizes, opacity=0.92),
          _query_scatter(True), _arc_lines(top_k_indices, top_ops)]

    stage_titles = [
        "Stage 1 — All chunks in vector space (grey)",
        "Stage 2 — Query vector placed in the same space",
        "Stage 3 — Cosine similarity rays to every chunk",
        "Stage 4 — Top-k retrieved (gold); others fade",
    ]
    frames = [go.Frame(data=fd, name=str(i),
                       layout=go.Layout(title_text=stage_titles[i]))
              for i, fd in enumerate([f0, f1, f2, f3])]

    _play_btn = dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 1300, "redraw": True},
                                  "fromcurrent": True, "mode": "immediate"}])
    _pause_btn = dict(label="⏸ Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])

    fig = go.Figure(
        data=f0,
        frames=frames,
        layout=go.Layout(
            title=dict(text=stage_titles[0], x=0.02),
            scene=dict(xaxis_title="LSA dim 1", yaxis_title="LSA dim 2",
                       zaxis_title="LSA dim 3", aspectmode="data"),
            height=560,
            margin=dict(l=0, r=0, t=60, b=60),
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=-0.05, x=0.5, xanchor="center", yanchor="top",
                buttons=[_play_btn, _pause_btn],
            )],
            sliders=[dict(
                currentvalue={"prefix": "Stage: ", "font": {"size": 13}},
                x=0.1, len=0.8, y=0.0,
                steps=[dict(label=f"Stage {i + 1}", method="animate",
                            args=[[str(i)], {"frame": {"duration": 0, "redraw": True},
                                             "mode": "immediate"}])
                       for i in range(4)],
            )],
        ),
    )
    return fig


def figure_hallucination_timeline() -> go.Figure:
    """
    3C — Animated bar-race showing illustrative hallucination rates across
    model generations. Educational/approximate numbers for classroom use.
    Bars for RAG-augmented models are green; base models are orange/red.
    """
    models = [
        "GPT-2 (2019)", "GPT-3 (2020)", "InstructGPT (2022)",
        "GPT-3.5 (2023)", "GPT-4 (2023)", "GPT-4 + RAG", "Claude-Sonnet + RAG",
    ]
    hal_rates = [82, 65, 47, 33, 19, 7, 4]
    is_rag = [False, False, False, False, False, True, True]
    colors = ["#EF553B" if not r else "#00CC96" for r in is_rag]

    # Build incremental frames (bar race: reveal one model at a time)
    frames = []
    for k in range(1, len(models) + 1):
        frames.append(go.Frame(
            data=[go.Bar(
                x=hal_rates[:k],
                y=models[:k],
                orientation="h",
                marker_color=colors[:k],
                text=[f"{v}%" for v in hal_rates[:k]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Hallucination rate: %{x}%<extra></extra>",
            )],
            name=str(k),
            layout=go.Layout(title_text=f"Hallucination rate — {models[k-1]}"),
        ))

    _play_btn = dict(label="▶ Play", method="animate",
                     args=[None, {"frame": {"duration": 900, "redraw": True},
                                  "fromcurrent": False, "mode": "immediate"}])
    _pause_btn = dict(label="⏸ Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])

    fig = go.Figure(
        data=[go.Bar(
            x=hal_rates[:1], y=models[:1], orientation="h",
            marker_color=colors[:1],
            text=[f"{hal_rates[0]}%"], textposition="outside",
        )],
        frames=frames,
        layout=go.Layout(
            title=dict(
                text="Illustrative hallucination rate by model generation "
                     "(approximate, for classroom discussion)",
                x=0.02,
            ),
            xaxis=dict(title="Estimated hallucination rate (%)", range=[0, 95]),
            yaxis=dict(autorange="reversed"),
            height=420,
            margin=dict(l=180, r=60, t=70, b=50),
            shapes=[dict(
                type="line", x0=15, x1=15, y0=-0.5, y1=6.5,
                line=dict(color="#4466FF", width=2, dash="dot"),
            )],
            annotations=[dict(
                x=15, y=3.5, text="RAG introduced →", showarrow=False,
                font=dict(color="#4466FF", size=11),
            )],
            updatemenus=[dict(
                type="buttons", showactive=False,
                y=-0.12, x=0.5, xanchor="center", yanchor="top",
                buttons=[_play_btn, _pause_btn],
            )],
            sliders=[dict(
                currentvalue={"prefix": "Model: ", "font": {"size": 12}},
                x=0.1, len=0.8, y=-0.04,
                steps=[dict(label=m[:14], method="animate",
                            args=[[str(i + 1)],
                                  {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}])
                       for i, m in enumerate(models)],
            )],
        ),
    )
    return fig


def figure_retrieval_gauge(margin: float, entropy_bits: float = 0.0,
                           template: str | None = None) -> go.Figure:
    """
    4A — Gauge + number indicator for retrieval confidence (top1 − top2 margin).
    Green zone: HIGH confidence; orange: MEDIUM; red: LOW.
    """
    if margin >= 0.15:
        label = "HIGH confidence"
        bar_color = "#00CC96"
    elif margin >= 0.05:
        label = "MEDIUM confidence"
        bar_color = "#FFA500"
    else:
        label = "LOW confidence"
        bar_color = "#EF553B"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(margin, 4),
        number={"suffix": "", "font": {"size": 28}},
        title={"text": f"Retrieval Margin (top1 − top2)<br><span style='font-size:13px'>{label}</span>"},
        delta={"reference": 0.10, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 0.5], "tickformat": ".2f", "nticks": 6},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ccc",
            "steps": [
                {"range": [0, 0.05], "color": "rgba(239,85,59,0.18)"},
                {"range": [0.05, 0.15], "color": "rgba(255,165,0,0.18)"},
                {"range": [0.15, 0.5], "color": "rgba(0,204,150,0.18)"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.8,
                "value": margin,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(t=60, b=20, l=30, r=30))
    return _themed(fig, template)
