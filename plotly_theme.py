"""Apply Plotly template + translucent ash / slate surfaces so Light mode matches the UI (no black plot boxes)."""

from __future__ import annotations

import plotly.graph_objects as go


def apply_plotly_theme(fig: go.Figure, template: str) -> go.Figure:
    """
    Streamlit [theme] is dark; without this, figures inherit dark paper/plot in Light appearance.
    Light: ash paper, white-ish plot area, slate axis labels. Dark: soft translucent navy.
    """
    if template == "plotly_white":
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(241, 245, 249, 0.96)",
            plot_bgcolor="rgba(255, 255, 255, 0.82)",
            font=dict(color="#0f172a"),
            title_font=dict(color="#0f172a"),
            legend=dict(font=dict(color="#334155")),
        )
        fig.update_xaxes(
            color="#334155",
            gridcolor="rgba(148, 163, 184, 0.4)",
            zerolinecolor="rgba(100, 116, 139, 0.35)",
            linecolor="rgba(71, 85, 105, 0.35)",
        )
        fig.update_yaxes(
            color="#334155",
            gridcolor="rgba(148, 163, 184, 0.4)",
            zerolinecolor="rgba(100, 116, 139, 0.35)",
            linecolor="rgba(71, 85, 105, 0.35)",
        )
        if any(getattr(tr, "type", None) == "scatterpolar" for tr in fig.data):
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    radialaxis=dict(
                        color="#334155",
                        gridcolor="rgba(148, 163, 184, 0.45)",
                        linecolor="rgba(100, 116, 139, 0.35)",
                    ),
                    angularaxis=dict(
                        color="#334155",
                        gridcolor="rgba(148, 163, 184, 0.35)",
                        linecolor="rgba(100, 116, 139, 0.3)",
                    ),
                )
            )
        if any(getattr(tr, "type", None) == "scatter3d" for tr in fig.data):
            ax = dict(
                backgroundcolor="rgba(255, 255, 255, 0.88)",
                color="#334155",
                gridcolor="rgba(148, 163, 184, 0.35)",
                showbackground=True,
            )
            fig.update_layout(
                scene=dict(
                    bgcolor="rgba(241, 245, 249, 0.95)",
                    xaxis=ax,
                    yaxis=ax,
                    zaxis=ax,
                )
            )
        if any(getattr(tr, "type", None) == "heatmap" for tr in fig.data):
            fig.update_layout(coloraxis=dict(colorbar=dict(tickfont=dict(color="#334155"), title_font=dict(color="#334155"))))
    else:
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(12, 17, 24, 0.82)",
            plot_bgcolor="rgba(15, 23, 42, 0.55)",
            font=dict(color="#e5e7eb"),
            title_font=dict(color="#f1f5f9"),
            legend=dict(font=dict(color="#cbd5e1")),
        )
        fig.update_xaxes(
            color="#cbd5e1",
            gridcolor="rgba(148, 163, 184, 0.25)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
        )
        fig.update_yaxes(
            color="#cbd5e1",
            gridcolor="rgba(148, 163, 184, 0.25)",
            zerolinecolor="rgba(148, 163, 184, 0.2)",
        )
        if any(getattr(tr, "type", None) == "scatterpolar" for tr in fig.data):
            fig.update_layout(
                polar=dict(
                    bgcolor="rgba(15, 23, 42, 0.45)",
                    radialaxis=dict(color="#cbd5e1", gridcolor="rgba(148, 163, 184, 0.3)"),
                    angularaxis=dict(color="#cbd5e1", gridcolor="rgba(148, 163, 184, 0.25)"),
                )
            )
        if any(getattr(tr, "type", None) == "scatter3d" for tr in fig.data):
            ax = dict(
                backgroundcolor="rgba(15, 23, 42, 0.5)",
                color="#cbd5e1",
                gridcolor="rgba(148, 163, 184, 0.25)",
                showbackground=True,
            )
            fig.update_layout(scene=dict(bgcolor="rgba(12, 17, 24, 0.75)", xaxis=ax, yaxis=ax, zaxis=ax))
    return fig
