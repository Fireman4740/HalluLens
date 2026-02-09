from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .config import CREATIVITY_COLOR_MAP, CREATIVITY_ORDER


def _empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 14},
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def build_creativity_scatter(
    df: pd.DataFrame,
    x_col: str = "creativity_composite",
    y_col: str = "hallucination_rate",
    color_by: str = "creativity_level",
    facet_by: str | None = None,
) -> Any:
    needed = {x_col, y_col}
    if not needed.issubset(df.columns):
        return _empty_figure("Insufficient data for scatter plot.")
    sub = df[[x_col, y_col, color_by] + ([facet_by] if facet_by else [])].dropna(subset=[x_col, y_col])
    if sub.empty:
        return _empty_figure("No rows available after filters.")
    fig = px.scatter(
        sub,
        x=x_col,
        y=y_col,
        color=color_by if color_by in sub.columns else None,
        facet_col=facet_by if facet_by in sub.columns else None,
        opacity=0.7,
        category_orders={"creativity_level": CREATIVITY_ORDER},
        color_discrete_map=CREATIVITY_COLOR_MAP if color_by == "creativity_level" else None,
    )
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=color_by if color_by in sub.columns else "",
    )
    return fig


def build_intent_boxplot(
    df: pd.DataFrame,
    y_col: str = "hallucination_rate",
) -> Any:
    if "creativity_level" not in df.columns or y_col not in df.columns:
        return _empty_figure("Insufficient data for intent boxplot.")
    sub = df[["creativity_level", y_col]].dropna()
    if sub.empty:
        return _empty_figure("No rows available after filters.")

    fig = px.box(
        sub,
        x="creativity_level",
        y=y_col,
        color="creativity_level",
        category_orders={"creativity_level": CREATIVITY_ORDER},
        color_discrete_map=CREATIVITY_COLOR_MAP,
        points="all",
    )
    fig.update_layout(
        xaxis_title="creativity_level",
        yaxis_title=y_col,
        legend_title="creativity_level",
    )
    return fig


def build_creativity_heatmap(
    df: pd.DataFrame,
    corr_cols: list[str],
    corr_method: str = "spearman",
) -> Any:
    available = [c for c in corr_cols if c in df.columns]
    if len(available) < 2:
        return _empty_figure("Insufficient numeric columns for correlation heatmap.")
    sub = df[available].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(how="all")
    if sub.empty:
        return _empty_figure("No numeric data available for heatmap.")
    corr = sub.corr(method=corr_method)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.to_numpy(),
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar={"title": f"{corr_method.title()} rho"},
        )
    )
    for i, y_label in enumerate(corr.index):
        for j, x_label in enumerate(corr.columns):
            fig.add_annotation(
                x=x_label,
                y=y_label,
                text=f"{corr.iloc[i, j]:.2f}",
                showarrow=False,
                font={"size": 10, "color": "black"},
            )
    fig.update_layout(
        xaxis_title="Metric",
        yaxis_title="Metric",
    )
    return fig


def build_glm_forest(glm_table: pd.DataFrame, top_n: int = 20) -> Any:
    needed = {"term", "odds_ratio", "ci_low", "ci_high", "p_value"}
    if glm_table.empty or not needed.issubset(glm_table.columns):
        return _empty_figure("No GLM output available.")

    sub = glm_table.copy()
    sub = sub[np.isfinite(sub["odds_ratio"]) & np.isfinite(sub["ci_low"]) & np.isfinite(sub["ci_high"])]
    if sub.empty:
        return _empty_figure("No finite GLM coefficients available.")
    sub = sub.sort_values("p_value").head(top_n).sort_values("odds_ratio")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["odds_ratio"],
            y=sub["term"],
            mode="markers",
            marker={"size": 9},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (sub["ci_high"] - sub["odds_ratio"]).clip(lower=0),
                "arrayminus": (sub["odds_ratio"] - sub["ci_low"]).clip(lower=0),
            },
            hovertemplate=(
                "term=%{y}<br>OR=%{x:.3f}<br>CI=[%{customdata[0]:.3f}, %{customdata[1]:.3f}]"
                "<br>p=%{customdata[2]:.3g}<extra></extra>"
            ),
            customdata=np.column_stack([sub["ci_low"], sub["ci_high"], sub["p_value"]]),
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Odds ratio",
        yaxis_title="GLM term",
        showlegend=False,
        height=max(400, 24 * len(sub) + 80),
    )
    return fig


def build_creativity_metrics_by_level_plot(
    df: pd.DataFrame,
    metric_cols: list[str],
    chart_type: str = "box",
) -> Any:
    if "creativity_level" not in df.columns:
        return _empty_figure("Insufficient data for creativity-level metric plot.")
    available = [c for c in metric_cols if c in df.columns]
    if not available:
        return _empty_figure("No creativity metric selected.")

    work = df[["creativity_level"] + available].copy()
    long_df = work.melt(
        id_vars=["creativity_level"],
        value_vars=available,
        var_name="creativity_metric",
        value_name="metric_value",
    ).dropna(subset=["metric_value", "creativity_level"])
    if long_df.empty:
        return _empty_figure("No rows available after filters.")

    if chart_type == "violin":
        fig = px.violin(
            long_df,
            x="metric_value",
            y="creativity_level",
            color="creativity_level",
            orientation="h",
            facet_col="creativity_metric",
            facet_col_wrap=3,
            box=True,
            points="all",
            category_orders={"creativity_level": CREATIVITY_ORDER},
            color_discrete_map=CREATIVITY_COLOR_MAP,
        )
    else:
        fig = px.box(
            long_df,
            x="metric_value",
            y="creativity_level",
            color="creativity_level",
            orientation="h",
            facet_col="creativity_metric",
            facet_col_wrap=3,
            points="all",
            category_orders={"creativity_level": CREATIVITY_ORDER},
            color_discrete_map=CREATIVITY_COLOR_MAP,
        )

    fig.update_layout(
        xaxis_title="Creativity metric value",
        yaxis_title="creativity_level",
        legend_title="creativity_level",
    )
    return fig
