from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def build_task_model_parameter_corr_plot(
    corr_df: pd.DataFrame,
    metric_order: list[str] | None = None,
) -> Any:
    required = {"model_name", "task", "metric", "r", "n"}
    if corr_df.empty or not required.issubset(corr_df.columns):
        return _empty_figure("Insufficient data for task/model correlation comparison.")

    work = corr_df.copy()
    work = work.dropna(subset=["model_name", "task", "metric", "r"])
    if work.empty:
        return _empty_figure("No rows available after filters.")

    if metric_order is None:
        metric_order = ["length_words", "temperature", "creativity_rank"]

    metric_values = [m for m in metric_order if m in work["metric"].astype(str).unique().tolist()]
    if not metric_values:
        metric_values = sorted(work["metric"].astype(str).unique().tolist())
    task_values = sorted(work["task"].astype(str).unique().tolist())
    model_values = sorted(work["model_name"].astype(str).unique().tolist())

    fig = px.bar(
        work,
        x="metric",
        y="r",
        color="model_name",
        facet_col="task",
        barmode="group",
        category_orders={
            "metric": metric_values,
            "task": task_values,
            "model_name": model_values,
        },
        hover_data={
            "metric": True,
            "task": True,
            "model_name": True,
            "n": True,
            "r": ":.3f",
            "p_value": ":.3g" if "p_value" in work.columns else False,
            "p_fdr_bh": ":.3g" if "p_fdr_bh" in work.columns else False,
        },
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_yaxes(range=[-1, 1], title_text="Spearman rho")
    fig.update_layout(
        xaxis_title="Parameter",
        legend_title="Model",
    )
    for annotation in fig.layout.annotations:
        txt = str(annotation.text)
        if txt.startswith("task="):
            annotation.text = txt.split("=", 1)[1]
    return fig


def build_task_parameter_heatmap_by_model(
    corr_df: pd.DataFrame,
    metric_order: list[str] | None = None,
) -> Any:
    required = {"model_name", "task", "metric", "r"}
    if corr_df.empty or not required.issubset(corr_df.columns):
        return _empty_figure("Insufficient data for task x parameter heatmap.")

    work = corr_df.copy()
    work = work.dropna(subset=["model_name", "task", "metric", "r"])
    if work.empty:
        return _empty_figure("No rows available after filters.")

    if metric_order is None:
        metric_order = ["length_words", "temperature", "creativity_rank"]

    metric_values = [m for m in metric_order if m in work["metric"].astype(str).unique().tolist()]
    if not metric_values:
        metric_values = sorted(work["metric"].astype(str).unique().tolist())
    task_values = sorted(work["task"].astype(str).unique().tolist())
    model_values = sorted(work["model_name"].astype(str).unique().tolist())

    fig = px.density_heatmap(
        work,
        x="metric",
        y="task",
        z="r",
        histfunc="avg",
        facet_col="model_name",
        category_orders={
            "metric": metric_values,
            "task": task_values,
            "model_name": model_values,
        },
        color_continuous_scale="RdBu",
        range_color=[-1, 1],
        hover_data={
            "metric": True,
            "task": True,
            "model_name": True,
            "r": ":.3f",
            "n": True if "n" in work.columns else False,
            "p_value": ":.3g" if "p_value" in work.columns else False,
            "p_fdr_bh": ":.3g" if "p_fdr_bh" in work.columns else False,
        },
    )
    fig.update_layout(
        xaxis_title="Parameter",
        yaxis_title="Task",
        coloraxis_colorbar={"title": "Spearman rho"},
    )
    for annotation in fig.layout.annotations:
        txt = str(annotation.text)
        if txt.startswith("model_name="):
            annotation.text = txt.split("=", 1)[1]
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


def build_creativity_score_distributions_plot(
    df: pd.DataFrame,
    ttct_col: str = "ttct_overall",
    ttcw_col: str = "ttcw_overall",
    level_col: str = "creativity_level",
    style: str = "violin",
    show_points: bool = False,
) -> tuple[Any, pd.DataFrame]:
    required = {ttct_col, ttcw_col, level_col}
    summary_cols = [
        "metric",
        "creativity_level",
        "n",
        "mean",
        "median",
        "std",
        "q25",
        "q75",
    ]
    if df.empty or not required.issubset(df.columns):
        return _empty_figure("Insufficient data for TTCT/TTCW distributions."), pd.DataFrame(columns=summary_cols)

    work = df[[level_col, ttct_col, ttcw_col]].copy()
    work[ttct_col] = pd.to_numeric(work[ttct_col], errors="coerce")
    work[ttcw_col] = pd.to_numeric(work[ttcw_col], errors="coerce")
    work[level_col] = work[level_col].astype(str)
    work = work.dropna(subset=[level_col], how="any")
    if work.empty:
        return _empty_figure("No rows available after filters."), pd.DataFrame(columns=summary_cols)

    observed_levels = sorted(work[level_col].dropna().astype(str).unique().tolist())
    level_order = [lvl for lvl in CREATIVITY_ORDER if lvl in observed_levels] + [
        lvl for lvl in observed_levels if lvl not in CREATIVITY_ORDER
    ]
    if not level_order:
        return _empty_figure("No creativity levels available."), pd.DataFrame(columns=summary_cols)

    style = str(style).strip().lower()
    if style not in {"violin", "ridgeline"}:
        style = "violin"

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.10,
        subplot_titles=["TTCT", "TTCW"],
    )

    metric_specs = [(ttct_col, "TTCT score", 1), (ttcw_col, "TTCW score", 2)]
    for metric_col, x_title, col_idx in metric_specs:
        for level in level_order:
            vals = work.loc[work[level_col].astype(str) == str(level), metric_col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            color = CREATIVITY_COLOR_MAP.get(str(level), "#666666")
            fig.add_trace(
                go.Violin(
                    x=vals,
                    y=[str(level)] * vals.size,
                    orientation="h",
                    name=str(level),
                    legendgroup=str(level),
                    showlegend=(col_idx == 1),
                    side="positive" if style == "ridgeline" else "both",
                    line={"color": color, "width": 1.2},
                    fillcolor=color,
                    opacity=0.32 if style == "ridgeline" else 0.60,
                    box_visible=(style == "violin"),
                    meanline={"visible": True, "color": "#222"},
                    points="all" if show_points else False,
                    marker={"size": 3.2, "opacity": 0.35},
                    jitter=0.03 if show_points else 0.0,
                    hovertemplate=(
                        f"{level_col}={str(level)}<br>"
                        + f"{metric_col}=%{{x:.3f}}<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )
        fig.update_xaxes(
            title_text=x_title,
            row=1,
            col=col_idx,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            linecolor="black",
        )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=level_order,
        title_text=level_col,
        row=1,
        col=1,
        showgrid=False,
        linecolor="black",
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=level_order,
        title_text="",
        row=1,
        col=2,
        showgrid=False,
        linecolor="black",
    )
    fig.update_layout(
        template="plotly_white",
        violinmode="overlay",
        title="Creativity Score Distributions by Creativity Level",
        font={"size": 15},
        margin={"l": 60, "r": 200, "t": 65, "b": 50},
        legend={
            "x": 1.01,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
            "itemsizing": "constant",
            "title": {"text": level_col},
        },
    )
    for annotation in fig.layout.annotations:
        annotation.font = {"size": 14}

    summary_rows: list[dict[str, Any]] = []
    for metric_col, _, _ in metric_specs:
        for level in level_order:
            vals = work.loc[work[level_col].astype(str) == str(level), metric_col].dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            summary_rows.append(
                {
                    "metric": metric_col,
                    "creativity_level": str(level),
                    "n": int(vals.size),
                    "mean": float(np.nanmean(vals)),
                    "median": float(np.nanmedian(vals)),
                    "std": float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "q25": float(np.nanquantile(vals, 0.25)),
                    "q75": float(np.nanquantile(vals, 0.75)),
                }
            )
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df["creativity_level"] = pd.Categorical(
            summary_df["creativity_level"],
            categories=level_order,
            ordered=True,
        )
        summary_df["metric"] = pd.Categorical(
            summary_df["metric"],
            categories=[ttct_col, ttcw_col],
            ordered=True,
        )
        summary_df = summary_df.sort_values(["metric", "creativity_level"]).reset_index(drop=True)
    return fig, summary_df[summary_cols] if not summary_df.empty else pd.DataFrame(columns=summary_cols)
