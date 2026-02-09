from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from .config import CREATIVITY_COLOR_MAP, CREATIVITY_ORDER


def build_line_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    facet_by: str | None,
    estimator: str,
    series_param_a: str,
    series_param_b: str,
    line_color_by: str,
    show_std: bool,
) -> Any:
    def _fmt(v: Any) -> str:
        if pd.isna(v):
            return "NA"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            if float(v).is_integer():
                return str(int(v))
            return f"{float(v):.3f}".rstrip("0").rstrip(".")
        return str(v)

    work = df.copy()
    work["_series_id"] = (
        f"{series_param_a}="
        + work[series_param_a].map(_fmt).astype(str)
        + " | "
        + f"{series_param_b}="
        + work[series_param_b].map(_fmt).astype(str)
    )

    group_cols = [x_axis, line_color_by, "_series_id"]
    if facet_by and facet_by not in group_cols:
        group_cols.append(facet_by)

    grouped = (
        work.groupby(group_cols, dropna=False)[y_axis]
        .agg(
            value="mean" if estimator == "mean" else "median",
            std="std",
            n="count",
        )
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    grouped["_line_id"] = grouped[line_color_by].astype(str) + " | " + grouped["_series_id"].astype(str)

    color_map = CREATIVITY_COLOR_MAP if line_color_by == "creativity_level" else None
    fig = px.line(
        grouped,
        x=x_axis,
        y="value",
        color=line_color_by,
        line_dash="_series_id",
        line_group="_line_id",
        facet_col=facet_by,
        error_y="std" if show_std else None,
        markers=True,
        hover_data={"n": True, "std": True, "_series_id": True},
        category_orders={"creativity_level": CREATIVITY_ORDER},
        color_discrete_map=color_map,
    )
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=f"{y_axis} ({estimator})" + (" ± std" if show_std else ""),
        legend_title=f"{line_color_by} / Paramètres",
    )
    return fig


def build_distribution_plot(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: str,
    y_axis: str,
    color_by: str | None,
    facet_by: str | None,
    show_points: bool,
) -> Any:
    points_mode = "all" if show_points else False
    if chart_type == "box":
        fig = px.box(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            points=points_mode,
            category_orders={"creativity_level": CREATIVITY_ORDER},
        )
    else:
        fig = px.violin(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            points=points_mode,
            box=True,
            category_orders={"creativity_level": CREATIVITY_ORDER},
        )
    fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, legend_title=color_by if color_by else "")
    return fig


def build_points_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    color_by: str | None,
    facet_by: str | None,
) -> Any:
    color_map = CREATIVITY_COLOR_MAP if color_by == "creativity_level" else None

    if pd.api.types.is_numeric_dtype(df[x_axis]):
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            opacity=0.45,
            category_orders={"creativity_level": CREATIVITY_ORDER},
            color_discrete_map=color_map,
        )
        fig.update_traces(marker={"size": 6})
    else:
        fig = px.strip(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            stripmode="overlay",
            category_orders={"creativity_level": CREATIVITY_ORDER},
            color_discrete_map=color_map,
        )
        fig.update_traces(jitter=0.35, marker={"opacity": 0.55, "size": 6})

    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title=color_by if color_by else "",
    )
    return fig
