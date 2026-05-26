from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from .config import CREATIVITY_COLOR_MAP, CREATIVITY_ORDER, KAPPA_COLOR_MAP, KAPPA_ORDER


def build_line_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    facet_by: str | None,
    estimator: str,
    series_param_a: str,
    series_param_b: str | None,
    line_color_by: str,
    show_std: bool,
    publication_style: bool = True,
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

    def _short_label(value: Any) -> str:
        text = "" if value is None else str(value)
        if "/" in text:
            text = text.split("/")[-1]
        return text.strip()

    work = df.copy()
    series_params = [series_param_a]
    if series_param_b:
        series_params.append(series_param_b)
    series_params = [
        c
        for c in series_params
        if c in work.columns and c not in {x_axis, line_color_by, facet_by}
    ]
    if not series_params:
        work["_series_id"] = "default"
    else:
        first = series_params[0]
        work["_series_id"] = work[first].map(_fmt).astype(str)
        for col in series_params[1:]:
            work["_series_id"] = work["_series_id"] + " · " + work[col].map(_fmt).astype(str)

    group_cols = [x_axis, line_color_by, "_series_id"]
    if facet_by and facet_by not in group_cols:
        group_cols.append(facet_by)

    grouped = (
        work.groupby(group_cols, dropna=False, observed=False)[y_axis]
        .agg(
            value="mean" if estimator == "mean" else "median",
            std="std",
            n="count",
        )
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)
    unique_series = sorted(grouped["_series_id"].astype(str).unique().tolist())
    series_code_map = {sid: f"S{i + 1}" for i, sid in enumerate(unique_series)}
    single_series = len(series_code_map) == 1
    grouped["_series_code"] = grouped["_series_id"].map(series_code_map)
    grouped["_line_id"] = grouped[line_color_by].astype(str) + " | " + grouped["_series_code"].astype(str)

    dash_cycle = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    symbol_cycle = ["circle", "square", "diamond", "triangle-up", "x", "cross"]
    line_dash_map = {
        code: dash_cycle[idx % len(dash_cycle)]
        for idx, code in enumerate(sorted(grouped["_series_code"].astype(str).unique().tolist()))
    }
    symbol_map = {
        code: symbol_cycle[idx % len(symbol_cycle)]
        for idx, code in enumerate(sorted(grouped["_series_code"].astype(str).unique().tolist()))
    }

    hover_data: dict[str, Any] = {"n": True, "_series_code": True, "_series_id": True}
    hover_data["std"] = True if show_std else False

    if line_color_by == "creativity_level":
        color_map = CREATIVITY_COLOR_MAP
    elif line_color_by == "kappa_level":
        color_map = KAPPA_COLOR_MAP
    else:
        color_map = None
    fig = px.line(
        grouped,
        x=x_axis,
        y="value",
        color=line_color_by,
        line_dash="_series_code",
        line_dash_map=line_dash_map,
        symbol="_series_code",
        symbol_map=symbol_map,
        line_group="_line_id",
        facet_col=facet_by,
        error_y="std" if show_std else None,
        markers=True,
        hover_data=hover_data,
        category_orders={"creativity_level": CREATIVITY_ORDER, "kappa_level": KAPPA_ORDER},
        color_discrete_map=color_map,
    )

    for trace in fig.data:
        raw_name = str(trace.name)
        parts = [p.strip() for p in raw_name.split(",")]
        color_value = _short_label(parts[0]) if parts else _short_label(raw_name)
        series_code = parts[1] if len(parts) > 1 else ""
        if single_series:
            trace.name = color_value
        else:
            trace.name = f"{color_value} · {series_code}" if series_code else color_value
        trace.line.width = 2.2 if publication_style else 1.8
        trace.marker.size = 7 if publication_style else 6

    if facet_by:
        for ann in fig.layout.annotations:
            ann.text = ann.text.replace(f"{facet_by}=", "")

    fig.update_layout(
        template="plotly_white",
        xaxis_title=x_axis,
        yaxis_title=f"{y_axis} ({estimator})" + (" ± std" if show_std else ""),
        legend_title=_short_label(line_color_by) if single_series else f"{_short_label(line_color_by)} · Série",
    )
    if publication_style:
        fig.update_layout(
            font={"size": 15},
            legend={
                "x": 1.01,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.8)",
                "itemsizing": "constant",
            },
            margin={"l": 50, "r": 220, "t": 40, "b": 45},
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black", zeroline=False)
    return fig


def build_distribution_plot(
    df: pd.DataFrame,
    chart_type: str,
    x_axis: str,
    y_axis: str,
    color_by: str | None,
    facet_by: str | None,
    show_points: bool,
    publication_style: bool = False,
) -> Any:
    def _short_label(value: Any) -> str:
        text = "" if value is None else str(value)
        if "/" in text:
            text = text.split("/")[-1]
        return text.strip()

    points_mode = "all" if show_points else False
    _cat_orders = {"creativity_level": CREATIVITY_ORDER, "kappa_level": KAPPA_ORDER}
    if chart_type == "box":
        fig = px.box(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            points=points_mode,
            category_orders=_cat_orders,
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
            category_orders=_cat_orders,
        )
    if facet_by:
        for ann in fig.layout.annotations:
            ann.text = ann.text.replace(f"{facet_by}=", "")
    for trace in fig.data:
        trace.name = _short_label(trace.name)
        if chart_type == "box":
            trace.line.width = 2 if publication_style else 1.2
        else:
            trace.line.width = 1.8 if publication_style else 1.0
    fig.update_layout(
        template="plotly_white" if publication_style else None,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title=_short_label(color_by) if color_by else "",
    )
    if publication_style:
        fig.update_layout(
            font={"size": 15},
            legend={
                "x": 1.01,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.8)",
                "itemsizing": "constant",
            },
            margin={"l": 50, "r": 220, "t": 40, "b": 45},
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black", zeroline=False)
    return fig


def build_points_plot(
    df: pd.DataFrame,
    x_axis: str,
    y_axis: str,
    color_by: str | None,
    facet_by: str | None,
    publication_style: bool = False,
) -> Any:
    def _short_label(value: Any) -> str:
        text = "" if value is None else str(value)
        if "/" in text:
            text = text.split("/")[-1]
        return text.strip()

    if color_by == "creativity_level":
        color_map = CREATIVITY_COLOR_MAP
    elif color_by == "kappa_level":
        color_map = KAPPA_COLOR_MAP
    else:
        color_map = None
    _cat_orders = {"creativity_level": CREATIVITY_ORDER, "kappa_level": KAPPA_ORDER}

    if pd.api.types.is_numeric_dtype(df[x_axis]):
        fig = px.scatter(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            opacity=0.45,
            category_orders=_cat_orders,
            color_discrete_map=color_map,
        )
        fig.update_traces(marker={"size": 7 if publication_style else 6})
    else:
        fig = px.strip(
            df,
            x=x_axis,
            y=y_axis,
            color=color_by,
            facet_col=facet_by,
            stripmode="overlay",
            category_orders=_cat_orders,
            color_discrete_map=color_map,
        )
        fig.update_traces(
            jitter=0.30 if publication_style else 0.35,
            marker={"opacity": 0.6 if publication_style else 0.55, "size": 7 if publication_style else 6},
        )

    if facet_by:
        for ann in fig.layout.annotations:
            ann.text = ann.text.replace(f"{facet_by}=", "")
    for trace in fig.data:
        trace.name = _short_label(trace.name)
    fig.update_layout(
        template="plotly_white" if publication_style else None,
        xaxis_title=x_axis,
        yaxis_title=y_axis,
        legend_title=_short_label(color_by) if color_by else "",
    )
    if publication_style:
        fig.update_layout(
            font={"size": 15},
            legend={
                "x": 1.01,
                "xanchor": "left",
                "y": 1.0,
                "yanchor": "top",
                "bgcolor": "rgba(255,255,255,0.8)",
                "itemsizing": "constant",
            },
            margin={"l": 50, "r": 220, "t": 40, "b": 45},
        )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black", zeroline=False)
    return fig



from .plotting_publication import (
    SPEARMAN_FAMILY_COLOR_MAP,
    build_claim_density_vs_hallucination_plot,
    build_mediation_path_diagram,
    build_per_model_temperature_effect_plot,
    build_prompt_variance_icc_plot,
    build_spearman_forest_plot,
)
