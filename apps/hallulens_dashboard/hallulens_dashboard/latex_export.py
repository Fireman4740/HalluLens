from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px

from .config import CREATIVITY_ORDER


TASK_FACET_ORDER = ["INTERVIEW", "LESSON_PLAN", "NEWS_ARTICLE"]


def _latex_escape(value: Any) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def _hex_from_plotly_color(color: str) -> str:
    raw = str(color).strip()
    if raw.startswith("#") and len(raw) == 7:
        return raw[1:].upper()
    m = re.fullmatch(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", raw, flags=re.IGNORECASE)
    if not m:
        return "1F77B4"
    r, g, b = (max(0, min(255, int(m.group(i)))) for i in (1, 2, 3))
    return f"{r:02X}{g:02X}{b:02X}"


def _ordered_values(values: list[str], key: str) -> list[str]:
    if key == "creativity_level":
        head = [v for v in CREATIVITY_ORDER if v in values]
        tail = [v for v in values if v not in CREATIVITY_ORDER]
        return head + sorted(tail)
    if key == "task":
        head = [v for v in TASK_FACET_ORDER if v in values]
        tail = [v for v in values if v not in TASK_FACET_ORDER]
        return head + sorted(tail)
    parsed: list[tuple[float, str]] = []
    for v in values:
        try:
            parsed.append((float(v), v))
        except Exception:
            parsed = []
            break
    if parsed:
        parsed.sort(key=lambda t: t[0])
        return [v for _, v in parsed]
    return sorted(values)


def build_main_chart_line_latex_groupplot(
    chart_table_df: pd.DataFrame,
    chart_cfg: dict[str, Any],
) -> str:
    if chart_table_df is None or chart_table_df.empty:
        return ""
    if str(chart_cfg.get("chart_type", "")).lower() != "line":
        return ""

    x_axis = str(chart_cfg.get("x_axis", ""))
    y_axis = str(chart_cfg.get("y_axis", ""))
    line_color_by = str(chart_cfg.get("line_color_by", ""))
    facet_by = chart_cfg.get("facet_by")
    estimator = "mean" if str(chart_cfg.get("line_estimator", "median")).lower() == "mean" else "median"
    y_value_col = f"{y_axis}_{estimator}"

    required = [x_axis, line_color_by, y_value_col]
    if isinstance(facet_by, str) and facet_by:
        required.append(facet_by)
    missing = [c for c in required if c not in chart_table_df.columns]
    if missing:
        return ""

    work = chart_table_df.copy()
    work[y_value_col] = pd.to_numeric(work[y_value_col], errors="coerce")
    work = work.dropna(subset=[x_axis, line_color_by, y_value_col])
    work = work[np.isfinite(work[y_value_col])].copy()
    if work.empty:
        return ""

    if isinstance(facet_by, str) and facet_by:
        work["_facet_key"] = work[facet_by].astype(str)
        facet_title = facet_by
    else:
        work["_facet_key"] = "all"
        facet_title = "facet"

    work["_x_key"] = work[x_axis].astype(str)
    work["_series_key"] = work[line_color_by].astype(str)

    x_values = _ordered_values(sorted(work["_x_key"].unique().tolist()), x_axis)
    facet_values = _ordered_values(sorted(work["_facet_key"].unique().tolist()), facet_title)
    series_values = _ordered_values(sorted(work["_series_key"].unique().tolist()), line_color_by)

    if not x_values or not facet_values or not series_values:
        return ""

    x_pos = {x: i + 1 for i, x in enumerate(x_values)}
    plotly_palette = px.colors.qualitative.Plotly
    series_colors = {
        series: _hex_from_plotly_color(plotly_palette[i % len(plotly_palette)])
        for i, series in enumerate(series_values)
    }
    series_color_name = {series: f"hlseries{i + 1}" for i, series in enumerate(series_values)}

    y_vals = work[y_value_col].to_numpy(dtype=float)
    y_min = float(np.nanmin(y_vals))
    y_max = float(np.nanmax(y_vals))
    span = max(y_max - y_min, 1e-6)
    pad = 0.05 * span
    y_lo = y_min - pad
    y_hi = y_max + pad

    if len(facet_values) <= 2:
        axis_width = "0.40\\textwidth"
    elif len(facet_values) == 3:
        axis_width = "0.29\\textwidth"
    else:
        axis_width = "0.23\\textwidth"

    xticks = ",".join(str(x_pos[x]) for x in x_values)
    xticklabels = ",".join(_latex_escape(x) for x in x_values)

    lines: list[str] = []
    lines.append("% Requires: \\usepackage{pgfplots}, \\usepgfplotslibrary{groupplots}, \\usepackage{xcolor}")
    lines.append("% Suggested: \\pgfplotsset{compat=1.18}")
    lines.append("\\begin{tikzpicture}")
    for series in series_values:
        lines.append(
            f"\\definecolor{{{series_color_name[series]}}}{{HTML}}{{{series_colors[series]}}}"
        )
    lines.append(
        "\\begin{groupplot}["
        + f"group style={{group size={len(facet_values)} by 1, horizontal sep=1.0cm}},"
        + f"width={axis_width},"
        + "height=0.30\\textwidth,"
        + "grid=major,"
        + "major grid style={draw=gray!20},"
        + "tick align=outside,"
        + f"xlabel={{{_latex_escape(x_axis)}}},"
        + f"ymin={y_lo:.6f},ymax={y_hi:.6f},"
        + f"xtick={{{xticks}}},"
        + f"xticklabels={{{xticklabels}}}"
        + "]"
    )

    for facet_idx, facet_value in enumerate(facet_values):
        axis_opts: list[str] = [f"title={{{_latex_escape(facet_value)}}}"]
        if facet_idx == 0:
            axis_opts.append(f"ylabel={{{_latex_escape(f'{y_axis} ({estimator})')}}}")
        if facet_idx == len(facet_values) - 1:
            axis_opts.append("legend style={at={(1.02,1.0)},anchor=north west,draw=none,fill=none,font=\\footnotesize}")
        lines.append("\\nextgroupplot[" + ",".join(axis_opts) + "]")

        facet_sub = work[work["_facet_key"] == facet_value].copy()
        for series in series_values:
            sub = facet_sub[facet_sub["_series_key"] == series].copy()
            if sub.empty:
                continue
            sub["_x_pos"] = sub["_x_key"].map(x_pos)
            sub = sub.sort_values("_x_pos")
            coords = " ".join(
                f"({int(row['_x_pos'])},{float(row[y_value_col]):.6f})"
                for _, row in sub.iterrows()
            )
            lines.append(
                "\\addplot+["
                + f"color={series_color_name[series]},"
                + "line width=1.1pt,mark=*,mark size=1.7pt"
                + f"] coordinates {{{coords}}};"
            )
            if facet_idx == len(facet_values) - 1:
                lines.append(f"\\addlegendentry{{{_latex_escape(series)}}}")

    lines.append("\\end{groupplot}")
    lines.append("\\end{tikzpicture}")
    return "\n".join(lines) + "\n"
