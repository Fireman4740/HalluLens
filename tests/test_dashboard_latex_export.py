from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.latex_export import build_main_chart_line_latex_groupplot


def test_build_main_chart_line_latex_groupplot_returns_groupplot_tex() -> None:
    chart_table_df = pd.DataFrame(
        {
            "creativity_level": [
                "FACTUAL",
                "HYBRID",
                "VERY_CREATIVE",
                "FACTUAL",
                "HYBRID",
                "VERY_CREATIVE",
            ]
            * 3,
            "task": (
                ["INTERVIEW"] * 6
                + ["LESSON_PLAN"] * 6
                + ["NEWS_ARTICLE"] * 6
            ),
            "model_name": (
                ["deepseek-v3.2"] * 3
                + ["gemini-3-flash-preview"] * 3
            )
            * 3,
            "hallucination_rate_median": [
                0.27,
                0.40,
                0.60,
                0.32,
                0.41,
                0.67,
                0.55,
                0.65,
                0.82,
                0.52,
                0.66,
                0.78,
                0.43,
                0.50,
                0.68,
                0.46,
                0.56,
                0.65,
            ],
            "hallucination_rate_mean": [
                0.28,
                0.41,
                0.61,
                0.33,
                0.42,
                0.68,
                0.56,
                0.66,
                0.83,
                0.53,
                0.67,
                0.79,
                0.44,
                0.51,
                0.69,
                0.47,
                0.57,
                0.66,
            ],
            "n": [20] * 18,
        }
    )
    chart_cfg = {
        "chart_type": "line",
        "x_axis": "creativity_level",
        "y_axis": "hallucination_rate",
        "line_estimator": "median",
        "line_color_by": "model_name",
        "facet_by": "task",
    }

    tex = build_main_chart_line_latex_groupplot(chart_table_df, chart_cfg)

    assert tex
    assert "\\begin{groupplot}" in tex
    assert "group size=3 by 1" in tex
    assert "title={INTERVIEW}" in tex
    assert "title={LESSON\\_PLAN}" in tex
    assert "title={NEWS\\_ARTICLE}" in tex
    assert "xticklabels={FACTUAL,HYBRID,VERY\\_CREATIVE}" in tex
    assert "\\addlegendentry{deepseek-v3.2}" in tex
    assert "\\addlegendentry{gemini-3-flash-preview}" in tex


def test_build_main_chart_line_latex_groupplot_returns_empty_for_invalid_cfg() -> None:
    chart_table_df = pd.DataFrame({"x": [1], "y": [2]})
    assert build_main_chart_line_latex_groupplot(chart_table_df, {"chart_type": "points"}) == ""
    assert (
        build_main_chart_line_latex_groupplot(
            chart_table_df,
            {"chart_type": "line", "x_axis": "x", "y_axis": "hallucination_rate", "line_color_by": "model"},
        )
        == ""
    )


def test_build_main_chart_line_latex_groupplot_orders_numeric_x_ascending() -> None:
    chart_table_df = pd.DataFrame(
        {
            "length_words": [1000, 250, 500, 1000, 250, 500],
            "task": ["INTERVIEW"] * 6,
            "model_name": ["model-a"] * 3 + ["model-b"] * 3,
            "hallucination_rate_median": [0.44, 0.40, 0.42, 0.55, 0.46, 0.50],
            "n": [8] * 6,
        }
    )
    chart_cfg = {
        "chart_type": "line",
        "x_axis": "length_words",
        "y_axis": "hallucination_rate",
        "line_estimator": "median",
        "line_color_by": "model_name",
        "facet_by": "task",
    }

    tex = build_main_chart_line_latex_groupplot(chart_table_df, chart_cfg)
    assert "xticklabels={250,500,1000}" in tex
