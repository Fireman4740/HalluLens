from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.creativity_plotting import (
    build_creativity_heatmap,
    build_creativity_metrics_by_level_plot,
    build_creativity_scatter,
    build_glm_forest,
    build_intent_boxplot,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "creativity_level": ["FACTUAL", "HYBRID", "VERY_CREATIVE", "FACTUAL", "HYBRID"],
            "hallucination_rate": [0.10, 0.22, 0.31, 0.12, 0.25],
            "ttct_overall": [2.2, 3.1, 4.4, 2.4, 3.0],
            "ttcw_overall": [3.8, 2.9, 2.1, 3.6, 2.7],
            "creativity_composite": [3.0, 3.0, 3.25, 3.0, 2.85],
            "task": ["INTERVIEW", "INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN", "NEWS_ARTICLE"],
            "length_words": [500, 500, 1000, 500, 1000],
        }
    )


def test_build_creativity_scatter_returns_non_empty_figure() -> None:
    fig = build_creativity_scatter(
        _sample_df(),
        x_col="creativity_composite",
        y_col="hallucination_rate",
        color_by="creativity_level",
        facet_by="task",
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_build_intent_boxplot_returns_non_empty_figure() -> None:
    fig = build_intent_boxplot(_sample_df(), y_col="hallucination_rate")
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_build_creativity_heatmap_returns_non_empty_figure() -> None:
    fig = build_creativity_heatmap(
        _sample_df(),
        corr_cols=["hallucination_rate", "ttct_overall", "ttcw_overall", "creativity_composite"],
        corr_method="spearman",
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_build_glm_forest_returns_non_empty_figure() -> None:
    glm_table = pd.DataFrame(
        {
            "term": ["Intercept", "C(creativity_level)[T.HYBRID]", "ttct_overall"],
            "odds_ratio": [0.9, 1.3, 1.1],
            "ci_low": [0.8, 1.1, 1.02],
            "ci_high": [1.0, 1.6, 1.2],
            "p_value": [0.2, 0.01, 0.03],
        }
    )
    fig = build_glm_forest(glm_table, top_n=10)
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_build_creativity_metrics_by_level_plot_returns_non_empty_figure() -> None:
    fig = build_creativity_metrics_by_level_plot(
        _sample_df(),
        metric_cols=["ttct_overall", "ttcw_overall", "creativity_composite"],
        chart_type="box",
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
