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
    build_creativity_score_distributions_plot,
    build_creativity_scatter,
    build_glm_forest,
    build_intent_boxplot,
    build_task_parameter_heatmap_by_model,
    build_task_model_parameter_corr_plot,
)
from hallulens_dashboard.plotting import (
    build_claim_density_vs_hallucination_plot,
    build_mediation_path_diagram,
    build_per_model_temperature_effect_plot,
    build_prompt_variance_icc_plot,
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


def test_build_creativity_score_distributions_plot_returns_figure_and_summary() -> None:
    fig, summary_df = build_creativity_score_distributions_plot(
        _sample_df(),
        ttct_col="ttct_overall",
        ttcw_col="ttcw_overall",
        level_col="creativity_level",
        style="violin",
        show_points=False,
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
    assert not summary_df.empty
    assert set(
        [
            "metric",
            "creativity_level",
            "n",
            "mean",
            "median",
            "std",
            "q25",
            "q75",
        ]
    ).issubset(summary_df.columns)
    assert set(summary_df["metric"].astype(str).tolist()) == {"ttct_overall", "ttcw_overall"}


def test_build_task_model_parameter_corr_plot_returns_non_empty_figure() -> None:
    corr_df = pd.DataFrame(
        {
            "model_name": ["grok-3-mini", "grok-3-mini", "mistral-small-creative", "mistral-small-creative"],
            "task": ["INTERVIEW", "NEWS_ARTICLE", "INTERVIEW", "NEWS_ARTICLE"],
            "metric": ["length_words", "temperature", "length_words", "temperature"],
            "r": [0.42, -0.21, 0.31, -0.18],
            "n": [24, 24, 24, 24],
            "p_value": [0.01, 0.11, 0.03, 0.15],
            "p_fdr_bh": [0.02, 0.11, 0.04, 0.15],
        }
    )
    fig = build_task_model_parameter_corr_plot(
        corr_df,
        metric_order=["length_words", "temperature", "creativity_rank"],
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_build_task_parameter_heatmap_by_model_returns_non_empty_figure() -> None:
    corr_df = pd.DataFrame(
        {
            "model_name": [
                "grok-3-mini",
                "grok-3-mini",
                "grok-3-mini",
                "mistral-small-creative",
                "mistral-small-creative",
                "mistral-small-creative",
            ],
            "task": [
                "INTERVIEW",
                "NEWS_ARTICLE",
                "LESSON_PLAN",
                "INTERVIEW",
                "NEWS_ARTICLE",
                "LESSON_PLAN",
            ],
            "metric": [
                "length_words",
                "temperature",
                "creativity_rank",
                "length_words",
                "temperature",
                "creativity_rank",
            ],
            "r": [0.42, -0.21, 0.18, 0.31, -0.18, 0.11],
            "n": [24, 24, 24, 24, 24, 24],
            "p_value": [0.01, 0.11, 0.09, 0.03, 0.15, 0.22],
            "p_fdr_bh": [0.02, 0.11, 0.10, 0.04, 0.15, 0.22],
        }
    )
    fig = build_task_parameter_heatmap_by_model(
        corr_df,
        metric_order=["length_words", "temperature", "creativity_rank"],
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_build_per_model_temperature_effect_plot_returns_figure_and_stats() -> None:
    rows: list[dict[str, float | str]] = []
    for model_name, base in [("grok-3-mini", 0.40), ("mistral-small-creative", 0.53)]:
        for temp in [0.25, 0.5, 0.75]:
            for i in range(12):
                rows.append(
                    {
                        "model_name": model_name,
                        "temperature": temp,
                        "hallucination_rate": float(base + 0.01 * (temp - 0.5) + 0.005 * ((i % 3) - 1)),
                    }
                )
    rng_df = pd.DataFrame(rows)
    fig, stats_df = build_per_model_temperature_effect_plot(
        rng_df,
        target_col="hallucination_rate",
        model_col="model_name",
        temperature_col="temperature",
        min_n_per_temp=5,
        n_boot=80,
        seed=11,
        n_cols=2,
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
    assert not stats_df.empty
    assert set(["model_name", "temperature", "n", "mean", "ci_low", "ci_high"]).issubset(stats_df.columns)


def test_build_claim_density_vs_hallucination_plot_returns_figure_and_summary() -> None:
    rows: list[dict[str, float | str]] = []
    for level, base_rate in [("FACTUAL", 0.38), ("HYBRID", 0.53), ("VERY_CREATIVE", 0.69)]:
        for i in range(40):
            response_len = float(120 + (i % 15))
            n_claims = float(4 + (i % 7))
            rows.append(
                {
                    "creativity_level": level,
                    "n_claims": n_claims,
                    "response_length_words": response_len,
                    "hallucination_rate": float(base_rate + 0.002 * (i % 9)),
                }
            )
    density_df = pd.DataFrame(rows)
    fig, summary_df = build_claim_density_vs_hallucination_plot(
        density_df,
        claims_col="n_claims",
        response_len_col="response_length_words",
        y_col="hallucination_rate",
        creativity_col="creativity_level",
        lowess_frac=0.45,
        min_points_for_lowess=10,
        marker_opacity=0.5,
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
    assert not summary_df.empty
    assert set(
        [
            "creativity_level",
            "n_points",
            "density_mean",
            "density_median",
            "hallucination_rate_mean",
            "hallucination_rate_median",
            "lowess_drawn",
        ]
    ).issubset(summary_df.columns)
    assert set(summary_df["creativity_level"].astype(str).tolist()) == {"FACTUAL", "HYBRID", "VERY_CREATIVE"}


def test_build_prompt_variance_icc_plot_returns_figure_prompt_means_and_summary() -> None:
    rows: list[dict[str, float | str]] = []
    for prompt_idx in range(40):
        prompt_id = f"p_{prompt_idx:03d}"
        base = 0.2 + 0.01 * (prompt_idx % 10)
        for run_idx in range(6):
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "hallucination_rate": float(base + 0.015 * ((run_idx % 3) - 1)),
                }
            )
    df = pd.DataFrame(rows)
    fig, prompt_means_df, summary_df = build_prompt_variance_icc_plot(
        df,
        target_col="hallucination_rate",
        prompt_col="prompt_id",
        n_bins=24,
        icc_value=0.595,
        show_kde=True,
        show_mixture=True,
        mixture_components=2,
        random_state=7,
    )
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
    assert not prompt_means_df.empty
    assert not summary_df.empty
    assert set(["prompt_id", "prompt_mean_hallucination_rate", "n_rows_per_prompt"]).issubset(prompt_means_df.columns)
    assert set(
        [
            "n_prompts",
            "grand_mean",
            "prompt_mean_std",
            "prompt_mean_var",
            "prompt_mean_q25",
            "prompt_mean_median",
            "prompt_mean_q75",
            "icc_annotated",
            "mixture_status",
            "mixture_components",
            "mixture_bic",
        ]
    ).issubset(summary_df.columns)
    assert int(summary_df.iloc[0]["n_prompts"]) == 40


def test_build_mediation_path_diagram_returns_figure_and_summary() -> None:
    mediation_df = pd.DataFrame(
        [
            {
                "n": 12469,
                "a_path": 0.0123,
                "a_p_value": 0.032,
                "b_path": 0.0204,
                "b_p_value": 0.014,
                "c_prime_direct": 0.1842,
                "c_prime_p_value": 1e-8,
                "c_total": 0.18445,
                "c_total_p_value": 1e-8,
                "indirect_ab": 0.00025,
                "indirect_ci_low": 0.00005,
                "indirect_ci_high": 0.00045,
                "indirect_p_boot": 0.012,
                "mediation_type": "partial_mediation",
            }
        ]
    )
    fig, summary_df = build_mediation_path_diagram(mediation_df)
    assert hasattr(fig, "data")
    assert not summary_df.empty
    assert set(
        [
            "n",
            "a_path",
            "a_p_value",
            "b_path",
            "b_p_value",
            "c_prime_direct",
            "c_prime_p_value",
            "c_total",
            "c_total_p_value",
            "indirect_ab",
            "indirect_ci_low",
            "indirect_ci_high",
            "indirect_p_boot",
            "direct_share_pct_abs",
            "indirect_share_pct_abs",
            "mediation_type",
        ]
    ).issubset(summary_df.columns)
