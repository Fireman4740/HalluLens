from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.analytics import build_spearman_forest_table  # noqa: E402
from hallulens_dashboard.plotting import build_spearman_forest_plot  # noqa: E402


def _sample_impact_df(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(2027)
    tasks = np.array(["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"])
    models = np.array(["grok-3-mini", "mistral-small-creative", "qwen3-30b-a3b"])
    levels = np.array(["FACTUAL", "HYBRID", "VERY_CREATIVE"])

    task = rng.choice(tasks, size=n, replace=True)
    model = rng.choice(models, size=n, replace=True)
    creativity_level = rng.choice(levels, size=n, replace=True)
    creativity_rank = np.vectorize({"FACTUAL": 0.0, "HYBRID": 1.0, "VERY_CREATIVE": 2.0}.get)(creativity_level)
    temperature = rng.choice([0.25, 0.5, 0.75], size=n, replace=True).astype(float)
    length_words = rng.choice([250, 500, 1000], size=n, replace=True).astype(float)
    response_length_words = length_words * rng.uniform(0.6, 1.2, size=n)
    response_length_tokens = response_length_words * rng.uniform(1.1, 1.6, size=n)
    prompt_length_words = rng.uniform(8, 35, size=n)
    n_claims = rng.poisson(8, size=n).astype(float)
    support_rate = np.clip(0.7 - 0.08 * creativity_rank + rng.normal(0, 0.07, size=n), 0, 1)

    task_effect = np.where(task == "LESSON_PLAN", 0.18, np.where(task == "INTERVIEW", -0.12, 0.0))
    model_effect = np.where(model == "mistral-small-creative", 0.05, np.where(model == "grok-3-mini", -0.02, 0.0))
    hallu = (
        0.46
        + task_effect
        + 0.10 * creativity_rank
        + 0.03 * (length_words / 1000.0)
        - 0.05 * temperature
        + model_effect
        + rng.normal(0, 0.05, size=n)
    )
    hallu = np.clip(hallu, 0, 1)

    return pd.DataFrame(
        {
            "hallucination_rate": hallu,
            "task": task,
            "model_name": model,
            "creativity_level": creativity_level,
            "creativity_rank": creativity_rank.astype(float),
            "temperature": temperature,
            "length_words": length_words,
            "response_length_words": response_length_words,
            "response_length_tokens": response_length_tokens,
            "prompt_length_words": prompt_length_words,
            "n_claims": n_claims,
            "support_rate": support_rate,
            "root_name": "longwiki-hybrid",
        }
    )


def test_build_spearman_forest_table_returns_ranked_rows_with_ci_and_families() -> None:
    df = _sample_impact_df()
    out = build_spearman_forest_table(
        df,
        target_col="hallucination_rate",
        min_n=20,
        n_boot=200,
        seed=17,
    )
    assert not out.empty
    assert set(
        [
            "factor",
            "modality",
            "spearman_rho",
            "abs_rho",
            "ci_low",
            "ci_high",
            "p_value",
            "factor_family",
            "row_label",
        ]
    ).issubset(out.columns)
    assert out["abs_rho"].is_monotonic_decreasing
    assert out["factor_family"].isin(["task", "creativity", "length", "temperature", "model", "other"]).all()
    finite_ci = out.dropna(subset=["ci_low", "ci_high"])
    if not finite_ci.empty:
        assert (finite_ci["ci_low"] <= finite_ci["ci_high"]).all()


def test_build_spearman_forest_plot_returns_non_empty_figure() -> None:
    df = _sample_impact_df()
    out = build_spearman_forest_table(
        df,
        target_col="hallucination_rate",
        min_n=20,
        n_boot=120,
        seed=5,
    )
    fig = build_spearman_forest_plot(out, top_n=20)
    assert hasattr(fig, "data")
    assert len(fig.data) > 0
