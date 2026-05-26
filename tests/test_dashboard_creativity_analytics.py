from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.creativity_analytics import (
    HAVE_STATSMODELS,
    build_creativity_corr_table,
    build_mediation_table,
    build_partial_corr_table,
    build_task_model_parameter_corr_table,
    fit_binomial_glm_stratified_by_model,
    fit_prompt_mixedlm_stratified_by_model,
    assess_creativity_level_homogeneity,
    fit_prompt_mixedlm,
    fit_binomial_glm,
)


def _make_corr_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    ttct = rng.normal(3.0, 0.6, n)
    ttcw = rng.normal(2.8, 0.7, n)
    hallu = 0.25 + 0.08 * ttct - 0.05 * ttcw + rng.normal(0, 0.03, n)
    hallu = np.clip(hallu, 0, 1)
    tasks = np.array(["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"])
    models = np.array(["grok-3-mini", "mistral-small-creative", "qwen-2.5"])
    creativity_levels = np.array(["FACTUAL", "HYBRID", "VERY_CREATIVE"])
    return pd.DataFrame(
        {
            "hallucination_rate": hallu,
            "ttct_overall": ttct,
            "ttcw_overall": ttcw,
            "creativity_composite": (ttct + ttcw) / 2.0,
            "creativity_rank": pd.cut(ttct, bins=3, labels=[0, 1, 2]).astype(float),
            "response_length_words": rng.integers(100, 600, n),
            "n_claim_rows": rng.integers(5, 25, n),
            "task": rng.choice(tasks, n),
            "model_name": rng.choice(models, n),
            "creativity_level": rng.choice(creativity_levels, n),
            "root_name": "longwiki-hybrid",
        }
    )


def _make_glm_df(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(456)
    levels = np.array(["FACTUAL", "HYBRID", "VERY_CREATIVE"])
    tasks = np.array(["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"])
    models = np.array(["grok-3-mini", "mistral-small-creative"])
    prompts = np.array([f"prompt_{i}" for i in range(40)])
    rows = []
    for _ in range(n):
        level = rng.choice(levels)
        task = rng.choice(tasks)
        model = rng.choice(models)
        prompt = rng.choice(prompts)
        temp = float(rng.choice([0.25, 0.5, 0.75]))
        length_words = int(rng.choice([500, 1000]))
        level_rank = int(np.where(levels == level)[0][0])
        task_effect = {"INTERVIEW": 0.00, "NEWS_ARTICLE": 0.05, "LESSON_PLAN": 0.02}[task]
        model_effect = {"grok-3-mini": 0.00, "mistral-small-creative": 0.02}[str(model)]
        creativity_interaction = {"grok-3-mini": 0.02, "mistral-small-creative": 0.06}[str(model)] * level_rank
        p = (
            0.20
            + creativity_interaction
            + task_effect
            + model_effect
            + (0.02 if length_words == 1000 else 0.0)
            + rng.normal(0, 0.03)
        )
        p = float(np.clip(p, 0.03, 0.93))
        n_claim_rows = 10
        hallucinated = int(rng.binomial(n=n_claim_rows, p=p))
        supported = n_claim_rows - hallucinated
        ttct = float(np.clip(2.0 + 0.9 * level_rank + rng.normal(0, 0.3), 1, 5))
        ttcw = float(np.clip(3.5 - 0.4 * level_rank + rng.normal(0, 0.4), 1, 5))
        rows.append(
            {
                "hallucination_rate": hallucinated / n_claim_rows,
                "n_claim_rows": n_claim_rows,
                "n_supported": supported,
                "creativity_level": level,
                "task": task,
                "temperature": temp,
                "length_words": length_words,
                "model_name": model,
                "prompt_id": prompt,
                "ttct_overall": ttct,
                "ttcw_overall": ttcw,
            }
        )
    return pd.DataFrame(rows)


def _make_mediation_df(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(789)
    creativity_rank = rng.choice([0.0, 1.0, 2.0], size=n, replace=True)
    length_words = rng.choice([50, 100, 250, 500, 1000], size=n, replace=True).astype(float)
    n_claims = 4.0 + 0.9 * creativity_rank + 0.004 * length_words + rng.normal(0.0, 1.0, n)
    n_claims = np.clip(n_claims, 1.0, None)
    hallu = (
        0.08
        + 0.013 * n_claims
        + 0.02 * creativity_rank
        + 0.00003 * length_words
        + rng.normal(0.0, 0.02, n)
    )
    hallu = np.clip(hallu, 0.0, 1.0)
    return pd.DataFrame(
        {
            "creativity_rank": creativity_rank,
            "n_claims": n_claims,
            "length_words": length_words,
            "hallucination_rate": hallu,
        }
    )


def _make_mixedlm_df(n: int = 360, n_prompts: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(159)
    levels = np.array(["FACTUAL", "HYBRID", "VERY_CREATIVE"])
    tasks = np.array(["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"])
    models = np.array(["grok-3-mini", "mistral-small-creative"])
    prompts = np.array([f"prompt_{i}" for i in range(n_prompts)])
    random_prompt_effect = {p: float(rng.normal(0.0, 0.035)) for p in prompts}

    rows = []
    for _ in range(n):
        level = str(rng.choice(levels))
        task = str(rng.choice(tasks))
        model = str(rng.choice(models))
        prompt = str(rng.choice(prompts))
        length_words = float(rng.choice([50, 100, 250, 500, 1000]))
        level_rank = {"FACTUAL": 0.0, "HYBRID": 1.0, "VERY_CREATIVE": 2.0}[level]
        task_effect = {"INTERVIEW": 0.00, "NEWS_ARTICLE": 0.03, "LESSON_PLAN": 0.015}[task]
        model_effect = {"grok-3-mini": 0.0, "mistral-small-creative": 0.012}[model]
        creativity_interaction = {"grok-3-mini": 0.010, "mistral-small-creative": 0.025}[model] * level_rank
        y = (
            0.14
            + creativity_interaction
            + task_effect
            + model_effect
            + 0.00005 * length_words
            + random_prompt_effect[prompt]
            + rng.normal(0.0, 0.025)
        )
        rows.append(
            {
                "hallucination_rate": float(np.clip(y, 0.0, 1.0)),
                "creativity_level": level,
                "task": task,
                "model_name": model,
                "length_words": length_words,
                "prompt_id": prompt,
            }
        )
    return pd.DataFrame(rows)


def _make_task_model_corr_df(n_per_group: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(31415)
    models = ["grok-3-mini", "mistral-small-creative"]
    tasks = ["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"]
    rows: list[dict[str, float | str]] = []
    for model_idx, model in enumerate(models):
        for task_idx, task in enumerate(tasks):
            for i in range(n_per_group):
                length_words = 250.0 + 35.0 * i + rng.normal(0.0, 4.0)
                temperature = 0.15 + 0.03 * (i % 10) + rng.normal(0.0, 0.005)
                creativity_rank = float(i % 3)
                hallu = (
                    0.16
                    + 0.00018 * length_words
                    - 0.055 * temperature
                    + 0.018 * creativity_rank
                    + 0.012 * task_idx
                    + 0.006 * model_idx
                    + rng.normal(0.0, 0.012)
                )
                rows.append(
                    {
                        "model_name": model,
                        "task": task,
                        "length_words": float(length_words),
                        "temperature": float(temperature),
                        "creativity_rank": creativity_rank,
                        "hallucination_rate": float(np.clip(hallu, 0.0, 1.0)),
                    }
                )
    return pd.DataFrame(rows)


def test_build_creativity_corr_table_has_valid_bounds_and_fdr() -> None:
    df = _make_corr_df()
    out = build_creativity_corr_table(
        df,
        target_col="hallucination_rate",
        metrics=["ttct_overall", "ttcw_overall", "creativity_composite"],
        n_boot=250,
        n_perm=400,
        seed=7,
    )
    assert not out.empty
    assert set(["metric", "method", "n", "r", "p_value", "ci_low", "ci_high", "p_fdr_bh"]).issubset(out.columns)
    finite_p = out["p_value"].dropna()
    assert ((finite_p >= 0.0) & (finite_p <= 1.0)).all()
    finite_ci = out.dropna(subset=["ci_low", "ci_high"])
    assert (finite_ci["ci_low"] <= finite_ci["ci_high"]).all()

    for _, grp in out.groupby("method"):
        ordered = grp.sort_values("p_value")
        fdr = ordered["p_fdr_bh"].to_numpy(dtype=float)
        fdr = fdr[np.isfinite(fdr)]
        if fdr.size > 1:
            assert np.all(np.diff(fdr) >= -1e-12)


def test_build_partial_corr_table_outputs_expected_columns() -> None:
    df = _make_corr_df()
    out = build_partial_corr_table(
        df,
        target_col="hallucination_rate",
        metrics=["ttct_overall", "ttcw_overall"],
        control_cols=("response_length_words", "n_claim_rows"),
        n_boot=200,
        seed=4,
    )
    assert set(
        ["metric", "controls", "n", "r_partial", "p_partial", "ci_low", "ci_high", "p_fdr_bh"]
    ).issubset(out.columns)
    if not out.empty:
        assert (out["n"] >= 8).all()
        assert out["r_partial"].apply(np.isfinite).all()


def test_fit_binomial_glm_returns_or_ci_p_when_available() -> None:
    df = _make_glm_df()
    out = fit_binomial_glm(df, target_rate_col="hallucination_rate")
    if not HAVE_STATSMODELS:
        assert out.empty
        return

    assert not out.empty
    assert set(["term", "odds_ratio", "ci_low", "ci_high", "p_value", "p_fdr_bh"]).issubset(out.columns)
    assert (out["odds_ratio"] > 0).all()
    assert ((out["p_value"] >= 0.0) & (out["p_value"] <= 1.0)).all()


def test_build_mediation_table_returns_effects_and_bootstrap_ci_when_available() -> None:
    df = _make_mediation_df()
    out = build_mediation_table(
        df,
        x_col="creativity_rank",
        mediator_col="n_claims",
        target_col="hallucination_rate",
        control_cols=("length_words",),
        n_boot=300,
        seed=42,
    )
    if not HAVE_STATSMODELS:
        assert out.empty
        return

    assert not out.empty
    assert set(
        [
            "a_path",
            "b_path",
            "c_prime_direct",
            "c_total",
            "indirect_ab",
            "indirect_ci_low",
            "indirect_ci_high",
            "indirect_p_boot",
            "sobel_p_value",
            "mediation_type",
            "n",
        ]
    ).issubset(out.columns)
    row = out.iloc[0]
    assert int(row["n"]) >= 20
    assert float(row["indirect_ci_low"]) <= float(row["indirect_ci_high"])
    if np.isfinite(row["indirect_p_boot"]):
        assert 0.0 <= float(row["indirect_p_boot"]) <= 1.0
    if np.isfinite(row["sobel_p_value"]):
        assert 0.0 <= float(row["sobel_p_value"]) <= 1.0


def test_fit_prompt_mixedlm_returns_icc_and_coefficients_when_available() -> None:
    df = _make_mixedlm_df()
    out = fit_prompt_mixedlm(df, target_col="hallucination_rate", group_col="prompt_id")
    if not HAVE_STATSMODELS:
        assert out.empty
        return

    assert not out.empty
    assert set(
        [
            "term",
            "coef",
            "ci_low",
            "ci_high",
            "p_value",
            "n_obs",
            "n_groups",
            "var_random_prompt",
            "var_residual",
            "icc",
            "converged",
        ]
    ).issubset(out.columns)
    row0 = out.iloc[0]
    assert int(row0["n_groups"]) >= 5
    assert np.isfinite(float(row0["var_residual"]))
    if np.isfinite(row0["icc"]):
        assert 0.0 <= float(row0["icc"]) <= 1.0


def test_build_task_model_parameter_corr_table_returns_task_model_breakdown() -> None:
    df = _make_task_model_corr_df()
    out = build_task_model_parameter_corr_table(
        df,
        target_col="hallucination_rate",
        metrics=["length_words", "temperature", "creativity_rank"],
        model_col="model_name",
        task_col="task",
        method="spearman",
        min_n=10,
    )
    assert not out.empty
    assert set(
        [
            "model_name",
            "task",
            "metric",
            "method",
            "n",
            "r",
            "abs_r",
            "p_value",
            "p_fdr_bh",
        ]
    ).issubset(out.columns)
    assert set(out["metric"].unique().tolist()) == {"length_words", "temperature", "creativity_rank"}
    assert set(out["model_name"].unique().tolist()) == {"grok-3-mini", "mistral-small-creative"}
    assert set(out["task"].unique().tolist()) == {"INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"}
    assert (out["n"] >= 10).all()
    assert ((out["r"] >= -1.0) & (out["r"] <= 1.0)).all()


def test_fit_binomial_glm_stratified_by_model_returns_expected_strata_when_available() -> None:
    df = _make_glm_df(n=260)
    out = fit_binomial_glm_stratified_by_model(
        df,
        target_rate_col="hallucination_rate",
        focus_model="grok-3-mini",
        model_col="model_name",
    )
    if not HAVE_STATSMODELS:
        assert out.empty
        return

    assert not out.empty
    assert set(
        [
            "stratum",
            "stratum_type",
            "n_models_in_stratum",
            "term",
            "odds_ratio",
            "p_value",
            "p_fdr_bh",
            "n_obs",
        ]
    ).issubset(out.columns)
    assert "single_model" in set(out["stratum_type"].unique().tolist())
    assert "pooled_other_models" in set(out["stratum_type"].unique().tolist())
    assert (out["n_obs"] >= 10).all()


def test_fit_prompt_mixedlm_stratified_by_model_returns_expected_strata_when_available() -> None:
    df = _make_mixedlm_df()
    out = fit_prompt_mixedlm_stratified_by_model(
        df,
        target_col="hallucination_rate",
        group_col="prompt_id",
        focus_model="grok-3-mini",
        model_col="model_name",
    )
    if not HAVE_STATSMODELS:
        assert out.empty
        return

    assert not out.empty
    assert set(
        [
            "stratum",
            "stratum_type",
            "term",
            "coef",
            "p_value",
            "n_obs",
            "n_groups",
        ]
    ).issubset(out.columns)
    assert "single_model" in set(out["stratum_type"].unique().tolist())


def test_test_creativity_level_homogeneity_returns_glm_and_mixed_rows_when_available() -> None:
    df = _make_glm_df(n=280)
    out = assess_creativity_level_homogeneity(
        df,
        target_rate_col="hallucination_rate",
        group_col="prompt_id",
        focus_model="grok-3-mini",
        model_col="model_name",
    )
    if not HAVE_STATSMODELS:
        assert out.empty
        return

    assert not out.empty
    assert set(
        [
            "model_type",
            "test",
            "model_groups",
            "lr_stat",
            "df_diff",
            "p_value",
            "conclusion",
            "status",
        ]
    ).issubset(out.columns)
    assert set(out["model_type"].unique().tolist()) == {"binomial_glm", "mixedlm"}
