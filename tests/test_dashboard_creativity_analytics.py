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
    build_partial_corr_table,
    fit_binomial_glm,
)


def _make_corr_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    ttct = rng.normal(3.0, 0.6, n)
    ttcw = rng.normal(2.8, 0.7, n)
    hallu = 0.25 + 0.08 * ttct - 0.05 * ttcw + rng.normal(0, 0.03, n)
    hallu = np.clip(hallu, 0, 1)
    return pd.DataFrame(
        {
            "hallucination_rate": hallu,
            "ttct_overall": ttct,
            "ttcw_overall": ttcw,
            "creativity_composite": (ttct + ttcw) / 2.0,
            "creativity_rank": pd.cut(ttct, bins=3, labels=[0, 1, 2]).astype(float),
            "response_length_words": rng.integers(100, 600, n),
            "n_claim_rows": rng.integers(5, 25, n),
        }
    )


def _make_glm_df(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(456)
    levels = np.array(["FACTUAL", "HYBRID", "VERY_CREATIVE"])
    tasks = np.array(["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"])
    models = np.array(["grok-3-mini", "mistral-small-creative"])
    rows = []
    for _ in range(n):
        level = rng.choice(levels)
        task = rng.choice(tasks)
        model = rng.choice(models)
        temp = float(rng.choice([0.25, 0.5, 0.75]))
        length_words = int(rng.choice([500, 1000]))
        level_rank = int(np.where(levels == level)[0][0])
        task_effect = {"INTERVIEW": 0.00, "NEWS_ARTICLE": 0.05, "LESSON_PLAN": 0.02}[task]
        p = 0.20 + 0.04 * level_rank + task_effect + (0.02 if length_words == 1000 else 0.0) + rng.normal(0, 0.03)
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
                "ttct_overall": ttct,
                "ttcw_overall": ttcw,
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
