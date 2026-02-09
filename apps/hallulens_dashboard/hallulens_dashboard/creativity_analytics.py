from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats

    HAVE_SCIPY = True
except Exception:
    stats = None
    HAVE_SCIPY = False

try:
    import statsmodels.api as sm
    import patsy
    from statsmodels.stats.multitest import multipletests

    HAVE_STATSMODELS = True
except Exception:
    sm = None
    patsy = None
    multipletests = None
    HAVE_STATSMODELS = False


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    if not valid_mask.any():
        return out
    p_valid = p[valid_mask]
    if multipletests is not None:
        _, p_adj, _, _ = multipletests(p_valid, alpha=0.05, method="fdr_bh")
        out[valid_mask] = p_adj
        return out

    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    p_adj = np.empty_like(q)
    p_adj[order] = np.clip(q, 0, 1)
    out[valid_mask] = p_adj
    return out


def _mean_ci(values: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(arr, [0.025, 0.975])
    return float(lo), float(hi)


def _bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if not HAVE_SCIPY:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if method == "spearman":
            res = stats.spearmanr(xb, yb, nan_policy="omit")
        else:
            res = stats.pearsonr(xb, yb)
        val = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        if np.isfinite(val):
            boot_stats.append(val)
    if not boot_stats:
        return np.nan, np.nan
    return _mean_ci(np.asarray(boot_stats, dtype=float))


def _permutation_pvalue_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_resamples: int,
    seed: int,
) -> float:
    if not HAVE_SCIPY:
        return np.nan

    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if method == "spearman":
        x_use = stats.rankdata(x)
        y_use = stats.rankdata(y)
    else:
        x_use = x
        y_use = y

    x_use = x_use - x_use.mean()
    y_use = y_use - y_use.mean()
    denom = np.sqrt(np.sum(x_use**2) * np.sum(y_use**2))
    if denom <= 0 or not np.isfinite(denom):
        return np.nan

    obs = float(np.dot(x_use, y_use) / denom)
    count = 0
    for _ in range(n_resamples):
        xp = rng.permutation(x_use)
        r = float(np.dot(xp, y_use) / denom)
        if abs(r) >= abs(obs):
            count += 1
    return float((count + 1) / (n_resamples + 1))


def _corr_stats(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> dict[str, float]:
    if not HAVE_SCIPY:
        return {"r": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    if method == "spearman":
        res = stats.spearmanr(x, y, nan_policy="omit")
    elif method == "pearson":
        res = stats.pearsonr(x, y)
    else:
        raise ValueError("method must be one of {'spearman', 'pearson'}")

    r = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    p_perm = _permutation_pvalue_corr(x, y, method=method, n_resamples=n_perm, seed=seed)
    ci_low, ci_high = _bootstrap_corr_ci(x, y, method=method, n_boot=n_boot, seed=seed)
    return {
        "r": r,
        "p_value": p_perm,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def build_creativity_corr_table(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    metrics: list[str] | None = None,
    n_boot: int = 5000,
    n_perm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    if metrics is None:
        metrics = [
            "ttct_fluency",
            "ttct_flexibility",
            "ttct_originality",
            "ttct_elaboration",
            "ttct_overall",
            "ttcw_coherence",
            "ttcw_descriptive_detail",
            "ttcw_ending",
            "ttcw_emotional_flexibility",
            "ttcw_overall",
            "creativity_composite",
            "creativity_rank",
        ]

    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in df.columns or target_col not in df.columns:
            continue
        sub = df[[metric, target_col]].dropna()
        if len(sub) < 4 or sub[metric].nunique() < 2 or sub[target_col].nunique() < 2:
            continue
        x = sub[metric].to_numpy(dtype=float)
        y = sub[target_col].to_numpy(dtype=float)
        for method in ("spearman", "pearson"):
            out = _corr_stats(x, y, method=method, n_boot=n_boot, n_perm=n_perm, seed=seed)
            rows.append(
                {
                    "metric": metric,
                    "target": target_col,
                    "method": method,
                    "n": int(len(sub)),
                    "r": out["r"],
                    "r2": out["r"] ** 2 if np.isfinite(out["r"]) else np.nan,
                    "p_value": out["p_value"],
                    "ci_low": out["ci_low"],
                    "ci_high": out["ci_high"],
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "metric",
                "target",
                "method",
                "n",
                "r",
                "r2",
                "p_value",
                "ci_low",
                "ci_high",
                "p_fdr_bh",
            ]
        )

    out_df = pd.DataFrame(rows)
    out_df["p_fdr_bh"] = _fdr_bh(out_df["p_value"].to_numpy(dtype=float))
    out_df = out_df.sort_values(["method", "p_fdr_bh", "metric"]).reset_index(drop=True)
    return out_df


def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(controls)), controls])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x @ coef
    return y - y_hat


def _partial_corr_rank_residualized(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols: tuple[str, ...],
) -> tuple[float, float]:
    if not HAVE_SCIPY:
        return np.nan, np.nan
    rank_df = sub.copy()
    for c in [x_col, y_col] + list(control_cols):
        rank_df[c] = rank_df[c].rank(method="average")

    controls = rank_df[list(control_cols)].to_numpy(dtype=float)
    rx = _residualize(rank_df[x_col].to_numpy(dtype=float), controls)
    ry = _residualize(rank_df[y_col].to_numpy(dtype=float), controls)
    res = stats.pearsonr(rx, ry)
    r = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
    p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
    return r, p


def _bootstrap_partial_ci(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    control_cols: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(sub)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = sub.iloc[idx]
        r, _ = _partial_corr_rank_residualized(boot, x_col, y_col, control_cols)
        if np.isfinite(r):
            vals.append(r)
    if not vals:
        return np.nan, np.nan
    return _mean_ci(np.asarray(vals, dtype=float))


def build_partial_corr_table(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    metrics: list[str] | None = None,
    control_cols: tuple[str, ...] = ("response_length_words", "n_claim_rows"),
    n_boot: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    if metrics is None:
        metrics = [
            "ttct_overall",
            "ttcw_overall",
            "creativity_composite",
            "creativity_rank",
        ]
    if not HAVE_SCIPY:
        return pd.DataFrame(
            columns=[
                "metric",
                "target",
                "n",
                "r_partial",
                "p_partial",
                "ci_low",
                "ci_high",
                "p_fdr_bh",
            ]
        )

    rows: list[dict[str, Any]] = []
    controls_available = tuple(c for c in control_cols if c in df.columns)
    if len(controls_available) == 0:
        return pd.DataFrame(
            columns=[
                "metric",
                "target",
                "n",
                "r_partial",
                "p_partial",
                "ci_low",
                "ci_high",
                "p_fdr_bh",
            ]
        )

    for metric in metrics:
        needed = [metric, target_col] + list(controls_available)
        if any(c not in df.columns for c in needed):
            continue
        sub = df[needed].dropna()
        if len(sub) < 8 or sub[metric].nunique() < 2:
            continue
        r, p = _partial_corr_rank_residualized(
            sub,
            x_col=metric,
            y_col=target_col,
            control_cols=controls_available,
        )
        ci_low, ci_high = _bootstrap_partial_ci(
            sub,
            x_col=metric,
            y_col=target_col,
            control_cols=controls_available,
            n_boot=n_boot,
            seed=seed,
        )
        rows.append(
            {
                "metric": metric,
                "target": target_col,
                "controls": ", ".join(controls_available),
                "n": int(len(sub)),
                "r_partial": r,
                "p_partial": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "metric",
                "target",
                "controls",
                "n",
                "r_partial",
                "p_partial",
                "ci_low",
                "ci_high",
                "p_fdr_bh",
            ]
        )
    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = _fdr_bh(out["p_partial"].to_numpy(dtype=float))
    out = out.sort_values(["p_fdr_bh", "metric"]).reset_index(drop=True)
    return out


def build_intent_contrast_table(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
) -> pd.DataFrame:
    required = [
        "title",
        "task",
        "length_words",
        "model_name",
        "temperature",
        "creativity_level",
        target_col,
    ]
    if any(c not in df.columns for c in required):
        return pd.DataFrame(
            columns=["contrast", "n_paired", "mean_diff", "median_diff", "ci_low", "ci_high"]
        )

    level_order = ["FACTUAL", "HYBRID", "VERY_CREATIVE"]
    contrast_values: dict[str, list[float]] = {}
    for _, sub in df.groupby(
        ["title", "task", "length_words", "model_name", "temperature"], dropna=False
    ):
        means = sub.groupby("creativity_level")[target_col].mean()
        levels = [lvl for lvl in level_order if lvl in means.index]
        for a, b in itertools.combinations(levels, 2):
            diff = float(means[b] - means[a])
            key = f"{b} - {a}"
            contrast_values.setdefault(key, []).append(diff)

    rows = []
    for contrast, vals in contrast_values.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            continue
        ci_low, ci_high = _mean_ci(arr)
        rows.append(
            {
                "contrast": contrast,
                "n_paired": int(arr.size),
                "mean_diff": float(np.nanmean(arr)),
                "median_diff": float(np.nanmedian(arr)),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["contrast", "n_paired", "mean_diff", "median_diff", "ci_low", "ci_high"]
        )
    return pd.DataFrame(rows).sort_values("contrast").reset_index(drop=True)


def fit_binomial_glm(
    df: pd.DataFrame,
    target_rate_col: str = "hallucination_rate",
) -> pd.DataFrame:
    out_cols = [
        "term",
        "coef",
        "odds_ratio",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "n_obs",
    ]
    if not HAVE_STATSMODELS:
        return pd.DataFrame(columns=out_cols)

    work, active_optional_terms = _prepare_binomial_glm_work(df, target_rate_col=target_rate_col)
    if work.empty:
        return pd.DataFrame(columns=out_cols)
    fit, _ = _fit_binomial_glm_internal(work, active_optional_terms)
    if fit is None:
        return pd.DataFrame(columns=out_cols)

    coef = fit.params
    se = fit.bse
    pvals = fit.pvalues
    out = pd.DataFrame(
        {
            "term": coef.index,
            "coef": coef.values,
            "odds_ratio": np.exp(coef.values),
            "ci_low": np.exp(coef.values - 1.96 * se.values),
            "ci_high": np.exp(coef.values + 1.96 * se.values),
            "p_value": pvals.values,
            "n_obs": len(work),
        }
    )
    out["p_fdr_bh"] = _fdr_bh(out["p_value"].to_numpy(dtype=float))
    out = out.sort_values("p_value").reset_index(drop=True)
    return out[out_cols]


def _prepare_binomial_glm_work(
    df: pd.DataFrame,
    target_rate_col: str = "hallucination_rate",
) -> tuple[pd.DataFrame, list[str]]:
    if not HAVE_STATSMODELS:
        return pd.DataFrame(), []

    required_base = [
        target_rate_col,
        "n_claim_rows",
        "n_supported",
        "creativity_level",
        "task",
        "temperature",
        "length_words",
        "model_name",
    ]
    if any(c not in df.columns for c in required_base):
        return pd.DataFrame(), []

    optional_numeric_terms = [c for c in ["ttct_overall", "ttcw_overall"] if c in df.columns]
    used_cols = required_base + optional_numeric_terms
    work = df[used_cols].copy()

    active_optional_terms: list[str] = []
    for term in optional_numeric_terms:
        s = pd.to_numeric(work[term], errors="coerce")
        if s.notna().any() and s.dropna().nunique() >= 2:
            work[term] = s
            active_optional_terms.append(term)
        else:
            work = work.drop(columns=[term], errors="ignore")

    work = work.dropna(
        subset=[
            "n_claim_rows",
            "n_supported",
            "creativity_level",
            "task",
            "temperature",
            "length_words",
            "model_name",
        ]
    )
    if work.empty:
        return pd.DataFrame(), []

    work["n_claim_rows"] = pd.to_numeric(work["n_claim_rows"], errors="coerce")
    work["n_supported"] = pd.to_numeric(work["n_supported"], errors="coerce")
    work = work.dropna(subset=["n_claim_rows", "n_supported"])
    work = work[work["n_claim_rows"] > 0].copy()
    if work.empty:
        return pd.DataFrame(), []

    work["hallucinated_claims"] = (
        work["n_claim_rows"] - work["n_supported"]
    ).clip(lower=0, upper=work["n_claim_rows"])
    work["supported_claims"] = work["n_supported"].clip(lower=0, upper=work["n_claim_rows"])
    work = work[work["hallucinated_claims"] + work["supported_claims"] > 0]
    if len(work) < 10:
        return pd.DataFrame(), []

    return work, active_optional_terms


def _fit_binomial_glm_internal(
    work: pd.DataFrame,
    active_optional_terms: list[str],
) -> tuple[Any | None, Any | None]:
    if not HAVE_STATSMODELS:
        return None, None
    if work.empty:
        return None, None
    y = np.column_stack([work["hallucinated_claims"], work["supported_claims"]])
    formula_parts = [
        "C(creativity_level) * C(task)",
        "temperature",
        "length_words",
        "C(model_name)",
    ]
    formula_parts.extend(active_optional_terms)
    formula = " + ".join(formula_parts)
    design = patsy.dmatrix(formula, work, return_type="dataframe")
    model = sm.GLM(y, design, family=sm.families.Binomial())
    fit = model.fit()
    return fit, design


def predict_binomial_glm_probabilities(
    df: pd.DataFrame,
    target_rate_col: str = "hallucination_rate",
    fixed_length_words: float | None = None,
    fixed_temperature: float | None = None,
    fixed_model_name: str | None = None,
) -> pd.DataFrame:
    out_cols = [
        "creativity_level",
        "task",
        "model_name",
        "temperature",
        "length_words",
        "predicted_hallucination_probability",
    ]
    if not HAVE_STATSMODELS:
        return pd.DataFrame(columns=out_cols)

    work, active_optional_terms = _prepare_binomial_glm_work(df, target_rate_col=target_rate_col)
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    fit, design = _fit_binomial_glm_internal(work, active_optional_terms)
    if fit is None or design is None:
        return pd.DataFrame(columns=out_cols)

    # Fixed reference point for interpretable probabilities.
    model_name = fixed_model_name
    if model_name is None or model_name not in work["model_name"].astype(str).unique().tolist():
        mode_model = work["model_name"].mode(dropna=True)
        model_name = str(mode_model.iloc[0]) if not mode_model.empty else str(work["model_name"].astype(str).iloc[0])
    if fixed_temperature is None:
        fixed_temperature = float(work["temperature"].median())
    if fixed_length_words is None:
        fixed_length_words = float(work["length_words"].median())

    levels = [lvl for lvl in ["FACTUAL", "HYBRID", "VERY_CREATIVE"] if lvl in work["creativity_level"].astype(str).unique().tolist()]
    tasks = sorted(work["task"].astype(str).unique().tolist())
    if not levels or not tasks:
        return pd.DataFrame(columns=out_cols)

    pred_rows: list[dict[str, Any]] = []
    optional_means = {
        term: float(pd.to_numeric(work[term], errors="coerce").mean())
        for term in active_optional_terms
    }
    for lvl in levels:
        for task in tasks:
            row = {
                "creativity_level": lvl,
                "task": task,
                "model_name": model_name,
                "temperature": float(fixed_temperature),
                "length_words": float(fixed_length_words),
            }
            row.update(optional_means)
            pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    try:
        design_pred = patsy.build_design_matrices([design.design_info], pred_df, return_type="dataframe")[0]
    except Exception:
        return pd.DataFrame(columns=out_cols)

    pred_prob = fit.predict(design_pred)
    out = pred_df[
        ["creativity_level", "task", "model_name", "temperature", "length_words"]
    ].copy()
    out["predicted_hallucination_probability"] = np.asarray(pred_prob, dtype=float)
    return out[out_cols].sort_values(["task", "creativity_level"]).reset_index(drop=True)
