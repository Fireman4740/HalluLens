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


def build_task_model_parameter_corr_table(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    metrics: list[str] | None = None,
    model_col: str = "model_name",
    task_col: str = "task",
    method: str = "spearman",
    min_n: int = 8,
) -> pd.DataFrame:
    if metrics is None:
        metrics = ["length_words", "temperature", "creativity_rank"]

    out_cols = [
        model_col,
        task_col,
        "metric",
        "method",
        "n",
        "r",
        "abs_r",
        "p_value",
        "p_fdr_bh",
    ]
    required = [model_col, task_col, target_col]
    if any(col not in df.columns for col in required):
        return pd.DataFrame(columns=out_cols)

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict[str, Any]] = []
    for metric in available_metrics:
        sub = df[[model_col, task_col, metric, target_col]].dropna(
            subset=[model_col, task_col, metric, target_col]
        )
        if sub.empty:
            continue

        for (model_name, task_name), grp in sub.groupby([model_col, task_col], dropna=False):
            if len(grp) < min_n:
                continue
            if grp[metric].nunique() < 2 or grp[target_col].nunique() < 2:
                continue

            x = grp[metric].to_numpy(dtype=float)
            y = grp[target_col].to_numpy(dtype=float)

            if method == "spearman":
                if HAVE_SCIPY:
                    res = stats.spearmanr(x, y, nan_policy="omit")
                    r = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                    p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
                else:
                    r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
                    p = np.nan
            elif method == "pearson":
                if HAVE_SCIPY:
                    res = stats.pearsonr(x, y)
                    r = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
                    p = float(res.pvalue) if hasattr(res, "pvalue") else float(res[1])
                else:
                    r = float(pd.Series(x).corr(pd.Series(y), method="pearson"))
                    p = np.nan
            else:
                raise ValueError("method must be one of {'spearman', 'pearson'}")

            if not np.isfinite(r):
                continue

            rows.append(
                {
                    model_col: str(model_name),
                    task_col: str(task_name),
                    "metric": metric,
                    "method": method,
                    "n": int(len(grp)),
                    "r": float(r),
                    "abs_r": abs(float(r)),
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                }
            )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = _fdr_bh(out["p_value"].to_numpy(dtype=float))
    out = out.sort_values([model_col, task_col, "abs_r", "metric"], ascending=[True, True, False, True])
    return out[out_cols].reset_index(drop=True)


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
    level_col: str = "creativity_level",
    level_order: list[str] | None = None,
) -> pd.DataFrame:
    if level_order is None:
        level_order = (
            ["FACTUAL", "HYBRID", "VERY_CREATIVE"]
            if level_col == "creativity_level"
            else ["LOW", "HIGH"]
        )
    required = [
        "title",
        "task",
        "length_words",
        "model_name",
        "temperature",
        level_col,
        target_col,
    ]
    if any(c not in df.columns for c in required):
        return pd.DataFrame(
            columns=["contrast", "n_paired", "mean_diff", "median_diff", "ci_low", "ci_high"]
        )

    contrast_values: dict[str, list[float]] = {}
    for _, sub in df.groupby(
        ["title", "task", "length_words", "model_name", "temperature"], dropna=False
    ):
        means = sub.groupby(level_col)[target_col].mean()
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


def _bootstrap_indirect_effect_ab(
    x: np.ndarray,
    m: np.ndarray,
    y: np.ndarray,
    controls: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(x)
    out: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        mb = m[idx]
        yb = y[idx]
        cb = controls[idx]
        xa = np.column_stack([np.ones(n), xb, cb])
        xb_full = np.column_stack([np.ones(n), xb, mb, cb])
        try:
            coef_a, *_ = np.linalg.lstsq(xa, mb, rcond=None)
            coef_b, *_ = np.linalg.lstsq(xb_full, yb, rcond=None)
        except Exception:
            continue
        a_boot = float(coef_a[1])
        b_boot = float(coef_b[2])
        ab = a_boot * b_boot
        if np.isfinite(ab):
            out.append(ab)
    return np.asarray(out, dtype=float)


def build_mediation_table(
    df: pd.DataFrame,
    x_col: str = "creativity_rank",
    mediator_col: str = "n_claims",
    target_col: str = "hallucination_rate",
    control_cols: tuple[str, ...] = ("length_words",),
    n_boot: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    out_cols = [
        "x",
        "mediator",
        "target",
        "controls",
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
        "sobel_z",
        "sobel_p_value",
        "prop_mediated",
        "mediation_type",
    ]
    if not HAVE_STATSMODELS:
        return pd.DataFrame(columns=out_cols)

    work = df.copy()
    if x_col not in work.columns and "creativity_level" in work.columns and x_col == "creativity_rank":
        work[x_col] = work["creativity_level"].map({"FACTUAL": 0, "HYBRID": 1, "VERY_CREATIVE": 2})

    mediator_used = mediator_col
    if mediator_used not in work.columns or work[mediator_used].dropna().nunique() < 2:
        fallback = "n_claim_rows"
        if fallback in work.columns and work[fallback].dropna().nunique() >= 2:
            mediator_used = fallback

    controls_available = tuple(c for c in control_cols if c in work.columns)
    needed = [x_col, mediator_used, target_col] + list(controls_available)
    if any(c not in work.columns for c in [x_col, mediator_used, target_col]):
        return pd.DataFrame(columns=out_cols)

    sub = work[needed].copy()
    for col in needed:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna()
    if (
        len(sub) < 20
        or sub[x_col].nunique() < 2
        or sub[mediator_used].nunique() < 2
        or sub[target_col].nunique() < 2
    ):
        return pd.DataFrame(columns=out_cols)

    x = sub[x_col].to_numpy(dtype=float)
    m = sub[mediator_used].to_numpy(dtype=float)
    y = sub[target_col].to_numpy(dtype=float)
    controls = (
        sub[list(controls_available)].to_numpy(dtype=float)
        if controls_available
        else np.zeros((len(sub), 0), dtype=float)
    )

    x_a = sm.add_constant(sub[[x_col] + list(controls_available)], has_constant="add")
    fit_a = sm.OLS(sub[mediator_used], x_a).fit()
    a_path = float(fit_a.params[x_col])
    a_se = float(fit_a.bse[x_col])
    a_p = float(fit_a.pvalues[x_col])

    x_b = sm.add_constant(
        sub[[x_col, mediator_used] + list(controls_available)], has_constant="add"
    )
    fit_b = sm.OLS(sub[target_col], x_b).fit()
    b_path = float(fit_b.params[mediator_used])
    b_se = float(fit_b.bse[mediator_used])
    b_p = float(fit_b.pvalues[mediator_used])
    c_prime = float(fit_b.params[x_col])
    c_prime_p = float(fit_b.pvalues[x_col])

    x_c = sm.add_constant(sub[[x_col] + list(controls_available)], has_constant="add")
    fit_c = sm.OLS(sub[target_col], x_c).fit()
    c_total = float(fit_c.params[x_col])
    c_total_p = float(fit_c.pvalues[x_col])

    indirect_ab = float(a_path * b_path)
    if np.isfinite(c_total) and abs(c_total) > 1e-12:
        prop_mediated = float(indirect_ab / c_total)
    else:
        prop_mediated = np.nan

    if HAVE_SCIPY:
        sobel_se = np.sqrt((b_path**2) * (a_se**2) + (a_path**2) * (b_se**2))
        if np.isfinite(sobel_se) and sobel_se > 0:
            sobel_z = float(indirect_ab / sobel_se)
            sobel_p = float(2.0 * stats.norm.sf(abs(sobel_z)))
        else:
            sobel_z = np.nan
            sobel_p = np.nan
    else:
        sobel_z = np.nan
        sobel_p = np.nan

    boot_indirect = _bootstrap_indirect_effect_ab(
        x=x,
        m=m,
        y=y,
        controls=controls,
        n_boot=n_boot,
        seed=seed,
    )
    if boot_indirect.size > 0:
        ci_low, ci_high = np.quantile(boot_indirect, [0.025, 0.975])
        p_left = float(np.mean(boot_indirect <= 0.0))
        p_right = float(np.mean(boot_indirect >= 0.0))
        p_boot = float(2.0 * min(p_left, p_right))
        p_boot = max(0.0, min(1.0, p_boot))
    else:
        ci_low = np.nan
        ci_high = np.nan
        p_boot = np.nan

    indirect_sig = (
        (np.isfinite(ci_low) and np.isfinite(ci_high) and ((ci_low > 0.0) or (ci_high < 0.0)))
        or (np.isfinite(p_boot) and p_boot < 0.05)
    )
    direct_sig = np.isfinite(c_prime_p) and c_prime_p < 0.05
    if indirect_sig and direct_sig:
        mediation_type = "partial_mediation"
    elif indirect_sig and not direct_sig:
        mediation_type = "full_mediation"
    elif (not indirect_sig) and direct_sig:
        mediation_type = "direct_only"
    else:
        mediation_type = "no_clear_effect"

    out = pd.DataFrame(
        [
            {
                "x": x_col,
                "mediator": mediator_used,
                "target": target_col,
                "controls": ", ".join(controls_available),
                "n": int(len(sub)),
                "a_path": a_path,
                "a_p_value": a_p,
                "b_path": b_path,
                "b_p_value": b_p,
                "c_prime_direct": c_prime,
                "c_prime_p_value": c_prime_p,
                "c_total": c_total,
                "c_total_p_value": c_total_p,
                "indirect_ab": indirect_ab,
                "indirect_ci_low": float(ci_low),
                "indirect_ci_high": float(ci_high),
                "indirect_p_boot": p_boot,
                "sobel_z": sobel_z,
                "sobel_p_value": sobel_p,
                "prop_mediated": prop_mediated,
                "mediation_type": mediation_type,
            }
        ]
    )
    return out[out_cols]


def fit_prompt_mixedlm(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    group_col: str = "prompt_id",
) -> pd.DataFrame:
    out_cols = [
        "term",
        "coef",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "n_obs",
        "n_groups",
        "var_random_prompt",
        "var_residual",
        "icc",
        "converged",
    ]
    if not HAVE_STATSMODELS:
        return pd.DataFrame(columns=out_cols)

    required = [target_col, "creativity_level", "task", "length_words", group_col]
    if any(c not in df.columns for c in required):
        return pd.DataFrame(columns=out_cols)

    has_kappa = (
        "kappa_level" in df.columns
        and df["kappa_level"].astype(str).replace("NA", pd.NA).dropna().nunique() >= 2
    )
    cols = required + (["kappa_level"] if has_kappa else [])
    work = df[cols].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work["length_words"] = pd.to_numeric(work["length_words"], errors="coerce")
    work[group_col] = work[group_col].astype(str)
    work["creativity_level"] = work["creativity_level"].astype(str)
    work["task"] = work["task"].astype(str)
    if has_kappa:
        work["kappa_level"] = work["kappa_level"].astype(str)
        work.loc[work["kappa_level"] == "NA", "kappa_level"] = np.nan
    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if (
        len(work) < 20
        or work[target_col].nunique() < 2
        or work[group_col].nunique() < 5
        or work["creativity_level"].nunique() < 2
        or work["task"].nunique() < 2
    ):
        return pd.DataFrame(columns=out_cols)

    kappa_term = " + C(kappa_level)" if has_kappa and work["kappa_level"].nunique() >= 2 else ""
    formula = f"{target_col} ~ C(creativity_level) * C(task) + length_words{kappa_term}"
    try:
        model = sm.MixedLM.from_formula(
            formula,
            data=work,
            groups=work[group_col],
            re_formula="1",
        )
        try:
            fit = model.fit(reml=True, method="lbfgs", disp=False)
        except Exception:
            fit = model.fit(reml=True, method="powell", disp=False)
    except Exception:
        return pd.DataFrame(columns=out_cols)

    params = fit.params
    pvals = fit.pvalues
    conf = fit.conf_int()

    var_random = np.nan
    try:
        cov_re = np.asarray(fit.cov_re, dtype=float)
        if cov_re.size > 0:
            var_random = float(cov_re.ravel()[0])
    except Exception:
        var_random = np.nan
    var_residual = float(fit.scale) if np.isfinite(fit.scale) else np.nan
    if np.isfinite(var_random) and np.isfinite(var_residual) and (var_random + var_residual) > 0:
        icc = float(var_random / (var_random + var_residual))
    else:
        icc = np.nan

    rows: list[dict[str, Any]] = []
    for term, coef in params.items():
        if term.lower().endswith("var"):
            continue
        ci_low = float(conf.loc[term, 0]) if term in conf.index else np.nan
        ci_high = float(conf.loc[term, 1]) if term in conf.index else np.nan
        p_value = float(pvals.get(term, np.nan))
        rows.append(
            {
                "term": term,
                "coef": float(coef),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": p_value,
                "n_obs": int(len(work)),
                "n_groups": int(work[group_col].nunique()),
                "var_random_prompt": var_random,
                "var_residual": var_residual,
                "icc": icc,
                "converged": bool(getattr(fit, "converged", False)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=out_cols)
    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = _fdr_bh(out["p_value"].to_numpy(dtype=float))
    out = out.sort_values("p_value").reset_index(drop=True)
    return out[out_cols]


def _build_model_strata(
    df: pd.DataFrame,
    model_col: str = "model_name",
    focus_model: str = "grok-3-mini",
) -> list[dict[str, Any]]:
    if model_col not in df.columns:
        return []
    work = df.copy()
    work[model_col] = work[model_col].astype(str)
    models = sorted(work[model_col].dropna().astype(str).unique().tolist())
    if not models:
        return []

    strata: list[dict[str, Any]] = []
    for model_name in models:
        sub = work[work[model_col].astype(str) == model_name].copy()
        if sub.empty:
            continue
        strata.append(
            {
                "stratum": model_name,
                "stratum_type": "single_model",
                "n_models_in_stratum": 1,
                "data": sub,
            }
        )

    if focus_model in models:
        other_sub = work[work[model_col].astype(str) != focus_model].copy()
        if not other_sub.empty:
            strata.append(
                {
                    "stratum": f"others_pooled_excluding_{focus_model}",
                    "stratum_type": "pooled_other_models",
                    "n_models_in_stratum": int(other_sub[model_col].nunique()),
                    "data": other_sub,
                }
            )
    return strata


def fit_binomial_glm_stratified_by_model(
    df: pd.DataFrame,
    target_rate_col: str = "hallucination_rate",
    focus_model: str = "grok-3-mini",
    model_col: str = "model_name",
) -> pd.DataFrame:
    out_cols = [
        "stratum",
        "stratum_type",
        "n_models_in_stratum",
        "term",
        "coef",
        "odds_ratio",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "n_obs",
    ]
    if not HAVE_STATSMODELS or model_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    rows: list[pd.DataFrame] = []
    for stratum in _build_model_strata(df, model_col=model_col, focus_model=focus_model):
        sub = stratum["data"]
        work, active_optional_terms = _prepare_binomial_glm_work(sub, target_rate_col=target_rate_col)
        if work.empty:
            continue
        fit, _ = _fit_binomial_glm_internal(work, active_optional_terms, include_model_term=False)
        if fit is None:
            continue

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
        out["stratum"] = str(stratum["stratum"])
        out["stratum_type"] = str(stratum["stratum_type"])
        out["n_models_in_stratum"] = int(stratum["n_models_in_stratum"])
        rows.append(out[out_cols].sort_values("p_value").reset_index(drop=True))

    if not rows:
        return pd.DataFrame(columns=out_cols)

    merged = pd.concat(rows, ignore_index=True)
    merged = merged.sort_values(["stratum_type", "stratum", "p_value"], ascending=[True, True, True])
    return merged[out_cols].reset_index(drop=True)


def fit_prompt_mixedlm_stratified_by_model(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    group_col: str = "prompt_id",
    focus_model: str = "grok-3-mini",
    model_col: str = "model_name",
) -> pd.DataFrame:
    out_cols = [
        "stratum",
        "stratum_type",
        "n_models_in_stratum",
        "term",
        "coef",
        "ci_low",
        "ci_high",
        "p_value",
        "p_fdr_bh",
        "n_obs",
        "n_groups",
        "var_random_prompt",
        "var_residual",
        "icc",
        "converged",
    ]
    if not HAVE_STATSMODELS or model_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    rows: list[pd.DataFrame] = []
    for stratum in _build_model_strata(df, model_col=model_col, focus_model=focus_model):
        sub = stratum["data"]
        out = fit_prompt_mixedlm(sub, target_col=target_col, group_col=group_col)
        if out.empty:
            continue
        out = out.copy()
        out["stratum"] = str(stratum["stratum"])
        out["stratum_type"] = str(stratum["stratum_type"])
        out["n_models_in_stratum"] = int(stratum["n_models_in_stratum"])
        rows.append(out[out_cols])

    if not rows:
        return pd.DataFrame(columns=out_cols)

    merged = pd.concat(rows, ignore_index=True)
    merged = merged.sort_values(["stratum_type", "stratum", "p_value"], ascending=[True, True, True])
    return merged[out_cols].reset_index(drop=True)


def assess_creativity_level_homogeneity(
    df: pd.DataFrame,
    target_rate_col: str = "hallucination_rate",
    group_col: str = "prompt_id",
    focus_model: str = "grok-3-mini",
    model_col: str = "model_name",
) -> pd.DataFrame:
    out_cols = [
        "model_type",
        "test",
        "focus_model",
        "model_groups",
        "n_obs",
        "n_groups",
        "llf_null",
        "llf_alt",
        "lr_stat",
        "df_diff",
        "p_value",
        "conclusion",
        "status",
    ]
    if not HAVE_STATSMODELS or model_col not in df.columns:
        return pd.DataFrame(columns=out_cols)

    def _conclusion(status: str, p_value: float) -> str:
        if status != "ok":
            return "unavailable"
        if np.isfinite(p_value) and p_value < 0.05:
            return "heterogeneous_effect"
        return "compatible_with_homogeneous_effect"

    rows: list[dict[str, Any]] = []

    # Binomial GLM homogeneity test: interaction creativity_level x model_bucket.
    glm_row: dict[str, Any] = {
        "model_type": "binomial_glm",
        "test": "LRT interaction C(creativity_level) x C(model_bucket)",
        "focus_model": focus_model,
        "model_groups": f"{focus_model} vs others_pooled",
        "n_obs": np.nan,
        "n_groups": np.nan,
        "llf_null": np.nan,
        "llf_alt": np.nan,
        "lr_stat": np.nan,
        "df_diff": np.nan,
        "p_value": np.nan,
        "status": "insufficient_data",
    }
    try:
        work, active_optional_terms = _prepare_binomial_glm_work(df, target_rate_col=target_rate_col)
        if not work.empty:
            work = work.copy()
            work["model_bucket"] = np.where(
                work[model_col].astype(str) == focus_model,
                focus_model,
                "__OTHER_MODELS__",
            )
            if (
                work["model_bucket"].nunique() >= 2
                and work["creativity_level"].nunique() >= 2
                and work["task"].nunique() >= 2
            ):
                y = np.column_stack([work["hallucinated_claims"], work["supported_claims"]])
                base_terms = [
                    "C(creativity_level)",
                    "C(task)",
                    "temperature",
                    "length_words",
                    "C(model_bucket)",
                ]
                alt_terms = [
                    "C(creativity_level) * C(model_bucket)",
                    "C(task)",
                    "temperature",
                    "length_words",
                ]
                base_terms.extend(active_optional_terms)
                alt_terms.extend(active_optional_terms)
                base_design = patsy.dmatrix(" + ".join(base_terms), work, return_type="dataframe")
                alt_design = patsy.dmatrix(" + ".join(alt_terms), work, return_type="dataframe")
                base_fit = sm.GLM(y, base_design, family=sm.families.Binomial()).fit()
                alt_fit = sm.GLM(y, alt_design, family=sm.families.Binomial()).fit()

                lr_stat = float(max(0.0, 2.0 * (alt_fit.llf - base_fit.llf)))
                df_diff = int(max(0, round(float(alt_fit.df_model - base_fit.df_model))))
                if df_diff > 0 and HAVE_SCIPY:
                    p_value = float(stats.chi2.sf(lr_stat, df_diff))
                else:
                    p_value = np.nan
                glm_row.update(
                    {
                        "n_obs": int(len(work)),
                        "llf_null": float(base_fit.llf),
                        "llf_alt": float(alt_fit.llf),
                        "lr_stat": lr_stat,
                        "df_diff": df_diff,
                        "p_value": p_value,
                        "status": "ok" if df_diff > 0 else "invalid_df_diff",
                    }
                )
    except Exception:
        glm_row["status"] = "fit_failed"

    glm_row["conclusion"] = _conclusion(str(glm_row["status"]), float(glm_row["p_value"]))
    rows.append(glm_row)

    # MixedLM homogeneity test: interaction creativity_level x model_bucket.
    mix_row: dict[str, Any] = {
        "model_type": "mixedlm",
        "test": "LRT interaction C(creativity_level) x C(model_bucket)",
        "focus_model": focus_model,
        "model_groups": f"{focus_model} vs others_pooled",
        "n_obs": np.nan,
        "n_groups": np.nan,
        "llf_null": np.nan,
        "llf_alt": np.nan,
        "lr_stat": np.nan,
        "df_diff": np.nan,
        "p_value": np.nan,
        "status": "insufficient_data",
    }
    try:
        required = [target_rate_col, "creativity_level", "task", "length_words", model_col, group_col]
        if all(c in df.columns for c in required):
            work = df[required].copy()
            work[target_rate_col] = pd.to_numeric(work[target_rate_col], errors="coerce")
            work["length_words"] = pd.to_numeric(work["length_words"], errors="coerce")
            work[group_col] = work[group_col].astype(str)
            work["creativity_level"] = work["creativity_level"].astype(str)
            work["task"] = work["task"].astype(str)
            work[model_col] = work[model_col].astype(str)
            work = work.replace([np.inf, -np.inf], np.nan).dropna()
            if not work.empty:
                work["model_bucket"] = np.where(
                    work[model_col].astype(str) == focus_model,
                    focus_model,
                    "__OTHER_MODELS__",
                )
                if (
                    len(work) >= 20
                    and work[group_col].nunique() >= 5
                    and work["model_bucket"].nunique() >= 2
                    and work["creativity_level"].nunique() >= 2
                    and work["task"].nunique() >= 2
                ):
                    base_formula = (
                        f"{target_rate_col} ~ C(creativity_level) + C(task) + length_words + C(model_bucket)"
                    )
                    alt_formula = (
                        f"{target_rate_col} ~ C(creativity_level) * C(model_bucket) + C(task) + length_words"
                    )
                    base_model = sm.MixedLM.from_formula(
                        base_formula,
                        data=work,
                        groups=work[group_col],
                        re_formula="1",
                    )
                    alt_model = sm.MixedLM.from_formula(
                        alt_formula,
                        data=work,
                        groups=work[group_col],
                        re_formula="1",
                    )
                    try:
                        base_fit = base_model.fit(reml=False, method="lbfgs", disp=False)
                    except Exception:
                        base_fit = base_model.fit(reml=False, method="powell", disp=False)
                    try:
                        alt_fit = alt_model.fit(reml=False, method="lbfgs", disp=False)
                    except Exception:
                        alt_fit = alt_model.fit(reml=False, method="powell", disp=False)

                    lr_stat = float(max(0.0, 2.0 * (alt_fit.llf - base_fit.llf)))
                    df_diff = int(max(0, len(alt_fit.fe_params) - len(base_fit.fe_params)))
                    if df_diff > 0 and HAVE_SCIPY:
                        p_value = float(stats.chi2.sf(lr_stat, df_diff))
                    else:
                        p_value = np.nan
                    mix_row.update(
                        {
                            "n_obs": int(len(work)),
                            "n_groups": int(work[group_col].nunique()),
                            "llf_null": float(base_fit.llf),
                            "llf_alt": float(alt_fit.llf),
                            "lr_stat": lr_stat,
                            "df_diff": df_diff,
                            "p_value": p_value,
                            "status": "ok" if df_diff > 0 else "invalid_df_diff",
                        }
                    )
    except Exception:
        mix_row["status"] = "fit_failed"

    mix_row["conclusion"] = _conclusion(str(mix_row["status"]), float(mix_row["p_value"]))
    rows.append(mix_row)

    return pd.DataFrame(rows)[out_cols]


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
    has_kappa = "kappa_level" in df.columns and df["kappa_level"].astype(str).replace("NA", pd.NA).dropna().nunique() >= 2
    used_cols = required_base + optional_numeric_terms + (["kappa_level"] if has_kappa else [])
    work = df[used_cols].copy()

    active_optional_terms: list[str] = []
    for term in optional_numeric_terms:
        s = pd.to_numeric(work[term], errors="coerce")
        if s.notna().any() and s.dropna().nunique() >= 2:
            work[term] = s
            active_optional_terms.append(term)
        else:
            work = work.drop(columns=[term], errors="ignore")

    if has_kappa:
        work["kappa_level"] = work["kappa_level"].astype(str)
        work.loc[work["kappa_level"] == "NA", "kappa_level"] = np.nan

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
    include_model_term: bool = True,
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
    ]
    if include_model_term:
        formula_parts.append("C(model_name)")
    if "kappa_level" in work.columns and work["kappa_level"].dropna().nunique() >= 2:
        formula_parts.append("C(kappa_level)")
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
