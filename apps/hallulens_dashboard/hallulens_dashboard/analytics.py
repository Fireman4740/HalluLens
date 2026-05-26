from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import CREATIVITY_ORDER

try:
    from scipy import stats

    HAVE_SCIPY = True
except Exception:
    stats = None
    HAVE_SCIPY = False


def numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def build_impact_summary(df: pd.DataFrame, x_axis: str, y_axis: str) -> pd.DataFrame:
    summary = (
        df.groupby(x_axis, dropna=False)[y_axis]
        .agg(n="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    if x_axis == "creativity_level":
        summary[x_axis] = pd.Categorical(summary[x_axis], categories=CREATIVITY_ORDER, ordered=True)
        summary = summary.sort_values(x_axis)
    elif numeric(summary[x_axis]):
        summary = summary.sort_values(x_axis)
    else:
        summary = summary.sort_values(x_axis, key=lambda s: s.astype(str))
    if not summary.empty:
        baseline = summary["mean"].iloc[0]
        summary["delta_vs_first_level"] = summary["mean"] - baseline
    return summary


def spearman_rho_p(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if HAVE_SCIPY:
        res = stats.spearmanr(x, y, nan_policy="omit")
        if hasattr(res, "statistic"):
            return float(res.statistic), float(res.pvalue)
        return float(res[0]), float(res[1])
    rho = x.corr(y, method="spearman")
    return float(rho), np.nan


def spearman_strength_label(rho: float) -> str:
    if not np.isfinite(rho):
        return "NA"
    abs_rho = abs(rho)
    if abs_rho < 0.10:
        return "très faible"
    if abs_rho < 0.30:
        return "faible"
    if abs_rho < 0.50:
        return "modérée"
    if abs_rho < 0.70:
        return "forte"
    return "très forte"


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    max_sample: int = 5000,
) -> tuple[float, float]:
    if not HAVE_SCIPY:
        return np.nan, np.nan
    if len(x) < 4:
        return np.nan, np.nan
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[valid]
    y_arr = y_arr[valid]
    if len(x_arr) < 4:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    if len(x_arr) > max_sample:
        keep = rng.choice(len(x_arr), size=max_sample, replace=False)
        x_arr = x_arr[keep]
        y_arr = y_arr[keep]

    x_rank = stats.rankdata(x_arr)
    y_rank = stats.rankdata(y_arr)
    n = len(x_rank)

    def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
        av = a - a.mean()
        bv = b - b.mean()
        denom = np.sqrt(np.sum(av * av) * np.sum(bv * bv))
        if denom <= 0 or not np.isfinite(denom):
            return np.nan
        return float(np.dot(av, bv) / denom)

    vals: list[float] = []
    for _ in range(max(0, int(n_boot))):
        idx = rng.integers(0, n, size=n)
        rho = _pearson_corr(x_rank[idx], y_rank[idx])
        if np.isfinite(rho):
            vals.append(rho)
    if not vals:
        return np.nan, np.nan
    lo, hi = np.quantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return float(lo), float(hi)


def _factor_family(factor: str) -> str:
    f = str(factor)
    if f == "task":
        return "task"
    if f in {"creativity_level", "creativity_rank"}:
        return "creativity"
    if f in {
        "length_words",
        "response_length_words",
        "response_length_tokens",
        "prompt_length_words",
        "n_claims",
        "n_claim_rows",
    }:
        return "length"
    if f == "temperature":
        return "temperature"
    if f == "model_name":
        return "model"
    return "other"


def build_spearman_forest_table(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    min_n: int = 20,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    out_cols = [
        "factor",
        "modality",
        "factor_type",
        "encoding",
        "n",
        "spearman_rho",
        "abs_rho",
        "ci_low",
        "ci_high",
        "p_value",
        "direction",
        "strength",
        "factor_family",
        "row_label",
    ]
    rows: list[dict[str, Any]] = []

    numeric_factors = [
        "temperature",
        "length_words",
        "response_length_words",
        "response_length_tokens",
        "prompt_length_words",
        "n_claims",
        "creativity_rank",
    ]
    categorical_factors = ["creativity_level", "kappa_level", "task", "model_name", "root_name"]

    for col in numeric_factors:
        if col not in df.columns:
            continue
        sub = df[[col, target_col]].dropna()
        if len(sub) < min_n or sub[col].nunique() < 2:
            continue
        x = sub[col].to_numpy(dtype=float)
        y = sub[target_col].to_numpy(dtype=float)
        rho, p = spearman_rho_p(sub[col], sub[target_col])
        ci_low, ci_high = _bootstrap_spearman_ci(x, y, n_boot=n_boot, seed=seed)
        rows.append(
            {
                "factor": col,
                "modality": "(global)",
                "factor_type": "numeric",
                "encoding": "raw ranks",
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "abs_rho": abs(float(rho)) if np.isfinite(rho) else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "direction": "positive" if rho >= 0 else "negative",
                "strength": spearman_strength_label(rho),
                "factor_family": _factor_family(col),
                "row_label": f"{col} | (global)",
            }
        )

    for col in categorical_factors:
        if col not in df.columns:
            continue
        sub = df[[col, target_col]].dropna().copy()
        if len(sub) < min_n or sub[col].nunique() < 2:
            continue
        levels = sorted(sub[col].astype(str).unique().tolist())
        for lvl in levels:
            x_bin = (sub[col].astype(str) == lvl).astype(int)
            if x_bin.nunique() < 2:
                continue
            x = x_bin.to_numpy(dtype=float)
            y = sub[target_col].to_numpy(dtype=float)
            rho, p = spearman_rho_p(x_bin, sub[target_col])
            ci_low, ci_high = _bootstrap_spearman_ci(x, y, n_boot=n_boot, seed=seed)
            rows.append(
                {
                    "factor": col,
                    "modality": lvl,
                    "factor_type": "categorical",
                    "encoding": "one-hot",
                    "n": int(len(sub)),
                    "spearman_rho": float(rho),
                    "abs_rho": abs(float(rho)) if np.isfinite(rho) else np.nan,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "direction": "positive" if rho >= 0 else "negative",
                    "strength": spearman_strength_label(rho),
                    "factor_family": _factor_family(col),
                    "row_label": f"{col} | {lvl}",
                }
            )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)
    out = out.sort_values(["abs_rho", "factor", "modality"], ascending=[False, True, True]).reset_index(drop=True)
    return out[out_cols]


def build_spearman_detailed(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    min_n: int = 20,
) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []

    numeric_factors = [
        "temperature",
        "length_words",
        "response_length_words",
        "response_length_tokens",
        "prompt_length_words",
        "n_claims",
        "creativity_rank",
    ]
    categorical_factors = ["creativity_level", "kappa_level", "task", "model_name", "root_name"]

    for col in numeric_factors:
        if col not in df.columns:
            continue
        sub = df[[col, target_col]].dropna()
        if len(sub) < min_n or sub[col].nunique() < 2:
            skipped.append(f"{col} (numérique): données insuffisantes")
            continue
        rho, p = spearman_rho_p(sub[col], sub[target_col])
        rows.append(
            {
                "factor": col,
                "modality": "(global)",
                "factor_type": "numeric",
                "encoding": "rangs de la variable",
                "n": int(len(sub)),
                "spearman_rho": float(rho),
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "direction": "positive" if rho >= 0 else "négative",
                "strength": spearman_strength_label(rho),
            }
        )

    for col in categorical_factors:
        if col not in df.columns:
            continue
        sub = df[[col, target_col]].dropna().copy()
        if len(sub) < min_n or sub[col].nunique() < 2:
            skipped.append(f"{col} (catégoriel): données insuffisantes")
            continue

        levels = sorted(sub[col].astype(str).unique().tolist())
        for lvl in levels:
            x_bin = (sub[col].astype(str) == lvl).astype(int)
            if x_bin.nunique() < 2:
                continue
            rho, p = spearman_rho_p(x_bin, sub[target_col])
            rows.append(
                {
                    "factor": col,
                    "modality": lvl,
                    "factor_type": "categorical",
                    "encoding": "one-hot: 1=modalité, 0=autres",
                    "n": int(len(sub)),
                    "spearman_rho": float(rho),
                    "p_value": float(p) if np.isfinite(p) else np.nan,
                    "direction": "positive" if rho >= 0 else "négative",
                    "strength": spearman_strength_label(rho),
                }
            )

    if not rows:
        return pd.DataFrame(), skipped

    out = pd.DataFrame(rows)
    out["abs_rho"] = out["spearman_rho"].abs()
    out = out.sort_values(["abs_rho", "factor", "modality"], ascending=[False, True, True]).reset_index(drop=True)
    return out, skipped
