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
        "support_rate",
        "creativity_rank",
    ]
    categorical_factors = ["creativity_level", "task", "model_name", "root_name"]

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
