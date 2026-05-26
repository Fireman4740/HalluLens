from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CREATIVITY_COLOR_MAP, CREATIVITY_ORDER


SPEARMAN_FAMILY_COLOR_MAP = {
    "task": "#1f77b4",
    "creativity": "#ff7f0e",
    "length": "#2ca02c",
    "temperature": "#7f7f7f",
    "model": "#9467bd",
    "other": "#111111",
}


def build_spearman_forest_plot(
    spearman_df: pd.DataFrame,
    top_n: int = 20,
) -> Any:
    required = {"row_label", "spearman_rho", "abs_rho", "ci_low", "ci_high", "factor_family", "p_value", "n"}
    if spearman_df is None or spearman_df.empty or not required.issubset(spearman_df.columns):
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(text="No Spearman forest data available.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    work = spearman_df.copy()
    work["abs_rho"] = pd.to_numeric(work["abs_rho"], errors="coerce")
    work["spearman_rho"] = pd.to_numeric(work["spearman_rho"], errors="coerce")
    work["ci_low"] = pd.to_numeric(work["ci_low"], errors="coerce")
    work["ci_high"] = pd.to_numeric(work["ci_high"], errors="coerce")
    work["n"] = pd.to_numeric(work["n"], errors="coerce")
    work["p_value"] = pd.to_numeric(work["p_value"], errors="coerce")
    work = work.dropna(subset=["spearman_rho", "abs_rho"])
    if work.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(text="No finite Spearman values.", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
        return fig

    top_n = max(1, int(top_n))
    work = work.sort_values("abs_rho", ascending=False).head(top_n).copy()
    work = work.iloc[::-1].reset_index(drop=True)
    y_order = work["row_label"].astype(str).tolist()

    fig = go.Figure()
    family_order = ["task", "creativity", "length", "temperature", "model", "other"]
    for family in family_order:
        sub = work[work["factor_family"].astype(str) == family].copy()
        if sub.empty:
            continue
        err_plus = (sub["ci_high"] - sub["spearman_rho"]).to_numpy(dtype=float)
        err_minus = (sub["spearman_rho"] - sub["ci_low"]).to_numpy(dtype=float)
        err_plus[~np.isfinite(err_plus)] = 0.0
        err_minus[~np.isfinite(err_minus)] = 0.0
        custom = np.column_stack(
            [
                sub["ci_low"].to_numpy(dtype=float),
                sub["ci_high"].to_numpy(dtype=float),
                sub["p_value"].to_numpy(dtype=float),
                sub["n"].to_numpy(dtype=float),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=sub["spearman_rho"],
                y=sub["row_label"],
                mode="markers",
                marker={
                    "size": 10,
                    "color": SPEARMAN_FAMILY_COLOR_MAP.get(family, SPEARMAN_FAMILY_COLOR_MAP["other"]),
                    "line": {"width": 0.5, "color": "white"},
                },
                error_x={
                    "type": "data",
                    "array": err_plus,
                    "arrayminus": err_minus,
                    "thickness": 1.6,
                    "width": 0,
                    "color": SPEARMAN_FAMILY_COLOR_MAP.get(family, SPEARMAN_FAMILY_COLOR_MAP["other"]),
                },
                name=family,
                customdata=custom,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + "rho=%{x:.3f}<br>"
                    + "95% CI=[%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
                    + "p=%{customdata[2]:.3e}<br>"
                    + "n=%{customdata[3]:.0f}"
                    + "<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=0.0, line_width=1.2, line_dash="dash", line_color="#666")
    fig.update_layout(
        template="plotly_white",
        title="Spearman Correlation Forest Plot",
        xaxis_title="Spearman rho",
        yaxis_title="Factor / modality",
        font={"size": 15},
        margin={"l": 330, "r": 40, "t": 60, "b": 50},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black")
    fig.update_yaxes(
        showgrid=False,
        linecolor="black",
        categoryorder="array",
        categoryarray=y_order,
    )
    return fig


def compute_per_model_temperature_effect_stats(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    model_col: str = "model_name",
    temperature_col: str = "temperature",
    min_n_per_temp: int = 8,
    n_boot: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    out_cols = [
        model_col,
        temperature_col,
        "n",
        "mean",
        "ci_low",
        "ci_high",
        "temp_levels_per_model",
    ]
    required = {target_col, model_col, temperature_col}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=out_cols)

    work = df[[model_col, temperature_col, target_col]].dropna().copy()
    work[temperature_col] = pd.to_numeric(work[temperature_col], errors="coerce")
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[temperature_col, target_col])
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    rng = np.random.default_rng(seed)

    def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan, np.nan
        mean = float(arr.mean())
        if arr.size < 2 or n_boot <= 1:
            return mean, np.nan, np.nan
        # Vectorized bootstrap in chunks: faster while avoiding large temporary allocations.
        max_cells = 2_000_000
        chunk_size = max(1, min(int(n_boot), int(max_cells / max(1, arr.size))))
        boot_chunks: list[np.ndarray] = []
        for start in range(0, int(n_boot), chunk_size):
            take = min(chunk_size, int(n_boot) - start)
            idx = rng.integers(0, arr.size, size=(take, arr.size))
            boot_chunks.append(arr[idx].mean(axis=1))
        boots = np.concatenate(boot_chunks, axis=0) if boot_chunks else np.asarray([mean], dtype=float)
        ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
        return mean, float(ci_low), float(ci_high)

    rows: list[dict[str, Any]] = []
    for (model_name, temp), grp in work.groupby([model_col, temperature_col], dropna=False, observed=False):
        n = int(len(grp))
        if n < min_n_per_temp:
            continue
        mean, ci_low, ci_high = _mean_ci(grp[target_col].to_numpy(dtype=float))
        rows.append(
            {
                model_col: str(model_name),
                temperature_col: float(temp),
                "n": n,
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return pd.DataFrame(columns=out_cols)

    temp_levels = (
        stats_df.groupby(model_col, dropna=False)[temperature_col]
        .nunique()
        .rename("temp_levels_per_model")
        .reset_index()
    )
    stats_df = stats_df.merge(temp_levels, on=model_col, how="left")
    stats_df = stats_df[stats_df["temp_levels_per_model"] >= 2].copy()
    if stats_df.empty:
        return pd.DataFrame(columns=out_cols)
    stats_df = stats_df.sort_values([model_col, temperature_col], ascending=[True, True]).reset_index(drop=True)
    return stats_df[out_cols]


def build_per_model_temperature_effect_plot_from_stats(
    stats_df: pd.DataFrame,
    model_col: str = "model_name",
    temperature_col: str = "temperature",
    target_col: str = "hallucination_rate",
    n_cols: int = 3,
) -> Any:
    needed = {model_col, temperature_col, "mean", "ci_low", "ci_high", "n", "temp_levels_per_model"}
    if stats_df is None or stats_df.empty or not needed.issubset(stats_df.columns):
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="Insufficient data per model/temperature for CI estimation.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    models = sorted(stats_df[model_col].astype(str).unique().tolist())
    n_cols = max(1, int(n_cols))
    n_rows = int(ceil(len(models) / n_cols))

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=models,
        shared_xaxes=False,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )
    for i, model_name in enumerate(models):
        row = i // n_cols + 1
        col = i % n_cols + 1
        sub = stats_df[stats_df[model_col].astype(str) == model_name].sort_values(temperature_col)
        x = sub[temperature_col].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        low = sub["ci_low"].to_numpy(dtype=float)
        high = sub["ci_high"].to_numpy(dtype=float)
        n_vals = sub["n"].to_numpy(dtype=float)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=high,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"m_{model_name}",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=low,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(31,119,180,0.18)",
                hoverinfo="skip",
                showlegend=False,
                legendgroup=f"m_{model_name}",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line={"color": "#1f77b4", "width": 2.2},
                marker={"size": 7, "color": "#1f77b4"},
                showlegend=(i == 0),
                name="Mean (95% CI)",
                legendgroup="mean_ci",
                customdata=np.column_stack([low, high, n_vals]),
                hovertemplate=(
                    "model=" + str(model_name) + "<br>"
                    + "temperature=%{x}<br>"
                    + "mean=%{y:.3f}<br>"
                    + "95% CI=[%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>"
                    + "n=%{customdata[2]:.0f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        template="plotly_white",
        title="Per-Model Temperature Effect on Hallucination Rate",
        font={"size": 14},
        margin={"l": 50, "r": 20, "t": 65, "b": 50},
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.05,
            "yanchor": "bottom",
        },
    )
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_xaxes(
                title_text=temperature_col,
                row=r,
                col=c,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                linecolor="black",
            )
            fig.update_yaxes(
                title_text=target_col,
                row=r,
                col=c,
                showgrid=True,
                gridcolor="rgba(0,0,0,0.08)",
                linecolor="black",
                zeroline=False,
            )
    return fig


def build_per_model_temperature_effect_plot(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    model_col: str = "model_name",
    temperature_col: str = "temperature",
    min_n_per_temp: int = 8,
    n_boot: int = 500,
    seed: int = 42,
    n_cols: int = 3,
) -> tuple[Any, pd.DataFrame]:
    stats_df = compute_per_model_temperature_effect_stats(
        df,
        target_col=target_col,
        model_col=model_col,
        temperature_col=temperature_col,
        min_n_per_temp=min_n_per_temp,
        n_boot=n_boot,
        seed=seed,
    )
    if stats_df.empty:
        out_cols = [
            model_col,
            temperature_col,
            "n",
            "mean",
            "ci_low",
            "ci_high",
            "temp_levels_per_model",
        ]
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="Insufficient data per model/temperature for CI estimation.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, pd.DataFrame(columns=out_cols)
    fig = build_per_model_temperature_effect_plot_from_stats(
        stats_df,
        model_col=model_col,
        temperature_col=temperature_col,
        target_col=target_col,
        n_cols=n_cols,
    )
    return fig, stats_df.copy()


def build_claim_density_vs_hallucination_plot(
    df: pd.DataFrame,
    claims_col: str = "n_claims",
    response_len_col: str = "response_length_words",
    y_col: str = "hallucination_rate",
    creativity_col: str = "creativity_level",
    lowess_frac: float = 0.45,
    min_points_for_lowess: int = 10,
    marker_opacity: float = 0.45,
    max_points: int = 12000,
    sample_seed: int = 42,
) -> tuple[Any, pd.DataFrame]:
    required = {claims_col, response_len_col, y_col, creativity_col}
    summary_cols = [
        creativity_col,
        "n_points",
        "density_mean",
        "density_median",
        "hallucination_rate_mean",
        "hallucination_rate_median",
        "lowess_drawn",
    ]
    if df is None or df.empty or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="No data available for claim density analysis.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, pd.DataFrame(columns=summary_cols)

    work = df[[claims_col, response_len_col, y_col, creativity_col]].copy()
    work[claims_col] = pd.to_numeric(work[claims_col], errors="coerce")
    work[response_len_col] = pd.to_numeric(work[response_len_col], errors="coerce")
    work[y_col] = pd.to_numeric(work[y_col], errors="coerce")
    work["claim_density_per_100_words"] = np.where(
        work[response_len_col] > 0,
        100.0 * work[claims_col] / work[response_len_col],
        np.nan,
    )
    work[creativity_col] = work[creativity_col].astype(str)
    work = work.dropna(subset=["claim_density_per_100_words", y_col, creativity_col]).copy()
    if work.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="No finite values for claim density and hallucination rate.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, pd.DataFrame(columns=summary_cols)

    lowess_frac = float(np.clip(lowess_frac, 0.05, 1.0))
    min_points_for_lowess = max(3, int(min_points_for_lowess))
    marker_opacity = float(np.clip(marker_opacity, 0.10, 1.0))

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

        have_lowess = True
    except Exception:
        sm_lowess = None
        have_lowess = False

    observed_levels = sorted(work[creativity_col].dropna().astype(str).unique().tolist())
    level_order = [lvl for lvl in CREATIVITY_ORDER if lvl in observed_levels] + [
        lvl for lvl in observed_levels if lvl not in CREATIVITY_ORDER
    ]

    plot_work = work.copy()
    if max_points and len(plot_work) > int(max_points):
        rng = np.random.default_rng(int(sample_seed))
        sampled_parts: list[pd.DataFrame] = []
        groups = list(plot_work.groupby(creativity_col, dropna=False))
        total = float(len(plot_work))
        for _, grp in groups:
            share = len(grp) / total if total > 0 else 0.0
            take = int(round(max(1, share * int(max_points))))
            if take >= len(grp):
                sampled_parts.append(grp)
            else:
                idx = rng.choice(len(grp), size=take, replace=False)
                sampled_parts.append(grp.iloc[idx])
        plot_work = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else plot_work
        if len(plot_work) > int(max_points):
            plot_work = plot_work.sample(n=int(max_points), random_state=int(sample_seed))

    fig = go.Figure()
    summary_rows: list[dict[str, Any]] = []

    for level in level_order:
        sub_plot = plot_work[plot_work[creativity_col].astype(str) == str(level)].copy()
        sub_full = work[work[creativity_col].astype(str) == str(level)].copy()
        if sub_plot.empty:
            continue
        x_vals = sub_plot["claim_density_per_100_words"].to_numpy(dtype=float)
        y_vals = sub_plot[y_col].to_numpy(dtype=float)
        color = CREATIVITY_COLOR_MAP.get(str(level), "#666666")
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name=str(level),
                legendgroup=str(level),
                marker={
                    "size": 6.5,
                    "opacity": marker_opacity,
                    "color": color,
                    "line": {"width": 0.4, "color": "white"},
                },
                hovertemplate=(
                    f"{creativity_col}={str(level)}<br>"
                    + "claim_density_per_100_words=%{x:.2f}<br>"
                    + f"{y_col}=%{{y:.3f}}<extra></extra>"
                ),
            )
        )

        lowess_drawn = False
        if have_lowess and len(sub_plot) >= min_points_for_lowess and sub_plot["claim_density_per_100_words"].nunique() >= 3:
            sorted_idx = np.argsort(x_vals)
            x_sorted = x_vals[sorted_idx]
            y_sorted = y_vals[sorted_idx]
            try:
                fitted = sm_lowess(y_sorted, x_sorted, frac=lowess_frac, return_sorted=True)
                if fitted is not None and len(fitted) >= 2:
                    fig.add_trace(
                        go.Scatter(
                            x=fitted[:, 0],
                            y=fitted[:, 1],
                            mode="lines",
                            line={"color": color, "width": 2.4},
                            legendgroup=str(level),
                            showlegend=False,
                            hovertemplate=(
                                f"{creativity_col}={str(level)} (LOWESS)<br>"
                                + "claim_density_per_100_words=%{x:.2f}<br>"
                                + f"{y_col}=%{{y:.3f}}<extra></extra>"
                            ),
                        )
                    )
                    lowess_drawn = True
            except Exception:
                lowess_drawn = False

        summary_rows.append(
            {
                creativity_col: str(level),
                "n_points": int(len(sub_full)),
                "density_mean": float(np.nanmean(sub_full["claim_density_per_100_words"].to_numpy(dtype=float))),
                "density_median": float(np.nanmedian(sub_full["claim_density_per_100_words"].to_numpy(dtype=float))),
                "hallucination_rate_mean": float(np.nanmean(sub_full[y_col].to_numpy(dtype=float))),
                "hallucination_rate_median": float(np.nanmedian(sub_full[y_col].to_numpy(dtype=float))),
                "lowess_drawn": bool(lowess_drawn),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df[creativity_col] = pd.Categorical(
            summary_df[creativity_col],
            categories=level_order,
            ordered=True,
        )
        summary_df = summary_df.sort_values(creativity_col).reset_index(drop=True)

    fig.update_layout(
        template="plotly_white",
        title="Claim Density vs. Hallucination Rate",
        xaxis_title="Claim density (n_claims / response_length_words x 100)",
        yaxis_title=y_col,
        font={"size": 15},
        margin={"l": 60, "r": 220, "t": 65, "b": 50},
        legend={
            "x": 1.01,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.8)",
            "itemsizing": "constant",
            "title": {"text": "creativity_level"},
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black", zeroline=False)
    if not have_lowess:
        fig.add_annotation(
            text="LOWESS unavailable (statsmodels missing): scatter only.",
            x=0.01,
            y=1.10,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            font={"size": 12, "color": "#666"},
        )
    return fig, summary_df[summary_cols] if not summary_df.empty else pd.DataFrame(columns=summary_cols)


def build_prompt_variance_icc_plot(
    df: pd.DataFrame,
    target_col: str = "hallucination_rate",
    prompt_col: str = "prompt_id",
    n_bins: int = 40,
    icc_value: float = 0.595,
    show_kde: bool = True,
    show_mixture: bool = False,
    mixture_components: int = 2,
    random_state: int = 42,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    required = {target_col, prompt_col}
    prompt_cols = [prompt_col, "prompt_mean_hallucination_rate", "n_rows_per_prompt"]
    summary_cols = [
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
    if df is None or df.empty or not required.issubset(df.columns):
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="No data available for prompt-level variance figure.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, pd.DataFrame(columns=prompt_cols), pd.DataFrame(columns=summary_cols)

    work = df[[prompt_col, target_col]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[prompt_col, target_col]).copy()
    if work.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="No finite prompt/target values for ICC figure.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, pd.DataFrame(columns=prompt_cols), pd.DataFrame(columns=summary_cols)

    prompt_means = (
        work.groupby(prompt_col, dropna=False, observed=False)[target_col]
        .agg(prompt_mean_hallucination_rate="mean", n_rows_per_prompt="count")
        .reset_index()
        .sort_values("prompt_mean_hallucination_rate")
        .reset_index(drop=True)
    )
    values = prompt_means["prompt_mean_hallucination_rate"].to_numpy(dtype=float)
    if values.size == 0:
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="No prompt means available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, prompt_means[prompt_cols], pd.DataFrame(columns=summary_cols)

    n_bins = max(8, int(n_bins))
    mixture_components = max(1, int(mixture_components))
    grand_mean = float(np.nanmean(values))
    prompt_std = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
    prompt_var = float(np.nanvar(values, ddof=1)) if values.size > 1 else 0.0

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=n_bins,
            histnorm="probability density",
            marker={"color": "#4c78a8", "line": {"width": 0.3, "color": "white"}},
            opacity=0.85,
            name="Prompt means",
            hovertemplate="mean_hallucination_rate=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
        )
    )

    x_grid = np.linspace(float(np.nanmin(values)), float(np.nanmax(values)), 300)
    kde_drawn = False
    if show_kde and values.size >= 5 and np.nanmax(values) > np.nanmin(values):
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(values)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=kde(x_grid),
                    mode="lines",
                    line={"color": "#222222", "width": 2.2},
                    name="KDE",
                    hovertemplate="x=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
                )
            )
            kde_drawn = True
        except Exception:
            kde_drawn = False

    mixture_status = "not_requested"
    mixture_bic = np.nan
    if show_mixture and values.size >= max(20, mixture_components * 10) and np.nanmax(values) > np.nanmin(values):
        try:
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(
                n_components=mixture_components,
                random_state=int(random_state),
                covariance_type="full",
            )
            arr = values.reshape(-1, 1)
            gmm.fit(arr)
            mixture_pdf = np.exp(gmm.score_samples(x_grid.reshape(-1, 1)))
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=mixture_pdf,
                    mode="lines",
                    line={"color": "#9467bd", "width": 2.0, "dash": "dot"},
                    name=f"GMM ({mixture_components})",
                    hovertemplate="x=%{x:.3f}<br>density=%{y:.3f}<extra></extra>",
                )
            )
            if hasattr(gmm, "weights_") and hasattr(gmm, "means_") and hasattr(gmm, "covariances_"):
                w = np.asarray(gmm.weights_, dtype=float)
                m = np.asarray(gmm.means_, dtype=float).reshape(-1)
                c = np.asarray(gmm.covariances_, dtype=float).reshape(-1)
                c = np.maximum(c, 1e-9)
                s = np.sqrt(c)
                for i in range(min(len(w), len(m), len(s))):
                    comp_pdf = w[i] * (1.0 / (s[i] * np.sqrt(2.0 * np.pi))) * np.exp(
                        -0.5 * ((x_grid - m[i]) / s[i]) ** 2
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_grid,
                            y=comp_pdf,
                            mode="lines",
                            line={"color": "#b89cd9", "width": 1.1, "dash": "dash"},
                            name=f"GMM comp {i + 1}",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )
            mixture_bic = float(gmm.bic(arr))
            mixture_status = "fitted"
        except Exception:
            mixture_status = "unavailable"
    elif show_mixture:
        mixture_status = "insufficient_points"

    fig.add_vline(x=grand_mean, line_width=2.0, line_dash="dash", line_color="#222222")
    fig.add_annotation(
        x=0.01,
        y=1.12,
        xref="paper",
        yref="paper",
        showarrow=False,
        xanchor="left",
        text=f"Grand mean={grand_mean:.3f} | ICC={float(icc_value):.3f}",
        font={"size": 12, "color": "#333"},
    )

    fig.update_layout(
        template="plotly_white",
        title="Prompt-Level Mean Hallucination Rate Distribution",
        xaxis_title="Prompt-level mean hallucination rate",
        yaxis_title="Density",
        barmode="overlay",
        font={"size": 15},
        margin={"l": 60, "r": 20, "t": 75, "b": 55},
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.03,
            "yanchor": "bottom",
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)", linecolor="black", zeroline=False)

    summary = pd.DataFrame(
        [
            {
                "n_prompts": int(values.size),
                "grand_mean": grand_mean,
                "prompt_mean_std": prompt_std,
                "prompt_mean_var": prompt_var,
                "prompt_mean_q25": float(np.nanquantile(values, 0.25)),
                "prompt_mean_median": float(np.nanmedian(values)),
                "prompt_mean_q75": float(np.nanquantile(values, 0.75)),
                "icc_annotated": float(icc_value),
                "mixture_status": mixture_status,
                "mixture_components": int(mixture_components if show_mixture else 0),
                "mixture_bic": mixture_bic,
                "kde_drawn": bool(kde_drawn),
            }
        ]
    )

    return (
        fig,
        prompt_means[prompt_cols],
        summary[
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
        ],
    )


def build_mediation_path_diagram(
    mediation_df: pd.DataFrame,
) -> tuple[Any, pd.DataFrame]:
    out_cols = [
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
    required = {
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
    }
    if mediation_df is None or mediation_df.empty or not required.issubset(mediation_df.columns):
        fig = go.Figure()
        fig.update_layout(template="plotly_white")
        fig.add_annotation(
            text="No mediation data available for path diagram.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig, pd.DataFrame(columns=out_cols)

    row = mediation_df.iloc[0]

    def _num(name: str) -> float:
        return float(pd.to_numeric(row.get(name), errors="coerce"))

    def _fmt_num(v: float, digits: int = 4) -> str:
        if not np.isfinite(v):
            return "NA"
        return f"{v:.{digits}f}"

    def _fmt_p(v: float) -> str:
        if not np.isfinite(v):
            return "NA"
        if v < 1e-4:
            return "<1e-4"
        return f"{v:.4f}"

    def _stars(v: float) -> str:
        if not np.isfinite(v):
            return ""
        if v < 0.001:
            return "***"
        if v < 0.01:
            return "**"
        if v < 0.05:
            return "*"
        return "n.s."

    a_path = _num("a_path")
    a_p = _num("a_p_value")
    b_path = _num("b_path")
    b_p = _num("b_p_value")
    c_prime = _num("c_prime_direct")
    c_prime_p = _num("c_prime_p_value")
    c_total = _num("c_total")
    c_total_p = _num("c_total_p_value")
    indirect_ab = _num("indirect_ab")
    indirect_ci_low = _num("indirect_ci_low")
    indirect_ci_high = _num("indirect_ci_high")
    indirect_p = _num("indirect_p_boot")
    n_obs = int(pd.to_numeric(row.get("n"), errors="coerce")) if np.isfinite(pd.to_numeric(row.get("n"), errors="coerce")) else np.nan
    mediation_type = str(row.get("mediation_type", "NA"))

    share_den = abs(c_prime) + abs(indirect_ab)
    if np.isfinite(share_den) and share_den > 0:
        direct_share = float(100.0 * abs(c_prime) / share_den)
        indirect_share = float(100.0 * abs(indirect_ab) / share_den)
    else:
        direct_share = np.nan
        indirect_share = np.nan

    # Node centers in normalized axes coordinates [0, 1].
    x_intent, y_intent = 0.12, 0.50
    x_claims, y_claims = 0.50, 0.82
    x_hallu, y_hallu = 0.88, 0.50
    node_w, node_h = 0.20, 0.14

    fig = go.Figure()
    node_style = {"line": {"color": "#2b2b2b", "width": 1.2}, "fillcolor": "rgba(245,245,245,0.98)"}
    for cx, cy in [(x_intent, y_intent), (x_claims, y_claims), (x_hallu, y_hallu)]:
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=cx - node_w / 2,
            x1=cx + node_w / 2,
            y0=cy - node_h / 2,
            y1=cy + node_h / 2,
            **node_style,
        )

    fig.add_annotation(
        x=x_intent,
        y=y_intent,
        xref="x",
        yref="y",
        text="Creative Intent<br>(creativity_rank)",
        showarrow=False,
        font={"size": 13, "color": "#111"},
    )
    fig.add_annotation(
        x=x_claims,
        y=y_claims,
        xref="x",
        yref="y",
        text="n_claims",
        showarrow=False,
        font={"size": 13, "color": "#111"},
    )
    fig.add_annotation(
        x=x_hallu,
        y=y_hallu,
        xref="x",
        yref="y",
        text="Hallucination Rate",
        showarrow=False,
        font={"size": 13, "color": "#111"},
    )

    # Direct path: Creative Intent -> Hallucination Rate
    fig.add_annotation(
        x=x_hallu - node_w / 2 + 0.01,
        y=y_hallu,
        xref="x",
        yref="y",
        ax=x_intent + node_w / 2 - 0.01,
        ay=y_intent,
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.3,
        arrowwidth=2.8,
        arrowcolor="#1f77b4",
    )
    direct_label = (
        f"c'={_fmt_num(c_prime)} ({_stars(c_prime_p)}), p={_fmt_p(c_prime_p)}"
        + (f"<br>|share|={_fmt_num(direct_share, 2)}%" if np.isfinite(direct_share) else "")
    )
    fig.add_annotation(
        x=0.50,
        y=0.40,
        xref="x",
        yref="y",
        text=direct_label,
        showarrow=False,
        font={"size": 12, "color": "#1f77b4"},
        align="center",
    )

    # Indirect subpaths: Creative Intent -> n_claims -> Hallucination Rate
    fig.add_annotation(
        x=x_claims - node_w / 2 + 0.01,
        y=y_claims - node_h / 6,
        xref="x",
        yref="y",
        ax=x_intent + node_w / 2 - 0.01,
        ay=y_intent + node_h / 5,
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=2.2,
        arrowcolor="#ff7f0e",
    )
    fig.add_annotation(
        x=0.33,
        y=0.72,
        xref="x",
        yref="y",
        text=f"a={_fmt_num(a_path)} ({_stars(a_p)}), p={_fmt_p(a_p)}",
        showarrow=False,
        font={"size": 11, "color": "#ff7f0e"},
    )

    fig.add_annotation(
        x=x_hallu - node_w / 2 + 0.01,
        y=y_hallu + node_h / 5,
        xref="x",
        yref="y",
        ax=x_claims + node_w / 2 - 0.01,
        ay=y_claims - node_h / 6,
        axref="x",
        ayref="y",
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=2.2,
        arrowcolor="#2ca02c",
    )
    fig.add_annotation(
        x=0.67,
        y=0.72,
        xref="x",
        yref="y",
        text=f"b={_fmt_num(b_path)} ({_stars(b_p)}), p={_fmt_p(b_p)}",
        showarrow=False,
        font={"size": 11, "color": "#2ca02c"},
    )

    indirect_label = (
        f"Indirect ab={_fmt_num(indirect_ab)} ({_stars(indirect_p)}), p_boot={_fmt_p(indirect_p)}"
        + f"<br>95% CI=[{_fmt_num(indirect_ci_low)}, {_fmt_num(indirect_ci_high)}]"
        + (f" | |share|={_fmt_num(indirect_share, 2)}%" if np.isfinite(indirect_share) else "")
    )
    fig.add_annotation(
        x=0.50,
        y=0.95,
        xref="x",
        yref="y",
        text=indirect_label,
        showarrow=False,
        font={"size": 11, "color": "#444"},
        align="center",
    )

    footer = (
        f"Total effect c={_fmt_num(c_total)} ({_stars(c_total_p)}), p={_fmt_p(c_total_p)}"
        + (f" | n={int(n_obs)}" if np.isfinite(n_obs) else "")
        + f" | type={mediation_type}"
    )
    fig.add_annotation(
        x=0.01,
        y=0.02,
        xref="x",
        yref="y",
        text=footer,
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font={"size": 11, "color": "#555"},
    )

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        template="plotly_white",
        title="Mediation Path Diagram: Creative Intent -> Hallucination Rate",
        margin={"l": 20, "r": 20, "t": 60, "b": 30},
        showlegend=False,
    )

    summary_df = pd.DataFrame(
        [
            {
                "n": n_obs,
                "a_path": a_path,
                "a_p_value": a_p,
                "b_path": b_path,
                "b_p_value": b_p,
                "c_prime_direct": c_prime,
                "c_prime_p_value": c_prime_p,
                "c_total": c_total,
                "c_total_p_value": c_total_p,
                "indirect_ab": indirect_ab,
                "indirect_ci_low": indirect_ci_low,
                "indirect_ci_high": indirect_ci_high,
                "indirect_p_boot": indirect_p,
                "direct_share_pct_abs": direct_share,
                "indirect_share_pct_abs": indirect_share,
                "mediation_type": mediation_type,
            }
        ]
    )
    return fig, summary_df[out_cols]
