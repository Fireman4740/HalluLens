from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html as components_html

from .analytics import (
    HAVE_SCIPY,
    build_impact_summary,
    build_spearman_detailed,
    build_spearman_forest_table,
)
from .claims_loading import (
    CLAIM_STATUS_ORDER,
    build_prompt_summary_from_claims,
    load_claims_explorer_dataset,
)
from .config import (
    ADV_DEFAULT_BOOTSTRAP_ITERS,
    ADV_DEFAULT_PERMUTATION_ITERS,
    ADV_DEFAULT_SEED,
    CREATIVITY_ORDER,
    CREATIVITY_SCORE_OPTIONS,
    DEFAULT_CHART_TYPE,
    DEFAULT_LINE_COLOR,
    DEFAULT_LINE_ESTIMATOR,
    DEFAULT_LINE_FACET,
    DEFAULT_SERIES_A,
    DEFAULT_SERIES_B,
    DEFAULT_X_AXIS,
    DEFAULT_Y_AXIS,
    HEATMAP_CORR_COLS_DEFAULT,
    LLM_EXPORT_DEFAULT_INCLUDE_ADVANCED,
    PAGE_CLAIMS_EXPLORER,
    PAGE_CREATIVITY,
    PAGE_EVALUATOR_AGREEMENT,
    PAGE_IMPACT,
    PAGE_LLM_EXPORT,
)
from .creativity_analytics import (
    HAVE_SCIPY as HAVE_SCIPY_ADV,
    HAVE_STATSMODELS,
    build_creativity_corr_table,
    build_intent_contrast_table,
    build_mediation_table,
    build_partial_corr_table,
    build_task_model_parameter_corr_table,
    fit_binomial_glm_stratified_by_model,
    fit_prompt_mixedlm_stratified_by_model,
    assess_creativity_level_homogeneity,
    fit_prompt_mixedlm,
    fit_binomial_glm,
    predict_binomial_glm_probabilities,
)
from .creativity_loading import load_creativity_dataset
from .creativity_plotting import (
    build_creativity_heatmap,
    build_creativity_metrics_by_level_plot,
    build_creativity_score_distributions_plot,
    build_creativity_scatter,
    build_glm_forest,
    build_intent_boxplot,
    build_task_parameter_heatmap_by_model,
    build_task_model_parameter_corr_plot,
)
from .data_loading import load_prompt_dataset
from .tr_relabeling import add_tr_relabeling_columns
from .evaluator_agreement import (
    build_component_pairwise_agreement_table,
    build_diversity_and_hallucination_stats,
    build_evaluator_claim_consensus,
    build_evaluator_support_tables,
    build_extraction_agreement,
    build_pairwise_agreement_table,
    build_pairwise_metric_matrix,
    load_evaluation_test_claims_dataset,
)
from .impact_figure_cache import (
    build_per_model_temp_figure_from_stats,
    figure_from_json,
    run_impact_claim_density_plot_cached,
    run_impact_mediation_diagram_cached,
    run_impact_per_model_temp_stats_cached,
    run_impact_prompt_variance_plot_cached,
)
from .latex_export import build_main_chart_line_latex_groupplot
from .plotting import (
    build_distribution_plot,
    build_line_plot,
    build_points_plot,
    build_spearman_forest_plot,
)
from .utils import apply_multiselect_filter, option_index, sorted_unique


@st.cache_data(show_spinner=True)
def load_prompt_dataset_cached(selected_roots: tuple[str, ...]) -> pd.DataFrame:
    return load_prompt_dataset(selected_roots)


@st.cache_data(show_spinner=True)
def load_creativity_dataset_cached(
    selected_roots: tuple[str, ...],
    strict_mode: bool = True,
    exclude_incomplete_runs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_creativity_dataset(
        selected_roots=selected_roots,
        strict_mode=strict_mode,
        exclude_incomplete_runs=exclude_incomplete_runs,
    )


@st.cache_data(show_spinner=True, persist=True)
def load_claims_explorer_dataset_cached(
    selected_roots: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_claims_explorer_dataset(selected_roots)


@st.cache_data(show_spinner=True, persist=True)
def load_evaluation_test_claims_dataset_cached(
    selected_roots: tuple[str, ...],
    version: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_evaluation_test_claims_dataset(selected_roots)


@st.cache_data(show_spinner=False, persist=True)
def build_component_pairwise_agreement_cached(
    claims_df: pd.DataFrame,
    component_col: str,
    min_overlap: int,
    by_task: bool,
    control_other_stages: bool,
) -> pd.DataFrame:
    return build_component_pairwise_agreement_table(
        claims_df,
        component_col=component_col,
        min_overlap=min_overlap,
        by_task=by_task,
        control_other_stages=control_other_stages,
    )


@st.cache_data(show_spinner=False, persist=True)
def build_extraction_agreement_cached(
    claims_df: pd.DataFrame,
    min_generations: int,
    by_task: bool,
) -> pd.DataFrame:
    return build_extraction_agreement(
        claims_df,
        min_generations=min_generations,
        by_task=by_task,
    )


@st.cache_data(show_spinner=False, persist=True)
def build_diversity_and_hallucination_stats_cached(
    claims_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return build_diversity_and_hallucination_stats(claims_df)


@st.cache_data(show_spinner=False, persist=True)
def run_impact_corr_cached(
    filtered_df: pd.DataFrame,
    corr_metrics: tuple[str, ...],
    n_boot: int,
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    return build_creativity_corr_table(
        filtered_df,
        target_col="hallucination_rate",
        metrics=list(corr_metrics),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_task_model_corr_cached(
    filtered_df: pd.DataFrame,
    metrics: tuple[str, ...],
    min_n: int,
) -> pd.DataFrame:
    return build_task_model_parameter_corr_table(
        filtered_df,
        target_col="hallucination_rate",
        metrics=list(metrics),
        model_col="model_name",
        task_col="task",
        method="spearman",
        min_n=min_n,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_spearman_forest_cached(
    filtered_df: pd.DataFrame,
    min_n: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return build_spearman_forest_table(
        filtered_df,
        target_col="hallucination_rate",
        min_n=min_n,
        n_boot=n_boot,
        seed=seed,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_partial_cached(
    filtered_df: pd.DataFrame,
    partial_metrics: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return build_partial_corr_table(
        filtered_df,
        target_col="hallucination_rate",
        metrics=list(partial_metrics),
        control_cols=("response_length_words", "n_claim_rows"),
        n_boot=n_boot,
        seed=seed,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_intent_cached(filtered_df: pd.DataFrame) -> pd.DataFrame:
    return build_intent_contrast_table(
        filtered_df,
        target_col="hallucination_rate",
        level_col="creativity_level",
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_kappa_intent_cached(filtered_df: pd.DataFrame) -> pd.DataFrame:
    return build_intent_contrast_table(
        filtered_df,
        target_col="hallucination_rate",
        level_col="kappa_level",
        level_order=["LOW", "HIGH"],
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_kappa_mediation_cached(
    filtered_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    work = filtered_df.copy()
    work["kappa_rank"] = work["kappa_level"].map({"LOW": 0, "HIGH": 1})
    return build_mediation_table(
        work,
        x_col="kappa_rank",
        mediator_col="n_claims",
        target_col="hallucination_rate",
        control_cols=("length_words",),
        n_boot=n_boot,
        seed=seed,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_glm_cached(filtered_df: pd.DataFrame) -> pd.DataFrame:
    return fit_binomial_glm(filtered_df, target_rate_col="hallucination_rate")


@st.cache_data(show_spinner=False, persist=True)
def run_impact_glm_stratified_cached(
    filtered_df: pd.DataFrame,
    focus_model: str,
) -> pd.DataFrame:
    return fit_binomial_glm_stratified_by_model(
        filtered_df,
        target_rate_col="hallucination_rate",
        focus_model=focus_model,
        model_col="model_name",
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_mediation_cached(
    filtered_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return build_mediation_table(
        filtered_df,
        x_col="creativity_rank",
        mediator_col="n_claims",
        target_col="hallucination_rate",
        control_cols=("length_words",),
        n_boot=n_boot,
        seed=seed,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_mixedlm_cached(filtered_df: pd.DataFrame) -> pd.DataFrame:
    return fit_prompt_mixedlm(
        filtered_df,
        target_col="hallucination_rate",
        group_col="prompt_id",
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_mixedlm_stratified_cached(
    filtered_df: pd.DataFrame,
    focus_model: str,
) -> pd.DataFrame:
    return fit_prompt_mixedlm_stratified_by_model(
        filtered_df,
        target_col="hallucination_rate",
        group_col="prompt_id",
        focus_model=focus_model,
        model_col="model_name",
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_homogeneity_cached(
    filtered_df: pd.DataFrame,
    focus_model: str,
) -> pd.DataFrame:
    return assess_creativity_level_homogeneity(
        filtered_df,
        target_rate_col="hallucination_rate",
        group_col="prompt_id",
        focus_model=focus_model,
        model_col="model_name",
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_advanced_analysis_cached(
    filtered_df: pd.DataFrame,
    corr_metrics: tuple[str, ...],
    partial_metrics: tuple[str, ...],
    n_boot: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corr_df = run_impact_corr_cached(
        filtered_df=filtered_df,
        corr_metrics=corr_metrics,
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    partial_df = run_impact_partial_cached(
        filtered_df=filtered_df,
        partial_metrics=partial_metrics,
        n_boot=n_boot,
        seed=seed,
    )
    intent_df = run_impact_intent_cached(filtered_df=filtered_df)
    glm_df = run_impact_glm_cached(filtered_df=filtered_df)
    mediation_df = run_impact_mediation_cached(
        filtered_df=filtered_df,
        n_boot=n_boot,
        seed=seed,
    )
    mixedlm_df = run_impact_mixedlm_cached(filtered_df=filtered_df)
    return corr_df, partial_df, intent_df, glm_df, mediation_df, mixedlm_df


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    if v != 0 and abs(v) < 1e-4:
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def _format_small_numeric_value(
    value: Any,
    small_threshold: float = 1e-4,
    fixed_digits: int = 6,
    sci_digits: int = 3,
) -> str:
    try:
        v = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(v):
        return "NA"
    if v == 0:
        return "0"
    if abs(v) < small_threshold:
        return f"{v:.{sci_digits}e}"
    if abs(v) >= 1 and float(v).is_integer():
        return str(int(v))
    return f"{v:.{fixed_digits}f}".rstrip("0").rstrip(".")


def _format_df_for_display(df: pd.DataFrame, small_threshold: float = 1e-4) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        out[col] = out[col].map(
            lambda v: _format_small_numeric_value(v, small_threshold=small_threshold)
        )
    return out


def _show_dataframe(
    df: pd.DataFrame,
    *,
    use_container_width: bool = True,
    hide_index: bool = True,
    small_threshold: float = 1e-4,
) -> None:
    st.dataframe(
        _format_df_for_display(df, small_threshold=small_threshold),
        use_container_width=use_container_width,
        hide_index=hide_index,
    )


def _to_csv_block(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "Aucune donnée disponible."
    out = df.head(max_rows).copy()
    out = _format_df_for_display(out, small_threshold=1e-4)
    return "```csv\n" + out.to_csv(index=False) + "```"


def _render_copy_button(text_to_copy: str, key: str) -> None:
    button_id = f"copy_btn_{key}"
    status_id = f"copy_status_{key}"
    payload = json.dumps(text_to_copy)
    html = f"""
    <div style="margin-top: 0.25rem;">
      <button id="{button_id}" style="padding: 0.45rem 0.9rem; border-radius: 0.5rem; border: 1px solid #999; cursor: pointer;">
        Copier le texte pour chatbot AI
      </button>
      <span id="{status_id}" style="margin-left: 0.6rem; font-size: 0.9rem; color: #666;"></span>
    </div>
    <script>
      const copyText = {payload};
      const btn = document.getElementById("{button_id}");
      const status = document.getElementById("{status_id}");
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(copyText);
          status.textContent = "Texte copié.";
        }} catch (err) {{
          status.textContent = "Copie impossible, utilisez le champ texte ci-dessous.";
        }}
      }});
    </script>
    """
    components_html(html, height=60)


@st.cache_data(show_spinner=False)
def build_llm_export_text_cached(
    selected_roots: tuple[str, ...],
    max_rows_per_table: int,
) -> str:
    project_context = """
HalluVSintention créative est un projet de recherche qui mesure l'impact des paramètres de génération LLM
(température, longueur, niveau créatif, tâche, modèle) sur les hallucinations. Le but est d'étudier l'impact de l'intention créative (définie par la combinaison de ces paramètres) sur les hallucinations, et d'identifier des configurations plus sûres.
Métriques hallucination au niveau prompt: n_claims, support_rate, hallucination_rate.
Données par run sous output/* avec run_config.json, generation.jsonl, output.csv et éventuellement creativity.jsonl.
""".strip()

    impact_df = load_prompt_dataset_cached(selected_roots)
    impact_metrics = impact_df[impact_df["metrics_available"]].copy() if not impact_df.empty else pd.DataFrame()

    # Impact summaries
    impact_rows = len(impact_metrics)
    impact_runs = int(impact_metrics["run_id"].nunique()) if not impact_metrics.empty else 0
    impact_prompts = int(impact_metrics["prompt_id"].nunique()) if not impact_metrics.empty else 0
    impact_hallu_mean = float(impact_metrics["hallucination_rate"].mean()) if not impact_metrics.empty else np.nan
    impact_support_mean = float(impact_metrics["support_rate"].mean()) if not impact_metrics.empty else np.nan

    impact_by_length = (
        build_impact_summary(impact_metrics, x_axis="length_words", y_axis="hallucination_rate")
        if not impact_metrics.empty and "length_words" in impact_metrics.columns
        else pd.DataFrame()
    )
    impact_by_temp = (
        build_impact_summary(impact_metrics, x_axis="temperature", y_axis="hallucination_rate")
        if not impact_metrics.empty and "temperature" in impact_metrics.columns
        else pd.DataFrame()
    )
    impact_by_task = (
        build_impact_summary(impact_metrics, x_axis="task", y_axis="hallucination_rate")
        if not impact_metrics.empty and "task" in impact_metrics.columns
        else pd.DataFrame()
    )
    spearman_df, spearman_skipped = (
        build_spearman_detailed(impact_metrics, target_col="hallucination_rate", min_n=20)
        if not impact_metrics.empty
        else (pd.DataFrame(), [])
    )
    spearman_top = spearman_df.head(max_rows_per_table) if not spearman_df.empty else pd.DataFrame()

    impact_task_model_metrics = [
        c
        for c in [
            "temperature",
            "length_words",
            "response_length_words",
            "response_length_tokens",
            "n_claims",
            "n_claim_rows",
            "creativity_rank",
            "ttct_overall",
            "ttcw_overall",
            "creativity_composite",
        ]
        if c in impact_metrics.columns and pd.api.types.is_numeric_dtype(impact_metrics[c])
    ]
    impact_task_model_corr = (
        run_impact_task_model_corr_cached(
            filtered_df=impact_metrics,
            metrics=tuple(impact_task_model_metrics),
            min_n=20,
        )
        if not impact_metrics.empty and impact_task_model_metrics
        else pd.DataFrame()
    )

    # Creativity page dataset (strict + complete runs)
    creativity_df, coverage_df = load_creativity_dataset(
        selected_roots=selected_roots,
        strict_mode=True,
        exclude_incomplete_runs=True,
        base_df=impact_df,
    )
    coverage_stats = coverage_df["status"].value_counts(dropna=False).to_dict() if not coverage_df.empty else {}
    complete_runs = int(coverage_stats.get("complete", 0))
    partial_runs = int(coverage_stats.get("partial", 0))
    missing_runs = int(coverage_stats.get("missing", 0))

    creativity_rows = len(creativity_df)
    creativity_runs = int(creativity_df["run_id"].nunique()) if not creativity_df.empty else 0
    creativity_hallu_mean = float(creativity_df["hallucination_rate"].mean()) if not creativity_df.empty else np.nan
    creativity_ttct_mean = float(creativity_df["ttct_overall"].mean()) if not creativity_df.empty else np.nan
    creativity_ttcw_mean = float(creativity_df["ttcw_overall"].mean()) if not creativity_df.empty else np.nan

    creativity_by_level = pd.DataFrame()
    if not creativity_df.empty and "creativity_level" in creativity_df.columns:
        creativity_by_level = (
            creativity_df.groupby("creativity_level", dropna=False)
            .agg(
                n=("creativity_level", "count"),
                hallucination_rate_mean=("hallucination_rate", "mean"),
                ttct_overall_mean=("ttct_overall", "mean"),
                ttcw_overall_mean=("ttcw_overall", "mean"),
                creativity_composite_mean=("creativity_composite", "mean"),
            )
            .reset_index()
            .sort_values("creativity_level")
        )
    creativity_by_task = (
        creativity_df.groupby("task", dropna=False)
        .agg(
            n=("task", "count"),
            hallucination_rate_mean=("hallucination_rate", "mean"),
            ttct_overall_mean=("ttct_overall", "mean"),
            ttcw_overall_mean=("ttcw_overall", "mean"),
            creativity_composite_mean=("creativity_composite", "mean"),
        )
        .reset_index()
        .sort_values(["hallucination_rate_mean", "n"], ascending=[False, False])
        if not creativity_df.empty and "task" in creativity_df.columns
        else pd.DataFrame()
    )
    creativity_corr_matrix = pd.DataFrame()
    creativity_corr_cols = [c for c in HEATMAP_CORR_COLS_DEFAULT if c in creativity_df.columns]
    if len(creativity_corr_cols) >= 2 and not creativity_df.empty:
        creativity_corr_matrix = creativity_df[creativity_corr_cols].corr(method="spearman")
        creativity_corr_matrix = creativity_corr_matrix.reset_index().rename(columns={"index": "metric"})

    # Responses & Claims page datasets
    claims_df, prompt_summary_df, claims_coverage_df = load_claims_explorer_dataset_cached(selected_roots)
    claims_runs = int(claims_df["run_id"].nunique()) if not claims_df.empty else 0
    claims_rows = int(len(claims_df))
    claims_prompts = int(prompt_summary_df["prompt_clean"].nunique()) if not prompt_summary_df.empty else 0
    claims_status_df = (
        claims_df["verification_status"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="count")
        if not claims_df.empty
        else pd.DataFrame()
    )
    claims_verified = (
        claims_df[
            claims_df["verification_status"].astype(str).isin(["supported", "hallucinated"])
        ].copy()
        if not claims_df.empty
        else pd.DataFrame()
    )
    claims_support_rate = float((claims_verified["verification_status"].astype(str) == "supported").mean()) if not claims_verified.empty else np.nan
    claims_hallucination_rate = float((claims_verified["verification_status"].astype(str) == "hallucinated").mean()) if not claims_verified.empty else np.nan
    claims_task_creativity_support = pd.DataFrame()
    if not claims_verified.empty:
        claims_task_creativity_support = (
            claims_verified.groupby(["task", "creativity_level"], dropna=False)
            .agg(
                n_verified=("claim_row_id", "count"),
                n_supported=("verification_status", lambda s: (s.astype(str) == "supported").sum()),
            )
            .reset_index()
        )
        claims_task_creativity_support["support_rate"] = np.where(
            claims_task_creativity_support["n_verified"] > 0,
            claims_task_creativity_support["n_supported"] / claims_task_creativity_support["n_verified"],
            np.nan,
        )
    claims_risky_prompts = pd.DataFrame()
    if not prompt_summary_df.empty:
        risky_cols = [
            c
            for c in [
                "root_name",
            "model_name",
            "task",
            "creativity_level",
            "kappa_level",
            "title",
            "n_verified",
            "n_supported",
                "n_hallucinated",
                "hallucination_rate_claim_level",
                "support_rate_claim_level",
                "claim_density_per_100_words",
            ]
            if c in prompt_summary_df.columns
        ]
        claims_risky_prompts = (
            prompt_summary_df.sort_values(
                by=["hallucination_rate_claim_level", "n_verified"],
                ascending=[False, False],
            )
            .head(max_rows_per_table)
            .copy()
        )
        if risky_cols:
            claims_risky_prompts = claims_risky_prompts[risky_cols]
    claims_diag_df = pd.DataFrame()
    if not prompt_summary_df.empty:
        diag_group_cols = [
            c
            for c in ["root_name", "model_name", "task", "creativity_level"]
            if c in prompt_summary_df.columns
        ]
        if not diag_group_cols:
            diag_group_cols = ["prompt_clean"]
        claims_diag_df = (
            prompt_summary_df.groupby(diag_group_cols, dropna=False)
            .agg(
                prompts=("prompt_clean", "nunique"),
                claims_verified=("n_verified", "sum"),
                hallucination_rate_mean=("hallucination_rate_claim_level", "mean"),
                precision_mean=("precision_mean", "mean"),
                recall_mean=("recall_mean", "mean"),
                f1_mean=("f1_mean", "mean"),
            )
            .reset_index()
            .sort_values(["hallucination_rate_mean", "claims_verified"], ascending=[False, False])
        )

    # Evaluator Agreement page datasets
    eval_claims_df, eval_runs_df = load_evaluation_test_claims_dataset_cached(selected_roots, version=2)
    eval_runs = int(eval_runs_df["run_id"].nunique()) if not eval_runs_df.empty else 0
    eval_claim_rows = int(len(eval_claims_df))
    eval_consensus_df = build_evaluator_claim_consensus(eval_claims_df) if not eval_claims_df.empty else pd.DataFrame()
    eval_evaluators = int(eval_consensus_df["evaluator_label"].nunique()) if not eval_consensus_df.empty else 0
    eval_comparable_claims = 0
    if not eval_consensus_df.empty:
        coverage_per_claim = eval_consensus_df.groupby("claim_key", dropna=False)["evaluator_label"].nunique()
        eval_comparable_claims = int((coverage_per_claim >= 2).sum())
    eval_support_overall, eval_support_by_task = build_evaluator_support_tables(eval_consensus_df)
    eval_component_specs = [
        ("claim_extractor_model", "Claim Extractor"),
        ("abstain_evaluator_model", "Abstain Evaluator"),
        ("verifier_model", "Verifier"),
        ("evaluator_label", "Final Pipeline Output"),
    ]
    eval_component_overall_frames: list[pd.DataFrame] = []
    eval_component_task_frames: list[pd.DataFrame] = []
    eval_interchange_rows: list[dict[str, Any]] = []

    def _interchangeability_flag(kappa_min: float, support_delta_max: float) -> str:
        if pd.isna(kappa_min) or pd.isna(support_delta_max):
            return "Insufficient data"
        if kappa_min >= 0.80 and support_delta_max <= 0.05:
            return "Likely interchangeable"
        if kappa_min >= 0.67 and support_delta_max <= 0.10:
            return "Partially interchangeable"
        return "Not interchangeable"

    for component_col, component_label in eval_component_specs:
        comp_overall = build_component_pairwise_agreement_table(
            eval_claims_df,
            component_col=component_col,
            min_overlap=50,
            by_task=False,
            control_other_stages=False,
        )
        comp_task = build_component_pairwise_agreement_table(
            eval_claims_df,
            component_col=component_col,
            min_overlap=50,
            by_task=True,
            control_other_stages=False,
        )
        if not comp_overall.empty:
            eval_component_overall_frames.append(comp_overall.assign(component_label=component_label))
        if not comp_task.empty:
            eval_component_task_frames.append(comp_task.assign(component_label=component_label))

        scopes: list[tuple[str, pd.DataFrame]] = []
        if not comp_overall.empty:
            scopes.append(("ALL", comp_overall))
        if not comp_task.empty:
            scopes.extend([(str(task_name), grp.copy()) for task_name, grp in comp_task.groupby("task", dropna=False)])
        for scope_name, scope_df in scopes:
            if scope_df.empty:
                continue
            kappa_min = pd.to_numeric(scope_df["kappa"], errors="coerce").min()
            support_delta_max = (
                pd.to_numeric(scope_df["support_rate_a"], errors="coerce")
                .sub(pd.to_numeric(scope_df["support_rate_b"], errors="coerce"))
                .abs()
                .max()
            )
            eval_interchange_rows.append(
                {
                    "component": component_label,
                    "scope": scope_name,
                    "n_pairs": int(len(scope_df)),
                    "kappa_mean": float(pd.to_numeric(scope_df["kappa"], errors="coerce").mean()),
                    "kappa_min": float(kappa_min) if pd.notna(kappa_min) else np.nan,
                    "agreement_mean": float(pd.to_numeric(scope_df["agreement_rate"], errors="coerce").mean()),
                    "max_support_delta": float(support_delta_max) if pd.notna(support_delta_max) else np.nan,
                    "min_overlap": int(pd.to_numeric(scope_df["n_overlap"], errors="coerce").min()),
                    "interchangeability": _interchangeability_flag(
                        float(kappa_min) if pd.notna(kappa_min) else np.nan,
                        float(support_delta_max) if pd.notna(support_delta_max) else np.nan,
                    ),
                }
            )

    eval_component_overall = (
        pd.concat(eval_component_overall_frames, ignore_index=True)
        if eval_component_overall_frames
        else pd.DataFrame()
    )
    eval_component_by_task = (
        pd.concat(eval_component_task_frames, ignore_index=True)
        if eval_component_task_frames
        else pd.DataFrame()
    )
    eval_interchange_df = pd.DataFrame(eval_interchange_rows)
    eval_components_coupled = False
    stage_cols = ["claim_extractor_model", "abstain_evaluator_model", "verifier_model"]
    if not eval_runs_df.empty and set(stage_cols).issubset(eval_runs_df.columns):
        ce = eval_runs_df["claim_extractor_model"].fillna("unknown").astype(str)
        ab = eval_runs_df["abstain_evaluator_model"].fillna("unknown").astype(str)
        vf = eval_runs_df["verifier_model"].fillna("unknown").astype(str)
        eval_components_coupled = bool(((ce == ab) & (ab == vf)).all())

    lines: list[str] = []
    lines.append("# HalluLens - Prompt prêt pour analyse par chatbot AI")
    lines.append("")
    lines.append("## Contexte projet")
    lines.append(project_context)
    lines.append("")
    lines.append("## Paramètres de génération du rapport")
    lines.append(f"- selected_roots: {', '.join(selected_roots)}")
    lines.append("")
    lines.append("## Revue globale des données calculées")
    lines.append(
        "- pages_incluses: Impact Hallucinations, Creativity Prism, Responses & Claims, Evaluator Agreement"
    )
    lines.append(
        "- objectif: consolider toutes les sorties statistiques et tableaux calculés dans l'interface pour analyse externe."
    )
    lines.append("")
    lines.append("## Résultats page Impact Hallucinations")
    lines.append(
        f"- rows_metrics_available: {impact_rows} | runs: {impact_runs} | unique_prompts: {impact_prompts}"
    )
    lines.append(
        f"- mean_hallucination_rate: {_format_number(impact_hallu_mean)} | mean_support_rate: {_format_number(impact_support_mean)}"
    )
    lines.append("")
    lines.append("### Impact summary par longueur")
    lines.append(_to_csv_block(impact_by_length, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Impact summary par température")
    lines.append(_to_csv_block(impact_by_temp, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Impact summary par tâche")
    lines.append(_to_csv_block(impact_by_task, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Spearman détaillé (top)")
    lines.append(_to_csv_block(spearman_top, max_rows=max_rows_per_table))
    if spearman_skipped:
        lines.append("")
        lines.append("Facteurs ignorés Spearman: " + "; ".join(spearman_skipped))
    lines.append("")
    lines.append("### Corrélations Spearman par tâche x modèle x paramètre")
    lines.append(_to_csv_block(impact_task_model_corr, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis")
    lines.append(
        "Section optionnelle: peut être ajoutée depuis le cache de la page Impact Hallucinations "
        "(aucun recalcul lourd dans la page LLM Export)."
    )
    lines.append("")
    lines.append("## Résultats page Creativity Prism")
    lines.append(
        f"- strict_complete_runs_only: true | complete_runs: {complete_runs} | partial_runs_excluded: {partial_runs} | missing_runs: {missing_runs}"
    )
    lines.append(
        f"- scored_rows: {creativity_rows} | scored_runs: {creativity_runs} | "
        f"mean_hallucination_rate: {_format_number(creativity_hallu_mean)} | "
        f"mean_ttct: {_format_number(creativity_ttct_mean)} | mean_ttcw: {_format_number(creativity_ttcw_mean)}"
    )
    lines.append("")
    lines.append("### Couverture des runs créativité")
    lines.append(_to_csv_block(coverage_df, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Résumé par creativity_level")
    lines.append(_to_csv_block(creativity_by_level, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Résumé créativité par tâche")
    lines.append(_to_csv_block(creativity_by_task, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Matrice de corrélations créativité (Spearman)")
    lines.append(_to_csv_block(creativity_corr_matrix, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("## Résultats page Responses & Claims")
    lines.append(
        f"- claim_rows: {claims_rows} | prompts: {claims_prompts} | runs_loaded: {claims_runs} | "
        f"support_rate_verified_claims: {_format_number(claims_support_rate)} | "
        f"hallucination_rate_verified_claims: {_format_number(claims_hallucination_rate)}"
    )
    lines.append("")
    lines.append("### Couverture des runs claims")
    lines.append(_to_csv_block(claims_coverage_df, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Distribution des statuts de vérification")
    lines.append(_to_csv_block(claims_status_df, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Support rate claims par tâche x créativité")
    lines.append(_to_csv_block(claims_task_creativity_support, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Prompts les plus à risque (claim-level)")
    lines.append(_to_csv_block(claims_risky_prompts, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Diagnostics agrégés claims")
    lines.append(_to_csv_block(claims_diag_df, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("## Résultats page Evaluator Agreement")
    lines.append(
        f"- evaluation_test_runs: {eval_runs} | claim_rows: {eval_claim_rows} | "
        f"evaluators: {eval_evaluators} | comparable_claims_ge_2_evaluators: {eval_comparable_claims} | "
        "pairwise_min_overlap: 50 | mode: comparable (no strict stage control)"
    )
    if eval_components_coupled:
        lines.append(
            "- note_identifiability: claim_extractor/abstain_evaluator/verifier sont couplés (même modèle par run), "
            "les effets par composant ne sont pas identifiables séparément."
        )
    lines.append("")
    lines.append("### Runs evaluation_test détectés")
    lines.append(_to_csv_block(eval_runs_df, max_rows=max_rows_per_table))
    lines.append("")
    
    # Add diversity, hallucination, and extraction agreement metrics
    eval_div_overall, eval_div_task, eval_halu_overall, eval_halu_task = build_diversity_and_hallucination_stats(
        eval_claims_df
    ) if not eval_claims_df.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    eval_extraction = build_extraction_agreement(
        eval_claims_df, min_generations=5, by_task=False
    ) if not eval_claims_df.empty else pd.DataFrame()
    eval_extraction_task = build_extraction_agreement(
        eval_claims_df, min_generations=5, by_task=True
    ) if not eval_claims_df.empty else pd.DataFrame()
    
    lines.append("### Diversité d'extraction (claims/génération) — global")
    lines.append(_to_csv_block(eval_div_overall, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Diversité d'extraction (claims/génération) — par tâche")
    lines.append(_to_csv_block(eval_div_task, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Taux d'hallucination par évaluateur — global")
    lines.append(_to_csv_block(eval_halu_overall, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Taux d'hallucination par évaluateur — par tâche")
    lines.append(_to_csv_block(eval_halu_task, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Accord d'extraction (Jaccard) — global")
    lines.append(_to_csv_block(eval_extraction, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Accord d'extraction (Jaccard) — par tâche")
    lines.append(_to_csv_block(eval_extraction_task, max_rows=max_rows_per_table))
    lines.append("")
    
    lines.append("### Support-rate par évaluateur (overall)")
    lines.append(_to_csv_block(eval_support_overall, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Support-rate par évaluateur et tâche")
    lines.append(_to_csv_block(eval_support_by_task, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Accord pairwise par composant (overall)")
    lines.append(_to_csv_block(eval_component_overall, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Accord pairwise par composant et tâche")
    lines.append(_to_csv_block(eval_component_by_task, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Diagnostic d'interchangeabilité par composant")
    lines.append(_to_csv_block(eval_interchange_df, max_rows=max_rows_per_table))
    lines.append("")
    lines.append("## Instructions pour le chatbot AI")
    lines.append(
        "Analyse ce rapport en identifiant: (1) les effets les plus robustes, "
        "(2) les facteurs potentiellement confondants, (3) l'accord inter-évaluateurs par composant "
        "et par tâche, (4) l'interchangeabilité réelle des modèles, (5) les recommandations expérimentales "
        "priorisées, (6) 5 prochaines analyses à lancer."
    )
    lines.append(
        "Utilise d'abord les tailles d'effet et intervalles de confiance, puis la significativité corrigée FDR."
    )
    return "\n".join(lines)


def _get_cached_impact_advanced_for_export(
    selected_roots: tuple[str, ...],
) -> dict[str, Any] | None:
    payload = st.session_state.get("impact_advanced_export_cache")
    if not isinstance(payload, dict):
        return None
    payload_roots = tuple(payload.get("selected_roots", ()))
    if payload_roots != tuple(selected_roots):
        return None
    required_df = [
        "corr_df",
        "partial_df",
        "intent_df",
        "kappa_intent_df",
        "glm_df",
        "mediation_df",
        "kappa_mediation_df",
        "mixedlm_df",
        "glm_strat_df",
        "mixedlm_strat_df",
        "homogeneity_df",
    ]
    if any(not isinstance(payload.get(k), pd.DataFrame) for k in required_df):
        return None
    return payload


def _append_cached_impact_advanced_to_export_text(
    export_text: str,
    payload: dict[str, Any] | None,
    include_advanced_analysis: bool,
    max_rows_per_table: int,
) -> str:
    if not include_advanced_analysis:
        return export_text

    lines = [export_text, "", "## Advanced analysis (cache Impact Hallucinations)"]
    if payload is None:
        lines.append(
            "- status: indisponible (aucun cache correspondant). "
            "Exécuter `Run advanced analysis` dans la page Impact Hallucinations, puis regénérer l'export."
        )
        return "\n".join(lines)

    seed = payload.get("seed", "NA")
    n_boot = payload.get("n_boot", "NA")
    n_perm = payload.get("n_perm", "NA")
    n_rows = payload.get("n_rows", "NA")
    generated_at = payload.get("generated_at", "NA")
    lines.append(
        f"- status: utilisé depuis cache | generated_at_utc: {generated_at} | "
        f"rows_analysed: {n_rows} | seed: {seed} | bootstrap: {n_boot} | permutation: {n_perm}"
    )
    lines.append("")
    lines.append("### Advanced analysis - robust correlations")
    lines.append(_to_csv_block(payload.get("corr_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - partial correlations")
    lines.append(_to_csv_block(payload.get("partial_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - paired contrasts (creativity_level)")
    lines.append(_to_csv_block(payload.get("intent_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - paired contrasts (kappa_level)")
    lines.append(_to_csv_block(payload.get("kappa_intent_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - GLM binomial (+ kappa_level si disponible)")
    lines.append(_to_csv_block(payload.get("glm_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - mediation (creativity_rank -> n_claims -> hallucination_rate | control length_words)")
    lines.append(_to_csv_block(payload.get("mediation_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - mediation (kappa_rank -> n_claims -> hallucination_rate | control length_words)")
    lines.append(_to_csv_block(payload.get("kappa_mediation_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - mixed model (hallucination_rate ~ creativity_level * task + length_words [+ kappa_level] + (1|prompt_id))")
    lines.append(_to_csv_block(payload.get("mixedlm_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - GLM stratifié par modèle")
    lines.append(_to_csv_block(payload.get("glm_strat_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - mixed model stratifié par modèle")
    lines.append(_to_csv_block(payload.get("mixedlm_strat_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - homogénéité creativity_level")
    lines.append(_to_csv_block(payload.get("homogeneity_df"), max_rows=max_rows_per_table))
    return "\n".join(lines)


def render_llm_export_page(selected_roots: tuple[str, ...]) -> None:
    st.subheader("LLM Export")
    st.caption(
        "Génère un texte structuré (contexte projet + résultats Impact/Creativity/Claims/Evaluator Agreement) "
        "prêt à coller dans un chatbot AI."
    )
    tab_generate, tab_help = st.tabs(["Génération", "Aide"])

    with tab_generate:
        cached_advanced = _get_cached_impact_advanced_for_export(selected_roots)
        include_advanced_analysis = st.checkbox(
            "Inclure advanced analysis (depuis cache Impact)",
            value=LLM_EXPORT_DEFAULT_INCLUDE_ADVANCED,
            key="llm_export_include_advanced",
            help=(
                "N'effectue pas de calcul lourd dans cette page: utilise uniquement les résultats "
                "déjà calculés dans Impact Hallucinations."
            ),
        )
        col_a, col_b = st.columns(2)
        max_rows_per_table = int(
            col_a.number_input(
                "Max lignes/table",
                min_value=5,
                max_value=200,
                value=20,
                step=1,
                key="llm_export_max_rows",
            )
        )
        col_b.caption(
            "Mode export: les analyses avancées sont reprises depuis le cache Impact. "
            "Les sections Claims/Evaluator sont consolidées depuis les datasets déjà chargés."
        )
        if include_advanced_analysis and cached_advanced is None:
            st.warning(
                "Advanced analysis non trouvée dans le cache pour les roots sélectionnés. "
                "Lancez `Run advanced analysis` dans Impact Hallucinations puis revenez ici."
            )
        elif include_advanced_analysis and cached_advanced is not None:
            st.success("Advanced analysis trouvée dans le cache Impact et sera incluse dans l'export.")

        if st.button("Générer le texte pour chatbot AI", type="primary", key="llm_export_generate_btn"):
            st.session_state["llm_export_ready"] = True

        if not st.session_state.get("llm_export_ready", False):
            st.info("Cliquez sur `Générer le texte pour chatbot AI`.")
            return

        progress_text = st.empty()
        progress_bar = st.progress(0.0)
        progress_text.info("Calcul 1/2: construction du package LLM de base...")
        export_text = build_llm_export_text_cached(
            selected_roots=selected_roots,
            max_rows_per_table=max_rows_per_table,
        )
        progress_bar.progress(0.7)
        progress_text.info("Calcul 2/2: ajout des résultats advanced depuis le cache Impact...")
        export_text = _append_cached_impact_advanced_to_export_text(
            export_text=export_text,
            payload=cached_advanced,
            include_advanced_analysis=include_advanced_analysis,
            max_rows_per_table=max_rows_per_table,
        )
        progress_bar.progress(1.0)
        progress_text.success("Génération du texte export terminée.")

        _render_copy_button(export_text, key="llm_export")
        st.download_button(
            "Télécharger le texte (.md)",
            data=export_text.encode("utf-8"),
            file_name="hallulens_llm_export.md",
            mime="text/markdown",
            key="llm_export_download_btn",
        )
        st.text_area(
            "Texte prêt à coller dans un chatbot AI",
            value=export_text,
            height=700,
            key="llm_export_textarea",
        )

    with tab_help:
        st.markdown(
            """
            ### Utilisation recommandée
            - 1) Sélectionnez les `Output roots` globaux dans la sidebar.
            - 2) Si vous voulez l'advanced analysis dans l'export, lancez d'abord **Run advanced analysis** dans la page Impact Hallucinations.
            - 3) Revenez ici et cliquez sur **Générer le texte pour chatbot AI**.
            - 4) Cliquez sur **Copier le texte pour chatbot AI**.
            - 5) Collez dans votre chatbot et demandez une analyse critique des résultats.

            ### Format du texte généré
            - Contexte projet HalluLens
            - Paramètres de génération du rapport
            - Résultats de la page Impact Hallucinations
            - Résultats de la page Creativity Prism
            - Résultats de la page Responses & Claims
            - Résultats de la page Evaluator Agreement
            - Instructions explicites au chatbot pour l'analyse
            """
        )


def render_impact_data_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    metrics_only = st.sidebar.checkbox("Metrics available only", value=True, key="impact_metrics_only")
    if metrics_only:
        df = df[df["metrics_available"]]

    root_filter = st.sidebar.multiselect(
        "Dataset",
        sorted_unique(df["root_name"]),
        default=sorted_unique(df["root_name"]),
        key="impact_root_filter",
    )
    model_filter = st.sidebar.multiselect(
        "Model",
        sorted_unique(df["model_name"]),
        default=sorted_unique(df["model_name"]),
        key="impact_model_filter",
    )
    task_filter = st.sidebar.multiselect(
        "Task",
        sorted_unique(df["task"]),
        default=sorted_unique(df["task"]),
        key="impact_task_filter",
    )
    creativity_filter = st.sidebar.multiselect(
        "Creativity",
        sorted_unique(df["creativity_level"]),
        default=sorted_unique(df["creativity_level"]),
        key="impact_creativity_filter",
    )
    kappa_options = sorted_unique(df["kappa_level"]) if "kappa_level" in df.columns else ["NA"]
    kappa_filter = st.sidebar.multiselect(
        "Kappa (truth regime)",
        kappa_options,
        default=kappa_options,
        key="impact_kappa_filter",
    )
    temperature_filter = st.sidebar.multiselect(
        "Temperature",
        sorted_unique(df["temperature"]),
        default=sorted_unique(df["temperature"]),
        key="impact_temperature_filter",
    )
    length_filter = st.sidebar.multiselect(
        "Length words",
        sorted_unique(df["length_words"]),
        default=sorted_unique(df["length_words"]),
        key="impact_length_filter",
    )

    response_words_min_filter = None
    response_words_max_filter = None
    if "response_length_words" in df.columns and df["response_length_words"].notna().any():
        words_min = int(np.nanmin(df["response_length_words"]))
        words_max = int(np.nanmax(df["response_length_words"]))
        if words_min < words_max:
            response_words_min_filter, response_words_max_filter = st.sidebar.slider(
                "Response length words range",
                min_value=words_min,
                max_value=words_max,
                value=(words_min, words_max),
                key="impact_response_words_range",
            )
        else:
            response_words_min_filter, response_words_max_filter = words_min, words_max

    response_tokens_min_filter = None
    response_tokens_max_filter = None
    if "response_length_tokens" in df.columns and df["response_length_tokens"].notna().any():
        tokens_min = int(np.nanmin(df["response_length_tokens"]))
        tokens_max = int(np.nanmax(df["response_length_tokens"]))
        if tokens_min < tokens_max:
            response_tokens_min_filter, response_tokens_max_filter = st.sidebar.slider(
                "Response length tokens range",
                min_value=tokens_min,
                max_value=tokens_max,
                value=(tokens_min, tokens_max),
                key="impact_response_tokens_range",
            )
        else:
            response_tokens_min_filter, response_tokens_max_filter = tokens_min, tokens_max

    if df["n_claims"].notna().any():
        min_claims = int(np.nanmin(df["n_claims"]))
        max_claims = int(np.nanmax(df["n_claims"]))
        min_claims_filter = st.sidebar.slider(
            "Min n_claims",
            min_value=min_claims,
            max_value=max_claims,
            value=min_claims,
            key="impact_min_claims",
        )
    else:
        min_claims_filter = 0

    filtered = df.copy()
    filtered = apply_multiselect_filter(filtered, "root_name", root_filter)
    filtered = apply_multiselect_filter(filtered, "model_name", model_filter)
    filtered = apply_multiselect_filter(filtered, "task", task_filter)
    filtered = apply_multiselect_filter(filtered, "creativity_level", creativity_filter)
    filtered = apply_multiselect_filter(filtered, "kappa_level", kappa_filter)
    filtered = apply_multiselect_filter(filtered, "temperature", temperature_filter)
    filtered = apply_multiselect_filter(filtered, "length_words", length_filter)
    filtered = filtered[filtered["n_claims"].fillna(0) >= min_claims_filter]
    if response_words_min_filter is not None and response_words_max_filter is not None:
        filtered = filtered[
            filtered["response_length_words"].fillna(response_words_min_filter).between(
                response_words_min_filter,
                response_words_max_filter,
                inclusive="both",
            )
        ]
    if response_tokens_min_filter is not None and response_tokens_max_filter is not None:
        filtered = filtered[
            filtered["response_length_tokens"].fillna(response_tokens_min_filter).between(
                response_tokens_min_filter,
                response_tokens_max_filter,
                inclusive="both",
            )
        ]
    return filtered


def render_impact_chart_config(plot_df: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("Chart config")
    chart_options = ["line", "points", "box", "violin"]
    chart_type = st.sidebar.selectbox(
        "Chart type",
        options=chart_options,
        index=option_index(chart_options, DEFAULT_CHART_TYPE),
        key="impact_chart_type",
    )

    y_options = [
        c
        for c in [
            "hallucination_rate",
            "support_rate",
            "n_claims",
            "response_length_words",
            "response_length_tokens",
        ]
        if c in plot_df.columns
    ]
    y_axis = st.sidebar.selectbox(
        "Y axis",
        options=y_options,
        index=option_index(y_options, DEFAULT_Y_AXIS),
        key="impact_y_axis",
    )

    x_options = [
        c
        for c in [
            "temperature",
            "length_words",
            "response_length_words",
            "response_length_tokens",
            "creativity_level",
            "kappa_level",
            "task",
            "model_name",
        ]
        if c in plot_df.columns
    ]
    x_axis = st.sidebar.selectbox(
        "X axis",
        options=x_options,
        index=option_index(x_options, DEFAULT_X_AXIS),
        key="impact_x_axis",
    )

    dims = [
        "temperature",
        "length_words",
        "response_length_words",
        "response_length_tokens",
        "creativity_level",
        "kappa_level",
        "task",
        "model_name",
        "root_name",
    ]
    estimator_options = ["mean", "median"]
    line_estimator = st.sidebar.selectbox(
        "Line estimator",
        options=estimator_options,
        index=option_index(estimator_options, DEFAULT_LINE_ESTIMATOR),
        key="impact_line_estimator",
    )
    publication_style = st.sidebar.checkbox(
        "Style publication académique",
        value=True,
        key="impact_publication_style",
    )

    cfg: dict[str, Any] = {
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "line_estimator": line_estimator,
        "publication_style": publication_style,
        "dims": dims,
    }

    if chart_type == "line":
        st.sidebar.markdown("#### Multi-lignes (comparaison)")
        st.sidebar.caption("Style de ligne = combinaison de paramètres. Légende compacte orientée publication.")

        facet_dims = ["None"] + [c for c in dims if c in plot_df.columns and c != x_axis]
        facet_by_raw = st.sidebar.selectbox(
            "Facet",
            options=facet_dims,
            index=option_index(facet_dims, DEFAULT_LINE_FACET),
            key="impact_facet_line",
        )
        facet_by = None if facet_by_raw == "None" else facet_by_raw

        color_dims = [c for c in dims if c in plot_df.columns and c not in {x_axis, facet_by}]
        if not color_dims:
            st.warning("Pas de dimension disponible pour colorer les lignes.")
            return {"invalid": True}

        if facet_by == "task" and "creativity_level" in color_dims:
            preferred_color = "creativity_level"
        else:
            preferred_color = DEFAULT_LINE_COLOR if DEFAULT_LINE_COLOR in color_dims else color_dims[0]

        line_color_by = st.sidebar.selectbox(
            "Couleur des lignes",
            options=color_dims,
            index=option_index(color_dims, preferred_color),
            key="impact_line_color_by",
        )

        series_candidates = [
            c
            for c in dims
            if c in plot_df.columns and c not in {x_axis, line_color_by, facet_by}
        ]
        if len(series_candidates) < 1:
            st.warning("Pas assez de paramètres pour construire les styles de lignes.")
            return {"invalid": True}

        default_a = DEFAULT_SERIES_A if DEFAULT_SERIES_A in series_candidates else series_candidates[0]
        default_b = "None"

        series_param_a = st.sidebar.selectbox(
            "Paramètre ligne A",
            options=series_candidates,
            index=option_index(series_candidates, default_a),
            key="impact_series_param_a",
        )
        series_candidates_b = ["None"] + [c for c in series_candidates if c != series_param_a]
        series_param_b_raw = st.sidebar.selectbox(
            "Paramètre ligne B",
            options=series_candidates_b,
            index=option_index(series_candidates_b, default_b),
            key="impact_series_param_b",
        )
        series_param_b = None if series_param_b_raw == "None" else series_param_b_raw
        show_std = st.sidebar.checkbox("Afficher écart-type (± std)", value=True, key="impact_show_std")

        cfg.update(
            {
                "facet_by": facet_by,
                "line_color_by": line_color_by,
                "series_param_a": series_param_a,
                "series_param_b": series_param_b,
                "show_std": show_std,
                "color_by": None,
                "show_points": False,
            }
        )
    elif chart_type == "points":
        optional_dims = ["None"] + [c for c in dims if c in plot_df.columns and c != x_axis]
        color_by_raw = st.sidebar.selectbox(
            "Color",
            options=optional_dims,
            index=min(1, len(optional_dims) - 1),
            key="impact_points_color",
        )
        facet_by_raw = st.sidebar.selectbox(
            "Facet",
            options=optional_dims,
            index=0,
            key="impact_points_facet",
        )
        color_by = None if color_by_raw == "None" else color_by_raw
        facet_by = None if facet_by_raw == "None" else facet_by_raw

        cfg.update(
            {
                "facet_by": facet_by,
                "color_by": color_by,
                "show_points": True,
                "show_std": False,
            }
        )
    else:
        optional_dims = ["None"] + [c for c in dims if c in plot_df.columns and c != x_axis]
        color_by_raw = st.sidebar.selectbox(
            "Color",
            options=optional_dims,
            index=min(1, len(optional_dims) - 1),
            key="impact_dist_color",
        )
        facet_by_raw = st.sidebar.selectbox(
            "Facet",
            options=optional_dims,
            index=0,
            key="impact_facet_dist",
        )
        color_by = None if color_by_raw == "None" else color_by_raw
        facet_by = None if facet_by_raw == "None" else facet_by_raw
        show_points = st.sidebar.checkbox("Show points (box/violin)", value=False, key="impact_show_points")

        cfg.update(
            {
                "facet_by": facet_by,
                "color_by": color_by,
                "show_points": show_points,
                "show_std": False,
            }
        )
    return cfg


def render_impact_kpis(filtered: pd.DataFrame) -> None:
    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    kpi_a.metric("Rows", f"{len(filtered):,}")
    kpi_b.metric("Unique prompts", f"{filtered['prompt_id'].nunique():,}")
    kpi_c.metric("Runs", f"{filtered['run_id'].nunique():,}")
    hallu_mean = filtered["hallucination_rate"].mean()
    kpi_d.metric("Mean hallucination", f"{hallu_mean:.3f}" if np.isfinite(hallu_mean) else "NA")


def _build_main_chart_table(df: pd.DataFrame, chart_cfg: dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    x_axis = chart_cfg.get("x_axis")
    y_axis = chart_cfg.get("y_axis")
    if not x_axis or not y_axis or x_axis not in df.columns or y_axis not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    chart_type = str(chart_cfg.get("chart_type", "")).lower()

    def _aggregate(group_cols: list[str], estimator: str) -> pd.DataFrame:
        if not group_cols:
            return pd.DataFrame()
        grouped = (
            work.groupby(group_cols, dropna=False, observed=False)[y_axis]
            .agg(_mean="mean", _median="median", _std="std", n="count")
            .reset_index()
        )
        grouped["_std"] = grouped["_std"].fillna(0.0)
        grouped = grouped.rename(
            columns={
                "_mean": f"{y_axis}_mean",
                "_median": f"{y_axis}_median",
                "_std": f"{y_axis}_std",
            }
        )
        estimator_col = f"{y_axis}_{estimator}"
        source_col = f"{y_axis}_{'mean' if estimator == 'mean' else 'median'}"
        grouped[estimator_col] = grouped[source_col]
        order_cols = [c for c in group_cols if c in grouped.columns]
        if order_cols:
            grouped = grouped.sort_values(order_cols).reset_index(drop=True)
        base_cols = (
            order_cols
            + ["n", estimator_col, f"{y_axis}_mean", f"{y_axis}_median", f"{y_axis}_std"]
        )
        unique_cols = []
        seen: set[str] = set()
        for c in base_cols:
            if c in grouped.columns and c not in seen:
                unique_cols.append(c)
                seen.add(c)
        return grouped[unique_cols]

    if chart_type == "line":
        line_color_by = chart_cfg.get("line_color_by")
        facet_by = chart_cfg.get("facet_by")
        series_a = chart_cfg.get("series_param_a")
        series_b = chart_cfg.get("series_param_b")
        estimator = "mean" if str(chart_cfg.get("line_estimator", "median")) == "mean" else "median"
        group_cols = [x_axis]
        for col in [line_color_by, facet_by, series_a, series_b]:
            if (
                isinstance(col, str)
                and col
                and col in work.columns
                and col not in group_cols
                and col != y_axis
            ):
                group_cols.append(col)
        return _aggregate(group_cols, estimator=estimator)

    color_by = chart_cfg.get("color_by")
    facet_by = chart_cfg.get("facet_by")
    group_cols = [x_axis]
    for col in [color_by, facet_by]:
        if (
            isinstance(col, str)
            and col
            and col in work.columns
            and col not in group_cols
            and col != y_axis
        ):
            group_cols.append(col)
    return _aggregate(group_cols, estimator="median")


def render_spearman_section(filtered: pd.DataFrame, show_header: bool = True) -> None:
    if show_header:
        st.subheader("Résumé Spearman détaillé vs hallucination_rate")
    else:
        st.markdown("#### Résumé Spearman détaillé vs hallucination_rate")
        if not HAVE_SCIPY:
            st.caption("SciPy indisponible: p-values non calculées, uniquement rho.")

    spearman_min_n = 20
    spearman_df, spearman_skipped = build_spearman_detailed(
        filtered,
        target_col="hallucination_rate",
        min_n=spearman_min_n,
    )

    if spearman_df.empty:
        st.info("Aucun facteur exploitable pour Spearman avec les filtres actuels.")
    else:
        display_cols = [
            "factor",
            "modality",
            "factor_type",
            "encoding",
            "n",
            "spearman_rho",
            "abs_rho",
            "p_value",
            "direction",
            "strength",
        ]
        _show_dataframe(spearman_df[display_cols], use_container_width=True, hide_index=True)

    if spearman_skipped:
        st.caption("Facteurs ignorés: " + "; ".join(spearman_skipped))


def render_spearman_forest_section(filtered: pd.DataFrame) -> None:
    st.subheader("Figure: Spearman Correlation Forest Plot")
    st.caption(
        "Forest plot horizontal (dot-and-whisker) trié par |rho|, avec IC bootstrap 95% "
        "et code couleur par famille de facteur."
    )
    fp_a, fp_b, fp_c = st.columns(3)
    forest_top_n = int(
        fp_a.slider(
            "Top factors",
            min_value=10,
            max_value=80,
            value=20,
            step=1,
            key="impact_spearman_forest_top_n",
        )
    )
    forest_n_boot = int(
        fp_b.number_input(
            "Bootstrap iterations (forest)",
            min_value=100,
            max_value=20_000,
            value=200,
            step=100,
            key="impact_spearman_forest_boot",
        )
    )
    forest_seed = int(
        fp_c.number_input(
            "Random seed (forest)",
            min_value=0,
            max_value=1_000_000_000,
            value=42,
            step=1,
            key="impact_spearman_forest_seed",
        )
    )
    forest_run = fp_a.button("Generate forest plot", key="impact_spearman_forest_run", type="primary")
    forest_reset = fp_b.button("Reset forest plot", key="impact_spearman_forest_reset")

    if forest_reset:
        st.session_state["impact_spearman_forest_ready"] = False
    if forest_run:
        st.session_state["impact_spearman_forest_ready"] = True

    if not st.session_state.get("impact_spearman_forest_ready", False):
        st.info(
            "Cliquez sur `Generate forest plot` pour lancer le calcul. "
            "Astuce: utilisez 200-500 itérations pendant l'exploration, puis 2000+ pour export final."
        )
        return

    with st.spinner("Computing Spearman forest plot..."):
        forest_df = run_spearman_forest_cached(
            filtered_df=filtered,
            min_n=20,
            n_boot=forest_n_boot,
            seed=forest_seed,
        )
    if forest_df.empty:
        st.info("Aucune donnée exploitable pour le forest plot Spearman.")
    else:
        forest_fig = build_spearman_forest_plot(forest_df, top_n=forest_top_n)
        forest_fig.update_layout(height=max(420, 22 * forest_top_n + 120))
        st.plotly_chart(forest_fig, use_container_width=True)
        forest_view = forest_df.sort_values("abs_rho", ascending=False).head(forest_top_n).copy()
        _show_dataframe(
            forest_view[
                [
                    "factor",
                    "modality",
                    "spearman_rho",
                    "abs_rho",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "strength",
                    "factor_family",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_task_model_corr_section(filtered: pd.DataFrame) -> None:
    st.markdown("#### Corrélations Spearman par tâche et par modèle")
    st.caption(
        "Comparaison par couple (model_name, task) entre `hallucination_rate` et "
        "`length_words`, `temperature`, `creativity_rank`."
    )
    task_model_metrics = tuple(
        c for c in ["length_words", "temperature", "creativity_rank"] if c in filtered.columns
    )
    task_model_min_n = 8
    task_model_corr_df = run_impact_task_model_corr_cached(
        filtered_df=filtered,
        metrics=task_model_metrics,
        min_n=task_model_min_n,
    )
    if task_model_corr_df.empty:
        st.info(
            "Données insuffisantes pour la corrélation par tâche/modèle "
            f"(min_n={task_model_min_n}, métriques={', '.join(task_model_metrics) if task_model_metrics else 'aucune'})."
        )
        return

    filter_a, filter_b = st.columns(2)
    task_values = sorted_unique(task_model_corr_df["task"])
    metric_values = sorted_unique(task_model_corr_df["metric"])
    hide_tasks = filter_a.multiselect(
        "Masquer tâches",
        options=task_values,
        default=[],
        key="impact_adv_task_model_hide_tasks",
    )
    hide_metrics = filter_b.multiselect(
        "Masquer paramètres",
        options=metric_values,
        default=[],
        key="impact_adv_task_model_hide_metrics",
    )

    task_model_view_df = task_model_corr_df.copy()
    if hide_tasks:
        task_model_view_df = task_model_view_df[
            ~task_model_view_df["task"].astype(str).isin([str(v) for v in hide_tasks])
        ]
    if hide_metrics:
        task_model_view_df = task_model_view_df[
            ~task_model_view_df["metric"].astype(str).isin([str(v) for v in hide_metrics])
        ]

    if task_model_view_df.empty:
        st.info("Aucune ligne restante après masquage des tâches/paramètres.")
        return

    _show_dataframe(
        task_model_view_df[
            [
                "model_name",
                "task",
                "metric",
                "n",
                "r",
                "p_value",
                "p_fdr_bh",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Graphique par tâche: modèles en couleurs, paramètres sur l'axe X.")
    corr_task_model_fig = build_task_model_parameter_corr_plot(
        task_model_view_df,
        metric_order=[m for m in task_model_metrics if m in task_model_view_df["metric"].astype(str).unique().tolist()],
    )
    st.plotly_chart(corr_task_model_fig, use_container_width=True)
    st.caption("Graphique task x paramètres: heatmap des corrélations, séparée par modèle.")
    corr_task_param_heatmap = build_task_parameter_heatmap_by_model(
        task_model_view_df,
        metric_order=[m for m in task_model_metrics if m in task_model_view_df["metric"].astype(str).unique().tolist()],
    )
    st.plotly_chart(corr_task_param_heatmap, use_container_width=True)


def _render_per_model_temperature_effect_section(filtered: pd.DataFrame) -> None:
    st.markdown("#### Figure: Per-Model Temperature Effect")
    st.caption(
        "Small-multiples (1 panel/model): hallucination_rate vs temperature; "
        "ligne = moyenne, zone ombrée = IC bootstrap 95%."
    )
    required = {"model_name", "temperature", "hallucination_rate"}
    if not required.issubset(filtered.columns):
        st.info("Colonnes requises absentes (`model_name`, `temperature`, `hallucination_rate`).")
        return

    c1, c2, c3, c4 = st.columns(4)
    min_n_per_temp = int(
        c1.number_input(
            "Min n per model/temperature",
            min_value=3,
            max_value=200,
            value=8,
            step=1,
            key="impact_temp_effect_min_n",
        )
    )
    temp_boot = int(
        c2.number_input(
            "Bootstrap iterations",
            min_value=50,
            max_value=20_000,
            value=300,
            step=50,
            key="impact_temp_effect_boot",
        )
    )
    temp_seed = int(
        c3.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000_000,
            value=42,
            step=1,
            key="impact_temp_effect_seed",
        )
    )
    n_cols = int(
        c4.selectbox(
            "Panels per row",
            options=[2, 3, 4],
            index=1,
            key="impact_temp_effect_n_cols",
        )
    )
    run_now = st.button("Generate per-model temperature figure", key="impact_temp_effect_run", type="primary")
    if not run_now and not st.session_state.get("impact_temp_effect_ready", False):
        st.info(
            "Cliquez sur `Generate per-model temperature figure` pour calculer la figure."
        )
        return
    st.session_state["impact_temp_effect_ready"] = True

    with st.spinner("Computing per-model temperature effect..."):
        temp_input = filtered[["model_name", "temperature", "hallucination_rate"]].copy()
        temp_stats = run_impact_per_model_temp_stats_cached(
            filtered_df=temp_input,
            min_n_per_temp=min_n_per_temp,
            n_boot=temp_boot,
            seed=temp_seed,
        )
    if temp_stats.empty:
        st.info("Données insuffisantes pour estimer cet effet avec les paramètres courants.")
        return
    fig_temp = build_per_model_temp_figure_from_stats(temp_stats, n_cols=n_cols)

    n_models = int(temp_stats["model_name"].nunique())
    n_rows = int(np.ceil(n_models / max(1, n_cols)))
    fig_temp.update_layout(height=max(420, 280 * n_rows + 140))
    st.plotly_chart(fig_temp, use_container_width=True)
    st.caption(
        "Lecture: si la pente reste proche de 0 dans chaque panel et les IC se recouvrent fortement, "
        "l'effet de température est nul/faible par modèle."
    )
    _show_dataframe(
        temp_stats[
            [
                "model_name",
                "temperature",
                "n",
                "mean",
                "ci_low",
                "ci_high",
                "temp_levels_per_model",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


def _render_claim_density_vs_hallucination_section(filtered: pd.DataFrame) -> None:
    st.markdown("#### Figure: Claim Density vs. Hallucination Rate")
    st.caption(
        "Scatter par prompt: densité de claims `n_claims / response_length_words x 100` "
        "vs `hallucination_rate`, coloré par `creativity_level`, avec LOWESS par niveau."
    )
    required = {"n_claims", "response_length_words", "hallucination_rate", "creativity_level"}
    if not required.issubset(filtered.columns):
        st.info(
            "Colonnes requises absentes (`n_claims`, `response_length_words`, "
            "`hallucination_rate`, `creativity_level`)."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    lowess_frac = float(
        c1.slider(
            "LOWESS frac",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            key="impact_claim_density_lowess_frac",
        )
    )
    min_points = int(
        c2.number_input(
            "Min points/level for LOWESS",
            min_value=3,
            max_value=10_000,
            value=10,
            step=1,
            key="impact_claim_density_min_points",
        )
    )
    marker_opacity = float(
        c3.slider(
            "Point opacity",
            min_value=0.10,
            max_value=0.95,
            value=0.45,
            step=0.05,
            key="impact_claim_density_marker_opacity",
        )
    )
    max_points = int(
        c4.number_input(
            "Max points rendered",
            min_value=1000,
            max_value=200_000,
            value=12_000,
            step=500,
            key="impact_claim_density_max_points",
        )
    )

    density_input = filtered[
        [
            "n_claims",
            "response_length_words",
            "hallucination_rate",
            "creativity_level",
        ]
    ].copy()
    density_valid_points = int(
        (
            pd.to_numeric(density_input["n_claims"], errors="coerce").notna()
            & (pd.to_numeric(density_input["response_length_words"], errors="coerce") > 0)
            & pd.to_numeric(density_input["hallucination_rate"], errors="coerce").notna()
            & density_input["creativity_level"].notna()
        ).sum()
    )
    fig_density_json, density_summary = run_impact_claim_density_plot_cached(
        filtered_df=density_input,
        lowess_frac=lowess_frac,
        min_points_for_lowess=min_points,
        marker_opacity=marker_opacity,
        max_points=max_points,
        sample_seed=42,
    )
    fig_density = figure_from_json(fig_density_json)
    fig_density.update_layout(height=560)
    st.plotly_chart(fig_density, use_container_width=True)
    if density_valid_points > max_points:
        st.caption(
            f"Affichage sous-échantillonné pour la fluidité ({max_points:,} points max affichés)."
        )

    if density_summary.empty:
        st.info("Pas assez de données exploitables pour cette figure.")
        return

    _show_dataframe(
        density_summary[
            [
                "creativity_level",
                "n_points",
                "density_mean",
                "density_median",
                "hallucination_rate_mean",
                "hallucination_rate_median",
                "lowess_drawn",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    no_lowess = density_summary[~density_summary["lowess_drawn"].astype(bool)]
    if not no_lowess.empty:
        missing_levels = ", ".join(no_lowess["creativity_level"].astype(str).tolist())
        st.caption(
            f"LOWESS non tracé pour: {missing_levels}. "
            "Augmentez la couverture ou réduisez `Min points/level for LOWESS`."
        )
    st.caption(
        "Lecture: si les courbes LOWESS restent séparées pour une densité comparable, "
        "l'écart d'hallucination n'est pas expliqué uniquement par la densité de claims."
    )


def _render_prompt_variance_icc_section(filtered: pd.DataFrame) -> None:
    st.markdown("#### Figure: ICC / Prompt-Level Variance")
    st.caption(
        "Distribution du taux moyen d'hallucination au niveau prompt "
        "(moyenné sur les runs/modèles), avec moyenne globale et annotation ICC."
    )
    required = {"prompt_id", "hallucination_rate"}
    if not required.issubset(filtered.columns):
        st.info("Colonnes requises absentes (`prompt_id`, `hallucination_rate`).")
        return

    c1, c2, c3, c4 = st.columns(4)
    n_bins = int(
        c1.slider(
            "Histogram bins",
            min_value=10,
            max_value=120,
            value=40,
            step=2,
            key="impact_icc_hist_bins",
        )
    )
    icc_value = float(
        c2.number_input(
            "ICC annotation",
            min_value=0.0,
            max_value=1.0,
            value=0.595,
            step=0.001,
            format="%.3f",
            key="impact_icc_annot_value",
        )
    )
    show_kde = bool(
        c3.checkbox(
            "Overlay KDE",
            value=True,
            key="impact_icc_show_kde",
        )
    )
    show_mixture = bool(
        c4.checkbox(
            "Overlay Gaussian mixture",
            value=False,
            key="impact_icc_show_mixture",
        )
    )

    c5, c6 = st.columns(2)
    mix_components = int(
        c5.selectbox(
            "Mixture components",
            options=[2, 3, 4],
            index=0,
            disabled=not show_mixture,
            key="impact_icc_mixture_components",
        )
    )
    mix_seed = int(
        c6.number_input(
            "Mixture random seed",
            min_value=0,
            max_value=1_000_000_000,
            value=42,
            step=1,
            disabled=not show_mixture,
            key="impact_icc_mixture_seed",
        )
    )

    icc_input = filtered[["prompt_id", "hallucination_rate"]].copy()
    fig_icc_json, prompt_means_df, icc_summary_df = run_impact_prompt_variance_plot_cached(
        filtered_df=icc_input,
        n_bins=n_bins,
        icc_value=icc_value,
        show_kde=show_kde,
        show_mixture=show_mixture,
        mixture_components=mix_components,
        random_state=mix_seed,
    )
    fig_icc = figure_from_json(fig_icc_json)
    fig_icc.update_layout(height=560)
    st.plotly_chart(fig_icc, use_container_width=True)

    if icc_summary_df.empty:
        st.info("Pas assez de données pour estimer la variance inter-prompts.")
        return

    summary_row = icc_summary_df.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Prompts", f"{int(summary_row.get('n_prompts', 0)):,}")
    m2.metric("Grand mean", _format_number(summary_row.get("grand_mean"), digits=4))
    m3.metric("Std (prompt means)", _format_number(summary_row.get("prompt_mean_std"), digits=4))
    m4.metric("ICC (annotated)", _format_number(summary_row.get("icc_annotated"), digits=3))

    _show_dataframe(
        icc_summary_df[
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
        use_container_width=True,
        hide_index=True,
    )
    if not prompt_means_df.empty:
        tail_n = min(5, len(prompt_means_df))
        st.caption(
            "Prompts extrêmes (moyenne hallucination): "
            f"bas {tail_n} et haut {tail_n} (utile pour analyser la variance captée par l'ICC)."
        )
        extremes = pd.concat(
            [
                prompt_means_df.head(tail_n),
                prompt_means_df.tail(tail_n),
            ],
            ignore_index=True,
        )
        _show_dataframe(
            extremes[
                [
                    "prompt_id",
                    "prompt_mean_hallucination_rate",
                    "n_rows_per_prompt",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    if show_mixture and str(summary_row.get("mixture_status", "")) != "fitted":
        st.caption(
            "Mixture non ajustée (`mixture_status` != fitted). "
            "Augmentez le nombre de points ou désactivez l'overlay mixture."
        )


def _render_impact_advanced_analysis_block(
    filtered: pd.DataFrame,
    selected_roots: tuple[str, ...],
    show_header: bool = True,
) -> None:
    if show_header:
        st.subheader("Advanced analysis")
    else:
        st.markdown("#### Advanced analysis")
    st.caption(
        "Robust analysis with bootstrap CI, permutation p-values, FDR correction, "
        "partial correlations, paired contrasts, and binomial GLM."
    )
    st.caption("Persistent cache activé (disque) pour les calculs avancés.")

    corr_metric_candidates = [
        c
        for c in [
            "temperature",
            "length_words",
            "response_length_words",
            "response_length_tokens",
            "prompt_length_words",
            "n_claims",
            "n_claim_rows",
            "creativity_rank",
            "ttct_overall",
            "ttcw_overall",
            "creativity_composite",
        ]
        if c in filtered.columns and pd.api.types.is_numeric_dtype(filtered[c])
    ]
    corr_metric_candidates = [
        c for c in corr_metric_candidates if filtered[c].dropna().nunique() >= 2
    ]

    default_corr = [
        c
        for c in [
            "temperature",
            "length_words",
            "response_length_words",
            "response_length_tokens",
            "n_claim_rows",
            "creativity_rank",
        ]
        if c in corr_metric_candidates
    ]
    controls = ("response_length_words", "n_claim_rows")
    default_partial = [c for c in ["temperature", "length_words", "creativity_rank"] if c in corr_metric_candidates and c not in controls]

    cfg_a, cfg_b, cfg_c, cfg_d = st.columns(4)
    seed = int(
        cfg_a.number_input(
            "Random seed",
            min_value=0,
            max_value=1_000_000_000,
            value=ADV_DEFAULT_SEED,
            step=1,
            key="impact_adv_seed",
        )
    )
    n_boot = int(
        cfg_b.number_input(
            "Bootstrap iterations",
            min_value=100,
            max_value=100_000,
            value=ADV_DEFAULT_BOOTSTRAP_ITERS,
            step=100,
            key="impact_adv_boot",
        )
    )
    n_perm = int(
        cfg_c.number_input(
            "Permutation iterations",
            min_value=100,
            max_value=200_000,
            value=ADV_DEFAULT_PERMUTATION_ITERS,
            step=100,
            key="impact_adv_perm",
        )
    )
    unstable_threshold = int(
        cfg_d.number_input(
            "Seuil n_claim_rows (instable)",
            min_value=1,
            max_value=200,
            value=10,
            step=1,
            key="impact_adv_unstable_threshold",
        )
    )

    pick_a, pick_b = st.columns(2)
    corr_metrics_selected = pick_a.multiselect(
        "Correlation metrics",
        options=corr_metric_candidates,
        default=default_corr if default_corr else corr_metric_candidates[: min(6, len(corr_metric_candidates))],
        key="impact_adv_corr_metrics",
    )
    partial_metrics_selected = pick_b.multiselect(
        "Partial-correlation metrics",
        options=[c for c in corr_metric_candidates if c not in controls],
        default=default_partial,
        key="impact_adv_partial_metrics",
    )
    focus_model = str(
        st.text_input(
            "Modèle de référence pour stratification (focus vs reste poolé)",
            value="grok-3-mini",
            key="impact_adv_focus_model",
        )
    ).strip()
    if not focus_model:
        focus_model = "grok-3-mini"

    if st.button("Run advanced analysis", type="primary", key="impact_run_advanced_button"):
        st.session_state["impact_run_advanced"] = True
    if st.button("Reset advanced analysis", key="impact_reset_advanced_button"):
        st.session_state["impact_run_advanced"] = False
    if st.button("Clear advanced cache", key="impact_clear_advanced_cache"):
        run_impact_corr_cached.clear()
        run_impact_task_model_corr_cached.clear()
        run_impact_partial_cached.clear()
        run_impact_intent_cached.clear()
        run_impact_kappa_intent_cached.clear()
        run_impact_glm_cached.clear()
        run_impact_glm_stratified_cached.clear()
        run_impact_mediation_cached.clear()
        run_impact_kappa_mediation_cached.clear()
        run_impact_mixedlm_cached.clear()
        run_impact_mixedlm_stratified_cached.clear()
        run_impact_homogeneity_cached.clear()
        run_impact_advanced_analysis_cached.clear()
        run_impact_per_model_temp_stats_cached.clear()
        run_impact_claim_density_plot_cached.clear()
        run_impact_prompt_variance_plot_cached.clear()
        run_impact_mediation_diagram_cached.clear()
        st.success("Advanced cache vidé.")

    if filtered.empty:
        st.warning("No rows available for advanced analysis.")
        return

    st.caption("Calculs rapides affichés automatiquement au chargement.")
    _render_task_model_corr_section(filtered)
    _render_per_model_temperature_effect_section(filtered)
    _render_claim_density_vs_hallucination_section(filtered)
    _render_prompt_variance_icc_section(filtered)

    if not st.session_state.get("impact_run_advanced", False):
        st.info(
            "Les calculs rapides sont affichés ci-dessus. "
            "Cliquez sur `Run advanced analysis` pour lancer les calculs lourds."
        )
        return

    if not corr_metrics_selected:
        st.warning("Select at least one metric for correlation analysis.")
        return
    if "n_claim_rows" in filtered.columns:
        unstable_rows = int((filtered["n_claim_rows"].fillna(0) < unstable_threshold).sum())
        if unstable_rows > 0:
            st.warning(
                f"Alerte stabilité: {unstable_rows} ligne(s) ont n_claim_rows < {unstable_threshold}. "
                "Les taux peuvent être instables."
            )

    if not HAVE_SCIPY_ADV:
        st.warning("SciPy is unavailable. Correlation and partial-correlation sections are disabled.")
    if not HAVE_STATSMODELS:
        st.warning("statsmodels is unavailable. GLM section is disabled.")

    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    total_steps = 11
    step_results = st.empty()
    completed_steps: list[str] = []

    def _push_step_result(label: str, df: pd.DataFrame) -> None:
        completed_steps.append(f"- {label}: {len(df):,} lignes")
        step_results.markdown("**Résultats disponibles au fur et à mesure**\n" + "\n".join(completed_steps))

    progress_text.info("Advanced analysis 1/9: robust correlations...")
    corr_df = run_impact_corr_cached(
        filtered_df=filtered,
        corr_metrics=tuple(corr_metrics_selected),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    _push_step_result("Robust correlations", corr_df)
    progress_bar.progress(1 / total_steps)

    progress_text.info("Advanced analysis 2/9: partial correlations...")
    partial_df = run_impact_partial_cached(
        filtered_df=filtered,
        partial_metrics=tuple(partial_metrics_selected),
        n_boot=n_boot,
        seed=seed,
    )
    _push_step_result("Partial correlations", partial_df)
    progress_bar.progress(2 / total_steps)

    progress_text.info("Advanced analysis 3/11: paired contrasts (creativity_level)...")
    intent_df = run_impact_intent_cached(filtered_df=filtered)
    _push_step_result("Paired contrasts creativity", intent_df)
    progress_bar.progress(3 / total_steps)

    progress_text.info("Advanced analysis 4/11: paired contrasts (kappa_level)...")
    kappa_intent_df = run_impact_kappa_intent_cached(filtered_df=filtered)
    _push_step_result("Paired contrasts kappa", kappa_intent_df)
    progress_bar.progress(4 / total_steps)

    progress_text.info("Advanced analysis 5/11: binomial GLM (+ kappa_level)...")
    glm_df = run_impact_glm_cached(filtered_df=filtered)
    _push_step_result("Binomial GLM", glm_df)
    progress_bar.progress(5 / total_steps)

    progress_text.info("Advanced analysis 6/11: mediation creativity_rank -> n_claims -> hallucination_rate...")
    mediation_df = run_impact_mediation_cached(
        filtered_df=filtered,
        n_boot=n_boot,
        seed=seed,
    )
    _push_step_result("Mediation creativity", mediation_df)
    progress_bar.progress(6 / total_steps)

    progress_text.info("Advanced analysis 7/11: mediation kappa_rank -> n_claims -> hallucination_rate...")
    kappa_mediation_df = run_impact_kappa_mediation_cached(
        filtered_df=filtered,
        n_boot=n_boot,
        seed=seed,
    )
    _push_step_result("Mediation kappa", kappa_mediation_df)
    progress_bar.progress(7 / total_steps)

    progress_text.info("Advanced analysis 8/11: mixed model (+ kappa_level)...")
    mixedlm_df = run_impact_mixedlm_cached(filtered_df=filtered)
    _push_step_result("Mixed model", mixedlm_df)
    progress_bar.progress(8 / total_steps)

    progress_text.info("Advanced analysis 9/11: GLM stratifié par modèle...")
    glm_strat_df = run_impact_glm_stratified_cached(
        filtered_df=filtered,
        focus_model=focus_model,
    )
    _push_step_result("GLM stratifié", glm_strat_df)
    progress_bar.progress(9 / total_steps)

    progress_text.info("Advanced analysis 10/11: mixed model stratifié par modèle...")
    mixedlm_strat_df = run_impact_mixedlm_stratified_cached(
        filtered_df=filtered,
        focus_model=focus_model,
    )
    _push_step_result("Mixed model stratifié", mixedlm_strat_df)
    progress_bar.progress(10 / total_steps)

    progress_text.info("Advanced analysis 11/11: test d'homogénéité creativity_level...")
    homogeneity_df = run_impact_homogeneity_cached(
        filtered_df=filtered,
        focus_model=focus_model,
    )
    _push_step_result("Homogénéité creativity_level", homogeneity_df)
    progress_bar.progress(1.0)
    progress_text.success("Advanced analysis terminée.")
    st.session_state["impact_advanced_export_cache"] = {
        "selected_roots": tuple(selected_roots),
        "corr_df": corr_df.copy(),
        "partial_df": partial_df.copy(),
        "intent_df": intent_df.copy(),
        "kappa_intent_df": kappa_intent_df.copy(),
        "glm_df": glm_df.copy(),
        "mediation_df": mediation_df.copy(),
        "kappa_mediation_df": kappa_mediation_df.copy(),
        "mixedlm_df": mixedlm_df.copy(),
        "glm_strat_df": glm_strat_df.copy(),
        "mixedlm_strat_df": mixedlm_strat_df.copy(),
        "homogeneity_df": homogeneity_df.copy(),
        "focus_model": str(focus_model),
        "seed": int(seed),
        "n_boot": int(n_boot),
        "n_perm": int(n_perm),
        "n_rows": int(len(filtered)),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }

    st.markdown("#### Robust correlations")
    if corr_df.empty:
        st.info("Insufficient data (or unavailable dependencies) for robust correlation analysis.")
    else:
        _show_dataframe(
            corr_df[
                [
                    "metric",
                    "method",
                    "r",
                    "ci_low",
                    "ci_high",
                    "p_fdr_bh",
                    "n",
                    "p_value",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Partial correlations (controls: response_length_words, n_claim_rows)")
    if partial_df.empty:
        st.info("Insufficient data (or unavailable dependencies) for partial correlations.")
    else:
        _show_dataframe(
            partial_df[
                [
                    "metric",
                    "r_partial",
                    "ci_low",
                    "ci_high",
                    "p_fdr_bh",
                    "n",
                    "p_partial",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Creativity-intention paired contrasts")
    if intent_df.empty:
        st.info("Paired contrasts unavailable with current filters.")
    else:
        _show_dataframe(intent_df, use_container_width=True, hide_index=True)

    st.markdown("#### Kappa-level paired contrasts (LOW vs HIGH)")
    if kappa_intent_df.empty:
        st.info("Kappa contrasts unavailable (need kappa_level column with LOW/HIGH values).")
    else:
        _show_dataframe(kappa_intent_df, use_container_width=True, hide_index=True)

    st.markdown("#### Binomial GLM (odds ratios)")
    if glm_df.empty:
        st.info("GLM unavailable (insufficient data or missing statsmodels).")
    else:
        _show_dataframe(
            glm_df[
                [
                    "term",
                    "odds_ratio",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "p_fdr_bh",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        forest = build_glm_forest(glm_df, top_n=20)
        st.plotly_chart(forest, use_container_width=True)

        st.markdown("#### Probas prédites (GLM)")
        st.caption(
            "Traduction des OR en probabilités prédites, à longueur/température/modèle fixés "
            "pour faciliter l'interprétation non-statistique."
        )
        pred_a, pred_b, pred_c = st.columns(3)
        fixed_length = float(
            pred_a.selectbox(
                "Longueur fixée (length_words)",
                options=sorted_unique(filtered["length_words"]) if "length_words" in filtered.columns else [500],
                key="impact_pred_fixed_length",
            )
        )
        fixed_temp = float(
            pred_b.selectbox(
                "Température fixée",
                options=sorted_unique(filtered["temperature"]) if "temperature" in filtered.columns else [0.5],
                key="impact_pred_fixed_temp",
            )
        )
        fixed_model = str(
            pred_c.selectbox(
                "Modèle fixé",
                options=sorted_unique(filtered["model_name"]) if "model_name" in filtered.columns else ["unknown"],
                key="impact_pred_fixed_model",
            )
        )
        if st.button("Voir probas prédites", key="impact_pred_button"):
            pred_df = predict_binomial_glm_probabilities(
                filtered,
                target_rate_col="hallucination_rate",
                fixed_length_words=fixed_length,
                fixed_temperature=fixed_temp,
                fixed_model_name=fixed_model,
            )
            if pred_df.empty:
                st.info("Impossible de calculer les probabilités prédites avec les données actuelles.")
            else:
                _show_dataframe(pred_df, use_container_width=True, hide_index=True)

    st.markdown(
        "#### Mediation: creativity_level -> n_claims -> hallucination_rate (control: length_words)"
    )
    if mediation_df.empty:
        st.info("Mediation analysis unavailable (insufficient data or missing statsmodels).")
    else:
        _show_dataframe(mediation_df, use_container_width=True, hide_index=True)
        st.markdown("##### Figure: Mediation Path Diagram")
        med_fig_json, med_summary = run_impact_mediation_diagram_cached(mediation_df=mediation_df)
        med_fig = figure_from_json(med_fig_json)
        med_fig.update_layout(height=520)
        st.plotly_chart(med_fig, use_container_width=True)
        if not med_summary.empty:
            _show_dataframe(
                med_summary[
                    [
                        "n",
                        "a_path",
                        "a_p_value",
                        "b_path",
                        "b_p_value",
                        "c_prime_direct",
                        "c_prime_p_value",
                        "indirect_ab",
                        "indirect_ci_low",
                        "indirect_ci_high",
                        "indirect_p_boot",
                        "direct_share_pct_abs",
                        "indirect_share_pct_abs",
                        "mediation_type",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )
        med = mediation_df.iloc[0]
        mediation_type = str(med.get("mediation_type", "NA"))
        st.caption(
            "Lecture: `indirect_ab` (a*b) quantifie l'effet indirect via le nombre de claims; "
            "`c_prime_direct` quantifie l'effet direct residual de créativité sur l'hallucination "
            "après contrôle du médiateur."
        )
        st.caption(
            "Dans le diagramme, `|share|` compare les parts de l'effet direct et indirect "
            "sur la base des magnitudes absolues (`|c'|` vs `|ab|`)."
        )
        st.info(
            "Conclusion automatique: "
            f"{mediation_type} | indirect_ci=[{_format_number(med.get('indirect_ci_low'))}, "
            f"{_format_number(med.get('indirect_ci_high'))}] | "
            f"indirect_p_boot={_format_number(med.get('indirect_p_boot'))}"
        )

    st.markdown(
        "#### Mediation: kappa_level -> n_claims -> hallucination_rate (control: length_words)"
    )
    if kappa_mediation_df.empty:
        st.info("Kappa mediation unavailable (need kappa_level with LOW/HIGH values or missing statsmodels).")
    else:
        _show_dataframe(kappa_mediation_df, use_container_width=True, hide_index=True)
        kmed = kappa_mediation_df.iloc[0]
        kmed_type = str(kmed.get("mediation_type", "NA"))
        st.caption(
            "kappa_rank: LOW=0, HIGH=1. Quantifie si la licence d'invention (κ) agit via le nombre "
            "de claims (effet indirect) ou directement sur le taux d'hallucination (effet direct)."
        )
        st.info(
            "Conclusion automatique kappa: "
            f"{kmed_type} | indirect_ci=[{_format_number(kmed.get('indirect_ci_low'))}, "
            f"{_format_number(kmed.get('indirect_ci_high'))}] | "
            f"indirect_p_boot={_format_number(kmed.get('indirect_p_boot'))}"
        )

    st.markdown(
        "#### Mixed model (random intercept prompt_id)"
    )
    if mixedlm_df.empty:
        st.info("Mixed model unavailable (insufficient data or missing statsmodels).")
    else:
        _show_dataframe(mixedlm_df, use_container_width=True, hide_index=True)
        mm = mixedlm_df.iloc[0]
        st.caption(
            "Modèle: hallucination_rate ~ creativity_level * task + length_words [+ kappa_level] + (1|prompt_id). "
            "kappa_level ajouté si colonne présente. "
            "L'ICC indique la part de variance attribuable aux différences inter-prompts."
        )
        st.info(
            f"ICC={_format_number(mm.get('icc'))} | "
            f"var_random_prompt={_format_number(mm.get('var_random_prompt'))} | "
            f"var_residual={_format_number(mm.get('var_residual'))}"
        )

    st.markdown("#### GLM binomial stratifié par modèle")
    st.caption(
        f"Régression répliquée par strate modèle (individuel + `others_pooled_excluding_{focus_model}`) "
        "avec focus sur les termes `creativity_level`."
    )
    if glm_strat_df.empty:
        st.info("GLM stratifié indisponible (données insuffisantes par strate).")
    else:
        glm_strat_view = glm_strat_df[
            glm_strat_df["term"].astype(str).str.contains("C\\(creativity_level\\)", regex=True)
        ].copy()
        if glm_strat_view.empty:
            glm_strat_view = glm_strat_df.copy()
        _show_dataframe(
            glm_strat_view[
                [
                    "stratum_type",
                    "stratum",
                    "n_models_in_stratum",
                    "term",
                    "odds_ratio",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "p_fdr_bh",
                    "n_obs",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Mixed model stratifié par modèle")
    st.caption(
        f"Modèle mixte répliqué par strate modèle (individuel + `others_pooled_excluding_{focus_model}`), "
        "avec extraction des coefficients liés à `creativity_level`."
    )
    if mixedlm_strat_df.empty:
        st.info("Mixed model stratifié indisponible (données insuffisantes par strate).")
    else:
        mixedlm_strat_view = mixedlm_strat_df[
            mixedlm_strat_df["term"].astype(str).str.contains("C\\(creativity_level\\)", regex=True)
        ].copy()
        if mixedlm_strat_view.empty:
            mixedlm_strat_view = mixedlm_strat_df.copy()
        _show_dataframe(
            mixedlm_strat_view[
                [
                    "stratum_type",
                    "stratum",
                    "n_models_in_stratum",
                    "term",
                    "coef",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "p_fdr_bh",
                    "n_obs",
                    "n_groups",
                    "icc",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Homogénéité de l'effet creativity_level")
    st.caption(
        f"Test interactionnel `creativity_level x model_bucket` pour `{focus_model}` vs reste poolé."
    )
    if homogeneity_df.empty:
        st.info("Tests d'homogénéité indisponibles (données insuffisantes ou dépendances manquantes).")
    else:
        _show_dataframe(
            homogeneity_df[
                [
                    "model_type",
                    "test",
                    "model_groups",
                    "n_obs",
                    "n_groups",
                    "lr_stat",
                    "df_diff",
                    "p_value",
                    "conclusion",
                    "status",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("#### Advanced exports")
    export_a, export_b, export_c, export_d = st.columns(4)
    export_a.download_button(
        "Download impact correlations CSV",
        data=corr_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_correlations.csv",
        mime="text/csv",
        disabled=corr_df.empty,
    )
    export_b.download_button(
        "Download impact GLM CSV",
        data=glm_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_glm.csv",
        mime="text/csv",
        disabled=glm_df.empty,
    )
    export_c.download_button(
        "Download impact mediation CSV",
        data=mediation_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_mediation.csv",
        mime="text/csv",
        disabled=mediation_df.empty,
    )
    export_d.download_button(
        "Download impact mixed model CSV",
        data=mixedlm_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_mixed_model.csv",
        mime="text/csv",
        disabled=mixedlm_df.empty,
    )
    export_e, export_f, export_g = st.columns(3)
    export_e.download_button(
        "Download impact stratified GLM CSV",
        data=glm_strat_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_glm_stratified.csv",
        mime="text/csv",
        disabled=glm_strat_df.empty,
    )
    export_f.download_button(
        "Download impact stratified mixed model CSV",
        data=mixedlm_strat_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_mixed_model_stratified.csv",
        mime="text/csv",
        disabled=mixedlm_strat_df.empty,
    )
    export_g.download_button(
        "Download impact homogeneity tests CSV",
        data=homogeneity_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_impact_advanced_homogeneity_tests.csv",
        mime="text/csv",
        disabled=homogeneity_df.empty,
    )


def _render_impact_statistics_fused_section(
    filtered: pd.DataFrame,
    selected_roots: tuple[str, ...],
) -> None:
    st.subheader("Analyse statistique fusionnée")
    st.caption(
        "Cette section regroupe le résumé Spearman détaillé et l'advanced analysis "
        "dans un même espace."
    )
    tab_results, tab_guide = st.tabs(["Résultats fusionnés", "Guide d'interprétation"])

    with tab_results:
        render_spearman_section(filtered, show_header=False)
        st.divider()
        _render_impact_advanced_analysis_block(
            filtered,
            selected_roots=selected_roots,
            show_header=False,
        )

    with tab_guide:
        st.markdown(
            """
            ### Que mesure chaque bloc ?
            - **response_length_words**: taille de la réponse du LLM en mots.
            - **response_length_tokens**: taille de la réponse du LLM en tokens (tiktoken si disponible, sinon approximation robuste).
            - **Spearman (rho)**: force et direction d'une relation monotone entre un facteur et `hallucination_rate`.
            - **Pearson (r)** dans l'advanced analysis: relation linéaire entre variables numériques.
            - **Bootstrap CI (95%)**: intervalle plausible de la corrélation, obtenu par rééchantillonnage.
            - **Permutation p-value**: significativité robuste calculée en permutant les valeurs (moins d'hypothèses paramétriques).
            - **FDR BH (`p_fdr_bh`)**: correction des p-values pour limiter les faux positifs quand on teste plusieurs métriques.
            - **Corrélation partielle (`r_partial`)**: corrélation entre une métrique et `hallucination_rate` en contrôlant `response_length_words` et `n_claim_rows`.
            - **Paired contrasts**: différence moyenne de `hallucination_rate` entre niveaux de `creativity_level` dans des groupes appariés.
            - **GLM binomial (odds ratio)**: effet ajusté des facteurs sur les odds d'hallucination claim-level.
            - **Modèle mixte (random intercept `prompt_id`)**:
              - modèle: `hallucination_rate ~ creativity_level * task + length_words + (1|prompt_id)`;
              - absorbe la variance intra-prompt via l'effet aléatoire;
              - `icc = var_random_prompt / (var_random_prompt + var_residual)` quantifie la part de variance inter-prompts.
            - **Mediation (creativity -> n_claims -> hallucination_rate)**:
              - `a_path`: effet de la créativité sur `n_claims`;
              - `b_path`: effet de `n_claims` sur `hallucination_rate` en contrôlant la créativité;
              - `indirect_ab`: effet indirect (mécanisme "quantité de claims");
              - `c_prime_direct`: effet direct résiduel (mécanisme "fiabilité par claim");
              - `indirect_ci_low/high` et `indirect_p_boot`: robustesse bootstrap de l'effet indirect;
              - `sobel_p_value`: test paramétrique complémentaire.

            ### Comment interpréter rapidement ?
            - **Direction**:
              - `rho/r > 0`: quand la métrique augmente, le taux d'hallucination tend à augmenter.
              - `rho/r < 0`: quand la métrique augmente, le taux d'hallucination tend à diminuer.
            - **Force de corrélation (règle pratique)**:
              - `|r| < 0.10`: très faible
              - `0.10 <= |r| < 0.30`: faible
              - `0.30 <= |r| < 0.50`: modérée
              - `|r| >= 0.50`: forte
            - **Confiance statistique**:
              - un `p_value` faible indique un signal compatible avec une association non aléatoire ;
              - un `p_fdr_bh < 0.05` est plus fiable en multi-tests ;
              - un CI qui reste loin de `0` suggère un effet plus robuste.
            - **GLM (odds ratio)**:
              - `OR > 1`: augmentation des odds d'hallucination ;
              - `OR < 1`: diminution des odds ;
              - si le CI95 de l'OR recouvre `1`, l'effet est moins concluant.
            - **Mediation**:
              - `indirect` significatif + `direct` non significatif: médiation complète (effet principalement via le nombre de claims).
              - `indirect` significatif + `direct` significatif: médiation partielle (quantité + qualité).
              - `indirect` non significatif + `direct` significatif: effet direct surtout qualitatif.
            - **Modèle mixte**:
              - ICC élevé: la variabilité inter-prompts domine, ce qui justifie le random effect;
              - comparer les IC/coefs au modèle sans effet aléatoire pour vérifier le gain de robustesse.

            ### Bonnes pratiques de lecture
            - Lire d'abord la **taille d'effet** (`r`, `r_partial`, `OR`), puis la significativité (`p`, `p_fdr_bh`).
            - Vérifier `n` avant de conclure (petits échantillons => résultats plus instables).
            - Ne pas inférer automatiquement une causalité: ces résultats montrent des associations conditionnelles.
            """
        )


def render_impact_page(selected_roots: tuple[str, ...]) -> None:
    with st.spinner("Loading prompt-level dataset..."):
        df = load_prompt_dataset_cached(selected_roots)
    if df.empty:
        st.warning("No compatible runs found (need run_config.json + generation.jsonl + output.csv).")
        return

    filtered = render_impact_data_filters(df)
    if filtered.empty:
        st.warning("No data after filters.")
        return

    plot_df = filtered.copy()
    plot_df["creativity_level"] = pd.Categorical(
        plot_df["creativity_level"],
        categories=CREATIVITY_ORDER,
        ordered=True,
    )

    chart_cfg = render_impact_chart_config(plot_df)
    if chart_cfg.get("invalid"):
        return

    render_impact_kpis(filtered)

    if chart_cfg["chart_type"] == "line":
        fig = build_line_plot(
            df=plot_df,
            x_axis=chart_cfg["x_axis"],
            y_axis=chart_cfg["y_axis"],
            facet_by=chart_cfg["facet_by"],
            estimator=chart_cfg["line_estimator"],
            series_param_a=chart_cfg["series_param_a"],
            series_param_b=chart_cfg["series_param_b"],
            line_color_by=chart_cfg["line_color_by"],
            show_std=chart_cfg["show_std"],
            publication_style=bool(chart_cfg.get("publication_style", True)),
        )
    elif chart_cfg["chart_type"] == "points":
        fig = build_points_plot(
            df=plot_df,
            x_axis=chart_cfg["x_axis"],
            y_axis=chart_cfg["y_axis"],
            color_by=chart_cfg["color_by"],
            facet_by=chart_cfg["facet_by"],
            publication_style=bool(chart_cfg.get("publication_style", True)),
        )
    else:
        fig = build_distribution_plot(
            df=plot_df,
            chart_type=chart_cfg["chart_type"],
            x_axis=chart_cfg["x_axis"],
            y_axis=chart_cfg["y_axis"],
            color_by=chart_cfg["color_by"],
            facet_by=chart_cfg["facet_by"],
            show_points=chart_cfg["show_points"],
            publication_style=bool(chart_cfg.get("publication_style", True)),
        )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)
    if chart_cfg["chart_type"] == "line":
        st.caption(
            "Légende compacte: couleur = variable choisie, style de ligne = série `S1`, `S2`, ... ; "
            "même code de style conservé entre facets."
        )
    elif bool(chart_cfg.get("publication_style", True)):
        st.caption("Style publication appliqué: template épuré, axes lisibles, grille légère et légende compacte.")

    st.markdown("#### Tableau du graphique principal")
    chart_table_df = _build_main_chart_table(plot_df, chart_cfg)
    if chart_table_df.empty:
        st.info("Impossible de générer le tableau pour le graphique courant.")
    else:
        max_rows_display = 5000
        if len(chart_table_df) > max_rows_display:
            st.caption(
                f"Tableau volumineux: affichage limité aux {max_rows_display:,} premières lignes "
                f"sur {len(chart_table_df):,} (CSV complet téléchargeable)."
            )
        st.download_button(
            "Download chart table CSV",
            data=chart_table_df.to_csv(index=False).encode("utf-8"),
            file_name="hallulens_impact_chart_table.csv",
            mime="text/csv",
            key="impact_chart_table_download",
        )
        latex_groupplot = build_main_chart_line_latex_groupplot(chart_table_df, chart_cfg)
        if latex_groupplot:
            st.download_button(
                "Download chart LaTeX (.tex)",
                data=latex_groupplot.encode("utf-8"),
                file_name="hallulens_impact_chart_groupplot.tex",
                mime="text/x-tex",
                key="impact_chart_latex_download",
            )
            st.caption(
                "Export PGFPlots/TikZ basé sur le tableau agrégé du graphique principal "
                "(layout groupplot compatible papier LaTeX)."
            )
        _show_dataframe(
            chart_table_df.head(max_rows_display),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Impact summary")
    summary = build_impact_summary(filtered, x_axis=chart_cfg["x_axis"], y_axis=chart_cfg["y_axis"])
    _show_dataframe(summary, use_container_width=True, hide_index=True)

    _render_impact_statistics_fused_section(filtered, selected_roots=selected_roots)
    st.divider()
    render_spearman_forest_section(filtered)

    st.subheader("Filtered data")
    download_cols = [
        c
        for c in [
            "root_name",
            "run_id",
            "model_name",
            "temperature",
            "length_words",
            "response_length_words",
            "response_length_tokens",
            "task",
            "creativity_level",
            "title",
            "n_claims",
            "support_rate",
            "hallucination_rate",
            "prompt",
            "generation",
        ]
        if c in filtered.columns
    ]
    table_df = filtered[download_cols].copy()
    st.download_button(
        "Download filtered CSV",
        data=table_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_filtered.csv",
        mime="text/csv",
    )
    _show_dataframe(table_df.head(1000), use_container_width=True, hide_index=True)


def _apply_creativity_sidebar_filters(
    df: pd.DataFrame,
    coverage_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Creativity Filters")
    st.sidebar.caption("Strict mode is active: only rows with hallucination metrics and valid creativity scores are kept.")

    root_filter = st.sidebar.multiselect(
        "Dataset",
        sorted_unique(df["root_name"]),
        default=sorted_unique(df["root_name"]),
        key="creativity_root_filter",
    )
    model_filter = st.sidebar.multiselect(
        "Model",
        sorted_unique(df["model_name"]),
        default=sorted_unique(df["model_name"]),
        key="creativity_model_filter",
    )
    task_filter = st.sidebar.multiselect(
        "Task",
        sorted_unique(df["task"]),
        default=sorted_unique(df["task"]),
        key="creativity_task_filter",
    )
    creativity_filter = st.sidebar.multiselect(
        "Creativity intention",
        sorted_unique(df["creativity_level"]),
        default=sorted_unique(df["creativity_level"]),
        key="creativity_intention_filter",
    )
    temperature_filter = st.sidebar.multiselect(
        "Temperature",
        sorted_unique(df["temperature"]),
        default=sorted_unique(df["temperature"]),
        key="creativity_temperature_filter",
    )
    length_filter = st.sidebar.multiselect(
        "Length words",
        sorted_unique(df["length_words"]),
        default=sorted_unique(df["length_words"]),
        key="creativity_length_filter",
    )

    if df["n_claim_rows"].notna().any():
        min_claims = int(np.nanmin(df["n_claim_rows"]))
        max_claims = int(np.nanmax(df["n_claim_rows"]))
        min_claims_filter = st.sidebar.slider(
            "Min n_claim_rows",
            min_value=min_claims,
            max_value=max_claims,
            value=min_claims,
            key="creativity_min_claim_rows",
        )
    else:
        min_claims_filter = 0

    ttct_min, ttct_max = 1.0, 5.0
    if df["ttct_overall"].notna().any():
        ttct_min = float(np.floor(np.nanmin(df["ttct_overall"]) * 10) / 10)
        ttct_max = float(np.ceil(np.nanmax(df["ttct_overall"]) * 10) / 10)
    ttct_range = st.sidebar.slider(
        "TTCT overall range",
        min_value=float(ttct_min),
        max_value=float(ttct_max),
        value=(float(ttct_min), float(ttct_max)),
        key="creativity_ttct_range",
    )

    ttcw_min, ttcw_max = 1.0, 5.0
    if df["ttcw_overall"].notna().any():
        ttcw_min = float(np.floor(np.nanmin(df["ttcw_overall"]) * 10) / 10)
        ttcw_max = float(np.ceil(np.nanmax(df["ttcw_overall"]) * 10) / 10)
    ttcw_range = st.sidebar.slider(
        "TTCW (TTWT) overall range",
        min_value=float(ttcw_min),
        max_value=float(ttcw_max),
        value=(float(ttcw_min), float(ttcw_max)),
        key="creativity_ttcw_range",
    )

    filtered = df.copy()
    filtered = apply_multiselect_filter(filtered, "root_name", root_filter)
    filtered = apply_multiselect_filter(filtered, "model_name", model_filter)
    filtered = apply_multiselect_filter(filtered, "task", task_filter)
    filtered = apply_multiselect_filter(filtered, "creativity_level", creativity_filter)
    filtered = apply_multiselect_filter(filtered, "temperature", temperature_filter)
    filtered = apply_multiselect_filter(filtered, "length_words", length_filter)
    filtered = filtered[filtered["n_claim_rows"].fillna(0) >= min_claims_filter]
    filtered = filtered[
        filtered["ttct_overall"].fillna(ttct_range[0]).between(ttct_range[0], ttct_range[1], inclusive="both")
    ]
    filtered = filtered[
        filtered["ttcw_overall"].fillna(ttcw_range[0]).between(ttcw_range[0], ttcw_range[1], inclusive="both")
    ]

    coverage_filtered = coverage_df.copy()
    if not coverage_filtered.empty:
        coverage_filtered = apply_multiselect_filter(coverage_filtered, "root_name", root_filter)
        coverage_filtered = apply_multiselect_filter(coverage_filtered, "model_name", model_filter)
        coverage_filtered = apply_multiselect_filter(coverage_filtered, "temperature", temperature_filter)
        coverage_filtered = apply_multiselect_filter(coverage_filtered, "length_words", length_filter)

    return filtered, coverage_filtered


def _render_creativity_kpis(filtered: pd.DataFrame, coverage_filtered: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    coverage_mean = coverage_filtered["coverage_pct"].mean() if not coverage_filtered.empty else np.nan
    c1.metric("Scored rows", f"{len(filtered):,}")
    c2.metric("Scored runs", f"{filtered['run_id'].nunique():,}")
    c3.metric("Coverage mean (%)", f"{coverage_mean:.1f}" if np.isfinite(coverage_mean) else "NA")
    c4.metric(
        "Mean hallucination",
        f"{filtered['hallucination_rate'].mean():.3f}" if len(filtered) else "NA",
    )
    c5.metric(
        "Mean TTCT",
        f"{filtered['ttct_overall'].mean():.3f}" if filtered["ttct_overall"].notna().any() else "NA",
    )
    c6.metric(
        "Mean TTCW (TTWT)",
        f"{filtered['ttcw_overall'].mean():.3f}" if filtered["ttcw_overall"].notna().any() else "NA",
    )


def _render_data_quality_block(coverage_filtered: pd.DataFrame) -> None:
    st.subheader("Data quality")
    if coverage_filtered.empty:
        st.info("No coverage information available for selected filters.")
        return

    statuses = coverage_filtered["status"].value_counts(dropna=False).to_dict()
    missing_n = int(statuses.get("missing", 0))
    partial_n = int(statuses.get("partial", 0))
    complete_n = int(statuses.get("complete", 0))

    if missing_n > 0 or partial_n > 0:
        st.warning(
            f"Coverage issues detected: {missing_n} missing run(s), {partial_n} partial run(s), "
            f"{complete_n} complete run(s)."
        )
    else:
        st.success(f"All selected runs are complete ({complete_n}).")

    display_cols = [
        "root_name",
        "model_name",
        "temperature",
        "length_words",
        "n_generation",
        "n_creativity_rows",
        "n_matched",
        "coverage_pct",
        "status",
    ]
    table = coverage_filtered[display_cols].sort_values(
        by=["status", "coverage_pct", "root_name", "model_name"],
        ascending=[True, False, True, True],
    )
    _show_dataframe(table, use_container_width=True, hide_index=True)


def _render_creativity_visual_block(filtered: pd.DataFrame) -> None:
    st.subheader("Visualizations")
    options = [c for c in CREATIVITY_SCORE_OPTIONS if c in filtered.columns]
    if not options:
        st.info("No creativity score columns available for plotting.")
        return

    controls_a, controls_b, controls_c = st.columns(3)
    x_score = controls_a.selectbox(
        "Creativity score on X",
        options=options,
        index=option_index(options, "creativity_composite"),
        key="creativity_x_score",
    )
    facet_options = ["None", "task", "length_words", "model_name"]
    facet_by_raw = controls_b.selectbox(
        "Scatter facet",
        options=facet_options,
        index=option_index(facet_options, "task"),
        key="creativity_scatter_facet",
    )
    corr_method = controls_c.selectbox(
        "Heatmap correlation method",
        options=["spearman", "pearson"],
        index=0,
        key="creativity_heatmap_method",
    )
    facet_by = None if facet_by_raw == "None" else facet_by_raw

    scatter = build_creativity_scatter(
        filtered,
        x_col=x_score,
        y_col="hallucination_rate",
        color_by="creativity_level",
        facet_by=facet_by,
    )
    scatter.update_layout(height=540)
    st.plotly_chart(scatter, use_container_width=True)

    # col_box.plotly_chart(box, use_container_width=True)

    heatmap_cols = [c for c in HEATMAP_CORR_COLS_DEFAULT if c in filtered.columns]
    heatmap = build_creativity_heatmap(filtered, corr_cols=heatmap_cols, corr_method=corr_method)
    heatmap.update_layout(height=550)
    st.plotly_chart(heatmap, use_container_width=True)

    st.markdown("#### Creativity metrics (X) vs creativity level classes (Y)")
    metric_defaults = [c for c in ["ttct_overall", "ttcw_overall", "creativity_composite"] if c in options]
    selected_metric_cols = st.multiselect(
        "Metrics to display on X axis",
        options=options,
        default=metric_defaults if metric_defaults else options[: min(3, len(options))],
        key="creativity_metrics_vs_level_cols",
    )
    metrics_plot_type = st.radio(
        "Metrics/class plot type",
        options=["box", "violin"],
        index=0,
        horizontal=True,
        key="creativity_metrics_vs_level_type",
    )
    metrics_by_level_fig = build_creativity_metrics_by_level_plot(
        filtered,
        metric_cols=selected_metric_cols,
        chart_type=metrics_plot_type,
    )
    metrics_by_level_fig.update_layout(height=650)
    st.plotly_chart(metrics_by_level_fig, use_container_width=True)

    st.markdown("#### Figure: Creativity Score Distributions")
    st.caption(
        "Manipulation check: distributions TTCT (gauche) et TTCW (droite) "
        "par `creativity_level`, pour visualiser déplacement et recouvrement."
    )
    dist_a, dist_b = st.columns(2)
    dist_style = dist_a.radio(
        "Distribution style",
        options=["violin", "ridgeline"],
        index=0,
        horizontal=True,
        key="creativity_score_dist_style",
    )
    dist_points = bool(
        dist_b.checkbox(
            "Show raw points",
            value=False,
            key="creativity_score_dist_points",
        )
    )
    dist_fig, dist_summary = build_creativity_score_distributions_plot(
        filtered,
        ttct_col="ttct_overall",
        ttcw_col="ttcw_overall",
        level_col="creativity_level",
        style=dist_style,
        show_points=dist_points,
    )
    dist_fig.update_layout(height=560)
    st.plotly_chart(dist_fig, use_container_width=True)
    if not dist_summary.empty:
        _show_dataframe(
            dist_summary[
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
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_creativity_exports(filtered: pd.DataFrame) -> None:
    st.subheader("Exports")
    col_a, col_b = st.columns(2)
    col_a.download_button(
        "Download filtered creativity CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_creativity_filtered.csv",
        mime="text/csv",
    )
    col_b.caption("Advanced analysis has been moved to the Impact Hallucinations page.")


def render_creativity_page(selected_roots: tuple[str, ...]) -> None:
    with st.spinner("Loading creativity + hallucination dataset..."):
        df, coverage_df = load_creativity_dataset_cached(
            selected_roots=selected_roots,
            strict_mode=True,
            exclude_incomplete_runs=True,
        )
    st.caption("Calculations include only complete runs. Partial or missing runs are excluded from computed metrics.")
    if coverage_df.empty:
        st.warning("No compatible runs found for creativity view.")
        return
    if df.empty:
        st.warning("No scored rows found with current strict mode (metrics + creativity scores).")
        _render_data_quality_block(coverage_df)
        return

    filtered, coverage_filtered = _apply_creativity_sidebar_filters(df, coverage_df)
    if filtered.empty:
        st.warning("No data after creativity filters.")
        _render_data_quality_block(coverage_filtered)
        return

    _render_creativity_kpis(filtered, coverage_filtered)
    _render_data_quality_block(coverage_filtered)
    _render_creativity_visual_block(filtered)
    _render_creativity_exports(filtered)

    st.subheader("Filtered scored data")
    preview_cols = [
        c
        for c in [
            "root_name",
            "run_id",
            "model_name",
            "temperature",
            "length_words",
            "task",
            "creativity_level",
            "hallucination_rate",
            "ttct_overall",
            "ttcw_overall",
            "creativity_composite",
            "prompt",
        ]
        if c in filtered.columns
    ]
    _show_dataframe(filtered[preview_cols].head(1000), use_container_width=True, hide_index=True)


def _claims_sidebar_filter_config(claims_df: pd.DataFrame) -> dict[str, Any]:
    st.sidebar.header("Claims Explorer Filters")
    with st.sidebar.form("claims_filters_form", clear_on_submit=False):
        root_filter = st.multiselect(
            "Dataset",
            sorted_unique(claims_df["root_name"]),
            default=sorted_unique(claims_df["root_name"]),
            key="claims_root_filter",
        )
        model_filter = st.multiselect(
            "Model",
            sorted_unique(claims_df["model_name"]),
            default=sorted_unique(claims_df["model_name"]),
            key="claims_model_filter",
        )
        task_filter = st.multiselect(
            "Task",
            sorted_unique(claims_df["task"]),
            default=sorted_unique(claims_df["task"]),
            key="claims_task_filter",
        )
        creativity_filter = st.multiselect(
            "Creativity",
            sorted_unique(claims_df["creativity_level"]),
            default=sorted_unique(claims_df["creativity_level"]),
            key="claims_creativity_filter",
        )
        kappa_options = sorted_unique(claims_df["kappa_level"]) if "kappa_level" in claims_df.columns else ["NA"]
        kappa_filter = st.multiselect(
            "Kappa",
            kappa_options,
            default=kappa_options,
            key="claims_kappa_filter",
        )
        temperature_filter = st.multiselect(
            "Temperature",
            sorted_unique(claims_df["temperature"]),
            default=sorted_unique(claims_df["temperature"]),
            key="claims_temperature_filter",
        )
        length_filter = st.multiselect(
            "Length words",
            sorted_unique(claims_df["length_words"]),
            default=sorted_unique(claims_df["length_words"]),
            key="claims_length_filter",
        )
        status_options = [s for s in CLAIM_STATUS_ORDER if s in claims_df["verification_status"].astype(str).unique()]
        status_filter = st.multiselect(
            "Verification status",
            options=status_options,
            default=status_options,
            key="claims_status_filter",
        )

        title_query = st.text_input("Title contains", value="", key="claims_title_query").strip()
        claim_query = st.text_input("Claim contains", value="", key="claims_claim_query").strip()
        prompt_query = st.text_input("Prompt contains", value="", key="claims_prompt_query").strip()
        submitted = st.form_submit_button("Apply claims filters")
    if submitted:
        st.sidebar.success("Claims filters applied.")

    return {
        "root_filter": tuple(root_filter),
        "model_filter": tuple(model_filter),
        "task_filter": tuple(task_filter),
        "creativity_filter": tuple(creativity_filter),
        "kappa_filter": tuple(kappa_filter),
        "temperature_filter": tuple(temperature_filter),
        "length_filter": tuple(length_filter),
        "status_filter": tuple(status_filter),
        "title_query": title_query,
        "claim_query": claim_query,
        "prompt_query": prompt_query,
    }


def _filter_claims_with_config(claims_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    filtered = claims_df.copy()
    filtered = apply_multiselect_filter(filtered, "root_name", list(cfg.get("root_filter", ())))
    filtered = apply_multiselect_filter(filtered, "model_name", list(cfg.get("model_filter", ())))
    filtered = apply_multiselect_filter(filtered, "task", list(cfg.get("task_filter", ())))
    filtered = apply_multiselect_filter(filtered, "creativity_level", list(cfg.get("creativity_filter", ())))
    filtered = apply_multiselect_filter(filtered, "kappa_level", list(cfg.get("kappa_filter", ())))
    filtered = apply_multiselect_filter(filtered, "temperature", list(cfg.get("temperature_filter", ())))
    filtered = apply_multiselect_filter(filtered, "length_words", list(cfg.get("length_filter", ())))
    filtered = apply_multiselect_filter(filtered, "verification_status", list(cfg.get("status_filter", ())))

    title_query = str(cfg.get("title_query", "")).strip()
    claim_query = str(cfg.get("claim_query", "")).strip()
    prompt_query = str(cfg.get("prompt_query", "")).strip()
    if title_query:
        filtered = filtered[
            filtered["title"].fillna("").astype(str).str.contains(title_query, case=False, regex=False)
        ]
    if claim_query:
        filtered = filtered[
            filtered["claim"].fillna("").astype(str).str.contains(claim_query, case=False, regex=False)
        ]
    if prompt_query:
        filtered = filtered[
            filtered["prompt"].fillna("").astype(str).str.contains(prompt_query, case=False, regex=False)
        ]
    return filtered


def _render_claims_coverage_block(coverage_df: pd.DataFrame) -> None:
    st.subheader("Run coverage")
    st.caption(
        "Vérifie la complétude des runs pour l'exploration claim-level "
        "(fichiers requis et volumes chargés)."
    )
    if coverage_df.empty:
        st.info("No run coverage data.")
        return
    status_counts = coverage_df["status"].value_counts(dropna=False).to_dict()
    loaded = int(status_counts.get("loaded", 0))
    missing = int(status_counts.get("missing_files", 0))
    if missing > 0:
        st.warning(f"{missing} run(s) have missing files. {loaded} run(s) loaded.")
    else:
        st.success(f"All runs ready. Loaded runs: {loaded}.")
    if "model_name" in coverage_df.columns:
        by_model = (
            coverage_df.groupby(["model_name", "status"], dropna=False)
            .size()
            .rename("n_runs")
            .reset_index()
            .sort_values(["status", "n_runs", "model_name"], ascending=[True, False, True])
        )
        st.caption("Résumé par modèle (chargé vs non chargé).")
        _show_dataframe(by_model, use_container_width=True, hide_index=True)

        missing_models = coverage_df[coverage_df["status"].astype(str) != "loaded"].copy()
        if not missing_models.empty:
            miss_summary = (
                missing_models.groupby("model_name", dropna=False)
                .agg(
                    n_runs_missing=("run_dir", "count"),
                    missing_files_examples=("missing_files", lambda s: "; ".join(sorted(set(str(v) for v in s if str(v))))),
                )
                .reset_index()
                .sort_values(["n_runs_missing", "model_name"], ascending=[False, True])
            )
            st.caption("Modèles non chargés et causes probables.")
            _show_dataframe(miss_summary, use_container_width=True, hide_index=True)

    cols = [
        c
        for c in [
            "root_name",
            "model_name",
            "run_name",
            "run_dir",
            "status",
            "n_prompts",
            "n_claim_rows",
            "missing_files",
        ]
        if c in coverage_df.columns
    ]
    _show_dataframe(coverage_df[cols], use_container_width=True, hide_index=True)


def _render_claims_kpis(filtered_claims: pd.DataFrame, prompt_summary: pd.DataFrame) -> None:
    st.caption(
        "Ces KPI résument la qualité factuelle au niveau claim sur le sous-ensemble filtré."
    )
    supported = int((filtered_claims["verification_status"].astype(str) == "supported").sum())
    hallucinated = int((filtered_claims["verification_status"].astype(str) == "hallucinated").sum())
    no_claim = int((filtered_claims["verification_status"].astype(str) == "no_claim").sum())
    unverified = int((filtered_claims["verification_status"].astype(str) == "unverified").sum())
    verified = supported + hallucinated
    support_rate = supported / verified if verified > 0 else np.nan
    halluc_rate = hallucinated / verified if verified > 0 else np.nan

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Claim rows", f"{len(filtered_claims):,}")
    c2.metric("Prompts", f"{len(prompt_summary):,}")
    c3.metric("Supported", f"{supported:,}")
    c4.metric("Hallucinated", f"{hallucinated:,}")
    c5.metric("Support rate", _format_number(support_rate, digits=4))
    c6.metric("Hallucination rate", _format_number(halluc_rate, digits=4))
    d1, d2 = st.columns(2)
    d1.caption(f"No-claim rows: {no_claim:,}")
    d2.caption(f"Unverified rows: {unverified:,}")


def _render_claims_global_analysis(filtered_claims: pd.DataFrame, prompt_summary: pd.DataFrame) -> None:
    st.markdown("#### Distribution des statuts de vérification")
    st.caption(
        "Montre la proportion de claims supportés, hallucinés, sans claim et non vérifiés."
    )
    status_counts = (
        filtered_claims["verification_status"]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("status")
        .reset_index(name="count")
    )
    fig_status = px.bar(status_counts, x="status", y="count", color="status")
    fig_status.update_layout(height=360, xaxis_title="Verification status", yaxis_title="Claim rows")
    st.plotly_chart(fig_status, use_container_width=True)

    st.markdown("#### Support rate par task et créativité")
    st.caption(
        "Heatmap du taux de support claim-level par combinaison tâche × intention créative. "
        "Vert = plus fiable, rouge = plus risqué."
    )
    verified = filtered_claims[filtered_claims["verification_status"].astype(str).isin(["supported", "hallucinated"])].copy()
    if verified.empty:
        st.info("No verified claims in current filters.")
    else:
        heat_df = (
            verified.groupby(["task", "creativity_level"], dropna=False)
            .agg(
                n_verified=("claim_row_id", "count"),
                n_supported=("verification_status", lambda s: (s.astype(str) == "supported").sum()),
            )
            .reset_index()
        )
        heat_df["support_rate"] = np.where(
            heat_df["n_verified"] > 0,
            heat_df["n_supported"] / heat_df["n_verified"],
            np.nan,
        )
        pivot = heat_df.pivot(index="task", columns="creativity_level", values="support_rate")
        if pivot.empty:
            st.info("Insufficient data for heatmap.")
        else:
            fig_heat = px.imshow(
                pivot,
                aspect="auto",
                zmin=0,
                zmax=1,
                color_continuous_scale="RdYlGn",
                text_auto=".2f",
            )
            fig_heat.update_layout(height=420)
            st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### Prompts les plus à risque (hallucination claim-level)")
    st.caption(
        "Classement des prompts avec le plus fort taux d'hallucination au niveau claims vérifiés."
    )
    if prompt_summary.empty:
        st.info("No prompt-level summary.")
    else:
        cols = [
            c
            for c in [
                "root_name",
                "model_name",
                "task",
                "creativity_level",
                "title",
                "n_verified",
                "n_supported",
                "n_hallucinated",
                "hallucination_rate_claim_level",
                "support_rate_claim_level",
                "claim_density_per_100_words",
            ]
            if c in prompt_summary.columns
        ]
        top = prompt_summary.sort_values(
            by=["hallucination_rate_claim_level", "n_verified"],
            ascending=[False, False],
        ).head(100)
        _show_dataframe(top[cols], use_container_width=True, hide_index=True)


def _render_prompt_drilldown(filtered_claims: pd.DataFrame, prompt_summary: pd.DataFrame) -> None:
    st.markdown("#### Prompt drill-down")
    st.caption(
        "Inspection fine prompt par prompt: filtre local par run/paramètres, "
        "puis analyse détaillée des claims et comparaison A/B de deux prompts."
    )
    if prompt_summary.empty:
        st.info("No prompts to inspect.")
        return

    with st.form("claims_drill_filters_form", clear_on_submit=False):
        drill_df = prompt_summary.copy()
        f1, f2, f3 = st.columns(3)
        run_filter = f1.multiselect(
            "Drill-down run_id",
            options=sorted_unique(drill_df["run_id"]) if "run_id" in drill_df.columns else [],
            default=sorted_unique(drill_df["run_id"]) if "run_id" in drill_df.columns else [],
            key="claims_drill_run_filter",
        )
        model_filter = f2.multiselect(
            "Drill-down model",
            options=sorted_unique(drill_df["model_name"]) if "model_name" in drill_df.columns else [],
            default=sorted_unique(drill_df["model_name"]) if "model_name" in drill_df.columns else [],
            key="claims_drill_model_filter",
        )
        task_filter = f3.multiselect(
            "Drill-down task",
            options=sorted_unique(drill_df["task"]) if "task" in drill_df.columns else [],
            default=sorted_unique(drill_df["task"]) if "task" in drill_df.columns else [],
            key="claims_drill_task_filter",
        )

        f4, f5, f6 = st.columns(3)
        creativity_filter = f4.multiselect(
            "Drill-down creativity",
            options=sorted_unique(drill_df["creativity_level"]) if "creativity_level" in drill_df.columns else [],
            default=sorted_unique(drill_df["creativity_level"]) if "creativity_level" in drill_df.columns else [],
            key="claims_drill_creativity_filter",
        )
        kappa_filter = f5.multiselect(
            "Drill-down kappa",
            options=sorted_unique(drill_df["kappa_level"]) if "kappa_level" in drill_df.columns else [],
            default=sorted_unique(drill_df["kappa_level"]) if "kappa_level" in drill_df.columns else [],
            key="claims_drill_kappa_filter",
        )
        temperature_filter = f6.multiselect(
            "Drill-down temperature",
            options=sorted_unique(drill_df["temperature"]) if "temperature" in drill_df.columns else [],
            default=sorted_unique(drill_df["temperature"]) if "temperature" in drill_df.columns else [],
            key="claims_drill_temperature_filter",
        )
        f7, _f8 = st.columns(2)
        length_filter = f7.multiselect(
            "Drill-down length_words",
            options=sorted_unique(drill_df["length_words"]) if "length_words" in drill_df.columns else [],
            default=sorted_unique(drill_df["length_words"]) if "length_words" in drill_df.columns else [],
            key="claims_drill_length_filter",
        )
        drill_submitted = st.form_submit_button("Apply drill-down filters")
    if drill_submitted:
        st.success("Drill-down filters applied.")

    drill_cfg = {
        "run_filter": tuple(run_filter),
        "model_filter": tuple(model_filter),
        "task_filter": tuple(task_filter),
        "creativity_filter": tuple(creativity_filter),
        "kappa_filter": tuple(kappa_filter),
        "temperature_filter": tuple(temperature_filter),
        "length_filter": tuple(length_filter),
    }
    drill_cache_key = (
        len(prompt_summary),
        tuple(sorted((k, str(v)) for k, v in drill_cfg.items())),
    )
    drill_cached = st.session_state.get("claims_drill_cache")
    if isinstance(drill_cached, dict) and drill_cached.get("cache_key") == drill_cache_key:
        drill_df = drill_cached["drill_df"]
    else:
        drill_df = prompt_summary.copy()
        drill_df = apply_multiselect_filter(drill_df, "run_id", list(run_filter))
        drill_df = apply_multiselect_filter(drill_df, "model_name", list(model_filter))
        drill_df = apply_multiselect_filter(drill_df, "task", list(task_filter))
        drill_df = apply_multiselect_filter(drill_df, "creativity_level", list(creativity_filter))
        drill_df = apply_multiselect_filter(drill_df, "kappa_level", list(kappa_filter))
        drill_df = apply_multiselect_filter(drill_df, "temperature", list(temperature_filter))
        drill_df = apply_multiselect_filter(drill_df, "length_words", list(length_filter))
        st.session_state["claims_drill_cache"] = {"cache_key": drill_cache_key, "drill_df": drill_df}

    if drill_df.empty:
        st.warning("No prompts after drill-down filters.")
        return

    opts_cache_key = (
        drill_cache_key,
        len(drill_df),
        int(drill_df["run_id"].nunique()) if "run_id" in drill_df.columns else 0,
    )
    opts_cached = st.session_state.get("claims_drill_opts_cache")
    if isinstance(opts_cached, dict) and opts_cached.get("cache_key") == opts_cache_key:
        opts = opts_cached["opts"]
    else:
        opts = drill_df.reset_index(drop=True).copy()
        opts["prompt_uid"] = opts["run_dir"].astype(str) + "::" + opts["prompt_clean"].astype(str)
        opts["prompt_label"] = (
            opts["title"].fillna("NA").astype(str)
            + " | "
            + opts["task"].fillna("NA").astype(str)
            + " | "
            + opts["creativity_level"].fillna("NA").astype(str)
            + " | len="
            + opts["length_words"].map(lambda v: _format_small_numeric_value(v))
            + " | k="
            + opts["kappa_level"].fillna("NA").astype(str)
            + " | run="
            + opts["run_id"].fillna("NA").astype(str).str[-18:]
            + " | H="
            + opts["hallucination_rate_claim_level"].map(lambda v: _format_small_numeric_value(v))
        )
        st.session_state["claims_drill_opts_cache"] = {
            "cache_key": opts_cache_key,
            "opts": opts,
        }

    prompt_lookup_signature = (
        len(filtered_claims),
        int(filtered_claims["claim_row_id"].nunique()) if "claim_row_id" in filtered_claims.columns else 0,
    )
    prompt_lookup_cached = st.session_state.get("claims_prompt_lookup_cache")
    if (
        isinstance(prompt_lookup_cached, dict)
        and prompt_lookup_cached.get("signature") == prompt_lookup_signature
    ):
        prompt_lookup = prompt_lookup_cached["lookup"]
    else:
        grouped = filtered_claims.groupby(["run_dir", "prompt_clean"], sort=False).groups
        prompt_lookup = {(str(run), str(prompt)): idx for (run, prompt), idx in grouped.items()}
        st.session_state["claims_prompt_lookup_cache"] = {
            "signature": prompt_lookup_signature,
            "lookup": prompt_lookup,
        }

    compare_mode = st.checkbox(
        "Comparer deux prompts/réponses",
        value=False,
        key="claims_drill_compare_mode",
    )
    sel_a_idx = st.selectbox(
        "Prompt A",
        options=opts.index.tolist(),
        format_func=lambda i: opts.loc[i, "prompt_label"],
        key="claims_prompt_selector_a",
    )

    def _extract_prompt_claims(sel_row: pd.Series) -> pd.DataFrame:
        key = (str(sel_row.get("run_dir", "")), str(sel_row.get("prompt_clean", "")))
        idx = prompt_lookup.get(key)
        if idx is None:
            return filtered_claims.iloc[0:0].copy()
        return filtered_claims.loc[idx].copy()

    def _render_prompt_card(sel_row: pd.Series, title: str, key_prefix: str) -> None:
        st.markdown(f"##### {title}")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Verified claims", f"{int(sel_row.get('n_verified', 0)):,}")
        k2.metric("Supported", f"{int(sel_row.get('n_supported', 0)):,}")
        k3.metric("Hallucinated", f"{int(sel_row.get('n_hallucinated', 0)):,}")
        k4.metric("No-claim rows", f"{int(sel_row.get('n_no_claim', 0)):,}")
        k5.metric("Hallucination rate", _format_number(sel_row.get("hallucination_rate_claim_level"), digits=4))
        st.caption(
            "Lecture: comparez la structure du prompt, la réponse générée et "
            "les claims hallucinés/supportés pour identifier les causes."
        )

        with st.expander(f"Prompt text ({title})", expanded=False):
            st.code(str(sel_row.get("prompt", "")), language="text")
        with st.expander(f"LLM response ({title})", expanded=True):
            st.code(str(sel_row.get("generation", "")), language="text")

        prompt_claims = _extract_prompt_claims(sel_row)
        cols = [
            c
            for c in [
                "verification_status",
                "kappa_level",
                "tr_label_candidate",
                "tr_reason",
                "tr_marker_hits",
                "tr_candidate_score",
                "is_supported",
                "sentence",
                "claim",
                "precision",
                "recall",
                "f1",
            ]
            if c in prompt_claims.columns
        ]
        _show_dataframe(prompt_claims[cols], use_container_width=True, hide_index=True)
        st.download_button(
            f"Download {title} claims CSV",
            data=prompt_claims.to_csv(index=False).encode("utf-8"),
            file_name=f"hallulens_prompt_claims_{key_prefix}.csv",
            mime="text/csv",
            key=f"claims_prompt_download_{key_prefix}",
        )

    sel_a = opts.loc[sel_a_idx]
    if not compare_mode or len(opts) < 2:
        _render_prompt_card(sel_a, title="Prompt A", key_prefix="a")
        return

    alt_options = [i for i in opts.index.tolist() if i != sel_a_idx]
    sel_b_idx = st.selectbox(
        "Prompt B",
        options=alt_options,
        format_func=lambda i: opts.loc[i, "prompt_label"],
        key="claims_prompt_selector_b",
    )
    sel_b = opts.loc[sel_b_idx]

    c_left, c_right = st.columns(2)
    with c_left:
        _render_prompt_card(sel_a, title="Prompt A", key_prefix="a")
    with c_right:
        _render_prompt_card(sel_b, title="Prompt B", key_prefix="b")


def _render_claim_rows_table(filtered_claims: pd.DataFrame) -> None:
    st.markdown("#### Claim-level explorer")
    st.caption(
        "Table détaillée de toutes les lignes de claims extraits avec le verdict de vérification."
    )
    ctl_a, ctl_b, ctl_c = st.columns(3)
    only_hallu = ctl_a.checkbox("Only hallucinated", value=False, key="claims_only_hallucinated")
    max_rows = int(
        ctl_b.number_input("Max rows shown", min_value=100, max_value=20000, value=2000, step=100, key="claims_max_rows")
    )
    sort_col = ctl_c.selectbox(
        "Sort by",
        options=[
            c
            for c in [
                "verification_status",
                "tr_label_candidate",
                "tr_candidate_score",
                "task",
                "creativity_level",
                "kappa_level",
                "title",
                "precision",
                "recall",
                "f1",
            ]
            if c in filtered_claims.columns
        ],
        index=0,
        key="claims_sort_col",
    )
    asc = st.checkbox("Ascending sort", value=True, key="claims_sort_asc")

    table = filtered_claims.copy()
    if only_hallu:
        table = table[table["verification_status"].astype(str) == "hallucinated"]
    if sort_col in table.columns:
        table = table.sort_values(sort_col, ascending=asc)

    cols = [
        c
        for c in [
            "root_name",
            "model_name",
            "task",
            "creativity_level",
            "kappa_level",
            "title",
            "verification_status",
            "tr_label_candidate",
            "tr_reason",
            "tr_marker_hits",
            "tr_candidate_score",
            "is_supported",
            "claim",
            "sentence",
            "precision",
            "recall",
            "f1",
            "overall_precision",
            "overall_recall",
            "overall_f1",
        ]
        if c in table.columns
    ]
    _show_dataframe(table[cols].head(max_rows), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered claim rows CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_claim_rows_filtered.csv",
        mime="text/csv",
        key="claims_rows_download",
    )


def _render_kappa_pd_candidates(filtered_claims: pd.DataFrame) -> None:
    st.markdown("#### κ-high HalluLens→PD candidates")
    st.caption(
        "Aide à l'audit qualitatif: repère les claims `κ high` que HalluLens marque comme "
        "`hallucinated`, mais qui ressemblent à de la divergence productive, du design pédagogique "
        "ou un gap d'oracle. Les labels restent des candidats à valider manuellement."
    )
    st.caption(
        "Cette section force automatiquement `Kappa = HIGH` et "
        "`Verification status = hallucinated`; les autres filtres sidebar restent appliqués."
    )

    required = {"verification_status", "kappa_level", "tr_label_candidate", "tr_candidate_score"}
    missing = sorted(required.difference(filtered_claims.columns))
    if missing:
        st.info(f"TR-aware relabeling unavailable. Missing columns: {', '.join(missing)}")
        return

    high_hallu = filtered_claims[
        filtered_claims["kappa_level"].astype(str).str.upper().eq("HIGH")
        & filtered_claims["verification_status"].astype(str).eq("hallucinated")
    ].copy()
    if high_hallu.empty:
        st.info(
            "No κ-high hallucinated claims after applying dataset/model/task/creativity/temperature/length/text filters. "
            "Try selecting the `kappa` output root and broadening model/task filters."
        )
        return

    label = high_hallu["tr_label_candidate"].astype(str)
    pd_count = int(label.eq("PD_candidate").sum())
    pedagogical_count = int(label.eq("pedagogical_design_or_oracle_gap").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("κ-high hallucinated", f"{len(high_hallu):,}")
    c2.metric("PD candidates", f"{pd_count:,}")
    c3.metric("Pedagogical/oracle-gap", f"{pedagogical_count:,}")

    candidate_labels = [
        "PD_candidate",
        "pedagogical_design_or_oracle_gap",
        "candidate_PD_or_oracle_gap",
        "stylistic_variation_candidate",
    ]
    c4, c5 = st.columns(2)
    selected_labels = c4.multiselect(
        "Candidate labels",
        options=[v for v in candidate_labels if v in label.unique().tolist()],
        default=[v for v in ["PD_candidate", "pedagogical_design_or_oracle_gap"] if v in label.unique().tolist()],
        key="claims_kappa_pd_candidate_labels",
    )
    top_n = int(
        c5.number_input(
            "Top N",
            min_value=2,
            max_value=1000,
            value=10,
            step=1,
            key="claims_kappa_pd_top_n",
        )
    )

    table = high_hallu[high_hallu["tr_label_candidate"].astype(str).isin(selected_labels)].copy()
    if table.empty:
        st.info("No candidate rows for the selected TR-aware labels.")
        return

    table = table.sort_values(
        ["tr_candidate_score", "task", "creativity_level", "title"],
        ascending=[False, True, True, True],
    )
    cols = [
        c
        for c in [
            "model_name",
            "task",
            "creativity_level",
            "kappa_level",
            "title",
            "verification_status",
            "claim",
            "sentence",
            "tr_label_candidate",
            "tr_reason",
            "tr_marker_hits",
            "tr_candidate_score",
        ]
        if c in table.columns
    ]
    _show_dataframe(table[cols].head(top_n), use_container_width=True, hide_index=True)
    st.download_button(
        "Download κ-high PD candidates CSV",
        data=table[cols].to_csv(index=False).encode("utf-8"),
        file_name="hallulens_kappa_high_pd_candidates.csv",
        mime="text/csv",
        key="claims_kappa_pd_download",
    )


def _render_verification_diagnostics(prompt_summary: pd.DataFrame) -> None:
    st.markdown("#### Verification diagnostics")
    st.caption(
        "Évalue la qualité de l'évaluation factuelle (precision/recall/f1) et "
        "les zones à risque selon les regroupements expérimentaux."
    )
    if prompt_summary.empty:
        st.info("No prompt summary for diagnostics.")
        return

    diag = prompt_summary.dropna(subset=["precision_mean", "recall_mean", "hallucination_rate_claim_level"]).copy()
    if not diag.empty:
        st.caption(
            "Scatter: chaque point = un prompt, couleur = hallucination claim-level."
        )
        fig_scatter = px.scatter(
            diag,
            x="precision_mean",
            y="recall_mean",
            color="hallucination_rate_claim_level",
            hover_data=[c for c in ["title", "task", "model_name", "n_verified"] if c in diag.columns],
            color_continuous_scale="Turbo",
        )
        fig_scatter.update_layout(height=470)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Precision/recall diagnostics unavailable for current filters.")

    run_cols = [
        c
        for c in ["root_name", "model_name", "task", "creativity_level"]
        if c in prompt_summary.columns
    ]
    if not run_cols:
        run_cols = ["prompt_clean"]
    run_diag = (
        prompt_summary.groupby(run_cols, dropna=False)
        .agg(
            prompts=("prompt_clean", "nunique"),
            claims_verified=("n_verified", "sum"),
            hallucination_rate_mean=("hallucination_rate_claim_level", "mean"),
            precision_mean=("precision_mean", "mean"),
            recall_mean=("recall_mean", "mean"),
            f1_mean=("f1_mean", "mean"),
        )
        .reset_index()
        .sort_values(["hallucination_rate_mean", "claims_verified"], ascending=[False, False])
    )
    st.caption(
        "Table agrégée par configuration pour comparer fiabilité et volume de claims vérifiés."
    )
    _show_dataframe(run_diag, use_container_width=True, hide_index=True)


def _ensure_claims_explorer_columns(claims_df: pd.DataFrame) -> pd.DataFrame:
    if claims_df is None or claims_df.empty:
        return claims_df
    normalized = claims_df.copy()
    if "kappa_level" not in normalized.columns:
        normalized["kappa_level"] = "NA"
    else:
        normalized["kappa_level"] = normalized["kappa_level"].fillna("NA")

    required_tr_cols = {"tr_label_candidate", "tr_reason", "tr_marker_hits", "tr_candidate_score"}
    if not required_tr_cols.issubset(normalized.columns):
        normalized = add_tr_relabeling_columns(normalized)
    return normalized


def _kappa_pd_candidate_filter_config(cfg: dict[str, Any]) -> dict[str, Any]:
    candidate_cfg = dict(cfg)
    candidate_cfg["kappa_filter"] = ("HIGH",)
    candidate_cfg["status_filter"] = ("hallucinated",)
    return candidate_cfg


def render_claims_explorer_page(selected_roots: tuple[str, ...]) -> None:
    with st.spinner("Loading LLM responses + claim-level verification data..."):
        claims_df, _prompt_summary_df, coverage_df = load_claims_explorer_dataset_cached(selected_roots)
    st.caption(
        "Cette page sert à auditer en profondeur le pipeline: prompt -> réponse LLM -> claims extraits -> vérification."
    )

    _render_claims_coverage_block(coverage_df)
    if claims_df.empty:
        st.warning("No claim-level data found. Requires runs with `run_config.json`, `generation.jsonl`, `output.csv`.")
        return
    claims_df = _ensure_claims_explorer_columns(claims_df)

    filter_cfg = _claims_sidebar_filter_config(claims_df)
    run_dir_signature = (
        tuple(sorted(claims_df["run_dir"].dropna().astype(str).unique().tolist()))
        if "run_dir" in claims_df.columns
        else tuple()
    )
    dataset_signature = (
        tuple(selected_roots),
        len(claims_df),
        int(claims_df["run_id"].nunique()) if "run_id" in claims_df.columns else 0,
        int(claims_df["claim_row_id"].nunique()) if "claim_row_id" in claims_df.columns else 0,
        run_dir_signature,
    )
    filter_signature = tuple(sorted((k, str(v)) for k, v in filter_cfg.items()))
    cache_key = (dataset_signature, filter_signature)
    cached = st.session_state.get("claims_filtered_cache")
    if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
        filtered_claims = cached["filtered_claims"]
        prompt_summary_filtered = cached["prompt_summary_filtered"]
    else:
        filtered_claims = _filter_claims_with_config(claims_df, filter_cfg)
        prompt_summary_filtered = build_prompt_summary_from_claims(filtered_claims)
        st.session_state["claims_filtered_cache"] = {
            "cache_key": cache_key,
            "filtered_claims": filtered_claims,
            "prompt_summary_filtered": prompt_summary_filtered,
        }

    if filtered_claims.empty:
        st.warning("No claim rows after filters.")
        return

    _render_claims_kpis(filtered_claims, prompt_summary_filtered)
    section = st.radio(
        "Section",
        options=["Vue globale", "Drill-down prompt", "Claim rows", "κ-high PD candidates", "Diagnostics"],
        horizontal=True,
        key="claims_active_section",
    )
    if section == "Vue globale":
        st.caption("Vue macro des statuts de vérification et du risque d'hallucination.")
        _render_claims_global_analysis(filtered_claims, prompt_summary_filtered)
    elif section == "Drill-down prompt":
        st.caption("Analyse micro au niveau prompt avec comparaison A/B de réponses.")
        _render_prompt_drilldown(filtered_claims, prompt_summary_filtered)
    elif section == "Claim rows":
        st.caption("Exploration ligne par ligne des claims extraits et de leur statut.")
        _render_claim_rows_table(filtered_claims)
    elif section == "κ-high PD candidates":
        st.caption("Extraction de candidats qualitatifs HalluLens hallucinated → PD sous κ high.")
        candidate_claims = _filter_claims_with_config(
            claims_df,
            _kappa_pd_candidate_filter_config(filter_cfg),
        )
        _render_kappa_pd_candidates(candidate_claims)
    else:
        st.caption("Diagnostics de performance du pipeline d'évaluation factuelle.")
        _render_verification_diagnostics(prompt_summary_filtered)


def render_evaluator_agreement_page(selected_roots: tuple[str, ...]) -> None:
    with st.spinner("Loading evaluation_test runs and claim-level verdicts..."):
        claims_df, runs_df = load_evaluation_test_claims_dataset_cached(selected_roots, version=2)

    st.caption(
        "Étude d'ablation complète des modèles évaluateurs du pipeline. "
        "Chaque run utilise les mêmes prompts/réponses; seul le modèle d'évaluation varie. "
        "Les métriques couvrent: diversité d'extraction, taux d'hallucination, accord pairwise et interchangeabilité."
    )

    if runs_df.empty or claims_df.empty:
        st.warning(
            "Aucun run `evaluation_test` exploitable trouvé (besoin de `run_config.json` + `output.csv` avec `is_supported`)."
        )
        return

    # ------------------------------------------------------------------ #
    # Sidebar filters                                                       #
    # ------------------------------------------------------------------ #
    st.sidebar.header("Evaluator Agreement Filters")
    root_filter = st.sidebar.multiselect(
        "Dataset (evaluation)",
        options=sorted_unique(runs_df["root_name"]),
        default=sorted_unique(runs_df["root_name"]),
        key="eval_agreement_root_filter",
    )
    generation_filter = st.sidebar.multiselect(
        "Generation model",
        options=sorted_unique(runs_df["generation_model"]),
        default=sorted_unique(runs_df["generation_model"]),
        key="eval_agreement_generation_filter",
    )
    evaluator_filter = st.sidebar.multiselect(
        "Evaluator signature",
        options=sorted_unique(runs_df["evaluator_label"]),
        default=sorted_unique(runs_df["evaluator_label"]),
        key="eval_agreement_evaluator_filter",
    )
    task_filter = st.sidebar.multiselect(
        "Task",
        options=sorted_unique(claims_df["task"]),
        default=sorted_unique(claims_df["task"]),
        key="eval_agreement_task_filter",
    )
    min_overlap = int(
        st.sidebar.slider(
            "Min overlap (pairwise)",
            min_value=5,
            max_value=5000,
            value=50,
            step=5,
            key="eval_agreement_min_overlap",
        )
    )
    min_generations = int(
        st.sidebar.slider(
            "Min shared generations (Jaccard)",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            key="eval_agreement_min_gen",
        )
    )
    control_other_stages = bool(
        st.sidebar.checkbox(
            "Strict stage control",
            value=False,
            key="eval_agreement_control_other_stages",
            help="Compare deux modèles d'un composant uniquement quand les autres étapes du pipeline sont identiques.",
        )
    )

    # ------------------------------------------------------------------ #
    # Apply filters                                                         #
    # ------------------------------------------------------------------ #
    runs_filtered = runs_df.copy()
    runs_filtered = apply_multiselect_filter(runs_filtered, "root_name", root_filter)
    runs_filtered = apply_multiselect_filter(runs_filtered, "generation_model", generation_filter)
    runs_filtered = apply_multiselect_filter(runs_filtered, "evaluator_label", evaluator_filter)
    if runs_filtered.empty:
        st.warning("Aucun run après filtres.")
        return

    claims_filtered = claims_df[claims_df["run_id"].isin(runs_filtered["run_id"])].copy()
    claims_filtered = apply_multiselect_filter(claims_filtered, "task", task_filter)
    if claims_filtered.empty:
        st.warning("Aucune claim row après filtres.")
        return

    # ------------------------------------------------------------------ #
    # Base analytics                                                        #
    # ------------------------------------------------------------------ #
    consensus_df = build_evaluator_claim_consensus(claims_filtered)
    if consensus_df.empty:
        st.warning("Impossible de construire les votes de consensus des évaluateurs.")
        return

    n_evaluators = int(consensus_df["evaluator_label"].nunique())
    if n_evaluators < 2:
        st.warning("Il faut au moins 2 évaluateurs distincts pour mesurer l'accord.")
        return

    pairwise_overall = build_pairwise_agreement_table(consensus_df, min_overlap=min_overlap, by_task=False)
    pairwise_task = build_pairwise_agreement_table(consensus_df, min_overlap=min_overlap, by_task=True)
    overall_support_df, task_support_df = build_evaluator_support_tables(consensus_df)

    component_specs = [
        ("claim_extractor_model", "Claim Extractor"),
        ("abstain_evaluator_model", "Abstain Evaluator"),
        ("verifier_model", "Verifier"),
        ("evaluator_label", "Final Pipeline Output"),
    ]
    component_results: dict[str, tuple[str, pd.DataFrame, pd.DataFrame]] = {}
    for component_col, component_label in component_specs:
        comp_overall = build_component_pairwise_agreement_cached(
            claims_filtered,
            component_col=component_col,
            min_overlap=min_overlap,
            by_task=False,
            control_other_stages=control_other_stages,
        )
        comp_task = build_component_pairwise_agreement_cached(
            claims_filtered,
            component_col=component_col,
            min_overlap=min_overlap,
            by_task=True,
            control_other_stages=control_other_stages,
        )
        component_results[component_col] = (component_label, comp_overall, comp_task)

    # Diversity + hallucination stats
    div_overall, div_task, halu_overall, halu_task = build_diversity_and_hallucination_stats_cached(claims_filtered)

    # Extraction Jaccard agreement
    extraction_overall = build_extraction_agreement_cached(claims_filtered, min_generations=min_generations, by_task=False)
    extraction_task = build_extraction_agreement_cached(claims_filtered, min_generations=min_generations, by_task=True)

    coverage_per_claim = consensus_df.groupby("claim_key", dropna=False)["evaluator_label"].nunique()
    comparable_claims = int((coverage_per_claim >= 2).sum())

    # ------------------------------------------------------------------ #
    # KPIs                                                                  #
    # ------------------------------------------------------------------ #
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Runs evaluation_test", f"{runs_filtered['run_id'].nunique():,}")
    k2.metric("Evaluators", f"{n_evaluators:,}")
    k3.metric("Total claims", f"{len(claims_filtered):,}")
    k4.metric("Comparable claims (≥2)", f"{comparable_claims:,}")
    k5.metric(
        "Moy. support rate",
        f"{overall_support_df['support_rate'].mean():.3f}" if not overall_support_df.empty else "NA",
    )

    # ------------------------------------------------------------------ #
    # Runs détectés + avertissement couplage                               #
    # ------------------------------------------------------------------ #
    with st.expander("Runs evaluation_test détectés", expanded=False):
        run_cols = [
            c for c in [
                "root_name", "run_name", "generation_model", "evaluator_label",
                "claim_extractor_model", "abstain_evaluator_model", "verifier_model",
                "n_claim_rows", "support_rate",
            ]
            if c in runs_filtered.columns
        ]
        _show_dataframe(
            runs_filtered.sort_values(["generation_model", "evaluator_label", "run_name"])[run_cols],
            use_container_width=True,
            hide_index=True,
        )

    stage_cols = ["claim_extractor_model", "abstain_evaluator_model", "verifier_model"]
    if set(stage_cols).issubset(runs_filtered.columns):
        ce = runs_filtered["claim_extractor_model"].fillna("unknown").astype(str)
        ab = runs_filtered["abstain_evaluator_model"].fillna("unknown").astype(str)
        vf = runs_filtered["verifier_model"].fillna("unknown").astype(str)
        if bool(((ce == ab) & (ab == vf)).all()):
            st.warning(
                "⚠ Les composants `claim_extractor`, `abstain_evaluator` et `verifier` utilisent "
                "toujours le même modèle dans les runs filtrés. Les effets par composant ne sont pas "
                "identifiables séparément — on mesure en réalité l'accord de la chaîne complète."
            )

    st.divider()

    # ================================================================== #
    # SECTION 1 : DIVERSITÉ D'EXTRACTION                                  #
    # ================================================================== #
    st.subheader("📊 Diversité d'extraction des claims")
    st.caption(
        "Nombre moyen de claims extraites par génération (phrase source). "
        "Un modèle plus *verbeux* génère plus de claims, ce qui peut réduire la précision "
        "ou augmenter artificiellement le taux d'hallucination."
    )

    if div_overall.empty:
        st.info("Données de diversité insuffisantes (colonne `generation_key` manquante).")
    else:
        col_div_a, col_div_b = st.columns(2)

        with col_div_a:
            fig_div = px.bar(
                div_overall,
                x="model",
                y="claims_per_gen_mean",
                error_y="claims_per_gen_std",
                color="model",
                text=div_overall["claims_per_gen_mean"].map(lambda x: f"{x:.2f}"),
                title="Claims extraites par génération (moy ± std)",
                labels={"model": "Modèle", "claims_per_gen_mean": "Claims / génération"},
            )
            fig_div.update_traces(textposition="outside")
            fig_div.update_layout(height=420, showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_div, use_container_width=True)

        with col_div_b:
            if not div_task.empty:
                fig_div_task = px.bar(
                    div_task,
                    x="task",
                    y="claims_per_gen_mean",
                    color="model",
                    barmode="group",
                    error_y="claims_per_gen_std",
                    title="Claims / génération par tâche et par extracteur",
                    labels={"task": "Tâche", "claims_per_gen_mean": "Claims / génération", "model": "Modèle"},
                )
                fig_div_task.update_layout(height=430, xaxis_tickangle=-10)
                st.plotly_chart(fig_div_task, use_container_width=True)

    st.divider()
    # SECTION 2 : TAUX D'HALLUCINATION FINAL                              #
    # ================================================================== #
    st.subheader("🔴 Taux d'hallucination par évaluateur")
    st.caption(
        "Le taux d'hallucination est `1 - support_rate`. Un modèle plus strict (tendance à marquer "
        "les claims comme non-supportées) produira un taux d'hallucination plus élevé, "
        "indépendamment de la qualité réelle du texte évalué."
    )

    if halu_overall.empty:
        st.info("Données de hallucination insuffisantes.")
    else:
        col_h_a, col_h_b = st.columns(2)

        with col_h_a:
            color_map = {row["evaluator_label"]: px.colors.qualitative.Plotly[i % 10]
                         for i, row in halu_overall.iterrows()}
            fig_halu = px.bar(
                halu_overall,
                x="evaluator_label",
                y="hallucination_rate",
                color="evaluator_label",
                text=halu_overall["hallucination_rate"].map(lambda x: f"{x:.3f}"),
                title="Taux d'hallucination global par évaluateur",
                labels={"evaluator_label": "Évaluateur", "hallucination_rate": "Hallucination rate"},
            )
            fig_halu.update_traces(textposition="outside")
            fig_halu.update_yaxes(range=[0, 1])
            fig_halu.update_layout(height=420, showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_halu, use_container_width=True)

        with col_h_b:
            fig_support_bar = px.bar(
                halu_overall,
                x="evaluator_label",
                y="support_rate",
                color="evaluator_label",
                text=halu_overall["support_rate"].map(lambda x: f"{x:.3f}"),
                title="Support rate (claims jugées correctes) par évaluateur",
                labels={"evaluator_label": "Évaluateur", "support_rate": "Support rate"},
            )
            fig_support_bar.update_traces(textposition="outside")
            fig_support_bar.update_yaxes(range=[0, 1])
            fig_support_bar.update_layout(height=420, showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_support_bar, use_container_width=True)

        if not halu_task.empty:
            col_ht_a, col_ht_b = st.columns(2)
            with col_ht_a:
                fig_halu_task = px.bar(
                    halu_task,
                    x="task",
                    y="hallucination_rate",
                    color="evaluator_label",
                    barmode="group",
                    title="Taux d'hallucination par tâche et évaluateur",
                    labels={"task": "Tâche", "hallucination_rate": "Hallucination rate", "evaluator_label": "Évaluateur"},
                )
                fig_halu_task.update_yaxes(range=[0, 1])
                fig_halu_task.update_layout(height=430)
                st.plotly_chart(fig_halu_task, use_container_width=True)

            with col_ht_b:
                fig_supp_task = px.bar(
                    halu_task,
                    x="task",
                    y="support_rate",
                    color="evaluator_label",
                    barmode="group",
                    title="Support rate par tâche et évaluateur",
                    labels={"task": "Tâche", "support_rate": "Support rate", "evaluator_label": "Évaluateur"},
                )
                fig_supp_task.update_yaxes(range=[0, 1])
                fig_supp_task.update_layout(height=430)
                st.plotly_chart(fig_supp_task, use_container_width=True)

        # Radar chart: comparaison multidimensionnelle par évaluateur
        if not halu_task.empty and halu_task["task"].nunique() >= 3:
            tasks_list = sorted(halu_task["task"].unique().tolist())
            import plotly.graph_objects as go
            fig_radar = go.Figure()
            for evaluator in halu_task["evaluator_label"].unique():
                ev_data = halu_task[halu_task["evaluator_label"] == evaluator]
                r_vals = [float(ev_data[ev_data["task"] == t]["hallucination_rate"].iloc[0])
                          if t in ev_data["task"].values else 0.0
                          for t in tasks_list]
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_vals + [r_vals[0]],
                    theta=tasks_list + [tasks_list[0]],
                    fill="toself",
                    name=evaluator,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 1])),
                title="Radar: taux d'hallucination par tâche et évaluateur",
                height=450,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # ================================================================== #
    # SECTION 3 : ACCORD D'EXTRACTION (JACCARD)                          #
    # ================================================================== #
    st.subheader("🔵 Accord d'extraction des claims (similarité de Jaccard)")
    st.caption(
        "Pour chaque paire de modèles, on calcule la similarité de Jaccard entre les ensembles "
        "de claims extraites d'une même génération (phrase source): "
        "`J = |A ∩ B| / |A ∪ B|`. Une valeur proche de 1 signifie que les deux modèles "
        "extraient les mêmes claims du même texte."
    )

    if extraction_overall.empty:
        st.info(
            "Pas assez de générations communes pour calculer la similarité de Jaccard. "
            "Essayez de réduire `Min shared generations`."
        )
    else:
        col_jac_a, col_jac_b = st.columns([1, 1])

        with col_jac_a:
            fig_jac = px.bar(
                extraction_overall.sort_values("jaccard_mean", ascending=False),
                x="pair_label",
                y="jaccard_mean",
                error_y="jaccard_std",
                color="pair_label",
                text=extraction_overall.sort_values("jaccard_mean", ascending=False)["jaccard_mean"].map(
                    lambda x: f"{x:.3f}"
                ),
                title="Similarité de Jaccard (extraction) — global",
                labels={"pair_label": "Paire", "jaccard_mean": "Jaccard moyen"},
            )
            fig_jac.update_traces(textposition="outside")
            fig_jac.update_yaxes(range=[0, 1])
            fig_jac.update_layout(height=420, showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_jac, use_container_width=True)

        with col_jac_b:
            # Heatmap Jaccard
            models_ext = sorted(
                list(set(extraction_overall["model_a"].tolist() + extraction_overall["model_b"].tolist()))
            )
            if len(models_ext) >= 2:
                mat_jac = pd.DataFrame(np.nan, index=models_ext, columns=models_ext, dtype=float)
                for lbl in models_ext:
                    mat_jac.loc[lbl, lbl] = 1.0
                for _, row in extraction_overall.iterrows():
                    v = float(row["jaccard_mean"]) if pd.notna(row["jaccard_mean"]) else np.nan
                    mat_jac.loc[str(row["model_a"]), str(row["model_b"])] = v
                    mat_jac.loc[str(row["model_b"]), str(row["model_a"])] = v
                fig_jac_heat = px.imshow(
                    mat_jac,
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    text_auto=".2f",
                    title="Heatmap Jaccard (extraction)",
                )
                fig_jac_heat.update_layout(height=420)
                st.plotly_chart(fig_jac_heat, use_container_width=True)

        if not extraction_task.empty:
            fig_jac_task = px.bar(
                extraction_task.sort_values(["task", "jaccard_mean"], ascending=[True, False]),
                x="task",
                y="jaccard_mean",
                color="pair_label",
                barmode="group",
                title="Jaccard par tâche",
                labels={"task": "Tâche", "jaccard_mean": "Jaccard moyen", "pair_label": "Paire"},
            )
            fig_jac_task.update_yaxes(range=[0, 1])
            fig_jac_task.update_layout(height=430)
            st.plotly_chart(fig_jac_task, use_container_width=True)

    st.divider()

    # ================================================================== #
    # SECTION 4 : ACCORD PAIRWISE PAR COMPOSANT DU PIPELINE              #
    # ================================================================== #
    st.subheader("🟡 Accord pairwise sur le verdict final (Kappa / Agreement rate)")
    st.caption(
        "**Note:** Comme les runs sont couplés (A-A-A vs B-B-B), les scores pour "
        "`Claim Extractor`, `Abstain Evaluator`, `Verifier` et `Final Pipeline Output` "
        "sont identiques — ils mesurent tous l'accord sur le verdict final `is_supported`. "
        "Utilisez la section Jaccard ci-dessus pour évaluer l'extraction spécifiquement."
    )
    if control_other_stages:
        st.caption("Mode strict activé: les autres étapes du pipeline sont contrôlées.")

    preferred_tasks = ["INTERVIEW", "NEWS_ARTICLE", "LESSON_PLAN"]
    component_tab_labels = [component_results[k][0] for k, _ in component_specs]
    component_tabs = st.tabs(component_tab_labels)

    for tab, (component_col, component_label) in zip(component_tabs, component_specs):
        with tab:
            _, comp_overall, comp_task = component_results[component_col]
            if comp_overall.empty and comp_task.empty:
                st.info(
                    f"Pas assez de recouvrement pour `{component_label}`. "
                    "Diminuez `Min overlap for pairwise metrics`."
                )
                continue

            metric = st.radio(
                f"Metric — {component_label}",
                options=["kappa", "agreement_rate"],
                index=0,
                horizontal=True,
                key=f"eval_agreement_metric_{component_col}",
            )
            pair_pool = sorted(
                pd.unique(
                    pd.concat(
                        [
                            comp_overall["pair_label"] if not comp_overall.empty else pd.Series(dtype=str),
                            comp_task["pair_label"] if not comp_task.empty else pd.Series(dtype=str),
                        ],
                        ignore_index=True,
                    )
                ).tolist()
            )
            selected_pairs = st.multiselect(
                f"Paires — {component_label}",
                options=pair_pool,
                default=pair_pool[: min(8, len(pair_pool))],
                key=f"eval_agreement_pairs_{component_col}",
            )

            available_tasks = sorted_unique(comp_task["task"]) if not comp_task.empty else []
            task_tabs = preferred_tasks + [t for t in available_tasks if t not in preferred_tasks]
            sub_tabs = st.tabs(["Ensemble"] + task_tabs)

            with sub_tabs[0]:
                view = comp_overall.copy()
                if selected_pairs:
                    view = apply_multiselect_filter(view, "pair_label", selected_pairs)
                if view.empty:
                    st.info("Aucune paire disponible.")
                else:
                    mat = build_pairwise_metric_matrix(view, metric_col=metric, a_col="model_a", b_col="model_b")
                    if not mat.empty:
                        fig_m = px.imshow(
                            mat,
                            aspect="auto",
                            color_continuous_scale="RdBu" if metric == "kappa" else "RdYlGn",
                            zmin=-1 if metric == "kappa" else 0,
                            zmax=1,
                            text_auto=".2f",
                        )
                        fig_m.update_layout(height=460, title=f"{metric} (ensemble) — {component_label}")
                        st.plotly_chart(fig_m, use_container_width=True)

            for idx, task_name in enumerate(task_tabs, start=1):
                with sub_tabs[idx]:
                    task_view = comp_task[comp_task["task"].astype(str) == str(task_name)].copy()
                    if selected_pairs:
                        task_view = apply_multiselect_filter(task_view, "pair_label", selected_pairs)
                    if task_view.empty:
                        st.info(f"Aucune paire pour la tâche `{task_name}`.")
                        continue
                    mat_task = build_pairwise_metric_matrix(task_view, metric_col=metric, a_col="model_a", b_col="model_b")
                    if not mat_task.empty:
                        fig_mt = px.imshow(
                            mat_task,
                            aspect="auto",
                            color_continuous_scale="RdBu" if metric == "kappa" else "RdYlGn",
                            zmin=-1 if metric == "kappa" else 0,
                            zmax=1,
                            text_auto=".2f",
                        )
                        fig_mt.update_layout(height=460, title=f"{metric} ({task_name}) — {component_label}")
                        st.plotly_chart(fig_mt, use_container_width=True)

    # ================================================================== #
    # SECTION 5 : EXPORTS                                                  #
    # ================================================================== #
    st.subheader("Exports")

    component_overall_frames: list[pd.DataFrame] = []
    component_task_frames: list[pd.DataFrame] = []
    for component_col, component_label in component_specs:
        comp_overall = component_results[component_col][1]
        comp_task = component_results[component_col][2]
        if not comp_overall.empty:
            component_overall_frames.append(comp_overall.assign(component_label=component_label))
        if not comp_task.empty:
            component_task_frames.append(comp_task.assign(component_label=component_label))
    component_overall_export = (
        pd.concat(component_overall_frames, ignore_index=True) if component_overall_frames else pd.DataFrame()
    )
    component_task_export = (
        pd.concat(component_task_frames, ignore_index=True) if component_task_frames else pd.DataFrame()
    )

    exp_cols = st.columns(5)
    exp_cols[0].download_button(
        "Pairwise overall CSV",
        data=component_overall_export.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_evaluator_component_pairwise_overall.csv",
        mime="text/csv",
        disabled=component_overall_export.empty,
    )
    exp_cols[1].download_button(
        "Pairwise by task CSV",
        data=component_task_export.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_evaluator_component_pairwise_by_task.csv",
        mime="text/csv",
        disabled=component_task_export.empty,
    )
    exp_cols[2].download_button(
        "Support by task CSV",
        data=task_support_df.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_evaluator_support_by_task.csv",
        mime="text/csv",
        disabled=task_support_df.empty,
    )
    exp_cols[3].download_button(
        "Diversity CSV",
        data=div_task.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_evaluator_diversity_by_task.csv",
        mime="text/csv",
        disabled=div_task.empty,
    )
    exp_cols[4].download_button(
        "Hallucination CSV",
        data=halu_task.to_csv(index=False).encode("utf-8"),
        file_name="hallulens_evaluator_hallucination_by_task.csv",
        mime="text/csv",
        disabled=halu_task.empty,
    )



def main() -> None:
    st.set_page_config(page_title="HalluLens Dashboard", layout="wide")
    st.title("HalluLens Dashboard")
    st.caption("Interactive analysis of hallucinations and creativity signals across experimental runs.")

    output_root = Path("output")
    root_options = sorted([p.as_posix() for p in output_root.iterdir() if p.is_dir()]) if output_root.exists() else []
    if not root_options:
        st.error("No output directories found under ./output")
        return

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        options=[PAGE_IMPACT, PAGE_CREATIVITY, PAGE_CLAIMS_EXPLORER, PAGE_EVALUATOR_AGREEMENT, PAGE_LLM_EXPORT],
        index=0,
        key="dashboard_page",
    )

    default_root = [r for r in root_options if r.endswith("longwiki-hybrid")] or root_options[:1]
    selected_roots = st.sidebar.multiselect(
        "Output roots",
        options=root_options,
        default=default_root,
        help="Runs are loaded from selected output roots.",
    )
    if not selected_roots:
        st.warning("Select at least one output root.")
        return

    selected_roots_tuple = tuple(sorted(selected_roots))
    if page == PAGE_IMPACT:
        st.subheader(PAGE_IMPACT)
        render_impact_page(selected_roots_tuple)
    elif page == PAGE_CREATIVITY:
        st.subheader(PAGE_CREATIVITY)
        render_creativity_page(selected_roots_tuple)
    elif page == PAGE_CLAIMS_EXPLORER:
        st.subheader(PAGE_CLAIMS_EXPLORER)
        render_claims_explorer_page(selected_roots_tuple)
    elif page == PAGE_EVALUATOR_AGREEMENT:
        st.subheader(PAGE_EVALUATOR_AGREEMENT)
        render_evaluator_agreement_page(selected_roots_tuple)
    else:
        render_llm_export_page(selected_roots_tuple)
