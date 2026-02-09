from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as components_html

from .analytics import HAVE_SCIPY, build_impact_summary, build_spearman_detailed
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
    PAGE_CREATIVITY,
    PAGE_IMPACT,
    PAGE_LLM_EXPORT,
)
from .creativity_analytics import (
    HAVE_SCIPY as HAVE_SCIPY_ADV,
    HAVE_STATSMODELS,
    build_creativity_corr_table,
    build_intent_contrast_table,
    build_partial_corr_table,
    fit_binomial_glm,
    predict_binomial_glm_probabilities,
)
from .creativity_loading import load_creativity_dataset
from .creativity_plotting import (
    build_creativity_heatmap,
    build_creativity_metrics_by_level_plot,
    build_creativity_scatter,
    build_glm_forest,
    build_intent_boxplot,
)
from .data_loading import load_prompt_dataset
from .plotting import build_distribution_plot, build_line_plot, build_points_plot
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


@st.cache_data(show_spinner=False)
def run_impact_advanced_analysis_cached(
    filtered_df: pd.DataFrame,
    corr_metrics: tuple[str, ...],
    partial_metrics: tuple[str, ...],
    n_boot: int,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    corr_df = build_creativity_corr_table(
        filtered_df,
        target_col="hallucination_rate",
        metrics=list(corr_metrics),
        n_boot=n_boot,
        n_perm=n_perm,
        seed=seed,
    )
    partial_df = build_partial_corr_table(
        filtered_df,
        target_col="hallucination_rate",
        metrics=list(partial_metrics),
        control_cols=("response_length_words", "n_claim_rows"),
        n_boot=n_boot,
        seed=seed,
    )
    intent_df = build_intent_contrast_table(
        filtered_df,
        target_col="hallucination_rate",
    )
    glm_df = fit_binomial_glm(filtered_df, target_rate_col="hallucination_rate")
    return corr_df, partial_df, intent_df, glm_df


def _format_number(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except Exception:
        return "NA"
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def _to_csv_block(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "Aucune donnée disponible."
    out = df.head(max_rows).copy()
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        out[col] = out[col].round(6)
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

    # Creativity page dataset (strict + complete runs)
    creativity_df, coverage_df = load_creativity_dataset(
        selected_roots=selected_roots,
        strict_mode=True,
        exclude_incomplete_runs=True,
        base_df=impact_df,
    )
    coverage_stats = {}
    if not coverage_df.empty:
        coverage_stats = coverage_df["status"].value_counts(dropna=False).to_dict()
    complete_runs = int(coverage_stats.get("complete", 0))
    partial_runs = int(coverage_stats.get("partial", 0))
    missing_runs = int(coverage_stats.get("missing", 0))

    creativity_rows = len(creativity_df)
    creativity_runs = int(creativity_df["run_id"].nunique()) if not creativity_df.empty else 0
    creativity_hallu_mean = (
        float(creativity_df["hallucination_rate"].mean()) if not creativity_df.empty else np.nan
    )
    creativity_ttct_mean = (
        float(creativity_df["ttct_overall"].mean()) if not creativity_df.empty else np.nan
    )
    creativity_ttcw_mean = (
        float(creativity_df["ttcw_overall"].mean()) if not creativity_df.empty else np.nan
    )

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

    lines: list[str] = []
    lines.append("# HalluLens - Prompt prêt pour analyse par chatbot AI")
    lines.append("")
    lines.append("## Contexte projet")
    lines.append(project_context)
    lines.append("")
    lines.append("## Paramètres de génération du rapport")
    lines.append(f"- selected_roots: {', '.join(selected_roots)}")
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
    lines.append("## Instructions pour le chatbot AI")
    lines.append(
        "Analyse ce rapport en identifiant: (1) les effets les plus robustes, "
        "(2) les facteurs potentiellement confondants, (3) les recommandations expérimentales "
        "priorisées, (4) les risques méthodologiques, (5) 5 prochaines analyses à lancer."
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
    required_df = ["corr_df", "partial_df", "intent_df", "glm_df"]
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
    lines.append("### Advanced analysis - paired contrasts")
    lines.append(_to_csv_block(payload.get("intent_df"), max_rows=max_rows_per_table))
    lines.append("")
    lines.append("### Advanced analysis - GLM binomial")
    lines.append(_to_csv_block(payload.get("glm_df"), max_rows=max_rows_per_table))
    return "\n".join(lines)


def render_llm_export_page(selected_roots: tuple[str, ...]) -> None:
    st.subheader("LLM Export")
    st.caption(
        "Génère un texte structuré (contexte projet + résultats Impact + résultats Creativity) "
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
            "Mode export: aucun recalcul avancé ici. "
            "Les résultats avancés sont repris uniquement depuis la page Impact."
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

        with st.spinner("Construction du package LLM..."):
            export_text = build_llm_export_text_cached(
                selected_roots=selected_roots,
                max_rows_per_table=max_rows_per_table,
            )
            export_text = _append_cached_impact_advanced_to_export_text(
                export_text=export_text,
                payload=cached_advanced,
                include_advanced_analysis=include_advanced_analysis,
                max_rows_per_table=max_rows_per_table,
            )

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

    cfg: dict[str, Any] = {
        "chart_type": chart_type,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "line_estimator": line_estimator,
        "dims": dims,
    }

    if chart_type == "line":
        st.sidebar.markdown("#### Multi-lignes (comparaison)")
        st.sidebar.caption("Style de ligne = combinaison des 2 paramètres. Couleur configurable.")

        facet_dims = ["None"] + [c for c in dims if c in plot_df.columns and c != x_axis]
        facet_by_raw = st.sidebar.selectbox(
            "Facet",
            options=facet_dims,
            index=option_index(facet_dims, DEFAULT_LINE_FACET),
            key="impact_facet_line",
        )
        facet_by = None if facet_by_raw == "None" else facet_by_raw

        color_dims = [c for c in dims if c in plot_df.columns and c != x_axis]
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

        series_candidates = [c for c in dims if c in plot_df.columns and c not in {x_axis, line_color_by}]
        if len(series_candidates) < 2:
            st.warning("Pas assez de paramètres pour construire des combinaisons de lignes.")
            return {"invalid": True}

        default_a = DEFAULT_SERIES_A if DEFAULT_SERIES_A in series_candidates else series_candidates[0]
        default_b = DEFAULT_SERIES_B if DEFAULT_SERIES_B in series_candidates and DEFAULT_SERIES_B != default_a else None
        if default_b is None:
            default_b = next(c for c in series_candidates if c != default_a)

        series_param_a = st.sidebar.selectbox(
            "Paramètre ligne A",
            options=series_candidates,
            index=option_index(series_candidates, default_a),
            key="impact_series_param_a",
        )
        series_candidates_b = [c for c in series_candidates if c != series_param_a]
        series_param_b = st.sidebar.selectbox(
            "Paramètre ligne B",
            options=series_candidates_b,
            index=option_index(series_candidates_b, default_b),
            key="impact_series_param_b",
        )
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
        st.dataframe(
            spearman_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )

    if spearman_skipped:
        st.caption("Facteurs ignorés: " + "; ".join(spearman_skipped))


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

    if st.button("Run advanced analysis", type="primary", key="impact_run_advanced_button"):
        st.session_state["impact_run_advanced"] = True
    if st.button("Reset advanced analysis", key="impact_reset_advanced_button"):
        st.session_state["impact_run_advanced"] = False

    if not st.session_state.get("impact_run_advanced", False):
        st.info("Click `Run advanced analysis` to compute advanced statistics on current filtered data.")
        return

    if filtered.empty:
        st.warning("No rows available for advanced analysis.")
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

    with st.spinner("Running advanced analysis on Impact Hallucinations..."):
        corr_df, partial_df, intent_df, glm_df = run_impact_advanced_analysis_cached(
            filtered_df=filtered,
            corr_metrics=tuple(corr_metrics_selected),
            partial_metrics=tuple(partial_metrics_selected),
            n_boot=n_boot,
            n_perm=n_perm,
            seed=seed,
        )
    st.session_state["impact_advanced_export_cache"] = {
        "selected_roots": tuple(selected_roots),
        "corr_df": corr_df.copy(),
        "partial_df": partial_df.copy(),
        "intent_df": intent_df.copy(),
        "glm_df": glm_df.copy(),
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
        st.dataframe(
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
        st.dataframe(
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
        st.dataframe(intent_df, use_container_width=True, hide_index=True)

    st.markdown("#### Binomial GLM (odds ratios)")
    if glm_df.empty:
        st.info("GLM unavailable (insufficient data or missing statsmodels).")
    else:
        st.dataframe(
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
                st.dataframe(pred_df, use_container_width=True, hide_index=True)

    st.markdown("#### Advanced exports")
    export_a, export_b = st.columns(2)
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
        )
    elif chart_cfg["chart_type"] == "points":
        fig = build_points_plot(
            df=plot_df,
            x_axis=chart_cfg["x_axis"],
            y_axis=chart_cfg["y_axis"],
            color_by=chart_cfg["color_by"],
            facet_by=chart_cfg["facet_by"],
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
        )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Impact summary")
    summary = build_impact_summary(filtered, x_axis=chart_cfg["x_axis"], y_axis=chart_cfg["y_axis"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    _render_impact_statistics_fused_section(filtered, selected_roots=selected_roots)

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
    st.dataframe(table_df.head(1000), use_container_width=True, hide_index=True)


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
    st.dataframe(table, use_container_width=True, hide_index=True)


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
    st.dataframe(filtered[preview_cols].head(1000), use_container_width=True, hide_index=True)


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
        options=[PAGE_IMPACT, PAGE_CREATIVITY, PAGE_LLM_EXPORT],
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
    else:
        render_llm_export_page(selected_roots_tuple)
