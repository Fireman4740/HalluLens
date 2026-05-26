from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .plotting_publication import (
    build_claim_density_vs_hallucination_plot,
    build_mediation_path_diagram,
    build_per_model_temperature_effect_plot_from_stats,
    build_prompt_variance_icc_plot,
    compute_per_model_temperature_effect_stats,
)


@st.cache_data(show_spinner=False, persist=True)
def run_impact_per_model_temp_stats_cached(
    filtered_df: pd.DataFrame,
    min_n_per_temp: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    return compute_per_model_temperature_effect_stats(
        filtered_df,
        target_col="hallucination_rate",
        model_col="model_name",
        temperature_col="temperature",
        min_n_per_temp=min_n_per_temp,
        n_boot=n_boot,
        seed=seed,
    )


@st.cache_data(show_spinner=False, persist=True)
def run_impact_claim_density_plot_cached(
    filtered_df: pd.DataFrame,
    lowess_frac: float,
    min_points_for_lowess: int,
    marker_opacity: float,
    max_points: int,
    sample_seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fig, summary_df = build_claim_density_vs_hallucination_plot(
        filtered_df,
        claims_col="n_claims",
        response_len_col="response_length_words",
        y_col="hallucination_rate",
        creativity_col="creativity_level",
        lowess_frac=lowess_frac,
        min_points_for_lowess=min_points_for_lowess,
        marker_opacity=marker_opacity,
        max_points=max_points,
        sample_seed=sample_seed,
    )
    return fig.to_plotly_json(), summary_df


@st.cache_data(show_spinner=False, persist=True)
def run_impact_prompt_variance_plot_cached(
    filtered_df: pd.DataFrame,
    n_bins: int,
    icc_value: float,
    show_kde: bool,
    show_mixture: bool,
    mixture_components: int,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    fig, prompt_means_df, summary_df = build_prompt_variance_icc_plot(
        filtered_df,
        target_col="hallucination_rate",
        prompt_col="prompt_id",
        n_bins=n_bins,
        icc_value=icc_value,
        show_kde=show_kde,
        show_mixture=show_mixture,
        mixture_components=mixture_components,
        random_state=random_state,
    )
    return fig.to_plotly_json(), prompt_means_df, summary_df


@st.cache_data(show_spinner=False, persist=True)
def run_impact_mediation_diagram_cached(
    mediation_df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fig, summary_df = build_mediation_path_diagram(mediation_df)
    return fig.to_plotly_json(), summary_df


def build_per_model_temp_figure_from_stats(
    stats_df: pd.DataFrame,
    n_cols: int,
) -> go.Figure:
    return build_per_model_temperature_effect_plot_from_stats(
        stats_df,
        model_col="model_name",
        temperature_col="temperature",
        target_col="hallucination_rate",
        n_cols=n_cols,
    )


def figure_from_json(fig_json: dict[str, Any]) -> go.Figure:
    return go.Figure(fig_json)
