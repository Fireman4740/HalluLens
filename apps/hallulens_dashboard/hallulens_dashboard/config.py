from __future__ import annotations

CREATIVITY_ORDER = ["FACTUAL", "HYBRID", "VERY_CREATIVE"]

CREATIVITY_COLOR_MAP = {
    "FACTUAL": "#1f77b4",
    "HYBRID": "#ff7f0e",
    "VERY_CREATIVE": "#2ca02c",
    "NA": "#7f7f7f",
}

DEFAULT_CHART_TYPE = "line"
DEFAULT_Y_AXIS = "hallucination_rate"
DEFAULT_X_AXIS = "length_words"
DEFAULT_LINE_ESTIMATOR = "median"
DEFAULT_LINE_FACET = "task"
DEFAULT_LINE_COLOR = "creativity_level"
DEFAULT_SERIES_A = "temperature"
DEFAULT_SERIES_B = "model_name"

PAGE_IMPACT = "Impact Hallucinations"
PAGE_CREATIVITY = "Creativity Prism"
PAGE_LLM_EXPORT = "LLM Export"

CREATIVITY_SCORE_OPTIONS = [
    "creativity_composite",
    "ttct_overall",
    "ttcw_overall",
    "ttct_fluency",
    "ttct_flexibility",
    "ttct_originality",
    "ttct_elaboration",
    "ttcw_coherence",
    "ttcw_descriptive_detail",
    "ttcw_ending",
    "ttcw_emotional_flexibility",
]

HEATMAP_CORR_COLS_DEFAULT = [
    "hallucination_rate",
    "support_rate",
    "n_claim_rows",
    "response_length_words",
    "ttct_overall",
    "ttcw_overall",
    "creativity_composite",
    "creativity_rank",
]

ADV_DEFAULT_BOOTSTRAP_ITERS = 5000
ADV_DEFAULT_PERMUTATION_ITERS = 10000
ADV_DEFAULT_SEED = 42

LLM_EXPORT_DEFAULT_INCLUDE_ADVANCED = False
LLM_EXPORT_DEFAULT_BOOTSTRAP_ITERS = 500
LLM_EXPORT_DEFAULT_PERMUTATION_ITERS = 2000
