from __future__ import annotations

from typing import Any

import pandas as pd


PD_MARKERS = [
    "imagined",
    "imagine",
    "fictional",
    "fictionalised",
    "fictionalized",
    "hypothetical",
    "suppose",
    "what if",
    "alternate",
    "invented",
    "scenario",
    "in a fictional",
    "in this imagined",
    "role-play",
    "creative exercise",
    "simulation",
    "mock",
]

PEDAGOGICAL_MARKERS = [
    "students will",
    "students should",
    "learning objective",
    "activity",
    "materials",
    "assessment",
    "worksheet",
    "discussion",
    "group work",
    "teacher asks",
    "teacher can",
    "exit ticket",
    "rubric",
    "lesson",
]

STYLISTIC_MARKERS = [
    "metaphor",
    "like a",
    "as if",
    "symbolizes",
    "symbolises",
    "personification",
]

FACTUAL_ASSERTION_MARKERS = [
    "was born",
    "died",
    "discovered",
    "invented",
    "founded",
    "won",
    "is located",
    "has a population",
    "served as",
    "became",
    "published",
    "graduated",
    "studied",
]


def _contains_any(text: str, markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker in text]


def _row_text(row: pd.Series) -> str:
    # Deliberately exclude prompt text: kappa-high prompts contain PD markers by design.
    parts = [str(row.get(col, "") or "") for col in ("claim", "sentence")]
    return " ".join(parts).lower()


def _status(row: pd.Series) -> str:
    value = row.get("verification_status", row.get("hallulens_status", row.get("status", "")))
    return str(value or "").strip().lower()


def _kappa(row: pd.Series) -> str:
    return str(row.get("kappa_level", "") or "").strip().upper()


def _task(row: pd.Series) -> str:
    return str(row.get("task", "") or "").strip().upper()


def _label_row(row: pd.Series) -> tuple[str, str, str, int]:
    status = _status(row)
    kappa = _kappa(row)
    task = _task(row)
    text = _row_text(row)

    pd_hits = _contains_any(text, PD_MARKERS)
    pedagogical_hits = _contains_any(text, PEDAGOGICAL_MARKERS)
    stylistic_hits = _contains_any(text, STYLISTIC_MARKERS)
    factual_hits = _contains_any(text, FACTUAL_ASSERTION_MARKERS)

    hit_groups: list[str] = []
    if pd_hits:
        hit_groups.append("pd:" + "|".join(pd_hits))
    if pedagogical_hits:
        hit_groups.append("pedagogical:" + "|".join(pedagogical_hits))
    if stylistic_hits:
        hit_groups.append("stylistic:" + "|".join(stylistic_hits))
    if factual_hits:
        hit_groups.append("factual:" + "|".join(factual_hits))
    marker_hits = "; ".join(hit_groups)

    if status == "supported":
        return "supported", "HalluLens claim is supported.", marker_hits, 0
    if status == "no_claim":
        return "no_claim", "HalluLens did not extract a verifiable claim.", marker_hits, 0
    if status != "hallucinated":
        return "unverified", "HalluLens claim is not verified.", marker_hits, 0

    if task == "LESSON_PLAN" and pedagogical_hits:
        return (
            "pedagogical_design_or_oracle_gap",
            "Lesson-plan wording describes classroom design rather than a Wikipedia-groundable fact.",
            marker_hits,
            90 + len(pedagogical_hits) + len(pd_hits),
        )

    if stylistic_hits and not factual_hits:
        return (
            "stylistic_variation_candidate",
            "Unsupported text appears stylistic or figurative rather than truth-conditional.",
            marker_hits,
            50 + len(stylistic_hits),
        )

    if kappa == "HIGH" and pd_hits:
        return (
            "PD_candidate",
            "Kappa-high prompt licenses invention and the claim/sentence contains explicit fictional or hypothetical framing.",
            marker_hits,
            100 + len(pd_hits),
        )

    if kappa == "HIGH" and factual_hits:
        return (
            "H_candidate",
            "Unsupported claim is framed as a real-world factual assertion even under kappa high.",
            marker_hits,
            10,
        )

    if kappa == "HIGH":
        return (
            "candidate_PD_or_oracle_gap",
            "Kappa-high unsupported claim without clear factual assertion markers; manual audit needed.",
            marker_hits,
            40,
        )

    return (
        "H_candidate",
        "Kappa-low or unspecified regime: unsupported claim remains a hallucination candidate.",
        marker_hits,
        10,
    )


def add_tr_relabeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        result = pd.DataFrame() if df is None else df.copy()
        for col, default in [
            ("tr_label_candidate", ""),
            ("tr_reason", ""),
            ("tr_marker_hits", ""),
            ("tr_candidate_score", 0),
        ]:
            if col not in result.columns:
                result[col] = default
        return result

    result = df.copy()
    labels = result.apply(_label_row, axis=1, result_type="expand")
    labels.columns = ["tr_label_candidate", "tr_reason", "tr_marker_hits", "tr_candidate_score"]
    for col in labels.columns:
        result[col] = labels[col]
    result["tr_candidate_score"] = pd.to_numeric(result["tr_candidate_score"], errors="coerce").fillna(0).astype(int)
    return result
