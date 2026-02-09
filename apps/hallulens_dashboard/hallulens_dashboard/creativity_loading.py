from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import load_prompt_dataset
from .utils import normalize_text

TTCT_COLS = [
    "ttct_fluency",
    "ttct_flexibility",
    "ttct_originality",
    "ttct_elaboration",
    "ttct_overall",
]

TTCW_COLS = [
    "ttcw_coherence",
    "ttcw_descriptive_detail",
    "ttcw_ending",
    "ttcw_emotional_flexibility",
    "ttcw_overall",
]

CREATIVITY_COLS = TTCT_COLS + TTCW_COLS + [
    "creativity_composite",
    "creativity_mode",
    "creativity_parse_ok",
    "creativity_error_count",
    "scored_row",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _error_count(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0
    return 1


def _extract_ttcw_question_scores(questions_scored: dict[str, Any]) -> dict[str, float]:
    mapped = {
        "ttcw_coherence": np.nan,
        "ttcw_descriptive_detail": np.nan,
        "ttcw_ending": np.nan,
        "ttcw_emotional_flexibility": np.nan,
    }
    for question_label, payload in questions_scored.items():
        key = str(question_label).strip().lower()
        score = _safe_float((payload or {}).get("score"))
        if "coherence" in key:
            mapped["ttcw_coherence"] = score
        elif "descriptive detail" in key:
            mapped["ttcw_descriptive_detail"] = score
        elif "ending" in key:
            mapped["ttcw_ending"] = score
        elif "emotional flexibility" in key:
            mapped["ttcw_emotional_flexibility"] = score
    return mapped


def _mean_skipna(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan
    return float(valid.mean())


def parse_creativity_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return pd.DataFrame(columns=["prompt_clean"] + CREATIVITY_COLS)

    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        return pd.DataFrame(columns=["prompt_clean"] + CREATIVITY_COLS)

    if not raw_text.strip():
        return pd.DataFrame(columns=["prompt_clean"] + CREATIVITY_COLS)

    for line_no, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
            parse_ok = True
        except Exception:
            rows.append(
                {
                    "prompt_clean": None,
                    "creativity_mode": None,
                    "ttct_fluency": np.nan,
                    "ttct_flexibility": np.nan,
                    "ttct_originality": np.nan,
                    "ttct_elaboration": np.nan,
                    "ttct_overall": np.nan,
                    "ttcw_coherence": np.nan,
                    "ttcw_descriptive_detail": np.nan,
                    "ttcw_ending": np.nan,
                    "ttcw_emotional_flexibility": np.nan,
                    "ttcw_overall": np.nan,
                    "creativity_composite": np.nan,
                    "creativity_parse_ok": False,
                    "creativity_error_count": 1,
                    "scored_row": False,
                    "line_no": line_no,
                }
            )
            continue

        creativity = obj.get("creativity") if isinstance(obj.get("creativity"), dict) else {}
        metrics = creativity.get("metrics") if isinstance(creativity.get("metrics"), dict) else {}
        ttct = metrics.get("ttct") if isinstance(metrics.get("ttct"), dict) else {}
        ttct_scores = ttct.get("scores") if isinstance(ttct.get("scores"), dict) else {}

        # Alias support: if ttcw is missing, fallback to ttwt.
        ttcw_root = metrics.get("ttcw")
        if not isinstance(ttcw_root, dict):
            ttcw_root = metrics.get("ttwt")
        if not isinstance(ttcw_root, dict):
            ttcw_root = {}
        questions_scored = (
            ttcw_root.get("questions_scored")
            if isinstance(ttcw_root.get("questions_scored"), dict)
            else {}
        )

        ttcw_scores = _extract_ttcw_question_scores(questions_scored)
        ttct_overall = _safe_float(ttct_scores.get("overall"))
        ttcw_overall = _mean_skipna(
            [
                ttcw_scores["ttcw_coherence"],
                ttcw_scores["ttcw_descriptive_detail"],
                ttcw_scores["ttcw_ending"],
                ttcw_scores["ttcw_emotional_flexibility"],
            ]
        )
        creativity_composite = _mean_skipna([ttct_overall, ttcw_overall])

        rows.append(
            {
                "prompt_clean": normalize_text(obj.get("prompt")),
                "creativity_mode": creativity.get("mode"),
                "ttct_fluency": _safe_float(ttct_scores.get("fluency")),
                "ttct_flexibility": _safe_float(ttct_scores.get("flexibility")),
                "ttct_originality": _safe_float(ttct_scores.get("originality")),
                "ttct_elaboration": _safe_float(ttct_scores.get("elaboration")),
                "ttct_overall": ttct_overall,
                "ttcw_coherence": ttcw_scores["ttcw_coherence"],
                "ttcw_descriptive_detail": ttcw_scores["ttcw_descriptive_detail"],
                "ttcw_ending": ttcw_scores["ttcw_ending"],
                "ttcw_emotional_flexibility": ttcw_scores["ttcw_emotional_flexibility"],
                "ttcw_overall": ttcw_overall,
                "creativity_composite": creativity_composite,
                "creativity_parse_ok": parse_ok,
                "creativity_error_count": _error_count(obj.get("errors")),
                "scored_row": bool(np.isfinite(ttct_overall) or np.isfinite(ttcw_overall)),
                "line_no": line_no,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["prompt_clean"] + CREATIVITY_COLS)

    out = pd.DataFrame(rows)
    for col in TTCT_COLS + TTCW_COLS + ["creativity_composite"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["creativity_parse_ok"] = out["creativity_parse_ok"].fillna(False).astype(bool)
    out["creativity_error_count"] = pd.to_numeric(
        out["creativity_error_count"], errors="coerce"
    ).fillna(0).astype(int)
    out["scored_row"] = out["scored_row"].fillna(False).astype(bool)
    return out


def _coverage_status(n_generation: int, n_creativity_rows: int, n_matched: int) -> str:
    if n_generation <= 0 or n_creativity_rows <= 0 or n_matched <= 0:
        return "missing"
    if n_matched >= n_generation:
        return "complete"
    return "partial"


def load_creativity_dataset(
    selected_roots: tuple[str, ...],
    strict_mode: bool = True,
    exclude_incomplete_runs: bool = False,
    base_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_df is None:
        base_df = load_prompt_dataset(selected_roots)
    else:
        base_df = base_df.copy()
    if base_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    creativity_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    run_cols = ["run_dir", "run_id", "root_name", "model_name", "temperature", "length_words"]
    run_meta = (
        base_df[run_cols]
        .drop_duplicates(subset=["run_dir"], keep="first")
        .reset_index(drop=True)
    )

    for _, run in run_meta.iterrows():
        run_dir = Path(str(run["run_dir"]))
        cpath = run_dir / "creativity.jsonl"
        run_subset = base_df[base_df["run_dir"] == str(run["run_dir"])]
        n_generation = int(run_subset["prompt_clean"].nunique())

        parsed = parse_creativity_jsonl(cpath)
        n_creativity_rows = int(len(parsed))

        matched_prompts = 0
        if not parsed.empty:
            parsed = parsed.copy()
            parsed["run_dir"] = str(run["run_dir"])
            creativity_frames.append(parsed)
            valid = parsed[
                parsed["creativity_parse_ok"]
                & parsed["prompt_clean"].notna()
                & parsed["prompt_clean"].ne("")
            ]
            if not valid.empty:
                matched_prompts = int(
                    run_subset["prompt_clean"]
                    .isin(valid["prompt_clean"].drop_duplicates())
                    .sum()
                )

        coverage_pct = float(100.0 * matched_prompts / n_generation) if n_generation > 0 else np.nan
        coverage_rows.append(
            {
                "run_dir": str(run["run_dir"]),
                "run_id": str(run["run_id"]),
                "root_name": run["root_name"],
                "model_name": run["model_name"],
                "temperature": run["temperature"],
                "length_words": run["length_words"],
                "n_generation": n_generation,
                "n_creativity_rows": n_creativity_rows,
                "n_matched": matched_prompts,
                "coverage_pct": coverage_pct,
                "status": _coverage_status(n_generation, n_creativity_rows, matched_prompts),
            }
        )

    coverage_df = pd.DataFrame(coverage_rows)
    if not creativity_frames:
        merged = base_df.copy()
        for col in CREATIVITY_COLS:
            if col not in merged.columns:
                merged[col] = np.nan
        merged["creativity_parse_ok"] = False
        merged["creativity_error_count"] = 0
        merged["scored_row"] = False
    else:
        creativity_df = pd.concat(creativity_frames, ignore_index=True)
        creativity_valid = creativity_df[creativity_df["prompt_clean"].notna()].copy()
        creativity_valid = creativity_valid.sort_values(
            by=["creativity_parse_ok", "scored_row", "line_no"],
            ascending=[False, False, True],
        )
        creativity_valid = creativity_valid.drop_duplicates(
            subset=["run_dir", "prompt_clean"],
            keep="first",
        )

        join_cols = [
            "run_dir",
            "prompt_clean",
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
            "creativity_mode",
            "creativity_parse_ok",
            "creativity_error_count",
            "scored_row",
        ]
        merged = base_df.merge(
            creativity_valid[join_cols],
            on=["run_dir", "prompt_clean"],
            how="left",
        )
        merged["creativity_parse_ok"] = (
            merged["creativity_parse_ok"].astype("boolean").fillna(False).astype(bool)
        )
        merged["creativity_error_count"] = pd.to_numeric(
            merged["creativity_error_count"], errors="coerce"
        ).fillna(0).astype(int)
        merged["scored_row"] = merged["scored_row"].astype("boolean").fillna(False).astype(bool)
        merged["creativity_mode"] = merged["creativity_mode"].fillna("NA")

    if exclude_incomplete_runs and not coverage_df.empty and not merged.empty:
        complete_run_dirs = coverage_df.loc[coverage_df["status"] == "complete", "run_dir"].astype(str).unique().tolist()
        merged = merged[merged["run_dir"].astype(str).isin(complete_run_dirs)].copy()

    if strict_mode:
        merged = merged[merged["metrics_available"] & merged["scored_row"]].copy()

    return merged.reset_index(drop=True), coverage_df.reset_index(drop=True)
