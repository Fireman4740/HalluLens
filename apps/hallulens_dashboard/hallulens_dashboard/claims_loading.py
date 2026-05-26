from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import build_run_frame, model_from_config, read_json
from .tr_relabeling import add_tr_relabeling_columns
from .utils import normalize_text, parse_bool

CLAIM_STATUS_ORDER = ["supported", "hallucinated", "no_claim", "unverified"]


def _status_from_row(claim: Any, is_supported_bool: Any) -> str:
    claim_text = "" if claim is None else str(claim).strip().lower()
    if claim_text in {"", "no claims", "no claim"}:
        return "no_claim"
    if is_supported_bool is True:
        return "supported"
    if is_supported_bool is False:
        return "hallucinated"
    return "unverified"


def _build_claim_rows_for_run(run_dir: Path, root_dir: Path) -> pd.DataFrame:
    prompt_frame = build_run_frame(run_dir=run_dir, root_dir=root_dir)
    if prompt_frame.empty:
        return pd.DataFrame()

    output_path = run_dir / "output.csv"
    if not output_path.exists():
        return pd.DataFrame()

    try:
        raw = pd.read_csv(output_path)
    except Exception:
        return pd.DataFrame()
    if "prompt" not in raw.columns:
        return pd.DataFrame()

    work = raw.copy()
    work["prompt"] = work["prompt"].fillna("").astype(str)
    work["prompt_clean"] = work["prompt"].map(normalize_text)
    work["claim"] = work.get("claim", "").fillna("").astype(str)
    work["sentence"] = work.get("sentence", "").fillna("").astype(str)
    if "is_supported" in work.columns:
        work["is_supported_bool"] = work["is_supported"].map(parse_bool)
    else:
        work["is_supported_bool"] = np.nan
    work["verification_status"] = [
        _status_from_row(c, s) for c, s in zip(work["claim"], work["is_supported_bool"])
    ]

    for col in [
        "precision",
        "recall",
        "f1",
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "n_claims",
        "k",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    prompt_meta_cols = [
        c
        for c in [
            "run_id",
            "run_dir",
            "root_name",
            "model_name",
            "temperature",
            "length_words",
            "task",
            "creativity_level",
            "kappa_level",
            "title",
            "prompt",
            "prompt_clean",
            "prompt_id",
            "generation",
            "response_length_words",
            "response_length_tokens",
            "support_rate",
            "hallucination_rate",
            "n_claim_rows",
            "n_supported",
            "metrics_available",
        ]
        if c in prompt_frame.columns
    ]
    prompt_meta = prompt_frame[prompt_meta_cols].drop_duplicates(
        subset=["run_dir", "prompt_clean"], keep="first"
    )

    work["run_dir"] = run_dir.as_posix()
    merged = work.merge(prompt_meta, on=["run_dir", "prompt_clean"], how="left", suffixes=("", "_prompt"))

    merged["run_id"] = merged.get("run_id", merged["run_dir"]).fillna(merged["run_dir"])
    merged["root_name"] = merged.get("root_name", root_dir.name).fillna(root_dir.name)
    merged["model_name"] = merged.get("model_name", "unknown").fillna("unknown")
    merged["task"] = merged.get("task", "NA").fillna("NA")
    merged["creativity_level"] = merged.get("creativity_level", "NA").fillna("NA")
    merged["title"] = merged.get("title", "").fillna("")
    merged["length_words"] = pd.to_numeric(merged.get("length_words"), errors="coerce")
    merged["temperature"] = pd.to_numeric(merged.get("temperature"), errors="coerce")
    merged["response_length_words"] = pd.to_numeric(
        merged.get("response_length_words"), errors="coerce"
    )
    merged["response_length_tokens"] = pd.to_numeric(
        merged.get("response_length_tokens"), errors="coerce"
    )

    merged = merged.reset_index(drop=True)
    merged["claim_row_index"] = np.arange(len(merged), dtype=int)
    merged["claim_row_id"] = merged["run_dir"].astype(str) + "::" + merged["claim_row_index"].astype(str)
    return merged


def build_prompt_summary_from_claims(claims_df: pd.DataFrame) -> pd.DataFrame:
    if claims_df is None or claims_df.empty:
        return pd.DataFrame()

    work = claims_df.copy()
    status = work["verification_status"].fillna("unverified").astype(str)
    work["is_supported_row"] = status.eq("supported").astype(int)
    work["is_hallucinated_row"] = status.eq("hallucinated").astype(int)
    work["is_no_claim_row"] = status.eq("no_claim").astype(int)
    work["is_unverified_row"] = status.eq("unverified").astype(int)
    work["is_verified_row"] = status.isin(["supported", "hallucinated"]).astype(int)

    group_cols = [
        c
        for c in [
            "run_id",
            "run_dir",
            "root_name",
            "model_name",
            "temperature",
            "length_words",
            "task",
            "creativity_level",
            "kappa_level",
            "title",
            "prompt",
            "prompt_clean",
            "prompt_id",
            "generation",
            "response_length_words",
            "response_length_tokens",
        ]
        if c in work.columns
    ]
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            n_rows=("claim_row_id", "count"),
            n_supported=("is_supported_row", "sum"),
            n_hallucinated=("is_hallucinated_row", "sum"),
            n_no_claim=("is_no_claim_row", "sum"),
            n_unverified=("is_unverified_row", "sum"),
            n_verified=("is_verified_row", "sum"),
            n_claims_reported_max=("n_claims", "max"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            overall_precision_mean=("overall_precision", "mean"),
            overall_recall_mean=("overall_recall", "mean"),
            overall_f1_mean=("overall_f1", "mean"),
        )
        .reset_index()
    )
    grouped["support_rate_claim_level"] = np.where(
        grouped["n_verified"] > 0,
        grouped["n_supported"] / grouped["n_verified"],
        np.nan,
    )
    grouped["hallucination_rate_claim_level"] = 1.0 - grouped["support_rate_claim_level"]
    grouped["claim_density_per_100_words"] = np.where(
        grouped["response_length_words"] > 0,
        100.0 * grouped["n_verified"] / grouped["response_length_words"],
        np.nan,
    )
    return grouped.sort_values(
        by=["hallucination_rate_claim_level", "n_verified"], ascending=[False, False]
    ).reset_index(drop=True)


def _build_run_coverage_row(run_dir: Path, root_dir: Path) -> dict[str, Any]:
    run_config = run_dir / "run_config.json"
    generation = run_dir / "generation.jsonl"
    output = run_dir / "output.csv"
    if not run_dir.is_dir():
        return {}

    model_name = "unknown"
    if run_config.exists():
        cfg = read_json(run_config)
        model_name = model_from_config(cfg)

    missing = []
    if not run_config.exists():
        missing.append("run_config.json")
    if not generation.exists():
        missing.append("generation.jsonl")
    if not output.exists():
        missing.append("output.csv")
    status = "ready" if not missing else "missing_files"
    return {
        "run_dir": run_dir.as_posix(),
        "run_name": run_dir.name,
        "root_name": root_dir.name,
        "model_name": model_name,
        "status": status,
        "missing_files": ", ".join(missing),
    }


def load_claims_explorer_dataset(
    selected_roots: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []

    for root in selected_roots:
        root_dir = Path(root)
        if not root_dir.exists() or not root_dir.is_dir():
            continue
        for run_dir in sorted(root_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            coverage = _build_run_coverage_row(run_dir, root_dir)
            if coverage:
                coverage_rows.append(coverage)
            run_df = _build_claim_rows_for_run(run_dir=run_dir, root_dir=root_dir)
            if not run_df.empty:
                frames.append(run_df)

    coverage_df = pd.DataFrame(coverage_rows)
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), coverage_df

    claims_df = pd.concat(frames, ignore_index=True)
    claims_df["verification_status"] = pd.Categorical(
        claims_df["verification_status"],
        categories=CLAIM_STATUS_ORDER,
        ordered=True,
    )
    claims_df["task"] = claims_df["task"].fillna("NA")
    claims_df["creativity_level"] = claims_df["creativity_level"].fillna("NA")
    claims_df["kappa_level"] = claims_df["kappa_level"].fillna("NA")
    claims_df["model_name"] = claims_df["model_name"].fillna("unknown")
    claims_df["root_name"] = claims_df["root_name"].fillna("NA")
    claims_df = add_tr_relabeling_columns(claims_df)

    prompt_summary_df = build_prompt_summary_from_claims(claims_df)

    if not coverage_df.empty:
        run_counts = (
            claims_df.groupby("run_dir", dropna=False)
            .agg(
                n_claim_rows=("claim_row_id", "count"),
                n_prompts=("prompt_clean", "nunique"),
            )
            .reset_index()
        )
        coverage_df = coverage_df.merge(run_counts, on="run_dir", how="left")
        coverage_df["status"] = np.where(
            coverage_df["n_claim_rows"].fillna(0) > 0,
            "loaded",
            coverage_df["status"],
        )
        coverage_df["n_claim_rows"] = coverage_df["n_claim_rows"].fillna(0).astype(int)
        coverage_df["n_prompts"] = coverage_df["n_prompts"].fillna(0).astype(int)

    return claims_df.reset_index(drop=True), prompt_summary_df.reset_index(drop=True), coverage_df.reset_index(drop=True)
