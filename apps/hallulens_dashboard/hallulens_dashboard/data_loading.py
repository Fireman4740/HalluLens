from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import hash_text, normalize_text, parse_bool, safe_float, safe_int

try:
    import tiktoken

    _TOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKEN_ENCODER = None


def response_token_length(text: Any) -> int:
    content = "" if text is None else str(text)
    if not content:
        return 0
    if _TOKEN_ENCODER is not None:
        try:
            return int(len(_TOKEN_ENCODER.encode(content)))
        except Exception:
            pass
    # Fallback approximation: words + punctuation chunks.
    return int(len(re.findall(r"\w+|[^\w\s]", content, flags=re.UNICODE)))


def model_from_config(config: dict[str, Any]) -> str:
    args = config.get("args") if isinstance(config.get("args"), dict) else {}
    candidate = config.get("model_name") or args.get("model") or args.get("llm")
    if not candidate:
        return "unknown"
    model_name = str(candidate)
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    return model_name


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate_output_metrics(output_csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(output_csv_path)
    except Exception:
        return pd.DataFrame(columns=["prompt_clean", "n_claims", "support_rate", "hallucination_rate"])

    if "prompt" not in df.columns:
        return pd.DataFrame(columns=["prompt_clean", "n_claims", "support_rate", "hallucination_rate"])

    work = df.copy()
    work["prompt_clean"] = work["prompt"].map(normalize_text)

    if "claim" in work.columns:
        work["claim_norm"] = work["claim"].astype(str).str.strip().str.lower()
        work["valid_claim"] = work["claim_norm"].ne("no claims")
    else:
        work["valid_claim"] = True

    if "is_supported" in work.columns:
        work["is_supported_bool"] = work["is_supported"].map(parse_bool)
    else:
        work["is_supported_bool"] = np.nan

    work["supported_claim"] = work["valid_claim"] & work["is_supported_bool"].eq(True)
    work["n_claims_num"] = pd.to_numeric(work.get("n_claims"), errors="coerce")

    grouped = (
        work.groupby("prompt_clean", dropna=False)
        .agg(
            n_claim_rows=("valid_claim", "sum"),
            n_supported=("supported_claim", "sum"),
            n_claims_reported=("n_claims_num", "max"),
        )
        .reset_index()
    )

    grouped["n_claims"] = grouped["n_claims_reported"].where(
        grouped["n_claims_reported"].notna(),
        grouped["n_claim_rows"],
    )
    grouped["support_rate"] = np.where(
        grouped["n_claim_rows"] > 0,
        grouped["n_supported"] / grouped["n_claim_rows"],
        np.nan,
    )
    grouped["hallucination_rate"] = 1.0 - grouped["support_rate"]
    grouped["metrics_available"] = grouped["hallucination_rate"].notna()
    return grouped[
        [
            "prompt_clean",
            "n_claims",
            "n_claim_rows",
            "n_supported",
            "support_rate",
            "hallucination_rate",
            "metrics_available",
        ]
    ]


def read_generation(generation_path: Path, fallback_length_words: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        with generation_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                creativity_level = obj.get("creativity_level") or obj.get("creativity") or "NA"
                rows.append(
                    {
                        "prompt": obj.get("prompt", ""),
                        "title": obj.get("title", ""),
                        "task": obj.get("task", "NA"),
                        "creativity_level": creativity_level,
                        "generation": obj.get("generation", ""),
                        "length_words": obj.get("length_words", fallback_length_words),
                    }
                )
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["prompt"] = df["prompt"].fillna("")
    df["generation"] = df["generation"].fillna("")
    df["task"] = df["task"].fillna("NA")
    df["creativity_level"] = df["creativity_level"].fillna("NA")
    df["prompt_clean"] = df["prompt"].map(normalize_text)
    df["prompt_id"] = df["prompt_clean"].map(hash_text)
    df["prompt_length_words"] = df["prompt_clean"].str.split().str.len()
    df["response_length_words"] = df["generation"].astype(str).str.split().str.len()
    df["response_length_tokens"] = df["generation"].map(response_token_length)
    df["length_words"] = pd.to_numeric(df["length_words"], errors="coerce")

    instance_ready = df[["title", "task", "creativity_level"]].notna().all(axis=1)
    keys = (
        df["title"].astype(str) + "||" + df["task"].astype(str) + "||" + df["creativity_level"].astype(str)
    )
    df["instance_id"] = np.where(instance_ready, keys.map(hash_text), None)
    df = df.drop_duplicates(subset=["prompt_clean"], keep="first").reset_index(drop=True)
    return df


def build_run_frame(run_dir: Path, root_dir: Path) -> pd.DataFrame:
    run_config_path = run_dir / "run_config.json"
    generation_path = run_dir / "generation.jsonl"
    output_csv_path = run_dir / "output.csv"

    if not (run_config_path.exists() and generation_path.exists() and output_csv_path.exists()):
        return pd.DataFrame()

    config = read_json(run_config_path)
    args = config.get("args") if isinstance(config.get("args"), dict) else {}

    model_name = model_from_config(config)
    temperature = safe_float(args.get("temperature"))
    length_words_target = safe_int(args.get("length_words"))

    df_gen = read_generation(generation_path, fallback_length_words=length_words_target)
    if df_gen.empty:
        return pd.DataFrame()

    df_metrics = aggregate_output_metrics(output_csv_path)
    frame = df_gen.merge(df_metrics, on="prompt_clean", how="left")

    run_id = run_dir.as_posix()
    root_name = root_dir.name
    frame["run_id"] = run_id
    frame["run_dir"] = run_dir.as_posix()
    frame["root"] = root_dir.as_posix()
    frame["root_name"] = root_name
    frame["model_name"] = model_name
    frame["temperature"] = temperature

    if np.isfinite(length_words_target):
        frame["length_words"] = frame["length_words"].fillna(length_words_target)
    frame["length_words"] = pd.to_numeric(frame["length_words"], errors="coerce")

    prompt_ids = sorted(frame["prompt_id"].dropna().astype(str).unique().tolist())
    frame["prompt_set_id"] = hash_text("\n".join(prompt_ids)) if prompt_ids else None

    instance_ids = sorted(frame["instance_id"].dropna().astype(str).unique().tolist())
    frame["instance_set_id"] = hash_text("\n".join(instance_ids)) if instance_ids else None

    frame["metrics_available"] = frame["hallucination_rate"].notna()
    frame["creativity_rank"] = frame["creativity_level"].map(
        {"FACTUAL": 0, "HYBRID": 1, "VERY_CREATIVE": 2}
    )
    return frame


def load_prompt_dataset(selected_roots: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for root in selected_roots:
        root_dir = Path(root)
        if not root_dir.exists() or not root_dir.is_dir():
            continue

        for run_dir in sorted(root_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_frame = build_run_frame(run_dir=run_dir, root_dir=root_dir)
            if not run_frame.empty:
                frames.append(run_frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for col in ["task", "model_name", "creativity_level", "root_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("NA")
    return df
