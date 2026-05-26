from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.claims_loading import load_claims_explorer_dataset
from hallulens_dashboard.tr_relabeling import add_tr_relabeling_columns


def _write_minimal_run(root: Path, run_name: str) -> Path:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "args": {
            "model": "test-model",
            "temperature": 0.5,
            "length_words": 50,
        }
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

    generation_row = {
        "prompt": "Explain who Ada Lovelace was.",
        "title": "Ada Lovelace",
        "task": "INTERVIEW",
        "creativity_level": "FACTUAL",
        "generation": "Ada Lovelace was an English mathematician and writer.",
        "length_words": 50,
    }
    (run_dir / "generation.jsonl").write_text(json.dumps(generation_row) + "\n", encoding="utf-8")

    output_df = pd.DataFrame(
        [
            {
                "prompt": generation_row["prompt"],
                "is_supported": True,
                "claim": "Ada Lovelace was an English mathematician.",
                "sentence": "Ada Lovelace was an English mathematician and writer.",
                "title": "Ada Lovelace",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "k": 4,
                "n_claims": 2,
                "overall_recall": 0.7,
                "overall_precision": 0.8,
                "overall_f1": 0.75,
            },
            {
                "prompt": generation_row["prompt"],
                "is_supported": False,
                "claim": "Ada Lovelace invented the smartphone.",
                "sentence": "Ada Lovelace invented the smartphone.",
                "title": "Ada Lovelace",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "k": 4,
                "n_claims": 2,
                "overall_recall": 0.7,
                "overall_precision": 0.8,
                "overall_f1": 0.75,
            },
            {
                "prompt": generation_row["prompt"],
                "is_supported": np.nan,
                "claim": "no claims",
                "sentence": "No additional claim.",
                "title": "Ada Lovelace",
                "precision": 0.8,
                "recall": 0.7,
                "f1": 0.75,
                "k": 4,
                "n_claims": 2,
                "overall_recall": 0.7,
                "overall_precision": 0.8,
                "overall_f1": 0.75,
            },
        ]
    )
    output_df.to_csv(run_dir / "output.csv", index=False)
    return run_dir


def test_load_claims_explorer_dataset_builds_claim_and_prompt_views(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    root.mkdir(parents=True, exist_ok=True)
    _write_minimal_run(root, "run_ok")

    claims_df, prompt_df, coverage_df = load_claims_explorer_dataset((root.as_posix(),))
    assert not claims_df.empty
    assert not prompt_df.empty
    assert not coverage_df.empty

    status_counts = claims_df["verification_status"].astype(str).value_counts().to_dict()
    assert status_counts.get("supported", 0) == 1
    assert status_counts.get("hallucinated", 0) == 1
    assert status_counts.get("no_claim", 0) == 1

    row = prompt_df.iloc[0]
    assert int(row["n_verified"]) == 2
    assert int(row["n_supported"]) == 1
    assert int(row["n_hallucinated"]) == 1
    assert float(row["support_rate_claim_level"]) == 0.5
    assert float(row["hallucination_rate_claim_level"]) == 0.5
    assert "kappa_level" in prompt_df.columns

    for col in ["tr_label_candidate", "tr_reason", "tr_marker_hits", "tr_candidate_score"]:
        assert col in claims_df.columns

    cov = coverage_df.iloc[0]
    assert str(cov["status"]) == "loaded"
    assert int(cov["n_claim_rows"]) == 3


def test_load_claims_explorer_dataset_reports_missing_files(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    root.mkdir(parents=True, exist_ok=True)
    run_missing = root / "run_missing"
    run_missing.mkdir(parents=True, exist_ok=True)
    (run_missing / "run_config.json").write_text("{}", encoding="utf-8")

    claims_df, prompt_df, coverage_df = load_claims_explorer_dataset((root.as_posix(),))
    assert claims_df.empty
    assert prompt_df.empty
    assert not coverage_df.empty
    assert str(coverage_df.iloc[0]["status"]) == "missing_files"


def test_add_tr_relabeling_columns_labels_core_cases() -> None:
    df = pd.DataFrame(
        [
            {
                "verification_status": "hallucinated",
                "kappa_level": "HIGH",
                "task": "NEWS_ARTICLE",
                "claim": "A hypothetical council creates a fictional archive.",
                "sentence": "In a hypothetical scenario, a fictional archive is created.",
            },
            {
                "verification_status": "hallucinated",
                "kappa_level": "HIGH",
                "task": "LESSON_PLAN",
                "claim": "Students will discuss the route in small groups.",
                "sentence": "Students will discuss the route in small groups during the activity.",
            },
            {
                "verification_status": "hallucinated",
                "kappa_level": "LOW",
                "task": "INTERVIEW",
                "claim": "Ada Lovelace invented the smartphone.",
                "sentence": "Ada Lovelace invented the smartphone.",
            },
            {
                "verification_status": "supported",
                "kappa_level": "HIGH",
                "task": "INTERVIEW",
                "claim": "Ada Lovelace was an English mathematician.",
                "sentence": "Ada Lovelace was an English mathematician.",
            },
            {
                "verification_status": "no_claim",
                "kappa_level": "HIGH",
                "task": "INTERVIEW",
                "claim": "no claims",
                "sentence": "",
            },
        ]
    )

    labeled = add_tr_relabeling_columns(df)

    assert labeled.loc[0, "tr_label_candidate"] == "PD_candidate"
    assert labeled.loc[1, "tr_label_candidate"] == "pedagogical_design_or_oracle_gap"
    assert labeled.loc[2, "tr_label_candidate"] == "H_candidate"
    assert labeled.loc[3, "tr_label_candidate"] == "supported"
    assert labeled.loc[4, "tr_label_candidate"] == "no_claim"
    assert int(labeled.loc[0, "tr_candidate_score"]) > int(labeled.loc[2, "tr_candidate_score"])
