from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.creativity_loading import load_creativity_dataset, parse_creativity_jsonl


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_parse_creativity_jsonl_supports_ttcw_ttwt_and_invalid_line(tmp_path: Path) -> None:
    path = tmp_path / "creativity.jsonl"
    valid_ttcw = {
        "prompt": "Prompt A",
        "response": "Response A",
        "creativity": {
            "mode": "creativityprism_repo",
            "metrics": {
                "ttct": {"scores": {"fluency": 4, "flexibility": 3, "originality": 5, "elaboration": 2, "overall": 3.5}},
                "ttcw": {
                    "questions_scored": {
                        "Coherence - Narrative Coherence": {"score": 4},
                        "Elaboration - Descriptive Detail": {"score": 3},
                        "Ending - Narrative Ending": {"score": 2},
                        "Flexibility - Emotional Flexibility": {"score": 5},
                    }
                },
            },
        },
        "errors": [],
    }
    valid_ttwt = {
        "prompt": "Prompt B",
        "response": "Response B",
        "creativity": {
            "mode": "creativityprism_repo",
            "metrics": {
                "ttct": {"scores": {"overall": 4}},
                "ttwt": {
                    "questions_scored": {
                        "Coherence - Narrative Coherence": {"score": 1},
                        "Elaboration - Descriptive Detail": {"score": 2},
                        "Ending - Narrative Ending": {"score": 3},
                        "Flexibility - Emotional Flexibility": {"score": 4},
                    }
                },
            },
        },
        "errors": [],
    }

    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(valid_ttcw) + "\n")
        handle.write(json.dumps(valid_ttwt) + "\n")
        handle.write("{not json}\n")

    parsed = parse_creativity_jsonl(path)
    assert len(parsed) == 3

    row_a = parsed[parsed["prompt_clean"] == "Prompt A"].iloc[0]
    assert bool(row_a["creativity_parse_ok"]) is True
    assert row_a["ttct_overall"] == 3.5
    assert row_a["ttcw_overall"] == 3.5
    assert bool(row_a["scored_row"]) is True

    row_b = parsed[parsed["prompt_clean"] == "Prompt B"].iloc[0]
    assert row_b["ttct_overall"] == 4.0
    assert row_b["ttcw_overall"] == 2.5  # alias ttwt -> ttcw

    invalid = parsed[parsed["creativity_parse_ok"] == False].iloc[0]  # noqa: E712
    assert pd.isna(invalid["prompt_clean"])
    assert invalid["creativity_error_count"] == 1
    assert bool(invalid["scored_row"]) is False


def test_load_creativity_dataset_join_coverage_and_strict_mode(tmp_path: Path) -> None:
    root = tmp_path / "output_root"
    run = root / "run_a"
    run.mkdir(parents=True, exist_ok=True)

    (run / "run_config.json").write_text(
        json.dumps({"model_name": "grok-3-mini", "args": {"temperature": 0.5, "length_words": 500}}),
        encoding="utf-8",
    )

    _write_jsonl(
        run / "generation.jsonl",
        [
            {
                "prompt": "Prompt A",
                "title": "Title A",
                "task": "INTERVIEW",
                "creativity_level": "FACTUAL",
                "generation": "Answer A",
                "length_words": 500,
            },
            {
                "prompt": "Prompt B",
                "title": "Title B",
                "task": "NEWS_ARTICLE",
                "creativity_level": "HYBRID",
                "generation": "Answer B",
                "length_words": 500,
            },
        ],
    )

    pd.DataFrame(
        [
            {"prompt": "Prompt A", "claim": "claim A", "is_supported": True, "n_claims": 1},
            {"prompt": "Prompt B", "claim": "claim B", "is_supported": False, "n_claims": 1},
        ]
    ).to_csv(run / "output.csv", index=False)

    _write_jsonl(
        run / "creativity.jsonl",
        [
            {
                "prompt": "Prompt A",
                "response": "Answer A",
                "creativity": {
                    "mode": "creativityprism_repo",
                    "metrics": {
                        "ttct": {"scores": {"overall": 3.0}},
                        "ttcw": {
                            "questions_scored": {
                                "Coherence - Narrative Coherence": {"score": 4},
                                "Elaboration - Descriptive Detail": {"score": 4},
                                "Ending - Narrative Ending": {"score": 4},
                                "Flexibility - Emotional Flexibility": {"score": 4},
                            }
                        },
                    },
                },
                "errors": [],
            }
        ],
    )

    merged_all, coverage = load_creativity_dataset((root.as_posix(),), strict_mode=False)
    assert len(merged_all) == 2
    assert "ttct_overall" in merged_all.columns
    assert "ttcw_overall" in merged_all.columns

    cov = coverage.iloc[0]
    assert cov["n_generation"] == 2
    assert cov["n_creativity_rows"] == 1
    assert cov["n_matched"] == 1
    assert cov["status"] == "partial"
    assert cov["coverage_pct"] == 50.0

    merged_strict, _ = load_creativity_dataset((root.as_posix(),), strict_mode=True)
    assert len(merged_strict) == 1
    assert bool(merged_strict.iloc[0]["scored_row"]) is True
    assert bool(merged_strict.iloc[0]["metrics_available"]) is True

    merged_complete_only, _ = load_creativity_dataset(
        (root.as_posix(),),
        strict_mode=True,
        exclude_incomplete_runs=True,
    )
    assert merged_complete_only.empty
