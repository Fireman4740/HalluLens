from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "apps" / "hallulens_dashboard"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from hallulens_dashboard.evaluator_agreement import (  # noqa: E402
    build_component_pairwise_agreement_table,
    build_evaluator_claim_consensus,
    build_evaluator_support_tables,
    build_pairwise_agreement_table,
    build_pairwise_metric_matrix,
)


def _sample_claim_votes() -> pd.DataFrame:
    rows = [
        # evaluator A
        {"run_id": "r1", "evaluator_label": "eval_a", "claim_key": "c1", "task": "INTERVIEW", "is_supported_int": 1},
        {"run_id": "r1", "evaluator_label": "eval_a", "claim_key": "c2", "task": "INTERVIEW", "is_supported_int": 1},
        {"run_id": "r1", "evaluator_label": "eval_a", "claim_key": "c3", "task": "NEWS_ARTICLE", "is_supported_int": 0},
        {"run_id": "r1", "evaluator_label": "eval_a", "claim_key": "c4", "task": "NEWS_ARTICLE", "is_supported_int": 0},
        # evaluator B
        {"run_id": "r2", "evaluator_label": "eval_b", "claim_key": "c1", "task": "INTERVIEW", "is_supported_int": 1},
        {"run_id": "r2", "evaluator_label": "eval_b", "claim_key": "c2", "task": "INTERVIEW", "is_supported_int": 0},
        {"run_id": "r2", "evaluator_label": "eval_b", "claim_key": "c3", "task": "NEWS_ARTICLE", "is_supported_int": 0},
        {"run_id": "r2", "evaluator_label": "eval_b", "claim_key": "c4", "task": "NEWS_ARTICLE", "is_supported_int": 1},
        # evaluator C
        {"run_id": "r3", "evaluator_label": "eval_c", "claim_key": "c1", "task": "INTERVIEW", "is_supported_int": 1},
        {"run_id": "r3", "evaluator_label": "eval_c", "claim_key": "c2", "task": "INTERVIEW", "is_supported_int": 1},
        {"run_id": "r3", "evaluator_label": "eval_c", "claim_key": "c3", "task": "NEWS_ARTICLE", "is_supported_int": 1},
        {"run_id": "r3", "evaluator_label": "eval_c", "claim_key": "c4", "task": "NEWS_ARTICLE", "is_supported_int": 1},
    ]
    return pd.DataFrame(rows)


def _sample_component_claim_votes() -> pd.DataFrame:
    rows = []
    run_specs = [
        ("r1", "eval_a", "ce_a", "ab_a", "vf_a"),
        ("r2", "eval_b", "ce_b", "ab_a", "vf_a"),
        ("r3", "eval_c", "ce_a", "ab_b", "vf_a"),
        ("r4", "eval_d", "ce_a", "ab_a", "vf_b"),
    ]
    claims = [
        ("c1", "INTERVIEW", [1, 1, 1, 0]),
        ("c2", "INTERVIEW", [1, 0, 1, 1]),
        ("c3", "NEWS_ARTICLE", [0, 0, 1, 0]),
        ("c4", "NEWS_ARTICLE", [0, 1, 1, 1]),
    ]
    for run_idx, (run_id, eval_label, ce, ab, vf) in enumerate(run_specs):
        for claim_key, task, votes in claims:
            rows.append(
                {
                    "run_id": run_id,
                    "claim_key": claim_key,
                    "task": task,
                    "is_supported_int": votes[run_idx],
                    "evaluator_label": eval_label,
                    "claim_extractor_model": ce,
                    "abstain_evaluator_model": ab,
                    "verifier_model": vf,
                    "root_name": "root",
                    "generation_model": "gen_a",
                    "q_generator_model": "qg_a",
                    "temperature": 0.5,
                    "length_words": 500,
                }
            )
    return pd.DataFrame(rows)


def test_build_evaluator_claim_consensus_shapes_votes() -> None:
    claims = _sample_claim_votes()
    consensus = build_evaluator_claim_consensus(claims)
    assert not consensus.empty
    assert set(
        ["evaluator_label", "claim_key", "task", "vote_mean", "is_supported_vote", "n_rows", "n_runs"]
    ).issubset(consensus.columns)
    assert consensus["claim_key"].nunique() == 4
    assert set(consensus["evaluator_label"].unique().tolist()) == {"eval_a", "eval_b", "eval_c"}
    assert ((consensus["is_supported_vote"] == 0) | (consensus["is_supported_vote"] == 1)).all()


def test_build_pairwise_agreement_table_computes_overlap_and_kappa() -> None:
    consensus = build_evaluator_claim_consensus(_sample_claim_votes())
    pairwise = build_pairwise_agreement_table(consensus, min_overlap=1, by_task=False)
    assert not pairwise.empty
    assert set(
        ["evaluator_a", "evaluator_b", "pair_label", "n_overlap", "agreement_rate", "kappa"]
    ).issubset(pairwise.columns)
    assert (pairwise["n_overlap"] >= 1).all()
    assert ((pairwise["agreement_rate"] >= 0.0) & (pairwise["agreement_rate"] <= 1.0)).all()
    finite_kappa = pairwise["kappa"].dropna()
    if not finite_kappa.empty:
        assert ((finite_kappa >= -1.0) & (finite_kappa <= 1.0)).all()


def test_build_pairwise_metric_matrix_is_square_and_symmetric() -> None:
    consensus = build_evaluator_claim_consensus(_sample_claim_votes())
    pairwise = build_pairwise_agreement_table(consensus, min_overlap=1, by_task=False)
    matrix = build_pairwise_metric_matrix(pairwise, metric_col="agreement_rate")
    assert not matrix.empty
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.index.tolist() == matrix.columns.tolist()
    vals = matrix.to_numpy(dtype=float)
    assert np.allclose(vals, vals.T, equal_nan=True)
    assert np.allclose(np.diag(vals), np.ones(len(matrix)), equal_nan=True)


def test_build_evaluator_support_tables_returns_overall_and_task_views() -> None:
    consensus = build_evaluator_claim_consensus(_sample_claim_votes())
    overall, by_task = build_evaluator_support_tables(consensus)
    assert not overall.empty
    assert not by_task.empty
    assert set(["evaluator_label", "n_claims", "support_rate"]).issubset(overall.columns)
    assert set(["evaluator_label", "task", "n_claims", "support_rate"]).issubset(by_task.columns)
    assert ((overall["support_rate"] >= 0.0) & (overall["support_rate"] <= 1.0)).all()


def test_build_component_pairwise_agreement_table_supports_pipeline_components() -> None:
    claims = _sample_component_claim_votes()
    ce_overall = build_component_pairwise_agreement_table(
        claims,
        component_col="claim_extractor_model",
        min_overlap=1,
        by_task=False,
    )
    assert not ce_overall.empty
    assert set(
        [
            "component",
            "model_a",
            "model_b",
            "pair_label",
            "task",
            "n_overlap",
            "agreement_rate",
            "kappa",
            "n_contexts",
        ]
    ).issubset(ce_overall.columns)
    assert set(ce_overall["task"].unique().tolist()) == {"ALL"}

    vf_by_task = build_component_pairwise_agreement_table(
        claims,
        component_col="verifier_model",
        min_overlap=1,
        by_task=True,
    )
    assert not vf_by_task.empty
    assert {"INTERVIEW", "NEWS_ARTICLE"}.issubset(set(vf_by_task["task"].unique().tolist()))
