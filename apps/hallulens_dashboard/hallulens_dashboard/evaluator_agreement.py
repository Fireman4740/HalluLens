from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loading import model_from_config, read_json
from .utils import hash_text, normalize_text, parse_bool, safe_float, safe_int


def _short_model_name(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return "unknown"
    if "/" in text:
        text = text.split("/")[-1]
    return text


def _mode_or_first(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if s.empty:
        return "NA"
    mode = s.mode(dropna=True)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(s.iloc[0])


def _cohen_kappa_binary(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return np.nan
    aa = np.asarray(a, dtype=int)
    bb = np.asarray(b, dtype=int)
    p0 = float((aa == bb).mean())
    pa = float(aa.mean())
    pb = float(bb.mean())
    pe = float(pa * pb + (1.0 - pa) * (1.0 - pb))
    if np.isclose(1.0 - pe, 0.0):
        if np.isclose(p0, 1.0):
            return 1.0
        return np.nan
    return float((p0 - pe) / (1.0 - pe))


def load_evaluation_test_claims_dataset(
    selected_roots: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    claim_frames: list[pd.DataFrame] = []
    run_rows: list[dict[str, Any]] = []

    for root in selected_roots:
        root_dir = Path(root)
        if not root_dir.exists() or not root_dir.is_dir():
            continue

        for run_dir in sorted(root_dir.iterdir()):
            if not run_dir.is_dir() or "evaluation_test" not in run_dir.name:
                continue

            run_config_path = run_dir / "run_config.json"
            output_path = run_dir / "output.csv"
            if not run_config_path.exists() or not output_path.exists():
                continue

            config = read_json(run_config_path)
            args = config.get("args") if isinstance(config.get("args"), dict) else {}
            generation_model = model_from_config(config)
            temperature = safe_float(args.get("temperature"))
            length_words = safe_int(args.get("length_words"))
            q_generator_model = _short_model_name(args.get("q_generator"))
            claim_extractor_model = _short_model_name(args.get("claim_extractor"))
            abstain_evaluator_model = _short_model_name(args.get("abstain_evaluator"))
            verifier_model = _short_model_name(args.get("verifier"))
            evaluator_signature = (
                f"a:{abstain_evaluator_model} | c:{claim_extractor_model} | v:{verifier_model}"
            )
            if (
                abstain_evaluator_model == claim_extractor_model
                and claim_extractor_model == verifier_model
            ):
                evaluator_label = verifier_model
            else:
                evaluator_label = evaluator_signature

            try:
                raw = pd.read_csv(output_path)
            except Exception:
                continue

            needed = ["prompt", "claim", "sentence", "title", "task", "is_supported"]
            for col in needed:
                if col not in raw.columns:
                    raw[col] = np.nan

            work = raw[needed].copy()
            work["is_supported_bool"] = work["is_supported"].map(parse_bool)
            work = work[work["is_supported_bool"].notna()].copy()
            if work.empty:
                continue

            work["prompt_norm"] = work["prompt"].map(normalize_text)
            work["claim_norm"] = work["claim"].map(normalize_text)
            work["sentence_norm"] = work["sentence"].map(normalize_text)
            work["title_norm"] = work["title"].map(normalize_text)
            work["task_norm"] = work["task"].map(normalize_text)
            work = work[
                ~work["claim_norm"].str.lower().isin({"", "no claims", "no claim"})
            ].copy()
            if work.empty:
                continue

            key_parts = (
                work["prompt_norm"]
                + "||"
                + work["claim_norm"]
                + "||"
                + work["sentence_norm"]
                + "||"
                + work["title_norm"]
                + "||"
                + work["task_norm"]
            )
            work["claim_key"] = [hash_text(v) for v in key_parts.tolist()]
            work["is_supported_int"] = work["is_supported_bool"].astype(int)

            # generation_key identifies the source sentence uniquely,
            # independently of which claims were extracted from it.
            gen_key_parts = (
                work["prompt_norm"]
                + "||"
                + work["sentence_norm"]
                + "||"
                + work["title_norm"]
                + "||"
                + work["task_norm"]
            )
            work["generation_key"] = [hash_text(v) for v in gen_key_parts.tolist()]

            run_id = run_dir.as_posix()
            frame = pd.DataFrame(
                {
                    "run_id": run_id,
                    "run_name": run_dir.name,
                    "run_dir": run_dir.as_posix(),
                    "root_name": root_dir.name,
                    "generation_model": generation_model,
                    "temperature": temperature,
                    "length_words": length_words,
                    "q_generator_model": q_generator_model,
                    "claim_extractor_model": claim_extractor_model,
                    "abstain_evaluator_model": abstain_evaluator_model,
                    "verifier_model": verifier_model,
                    "evaluator_signature": evaluator_signature,
                    "evaluator_label": evaluator_label,
                    "task": work["task_norm"].replace("", "NA"),
                    "title": work["title_norm"].replace("", "NA"),
                    "prompt_norm": work["prompt_norm"],
                    "claim_norm": work["claim_norm"],
                    "sentence_norm": work["sentence_norm"],
                    "claim_key": work["claim_key"],
                    "generation_key": work["generation_key"],
                    "is_supported_int": work["is_supported_int"],
                }
            )
            frame = frame.drop_duplicates(subset=["run_id", "claim_key"], keep="first")
            if frame.empty:
                continue

            claim_frames.append(frame)
            run_rows.append(
                {
                    "run_id": run_id,
                    "run_name": run_dir.name,
                    "root_name": root_dir.name,
                    "generation_model": generation_model,
                    "temperature": temperature,
                    "length_words": length_words,
                    "q_generator_model": q_generator_model,
                    "claim_extractor_model": claim_extractor_model,
                    "abstain_evaluator_model": abstain_evaluator_model,
                    "verifier_model": verifier_model,
                    "evaluator_signature": evaluator_signature,
                    "evaluator_label": evaluator_label,
                    "n_claim_rows": int(len(frame)),
                    "support_rate": float(frame["is_supported_int"].mean()),
                }
            )

    if not claim_frames:
        return pd.DataFrame(), pd.DataFrame()

    claims_df = pd.concat(claim_frames, ignore_index=True)
    runs_df = pd.DataFrame(run_rows).drop_duplicates(subset=["run_id"]).reset_index(drop=True)
    return claims_df, runs_df


def build_evaluator_claim_consensus(claims_df: pd.DataFrame) -> pd.DataFrame:
    out_cols = [
        "evaluator_label",
        "claim_key",
        "task",
        "vote_mean",
        "is_supported_vote",
        "n_rows",
        "n_runs",
        "intra_disagreement",
    ]
    if claims_df.empty:
        return pd.DataFrame(columns=out_cols)

    grouped = (
        claims_df.groupby(["evaluator_label", "claim_key"], dropna=False)
        .agg(
            vote_mean=("is_supported_int", "mean"),
            n_rows=("is_supported_int", "size"),
            n_runs=("run_id", "nunique"),
            task=("task", _mode_or_first),
        )
        .reset_index()
    )
    grouped["is_supported_vote"] = (grouped["vote_mean"] >= 0.5).astype(int)
    grouped["intra_disagreement"] = ((grouped["vote_mean"] > 0.0) & (grouped["vote_mean"] < 1.0)).astype(int)
    return grouped[out_cols]


def build_pairwise_agreement_table(
    consensus_df: pd.DataFrame,
    min_overlap: int = 20,
    by_task: bool = False,
) -> pd.DataFrame:
    out_cols = [
        "evaluator_a",
        "evaluator_b",
        "pair_label",
        "task",
        "n_overlap",
        "agreement_rate",
        "disagreement_rate",
        "kappa",
        "support_rate_a",
        "support_rate_b",
    ]
    if consensus_df.empty:
        return pd.DataFrame(columns=out_cols)

    labels = sorted(consensus_df["evaluator_label"].dropna().astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    for i, eva in enumerate(labels):
        left = consensus_df[consensus_df["evaluator_label"].astype(str) == eva][
            ["claim_key", "task", "is_supported_vote"]
        ].copy()
        left = left.rename(columns={"task": "task_a", "is_supported_vote": "vote_a"})
        for evb in labels[i + 1 :]:
            right = consensus_df[consensus_df["evaluator_label"].astype(str) == evb][
                ["claim_key", "task", "is_supported_vote"]
            ].copy()
            right = right.rename(columns={"task": "task_b", "is_supported_vote": "vote_b"})
            merged = left.merge(right, on="claim_key", how="inner")
            if merged.empty:
                continue
            merged["task"] = np.where(
                merged["task_a"].astype(str) == merged["task_b"].astype(str),
                merged["task_a"].astype(str),
                merged["task_a"].astype(str),
            )
            groups = [("ALL", merged)] if not by_task else list(merged.groupby("task", dropna=False))
            for task_name, grp in groups:
                n = int(len(grp))
                if n < min_overlap:
                    continue
                agreement = float((grp["vote_a"].to_numpy(dtype=int) == grp["vote_b"].to_numpy(dtype=int)).mean())
                kappa = _cohen_kappa_binary(
                    grp["vote_a"].to_numpy(dtype=int),
                    grp["vote_b"].to_numpy(dtype=int),
                )
                rows.append(
                    {
                        "evaluator_a": eva,
                        "evaluator_b": evb,
                        "pair_label": f"{eva} vs {evb}",
                        "task": str(task_name),
                        "n_overlap": n,
                        "agreement_rate": agreement,
                        "disagreement_rate": 1.0 - agreement,
                        "kappa": kappa,
                        "support_rate_a": float(grp["vote_a"].mean()),
                        "support_rate_b": float(grp["vote_b"].mean()),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)
    sort_cols = ["task", "agreement_rate", "n_overlap"] if by_task else ["agreement_rate", "n_overlap"]
    out = out.sort_values(sort_cols, ascending=[True, False, False] if by_task else [False, False]).reset_index(drop=True)
    return out[out_cols]


def build_pairwise_metric_matrix(
    pairwise_df: pd.DataFrame,
    metric_col: str,
    a_col: str = "evaluator_a",
    b_col: str = "evaluator_b",
) -> pd.DataFrame:
    if pairwise_df.empty or metric_col not in pairwise_df.columns:
        return pd.DataFrame()
    if a_col not in pairwise_df.columns or b_col not in pairwise_df.columns:
        return pd.DataFrame()
    labels = sorted(
        pd.unique(
            pd.concat(
                [
                    pairwise_df[a_col].astype(str),
                    pairwise_df[b_col].astype(str),
                ],
                ignore_index=True,
            )
        ).tolist()
    )
    if not labels:
        return pd.DataFrame()
    mat = pd.DataFrame(np.nan, index=labels, columns=labels, dtype=float)
    for lbl in labels:
        if metric_col in {"agreement_rate", "kappa"}:
            mat.loc[lbl, lbl] = 1.0
    for _, row in pairwise_df.iterrows():
        a = str(row[a_col])
        b = str(row[b_col])
        v = float(row[metric_col]) if pd.notna(row[metric_col]) else np.nan
        mat.loc[a, b] = v
        mat.loc[b, a] = v
    return mat


def build_component_pairwise_agreement_table(
    claims_df: pd.DataFrame,
    component_col: str,
    min_overlap: int = 20,
    by_task: bool = False,
    control_other_stages: bool = False,
) -> pd.DataFrame:
    out_cols = [
        "component",
        "model_a",
        "model_b",
        "pair_label",
        "task",
        "n_overlap",
        "agreement_rate",
        "disagreement_rate",
        "kappa",
        "support_rate_a",
        "support_rate_b",
        "n_contexts",
    ]
    required = {
        "run_id",
        "claim_key",
        "task",
        "is_supported_int",
        component_col,
    }
    if claims_df.empty or not required.issubset(claims_df.columns):
        return pd.DataFrame(columns=out_cols)

    pipeline_stage_cols = ["claim_extractor_model", "abstain_evaluator_model", "verifier_model"]
    other_stage_cols: list[str] = []
    if control_other_stages:
        if component_col in pipeline_stage_cols:
            other_stage_cols = [c for c in pipeline_stage_cols if c in claims_df.columns and c != component_col]
        # For evaluator_label, we don't control for the individual stages because evaluator_label
        # is defined by them. Controlling for them would mean evaluator_label is constant per group.
    context_candidates = [
        "root_name",
        "generation_model",
        "q_generator_model",
        "temperature",
        "length_words",
    ] + other_stage_cols
    context_cols = [c for c in context_candidates if c in claims_df.columns]

    run_meta = claims_df[["run_id", component_col] + context_cols].drop_duplicates(subset=["run_id"]).copy()
    run_meta[component_col] = run_meta[component_col].fillna("unknown").astype(str)
    claim_votes = claims_df[["run_id", "claim_key", "task", "is_supported_int"]].copy()
    claim_votes["task"] = claim_votes["task"].fillna("NA").astype(str)
    claim_votes["is_supported_int"] = pd.to_numeric(claim_votes["is_supported_int"], errors="coerce")
    claim_votes = claim_votes.dropna(subset=["is_supported_int"])
    claim_votes["is_supported_int"] = claim_votes["is_supported_int"].astype(int).clip(lower=0, upper=1)

    work = claim_votes.merge(run_meta, on="run_id", how="inner")
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    if context_cols:
        context_groups = list(work.groupby(context_cols, dropna=False))
    else:
        context_groups = [(("ALL",), work)]

    accum: dict[tuple[str, str, str], dict[str, Any]] = {}
    for _, ctx in context_groups:
        if ctx.empty:
            continue
        vote_df = (
            ctx.groupby([component_col, "claim_key", "task"], dropna=False)
            .agg(vote_mean=("is_supported_int", "mean"))
            .reset_index()
        )
        vote_df["vote"] = (vote_df["vote_mean"] >= 0.5).astype(int)
        models = sorted(vote_df[component_col].dropna().astype(str).unique().tolist())
        if len(models) < 2:
            continue

        for i, model_a in enumerate(models):
            left = vote_df[vote_df[component_col].astype(str) == model_a][
                ["claim_key", "task", "vote"]
            ].rename(columns={"vote": "vote_a"})
            for model_b in models[i + 1 :]:
                right = vote_df[vote_df[component_col].astype(str) == model_b][
                    ["claim_key", "task", "vote"]
                ].rename(columns={"vote": "vote_b"})
                merged = left.merge(right, on=["claim_key", "task"], how="inner")
                if merged.empty:
                    continue

                groups = [("ALL", merged)] if not by_task else list(merged.groupby("task", dropna=False))
                for task_name, grp in groups:
                    if grp.empty:
                        continue
                    key = (str(model_a), str(model_b), str(task_name))
                    rec = accum.setdefault(
                        key,
                        {
                            "n11": 0,
                            "n10": 0,
                            "n01": 0,
                            "n00": 0,
                            "n_contexts": 0,
                        },
                    )
                    a = grp["vote_a"].to_numpy(dtype=int)
                    b = grp["vote_b"].to_numpy(dtype=int)
                    rec["n11"] += int(((a == 1) & (b == 1)).sum())
                    rec["n10"] += int(((a == 1) & (b == 0)).sum())
                    rec["n01"] += int(((a == 0) & (b == 1)).sum())
                    rec["n00"] += int(((a == 0) & (b == 0)).sum())
                    rec["n_contexts"] += 1

    rows: list[dict[str, Any]] = []
    for (model_a, model_b, task_name), rec in accum.items():
        n11 = int(rec["n11"])
        n10 = int(rec["n10"])
        n01 = int(rec["n01"])
        n00 = int(rec["n00"])
        n = n11 + n10 + n01 + n00
        if n < min_overlap:
            continue
        agreement = float((n11 + n00) / n)
        pa = float((n11 + n10) / n)
        pb = float((n11 + n01) / n)
        pe = float(pa * pb + (1.0 - pa) * (1.0 - pb))
        if np.isclose(1.0 - pe, 0.0):
            kappa = 1.0 if np.isclose(agreement, 1.0) else np.nan
        else:
            kappa = float((agreement - pe) / (1.0 - pe))
        rows.append(
            {
                "component": component_col,
                "model_a": model_a,
                "model_b": model_b,
                "pair_label": f"{model_a} vs {model_b}",
                "task": task_name,
                "n_overlap": int(n),
                "agreement_rate": agreement,
                "disagreement_rate": 1.0 - agreement,
                "kappa": kappa,
                "support_rate_a": pa,
                "support_rate_b": pb,
                "n_contexts": int(rec["n_contexts"]),
            }
        )

    if not rows:
        return pd.DataFrame(columns=out_cols)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["task", "agreement_rate", "n_overlap", "pair_label"] if by_task else ["agreement_rate", "n_overlap", "pair_label"],
        ascending=[True, False, False, True] if by_task else [False, False, True],
    ).reset_index(drop=True)
    return out[out_cols]


def build_evaluator_support_tables(consensus_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_cols = ["evaluator_label", "n_claims", "support_rate"]
    by_task_cols = ["evaluator_label", "task", "n_claims", "support_rate"]
    if consensus_df.empty:
        return pd.DataFrame(columns=overall_cols), pd.DataFrame(columns=by_task_cols)

    overall = (
        consensus_df.groupby("evaluator_label", dropna=False)
        .agg(
            n_claims=("claim_key", "count"),
            support_rate=("is_supported_vote", "mean"),
        )
        .reset_index()
        .sort_values(["support_rate", "n_claims"], ascending=[False, False])
        .reset_index(drop=True)
    )
    by_task = (
        consensus_df.groupby(["evaluator_label", "task"], dropna=False)
        .agg(
            n_claims=("claim_key", "count"),
            support_rate=("is_supported_vote", "mean"),
        )
        .reset_index()
        .sort_values(["task", "support_rate", "n_claims"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return overall[overall_cols], by_task[by_task_cols]


# ---------------------------------------------------------------------------
# Extraction agreement (Jaccard similarity)
# ---------------------------------------------------------------------------

def build_extraction_agreement(
    claims_df: pd.DataFrame,
    min_generations: int = 5,
    by_task: bool = False,
) -> pd.DataFrame:
    """Compare claim extractors by computing Jaccard similarity between the sets
    of claims extracted from the same source generation (sentence).

    For each pair of models (A, B), for each generation_key both models saw, we
    compute:
        jaccard = |claims_A ∩ claims_B| / |claims_A ∪ claims_B|

    We then average this across all shared generation_keys.

    Returns one row per pair (optionally per task).
    """
    out_cols = [
        "model_a",
        "model_b",
        "pair_label",
        "task",
        "n_generations",
        "jaccard_mean",
        "jaccard_std",
        "claims_per_gen_a",
        "claims_per_gen_b",
    ]
    required = {"run_id", "claim_key", "generation_key", "task", "claim_extractor_model"}
    if claims_df.empty or not required.issubset(claims_df.columns):
        return pd.DataFrame(columns=out_cols)

    work = claims_df[["claim_extractor_model", "generation_key", "claim_key", "task"]].copy()
    work["claim_extractor_model"] = work["claim_extractor_model"].fillna("unknown").astype(str)
    work["task"] = work["task"].fillna("NA").astype(str)

    models = sorted(work["claim_extractor_model"].unique().tolist())
    if len(models) < 2:
        return pd.DataFrame(columns=out_cols)

    scopes: list[tuple[str, pd.DataFrame]] = [("ALL", work)]
    if by_task:
        scopes = [(str(t), g.copy()) for t, g in work.groupby("task", dropna=False)]

    rows: list[dict[str, Any]] = []
    for scope_name, scope_work in scopes:
        # Build per-model claim sets per generation_key
        model_gen_claims: dict[str, dict[str, set]] = {}
        for model, group in scope_work.groupby("claim_extractor_model", dropna=False):
            gen_sets: dict[str, set] = {}
            for gen_key, gen_group in group.groupby("generation_key", dropna=False):
                gen_sets[str(gen_key)] = set(gen_group["claim_key"].tolist())
            model_gen_claims[str(model)] = gen_sets

        for i, model_a in enumerate(models):
            sets_a = model_gen_claims.get(model_a, {})
            for model_b in models[i + 1:]:
                sets_b = model_gen_claims.get(model_b, {})
                shared_gens = set(sets_a.keys()) & set(sets_b.keys())
                if len(shared_gens) < min_generations:
                    continue
                jaccards: list[float] = []
                counts_a: list[int] = []
                counts_b: list[int] = []
                for gen_key in shared_gens:
                    ca = sets_a[gen_key]
                    cb = sets_b[gen_key]
                    union = ca | cb
                    intersection = ca & cb
                    if union:
                        jaccards.append(len(intersection) / len(union))
                    counts_a.append(len(ca))
                    counts_b.append(len(cb))
                if not jaccards:
                    continue
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "pair_label": f"{model_a} vs {model_b}",
                        "task": scope_name,
                        "n_generations": int(len(jaccards)),
                        "jaccard_mean": float(np.mean(jaccards)),
                        "jaccard_std": float(np.std(jaccards)),
                        "claims_per_gen_a": float(np.mean(counts_a)),
                        "claims_per_gen_b": float(np.mean(counts_b)),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=out_cols)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["task", "jaccard_mean"] if by_task else ["jaccard_mean"],
        ascending=[True, False] if by_task else [False],
    ).reset_index(drop=True)
    return out[out_cols]


# ---------------------------------------------------------------------------
# Diversity & hallucination statistics
# ---------------------------------------------------------------------------

def build_diversity_and_hallucination_stats(
    claims_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return four DataFrames:
    1. overall_diversity  : avg/std claims per generation_key per claim_extractor_model
    2. task_diversity     : same, broken down by task
    3. overall_hallucination : hallucination_rate = 1 - support_rate per evaluator_label
    4. task_hallucination    : same, broken down by task
    """
    diversity_overall_cols = ["model", "n_generations", "claims_per_gen_mean", "claims_per_gen_std", "total_claims"]
    diversity_task_cols = ["model", "task", "n_generations", "claims_per_gen_mean", "claims_per_gen_std", "total_claims"]
    halu_overall_cols = ["evaluator_label", "n_claims", "support_rate", "hallucination_rate"]
    halu_task_cols = ["evaluator_label", "task", "n_claims", "support_rate", "hallucination_rate"]

    empty = (
        pd.DataFrame(columns=diversity_overall_cols),
        pd.DataFrame(columns=diversity_task_cols),
        pd.DataFrame(columns=halu_overall_cols),
        pd.DataFrame(columns=halu_task_cols),
    )
    if claims_df.empty:
        return empty

    required_div = {"claim_extractor_model", "generation_key", "claim_key"}
    required_halu = {"evaluator_label", "is_supported_int", "claim_key"}
    if not required_div.issubset(claims_df.columns) or not required_halu.issubset(claims_df.columns):
        return empty

    # --- Diversity ---
    div_work = claims_df[["claim_extractor_model", "generation_key", "claim_key", "task"]].copy()
    div_work["claim_extractor_model"] = div_work["claim_extractor_model"].fillna("unknown").astype(str)
    div_work["task"] = div_work["task"].fillna("NA").astype(str)

    div_grouped = (
        div_work.groupby(["claim_extractor_model", "generation_key"], dropna=False)["claim_key"]
        .nunique()
        .reset_index(name="n_claims_in_gen")
    )
    overall_div = (
        div_grouped.groupby("claim_extractor_model", dropna=False)
        .agg(
            n_generations=("generation_key", "nunique"),
            claims_per_gen_mean=("n_claims_in_gen", "mean"),
            claims_per_gen_std=("n_claims_in_gen", "std"),
            total_claims=("n_claims_in_gen", "sum"),
        )
        .reset_index()
        .rename(columns={"claim_extractor_model": "model"})
        .sort_values("claims_per_gen_mean", ascending=False)
        .reset_index(drop=True)
    )

    task_div_grouped = (
        div_work.groupby(["claim_extractor_model", "task", "generation_key"], dropna=False)["claim_key"]
        .nunique()
        .reset_index(name="n_claims_in_gen")
    )
    task_div = (
        task_div_grouped.groupby(["claim_extractor_model", "task"], dropna=False)
        .agg(
            n_generations=("generation_key", "nunique"),
            claims_per_gen_mean=("n_claims_in_gen", "mean"),
            claims_per_gen_std=("n_claims_in_gen", "std"),
            total_claims=("n_claims_in_gen", "sum"),
        )
        .reset_index()
        .rename(columns={"claim_extractor_model": "model"})
        .sort_values(["task", "claims_per_gen_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # --- Hallucination ---
    halu_work = claims_df[["evaluator_label", "is_supported_int", "claim_key", "task"]].copy()
    halu_work["evaluator_label"] = halu_work["evaluator_label"].fillna("unknown").astype(str)
    halu_work["task"] = halu_work["task"].fillna("NA").astype(str)
    halu_work["is_supported_int"] = pd.to_numeric(halu_work["is_supported_int"], errors="coerce")

    overall_halu = (
        halu_work.groupby("evaluator_label", dropna=False)
        .agg(
            n_claims=("claim_key", "count"),
            support_rate=("is_supported_int", "mean"),
        )
        .reset_index()
        .sort_values("support_rate", ascending=True)
        .reset_index(drop=True)
    )
    overall_halu["hallucination_rate"] = 1.0 - overall_halu["support_rate"]

    task_halu = (
        halu_work.groupby(["evaluator_label", "task"], dropna=False)
        .agg(
            n_claims=("claim_key", "count"),
            support_rate=("is_supported_int", "mean"),
        )
        .reset_index()
        .sort_values(["task", "support_rate"], ascending=[True, True])
        .reset_index(drop=True)
    )
    task_halu["hallucination_rate"] = 1.0 - task_halu["support_rate"]

    return (
        overall_div[diversity_overall_cols],
        task_div[diversity_task_cols],
        overall_halu[halu_overall_cols],
        task_halu[halu_task_cols],
    )
