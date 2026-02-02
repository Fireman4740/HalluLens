# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import json
import hashlib
import re
from pathlib import Path

# Avoid multiprocess resource_tracker shutdown errors on Python 3.12
os.environ.setdefault("MP_NO_RESOURCE_TRACKER", "1")

# Ensure project root is on sys.path to allow imports from 'tasks' and 'utils'
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd

try:
    from tasks.longwiki.facthalu import FactHalu
    from utils import exp
    from utils import generate_question as qa
    from utils import generate_hybrid_prompt as hybrid_qa
except ModuleNotFoundError:
    # Fallback in case the script is executed from a different CWD
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from tasks.longwiki.facthalu import FactHalu
    from utils import exp
    from utils import generate_question as qa
    from utils import generate_hybrid_prompt as hybrid_qa

TASKNAME = "longwiki"

def _slug(value: str) -> str:
    value = value or "default"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    value = value.strip("_")
    return value or "default"

def _resolve_run_namespace(args, model_name: str) -> str:
    namespace = getattr(args, "run_namespace", "auto")
    if namespace == "none":
        return ""
    if namespace == "auto":
        config = {
            "exp_mode": args.exp_mode,
            "model": args.model,
            "q_generator": args.q_generator,
            "claim_extractor": args.claim_extractor,
            "abstain_evaluator": args.abstain_evaluator,
            "verifier": args.verifier,
            "inference_method": args.inference_method,
            "N": args.N,
            "k": args.k,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_workers": args.max_workers,
            "tasks": args.tasks,
            "creativity": args.creativity,
            "length_words": args.length_words,
            "static_user_prompt": args.static_user_prompt,
            "low_level": args.low_level,
            "high_level": args.high_level,
            "prompt_seed": args.prompt_seed,
            "use_lm_studio": args.use_lm_studio,
            "lm_studio_url": args.lm_studio_url,
            "lm_studio_model": args.lm_studio_model,
            "db_path": args.db_path,
        }
        digest = hashlib.sha1(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        return f"{_slug(args.exp_mode)}_{_slug(model_name)}_{digest}"
    return _slug(namespace)

def _resolve_eval_cache_path(args, base_path: str, model_name: str) -> str:
    cache_root = (
        f"{base_path}/data/longwiki/.cache"
        if args.eval_cache_path is None
        else args.eval_cache_path
    )
    namespace = args.cache_namespace
    if namespace == "none":
        return cache_root
    if namespace == "auto":
        run_namespace = _resolve_run_namespace(args, model_name)
        if run_namespace:
            namespace = run_namespace
        else:
            config = {
                "model_name": model_name,
                "exp_mode": args.exp_mode,
                "claim_extractor": args.claim_extractor,
                "verifier": args.verifier,
                "abstain_evaluator": args.abstain_evaluator,
                "k": args.k,
                "db_path": args.db_path,
            }
            digest = hashlib.sha1(
                json.dumps(config, sort_keys=True).encode("utf-8")
            ).hexdigest()[:8]
            namespace = f"{_slug(args.exp_mode)}_{_slug(model_name)}_{digest}"
    return os.path.join(cache_root, namespace)

def _resolve_output_folder(args, model_name: str) -> Path:
    suffix = getattr(args, "output_suffix", "") or _resolve_run_namespace(args, model_name)
    suffix = _slug(suffix) if suffix else ""
    suffix_part = f"__{suffix}" if suffix else ""
    return Path(f"output/{TASKNAME}-{args.exp_mode}/{model_name}{suffix_part}")

def _resolve_qa_output_path(args, model_name: str) -> str:
    suffix = getattr(args, "output_suffix", "") or _resolve_run_namespace(args, model_name)
    suffix = _slug(suffix) if suffix else ""
    suffix_part = f"__{suffix}" if suffix else ""
    if args.exp_mode == "hybrid":
        return f"data/longwiki/save/hybrid_{model_name}{suffix_part}.jsonl"
    return f"data/longwiki/save/longwiki_{model_name}{suffix_part}.jsonl"

def run_eval(args):
    model_name = args.model.split("/")[-1]
    output_folder = _resolve_output_folder(args, model_name)
    output_csv = output_folder / "output.csv"
    generations_file_path = output_folder / "generation.jsonl"
    output_folder.mkdir(parents=True, exist_ok=True)
    if not generations_file_path.exists():
        print(
            "Missing generations file: {}. Run --do_inference first or provide a valid generation.jsonl before --do_eval.".format(
                generations_file_path
            )
        )
        return
    base_path = os.path.dirname(os.path.abspath(__name__))
    eval_cache_path = _resolve_eval_cache_path(args, base_path, model_name)
    os.makedirs(eval_cache_path, exist_ok=True)

    facthalu = FactHalu(
        generations_file_path,
        output_csv,
        abstain_evaluator=args.abstain_evaluator,
        refusal_evaluator=args.abstain_evaluator,
        claim_extractor=args.claim_extractor,
        verifier=args.verifier,
        k=args.k,
        eval_cache_path=eval_cache_path,
        db_path=args.db_path,
        args=args,
    )

    # save all evalaution details
    eval_details = {
        "output_csv": str(output_csv),
        "abstain_evaluator": args.abstain_evaluator,
        "claim_extractor": args.claim_extractor,
        "verifier": args.verifier,
        "k": args.k,
        "evalauted_model": model_name,
        "exp_mode": args.exp_mode,
        "eval_time": str(pd.Timestamp.now()),
    }

    with open(output_folder / "eval_details.json", "w") as f:
        json.dump(eval_details, f)

    facthalu.run()


def save_run_config(args, qa_output_path: str):
    model_name = args.model.split("/")[-1]
    output_folder = _resolve_output_folder(args, model_name)
    output_folder.mkdir(parents=True, exist_ok=True)

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    env_vars = {
        "OPENROUTER_API_KEY": "set" if openrouter_api_key else None,
        "OPENROUTER_HTTP_REFERER": os.getenv("OPENROUTER_HTTP_REFERER"),
        "OPENROUTER_APP_TITLE": os.getenv("OPENROUTER_APP_TITLE"),
        "OPENROUTER_MODEL": os.getenv("OPENROUTER_MODEL"),
        "USE_LM_STUDIO": os.getenv("USE_LM_STUDIO"),
        "LM_STUDIO_URL": os.getenv("LM_STUDIO_URL"),
        "LM_STUDIO_MODEL": os.getenv("LM_STUDIO_MODEL"),
        "EXP_MODE": os.getenv("EXP_MODE"),
        "N": os.getenv("N"),
        "DB_PATH": os.getenv("DB_PATH"),
        "MODEL_RESPONSE": os.getenv("MODEL_RESPONSE"),
        "MODEL_PROMPT": os.getenv("MODEL_PROMPT"),
        "MODEL_EVAL": os.getenv("MODEL_EVAL"),
        "ABSTAIN_EVALUATOR": os.getenv("ABSTAIN_EVALUATOR"),
        "CLAIM_EXTRACTOR": os.getenv("CLAIM_EXTRACTOR"),
        "VERIFIER": os.getenv("VERIFIER"),
        "INFERENCE_METHOD": os.getenv("INFERENCE_METHOD"),
        "TEMPERATURE": os.getenv("TEMPERATURE"),
        "MAX_TOKENS": os.getenv("MAX_TOKENS"),
        "MAX_WORKERS": os.getenv("MAX_WORKERS"),
        "TASKS": os.getenv("TASKS"),
        "CREATIVITY": os.getenv("CREATIVITY"),
        "LENGTH_WORDS": os.getenv("LENGTH_WORDS"),
        "LOW_LEVEL": os.getenv("LOW_LEVEL"),
        "HIGH_LEVEL": os.getenv("HIGH_LEVEL"),
        "STATIC_USER_PROMPT": os.getenv("STATIC_USER_PROMPT"),
        "K": os.getenv("K"),
        "EVAL_CACHE_PATH": os.getenv("EVAL_CACHE_PATH"),
    }

    config = {
        "timestamp": str(pd.Timestamp.now()),
        "task": TASKNAME,
        "exp_mode": args.exp_mode,
        "model_name": model_name,
        "qa_output_path": qa_output_path,
        "args": vars(args),
        "env": env_vars,
    }

    with open(output_folder / "run_config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(output_folder / "run_history.jsonl", "a") as f:
        f.write(json.dumps(config, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_mode", type=str, default="", help="longwiki or hybrid")

    parser.add_argument("--do_generate_prompt", default=False, action="store_true")
    parser.add_argument("--do_inference", default=False, action="store_true")
    parser.add_argument("--do_eval", default=False, action="store_true")
    parser.add_argument("--do_extract_only", default=False, action="store_true")

    parser.add_argument(
        "--model",
        type=str,
        default="deepseek/deepseek-v3.2",
        help='model that is being "TESTED"',
    )
    parser.add_argument(
        "--q_generator",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="model that is used for question generation",
    )

    parser.add_argument(
        "--claim_extractor",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="model that is used for claim extraction",
    )
    parser.add_argument(
        "--abstain_evaluator",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="model that is used for abstantion evaluation",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="deepseek/deepseek-v3.2",
        help="model that is used for final verification",
    )

    parser.add_argument(
        "--inference_method",
        type=str,
        default="custom",
        help="vllm, openai, or custom (OpenRouter)",
    )
    parser.add_argument("--eval_cache_path", type=str, default=None)
    parser.add_argument(
        "--cache_namespace",
        type=str,
        default="auto",
        help="Namespace under eval_cache_path: auto | none | <name>",
    )
    parser.add_argument(
        "--run_namespace",
        type=str,
        default="auto",
        help="Namespace for outputs/prompts: auto | none | <name>",
    )
    parser.add_argument(
        "--db_path", type=str, default="data/wiki_data/.cache/enwiki-20230401.db"
    )
    parser.add_argument("--N", type=int, default=250)

    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for inference model (0.0 = deterministic, higher = more random)",
    )
    parser.add_argument("--max_workers", type=int, default=64)
    parser.add_argument(
        "--prompt_max_workers",
        type=int,
        default=50,
        help="Max workers for prompt generation",
    )
    parser.add_argument(
        "--eval_max_workers",
        type=int,
        default=16,
        help="Max workers for evaluation (abstain/extract/verify)",
    )
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="Force using legacy cache by index even when prompt hashes mismatch",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Append a suffix to the output folder name to isolate runs",
    )
    # Hybrid prompt generation arguments
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Task types for hybrid prompts (e.g., INTERVIEW NEWS_ARTICLE)",
    )
    parser.add_argument(
        "--creativity",
        nargs="+",
        default=None,
        help="Creativity levels for hybrid prompts (e.g., FACTUAL HYBRID VERY_CREATIVE)",
    )
    parser.add_argument(
        "--length_words",
        type=int,
        default=500,
        help="Target word count for hybrid prompts",
    )
    parser.add_argument(
        "--prompt_seed",
        type=int,
        default=None,
        help="Random seed for deterministic subject selection in hybrid prompts",
    )
    parser.add_argument(
        "--static_user_prompt",
        action="store_true",
        help="Use deterministic prompt template for hybrid prompt generation",
    )
    parser.add_argument(
        "--low_level",
        type=int,
        default=5,
        help="Minimum h_score_cat for hybrid prompts",
    )
    parser.add_argument(
        "--high_level",
        type=int,
        default=10,
        help="Maximum h_score_cat for hybrid prompts",
    )
    parser.add_argument(
        "--use_lm_studio",
        action="store_true",
        help="Use LM Studio instead of OpenRouter",
    )
    parser.add_argument(
        "--lm_studio_url",
        type=str,
        default="http://10.10.12.21:1234/v1/chat/completions",
        help="LM Studio API URL",
    )
    parser.add_argument(
        "--lm_studio_model",
        type=str,
        default="openai/gpt-oss-20b",
        help="LM Studio model name",
    )

    args = parser.parse_args()

    # Set LM Studio environment variables if requested
    if args.use_lm_studio:
        os.environ["USE_LM_STUDIO"] = "true"
        os.environ["LM_STUDIO_URL"] = args.lm_studio_url
        os.environ["LM_STUDIO_MODEL"] = args.lm_studio_model
        print(
            f"Using LM Studio at {args.lm_studio_url} with model {args.lm_studio_model}"
        )

    # save all args details in
    base_path = os.path.dirname(os.path.abspath(__name__))
    model_name = args.model.split("/")[-1]

    # Determine output path based on mode
    QA_OUTPUT_PATH = _resolve_qa_output_path(args, model_name)

    save_run_config(args, QA_OUTPUT_PATH)

    if args.do_generate_prompt:
        if os.path.exists(QA_OUTPUT_PATH):
            print("using existing qa file")
            all_prompts = pd.read_json(QA_OUTPUT_PATH, lines=True)
            if len(all_prompts) >= args.N:
                all_prompts = all_prompts.head(args.N)
            else:
                print(
                    f"Warning: existing file has {len(all_prompts)} prompts, expected {args.N}"
                )
        else:
            if args.exp_mode == "longwiki":
                wiki_input_path = (
                    f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl"
                )
                print(wiki_input_path)
                QAs = qa.longform_QA_generation_run_batch(
                    wiki_input_path=f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl",
                    N=args.N,
                    q_generator=args.q_generator,
                    output_path=QA_OUTPUT_PATH,
                    from_scratch=False,
                    max_workers=args.prompt_max_workers,
                )
                all_prompts = pd.DataFrame(QAs)
            elif args.exp_mode == "hybrid":
                wiki_input_path = (
                    f"{base_path}/data/wiki_data/doc_goodwiki_h_score.jsonl"
                )
                print(f"Generating HYBRID prompts from {wiki_input_path}")

                # Determine which model to use for prompt generation
                if args.use_lm_studio:
                    q_generator_model = f"lm-studio/{args.lm_studio_model}"
                else:
                    q_generator_model = args.q_generator

                QAs = hybrid_qa.hybrid_prompt_generation_run_batch(
                    wiki_input_path=wiki_input_path,
                    N=args.N,
                    q_generator=q_generator_model,
                    output_path=QA_OUTPUT_PATH,
                    from_scratch=False,
                    low_level=args.low_level,
                    high_level=args.high_level,
                    tasks=args.tasks,
                    creativity_levels=args.creativity,
                    length_words=args.length_words,
                    static_user_prompt=args.static_user_prompt,
                    max_workers=args.prompt_max_workers,
                    seed=args.prompt_seed,
                )
                all_prompts = pd.DataFrame(QAs)
                print(f"Generated {len(all_prompts)} hybrid prompts")
            else:
                raise NotImplementedError(f"Mode {args.exp_mode} not implemented")

    # RUN INFERENCE
    if args.do_inference:
        all_prompts = pd.read_json(QA_OUTPUT_PATH, lines=True)
        if len(all_prompts) < args.N:
            print(f"Warning: only {len(all_prompts)} prompts available, using all")
        else:
            all_prompts = all_prompts.head(args.N)

        print(f"Start Inference for {args.model} ", args.exp_mode, len(all_prompts))

        output_folder = _resolve_output_folder(args, model_name)
        output_folder.mkdir(parents=True, exist_ok=True)
        generations_file_path = output_folder / "generation.jsonl"
        exp.run_exp(
            task=f"{TASKNAME}-{args.exp_mode}",
            model_path=args.model,
            all_prompts=all_prompts,
            generations_file_path=generations_file_path,
            inference_method=args.inference_method,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_workers=args.max_workers,
        )

        print("\n***Inference completed")

    # RUN EVALUATION:
    if args.do_eval:
        print("============= [[ {} ]] =================".format(args.exp_mode))
        print(f"Running evaluation for {model_name};")
        print(f"** Refusal Evaluator: {args.abstain_evaluator}")
        print(f"** Claim Extractor: {args.claim_extractor}")
        print(f"** Verifier: {args.verifier}")
        print("=========================================")
        run_eval(args)

        print("\n***Evaluation completed")
