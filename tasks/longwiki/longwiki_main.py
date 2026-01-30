# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import json
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


def run_eval(args):
    model_name = args.model.split("/")[-1]
    output_folder = Path(f"output/{TASKNAME}-{args.exp_mode}/{model_name}")
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
    eval_cache_path = (
        f"{base_path}/data/longwiki/.cache"
        if args.eval_cache_path is None
        else args.eval_cache_path
    )

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
    if args.exp_mode == "hybrid":
        QA_OUTPUT_PATH = f"data/longwiki/save/hybrid_{model_name}.jsonl"
    else:
        QA_OUTPUT_PATH = f"data/longwiki/save/longwiki_{model_name}.jsonl"

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

        exp.run_exp(
            task=f"{TASKNAME}-{args.exp_mode}",
            model_path=args.model,
            all_prompts=all_prompts,
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
