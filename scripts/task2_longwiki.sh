# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

GPU="${GPU:-0}"
if [[ -f ".env" ]]; then
    set -a
    . ./.env
    set +a
fi
MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    # "meta-llama/Llama-3.1-70B-Instruct"
    # "meta-llama/Llama-3.1-405B-Instruct-FP8"
    # "meta-llama/Llama-3.3-70B-Instruct"
    # "google/gemma-2-9b-it"
    # "google/gemma-2-27b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "mistralai/Mistral-7B-Instruct-v0.3"
    # "mistralai/Mistral-Nemo-Instruct-2407"
    # "claude-3-sonnet"
    # "claude-3-haiku"
    # "gpt-4o"
)

# Inference parameters (can be overridden via environment variables)
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_WORKERS="${MAX_WORKERS:-64}"
INFERENCE_METHOD="${INFERENCE_METHOD:-vllm}"

EXP_MODE="${EXP_MODE:-longwiki}"
N="${N:-5}"
DB_PATH="${DB_PATH:-/private/home/yejinbang/facthalu/data/wiki_data/.cache/enwiki-20230401.db}"
MODEL_PROMPT="${MODEL_PROMPT:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_EVAL="${MODEL_EVAL:-meta-llama/Llama-3.1-8B-Instruct}"
ABSTAIN_EVALUATOR="${ABSTAIN_EVALUATOR:-${MODEL_EVAL}}"
CLAIM_EXTRACTOR="${CLAIM_EXTRACTOR:-${MODEL_EVAL}}"
VERIFIER="${VERIFIER:-${MODEL_EVAL}}"
TASKS="${TASKS:-}"
CREATIVITY="${CREATIVITY:-}"
LENGTH_WORDS="${LENGTH_WORDS:-500}"
LOW_LEVEL="${LOW_LEVEL:-5}"
HIGH_LEVEL="${HIGH_LEVEL:-10}"
K="${K:-32}"
EVAL_CACHE_PATH="${EVAL_CACHE_PATH:-}"

STATIC_USER_PROMPT="${STATIC_USER_PROMPT:-false}"
STATIC_ARGS=()
if [[ "${STATIC_USER_PROMPT}" == "true" ]]; then
    STATIC_ARGS+=(--static_user_prompt)
fi

TASKS_ARGS=()
if [[ -n "${TASKS}" ]]; then
    read -r -a TASKS_ARR <<< "${TASKS}"
    TASKS_ARGS+=(--tasks "${TASKS_ARR[@]}")
fi

CREATIVITY_ARGS=()
if [[ -n "${CREATIVITY}" ]]; then
    read -r -a CREATIVITY_ARR <<< "${CREATIVITY}"
    CREATIVITY_ARGS+=(--creativity "${CREATIVITY_ARR[@]}")
fi

EVAL_CACHE_ARGS=()
if [[ -n "${EVAL_CACHE_PATH}" ]]; then
    EVAL_CACHE_ARGS+=(--eval_cache_path "${EVAL_CACHE_PATH}")
fi

for MODEL in "${MODELS[@]}"
do  
    CUDA_VISIBLE_DEVICES=$GPU python3 -m tasks.longwiki.longwiki_main \
        --exp_mode $EXP_MODE \
        --do_generate_prompt \
        --do_inference \
        --do_eval \
        --model $MODEL\
        --inference_method "${INFERENCE_METHOD}" \
        --N "${N}" \
        --db_path "${DB_PATH}" \
        --q_generator "${MODEL_PROMPT}" \
        --abstain_evaluator "${ABSTAIN_EVALUATOR}" \
        --claim_extractor "${CLAIM_EXTRACTOR}" \
        --verifier "${VERIFIER}" \
        --k "${K}" \
        --temperature "${TEMPERATURE}" \
        --max_tokens "${MAX_TOKENS}" \
        --max_workers "${MAX_WORKERS}" \
        --length_words "${LENGTH_WORDS}" \
        --low_level "${LOW_LEVEL}" \
        --high_level "${HIGH_LEVEL}" \
        "${STATIC_ARGS[@]}" \
        "${TASKS_ARGS[@]}" \
        "${CREATIVITY_ARGS[@]}" \
        "${EVAL_CACHE_ARGS[@]}"
done
