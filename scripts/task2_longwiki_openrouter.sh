#!/usr/bin/env bash

# Runs LongWiki end-to-end with OpenRouter only.
# - Prompt generation: MODEL_PROMPT (default: openai/gpt-oss-safeguard-20b)
# - Response generation: MODEL_RESPONSE (default: mistral-small-creative)
# - Evaluation: MODEL_EVAL (default: openai/gpt-oss-safeguard-20b)
# - Tasks: set TASKS="INTERVIEW NEWS_ARTICLE ..." to pass --tasks

set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  . ./.env
  set +a
fi

EXP_MODE="${EXP_MODE:-longwiki}"
N="${N:-1}"
DB_PATH="${DB_PATH:-data/wiki_data/.cache/enwiki-20230401.db}"

MODEL_RESPONSE="${MODEL_RESPONSE:-mistralai/mistral-small-creative}"
MODEL_PROMPT="${MODEL_PROMPT:-openai/gpt-oss-safeguard-20b}"
MODEL_EVAL="${MODEL_EVAL:-openai/gpt-oss-safeguard-20b}"
ABSTAIN_EVALUATOR="${ABSTAIN_EVALUATOR:-${MODEL_EVAL}}"
CLAIM_EXTRACTOR="${CLAIM_EXTRACTOR:-${MODEL_EVAL}}"
VERIFIER="${VERIFIER:-${MODEL_EVAL}}"

# Inference parameters (can be overridden via environment variables)
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_WORKERS="${MAX_WORKERS:-64}"
LENGTH_WORDS="${LENGTH_WORDS:-500}"
LOW_LEVEL="${LOW_LEVEL:-5}"
HIGH_LEVEL="${HIGH_LEVEL:-10}"
K="${K:-32}"
INFERENCE_METHOD="${INFERENCE_METHOD:-custom}"
EVAL_CACHE_PATH="${EVAL_CACHE_PATH:-}"
USE_LM_STUDIO="${USE_LM_STUDIO:-false}"
LM_STUDIO_URL="${LM_STUDIO_URL:-http://10.10.12.21:1234/v1/chat/completions}"
LM_STUDIO_MODEL="${LM_STUDIO_MODEL:-openai/gpt-oss-20b}"

STATIC_USER_PROMPT="${STATIC_USER_PROMPT:-false}"

STATIC_ARGS=()
if [[ "${STATIC_USER_PROMPT}" == "true" ]]; then
  STATIC_ARGS+=(--static_user_prompt)
fi

TASKS_ARGS=()
if [[ -n "${TASKS:-}" ]]; then
  read -r -a TASKS_ARR <<< "${TASKS}"
  TASKS_ARGS+=(--tasks "${TASKS_ARR[@]}")
fi

CREATIVITY_ARGS=()
if [[ -n "${CREATIVITY:-}" ]]; then
  read -r -a CREATIVITY_ARR <<< "${CREATIVITY}"
  CREATIVITY_ARGS+=(--creativity "${CREATIVITY_ARR[@]}")
fi

LM_STUDIO_ARGS=()
if [[ "${USE_LM_STUDIO}" == "true" ]]; then
  LM_STUDIO_ARGS+=(--use_lm_studio --lm_studio_url "${LM_STUDIO_URL}" --lm_studio_model "${LM_STUDIO_MODEL}")
fi

EVAL_CACHE_ARGS=()
if [[ -n "${EVAL_CACHE_PATH}" ]]; then
  EVAL_CACHE_ARGS+=(--eval_cache_path "${EVAL_CACHE_PATH}")
fi

python -m tasks.longwiki.longwiki_main \
  --exp_mode "${EXP_MODE}" \
  --do_generate_prompt \
  --do_inference \
  --do_eval \
  --model "${MODEL_RESPONSE}" \
  --q_generator "${MODEL_PROMPT}" \
  --abstain_evaluator "${ABSTAIN_EVALUATOR}" \
  --claim_extractor "${CLAIM_EXTRACTOR}" \
  --verifier "${VERIFIER}" \
  --db_path "${DB_PATH}" \
  --N "${N}" \
  --k "${K}" \
  --temperature "${TEMPERATURE}" \
  --max_tokens "${MAX_TOKENS}" \
  --max_workers "${MAX_WORKERS}" \
  --length_words "${LENGTH_WORDS}" \
  --low_level "${LOW_LEVEL}" \
  --high_level "${HIGH_LEVEL}" \
  --inference_method "${INFERENCE_METHOD}" \
  ${STATIC_ARGS[@]+"${STATIC_ARGS[@]}"} \
  ${TASKS_ARGS[@]+"${TASKS_ARGS[@]}"} \
  ${CREATIVITY_ARGS[@]+"${CREATIVITY_ARGS[@]}"} \
  ${LM_STUDIO_ARGS[@]+"${LM_STUDIO_ARGS[@]}"} \
  ${EVAL_CACHE_ARGS[@]+"${EVAL_CACHE_ARGS[@]}"}
