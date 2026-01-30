#!/usr/bin/env bash

# Runs LongWiki end-to-end with OpenRouter only.
# - Prompt generation: deepseek/deepseek-v3.2
# - Response generation: mistral-small-creative
# - Evaluation: deepseek/deepseek-v3.2

set -euo pipefail

EXP_MODE=longwiki
N=1
DB_PATH="data/wiki_data/.cache/enwiki-20230401.db"

MODEL_RESPONSE="mistralai/mistral-small-creative"
MODEL_PROMPT="deepseek/deepseek-v3.2"
MODEL_EVAL="deepseek/deepseek-v3.2"

# Inference parameters (can be overridden via environment variables)
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_WORKERS="${MAX_WORKERS:-64}"

python -m tasks.longwiki.longwiki_main \
  --exp_mode "${EXP_MODE}" \
  --do_generate_prompt \
  --do_inference \
  --do_eval \
  --model "${MODEL_RESPONSE}" \
  --q_generator "${MODEL_PROMPT}" \
  --abstain_evaluator "${MODEL_EVAL}" \
  --claim_extractor "${MODEL_EVAL}" \
  --verifier "${MODEL_EVAL}" \
  --db_path "${DB_PATH}" \
  --N "${N}" \
  --temperature "${TEMPERATURE}" \
  --max_tokens "${MAX_TOKENS}" \
  --max_workers "${MAX_WORKERS}"
