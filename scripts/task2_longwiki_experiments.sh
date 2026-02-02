#!/usr/bin/env bash
set -euo pipefail

load_dotenv() {
  local dotenv_file="${1:-.env}"
  [[ -f "${dotenv_file}" ]] || return 0

  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ "${line}" =~ ^[[:space:]]*$ ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue

    if [[ "${line}" == export[[:space:]]* ]]; then
      line="${line#export }"
    fi

    [[ "${line}" == *"="* ]] || continue
    local key="${line%%=*}"
    local val="${line#*=}"

    key="${key%%[[:space:]]*}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    val="${val#"${val%%[![:space:]]*}"}"

    if [[ "${val}" == \"*\" ]]; then
      val="${val#\"}"
      val="${val%%\"*}"
    elif [[ "${val}" == \'*\' ]]; then
      val="${val#\'}"
      val="${val%%\'*}"
    else
      val="${val%%[[:space:]]#*}"
      val="${val%"${val##*[![:space:]]}"}"
    fi

    export "${key}=${val}"
  done < "${dotenv_file}"
}

slug() {
  local value="${1:-default}"
  value="$(printf '%s' "${value}" | sed -E 's/[^a-zA-Z0-9._-]+/_/g; s/^_+|_+$//g')"
  [[ -z "${value}" ]] && value="default"
  printf '%s' "${value}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

load_dotenv ".env"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Error: python not found in PATH." >&2
    exit 1
  fi
fi

# Core config
EXP_MODE="${EXP_MODE:-hybrid}"
N="${N:-300}"
DB_PATH="${DB_PATH:-data/wiki_data/.cache/enwiki-20230401.db}"

MODEL_PROMPT="${MODEL_PROMPT:-openai/gpt-oss-safeguard-20b}"
MODEL_EVAL="${MODEL_EVAL:-openai/gpt-oss-safeguard-20b}"
ABSTAIN_EVALUATOR="${ABSTAIN_EVALUATOR:-${MODEL_EVAL}}"
CLAIM_EXTRACTOR="${CLAIM_EXTRACTOR:-${MODEL_EVAL}}"
VERIFIER="${VERIFIER:-${MODEL_EVAL}}"

INFERENCE_METHOD="${INFERENCE_METHOD:-custom}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
MAX_WORKERS="${MAX_WORKERS:-64}"
PROMPT_MAX_WORKERS="${PROMPT_MAX_WORKERS:-50}"
EVAL_MAX_WORKERS="${EVAL_MAX_WORKERS:-16}"

LENGTH_WORDS_LIST="${LENGTH_WORDS_LIST:-${LENGTH_WORDS:-500}}"
INFER_TEMPS="${INFER_TEMPS:-${TEMPERATURE}}"
INFER_MODELS="${INFER_MODELS:-${MODEL_RESPONSE:-}}"

LENGTH_WORDS_LIST="${LENGTH_WORDS_LIST//,/ }"
INFER_TEMPS="${INFER_TEMPS//,/ }"
INFER_MODELS="${INFER_MODELS//,/ }"

STATIC_USER_PROMPT="${STATIC_USER_PROMPT:-true}"
PROMPT_SEED="${PROMPT_SEED:-42}"

LOW_LEVEL="${LOW_LEVEL:-5}"
HIGH_LEVEL="${HIGH_LEVEL:-10}"
K="${K:-32}"

PROMPT_MODEL_TAG="${PROMPT_MODEL_TAG:-promptset}"
RUN_NAMESPACE_BASE="${RUN_NAMESPACE_BASE:-static_prompts}"
FORCE_REGEN="${FORCE_REGEN:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

if [[ -z "${INFER_MODELS}" ]]; then
  echo "Error: set INFER_MODELS or MODEL_RESPONSE in .env." >&2
  exit 1
fi

PREFIX="longwiki"
if [[ "${EXP_MODE}" == "hybrid" ]]; then
  PREFIX="hybrid"
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

STATIC_ARGS=()
if [[ "${STATIC_USER_PROMPT}" == "true" ]]; then
  STATIC_ARGS+=(--static_user_prompt)
fi

LM_STUDIO_ARGS=()
if [[ "${USE_LM_STUDIO:-false}" == "true" ]]; then
  LM_STUDIO_ARGS+=(--use_lm_studio)
  LM_STUDIO_ARGS+=(--lm_studio_url "${LM_STUDIO_URL:-http://10.10.12.21:1234/v1/chat/completions}")
  LM_STUDIO_ARGS+=(--lm_studio_model "${LM_STUDIO_MODEL:-openai/gpt-oss-20b}")
fi

EVAL_CACHE_ARGS=()
if [[ -n "${EVAL_CACHE_PATH:-}" ]]; then
  EVAL_CACHE_ARGS+=(--eval_cache_path "${EVAL_CACHE_PATH}")
fi

CACHE_NAMESPACE_ARGS=()
if [[ -n "${CACHE_NAMESPACE:-}" ]]; then
  CACHE_NAMESPACE_ARGS+=(--cache_namespace "${CACHE_NAMESPACE}")
fi

prompt_file_for() {
  local run_ns="$1"
  local suffix
  suffix="$(slug "${run_ns}")"
  local suffix_part=""
  if [[ -n "${suffix}" ]]; then
    suffix_part="__${suffix}"
  fi
  printf '%s\n' "data/longwiki/save/${PREFIX}_${PROMPT_MODEL_TAG}${suffix_part}.jsonl"
}

qa_path_for() {
  local model="$1"
  local run_ns="$2"
  local model_base="${model##*/}"
  local suffix
  suffix="$(slug "${run_ns}")"
  local suffix_part=""
  if [[ -n "${suffix}" ]]; then
    suffix_part="__${suffix}"
  fi
  printf '%s\n' "data/longwiki/save/${PREFIX}_${model_base}${suffix_part}.jsonl"
}

generate_prompts_for_length() {
  local length="$1"
  local run_ns="${RUN_NAMESPACE_BASE}_len${length}"
  local prompt_file
  prompt_file="$(prompt_file_for "${run_ns}")"

  if [[ -f "${prompt_file}" && "${FORCE_REGEN}" != "true" ]]; then
    echo "Using existing prompts: ${prompt_file}" >&2
    printf '%s\n' "${prompt_file}"
    return 0
  fi

  echo "Generating prompts (len=${length}, static=${STATIC_USER_PROMPT}, seed=${PROMPT_SEED})" >&2
  "${PYTHON_BIN}" -m tasks.longwiki.longwiki_main \
    --exp_mode "${EXP_MODE}" \
    --do_generate_prompt \
    --model "${PROMPT_MODEL_TAG}" \
    --q_generator "${MODEL_PROMPT}" \
    --db_path "${DB_PATH}" \
    --N "${N}" \
    --prompt_max_workers "${PROMPT_MAX_WORKERS}" \
    --length_words "${length}" \
    --low_level "${LOW_LEVEL}" \
    --high_level "${HIGH_LEVEL}" \
    --run_namespace "${run_ns}" \
    --prompt_seed "${PROMPT_SEED}" \
    "${STATIC_ARGS[@]}" \
    "${TASKS_ARGS[@]}" \
    "${CREATIVITY_ARGS[@]}" \
    "${LM_STUDIO_ARGS[@]}" 1>&2

  if [[ ! -f "${prompt_file}" ]]; then
    echo "Error: prompts file not found: ${prompt_file}" >&2
    return 1
  fi

  printf '%s\n' "${prompt_file}"
}

run_inference_eval() {
  local length="$1"
  local model="$2"
  local temp="$3"
  local model_base="${model##*/}"
  local run_ns="${RUN_NAMESPACE_BASE}_len${length}_m${model_base}_t${temp}"
  local qa_path
  qa_path="$(qa_path_for "${model}" "${run_ns}")"

  if [[ ! -f "${qa_path}" ]]; then
    mkdir -p "$(dirname "${qa_path}")"
    cp -f "$4" "${qa_path}"
  fi

  local args=(
    --exp_mode "${EXP_MODE}"
    --do_inference
    --model "${model}"
    --q_generator "${MODEL_PROMPT}"
    --abstain_evaluator "${ABSTAIN_EVALUATOR}"
    --claim_extractor "${CLAIM_EXTRACTOR}"
    --verifier "${VERIFIER}"
    --db_path "${DB_PATH}"
    --N "${N}"
    --k "${K}"
    --temperature "${temp}"
    --max_tokens "${MAX_TOKENS}"
    --max_workers "${MAX_WORKERS}"
    --prompt_max_workers "${PROMPT_MAX_WORKERS}"
    --eval_max_workers "${EVAL_MAX_WORKERS}"
    --length_words "${length}"
    --low_level "${LOW_LEVEL}"
    --high_level "${HIGH_LEVEL}"
    --inference_method "${INFERENCE_METHOD}"
    --run_namespace "${run_ns}"
    --prompt_seed "${PROMPT_SEED}"
  )

  if [[ "${SKIP_EVAL}" != "true" ]]; then
    args+=(--do_eval)
  fi

  "${PYTHON_BIN}" -m tasks.longwiki.longwiki_main \
    "${args[@]}" \
    "${STATIC_ARGS[@]}" \
    "${TASKS_ARGS[@]}" \
    "${CREATIVITY_ARGS[@]}" \
    "${LM_STUDIO_ARGS[@]}" \
    "${EVAL_CACHE_ARGS[@]}" \
    "${CACHE_NAMESPACE_ARGS[@]}"
}

for length in ${LENGTH_WORDS_LIST}; do
  prompts_file="$(generate_prompts_for_length "${length}")"
  for model in ${INFER_MODELS}; do
    for temp in ${INFER_TEMPS}; do
      echo "Run: len=${length} model=${model} temp=${temp}"
      run_inference_eval "${length}" "${model}" "${temp}" "${prompts_file}"
    done
  done
done
