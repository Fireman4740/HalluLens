#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAIN_SCRIPT="${SCRIPT_DIR}/task2_longwiki_openrouter.sh"
DOTENV_FILE="${ROOT_DIR}/.env"

normalize_file() {
  local path="$1"
  [[ -f "${path}" ]] || return 0

  if command -v perl >/dev/null 2>&1; then
    perl -pi -e 's/\r$//' "${path}"
  else
    local tmp
    tmp="$(mktemp)"
    tr -d '\r' < "${path}" > "${tmp}"
    cat "${tmp}" > "${path}"
    rm -f "${tmp}"
  fi
}

# Normalize line endings to avoid CRLF errors on WSL.
normalize_file "${MAIN_SCRIPT}"
normalize_file "${DOTENV_FILE}"

exec bash "${MAIN_SCRIPT}" "$@"
