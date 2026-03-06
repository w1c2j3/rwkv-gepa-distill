#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Missing ${VENV_PYTHON}. Run 'bash scripts/bootstrap.sh' first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

exec "${VENV_PYTHON}" -u scripts/scheduler.py batch-100 "$@"
