#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Missing ${VENV_DIR}. Run 'bash scripts/bootstrap.sh' first." >&2
  exit 1
fi

cd "${ROOT_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m distill_gepa.optimize_prompt
python -m distill_gepa.generate_small_dataset

echo "Smoke test outputs:"
echo "  ${ROOT_DIR}/artifacts/best_prompt.txt"
echo "  ${ROOT_DIR}/artifacts/optimization_report.json"
echo "  ${ROOT_DIR}/data/distill/distill_small.jsonl"
