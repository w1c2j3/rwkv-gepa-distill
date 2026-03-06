#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
UV_CACHE_DIR="${ROOT_DIR}/.uv-cache"

pick_python() {
  if command -v python3.11 >/dev/null 2>&1; then
    echo "python3.11"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi

  echo "Unable to find python3.11 or python3." >&2
  exit 1
}

ensure_venv() {
  local python_bin="$1"

  if command -v uv >/dev/null 2>&1; then
    mkdir -p "${UV_CACHE_DIR}"
    UV_CACHE_DIR="${UV_CACHE_DIR}" uv venv \
      --python "${python_bin}" \
      --seed \
      --allow-existing \
      --no-managed-python \
      "${VENV_DIR}"
    return 0
  fi

  if [[ -d "${VENV_DIR}" ]]; then
    echo "Existing virtual environment is missing pip and uv is unavailable." >&2
    echo "Remove ${VENV_DIR} and rerun bootstrap on a system with working venv support." >&2
    exit 1
  fi

  echo "Creating virtual environment at ${VENV_DIR}"
  "${python_bin}" -m venv "${VENV_DIR}"
}

PYTHON_BIN="$(pick_python)"
if [[ "${PYTHON_BIN}" != "python3.11" ]]; then
  echo "python3.11 not found; falling back to $("${PYTHON_BIN}" --version 2>&1)" >&2
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  ensure_venv "${PYTHON_BIN}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python - <<'PY'
import sys

major, minor = sys.version_info[:2]
if major != 3 or minor < 11:
    raise SystemExit("Python 3.11+ is required for this project.")
print(f"Using Python {major}.{minor}")
PY

if ! python -m pip --version >/dev/null 2>&1; then
  ensure_venv "${PYTHON_BIN}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

python -m pip install --upgrade pip
python -m pip install datasets gepa litellm python-dotenv pyyaml orjson tenacity tqdm

echo "Bootstrap complete."
