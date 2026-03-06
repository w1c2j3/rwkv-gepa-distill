#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor"

mkdir -p "${VENDOR_DIR}"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [[ -d "${target_dir}/.git" ]]; then
    echo "Skipping existing repo: ${target_dir}"
    return 0
  fi

  echo "Cloning ${repo_url} -> ${target_dir}"
  git clone --depth 1 "${repo_url}" "${target_dir}"
}

clone_if_missing "https://github.com/BlinkDL/RWKV-LM" "${VENDOR_DIR}/RWKV-LM"
clone_if_missing "https://github.com/BlinkDL/Albatross" "${VENDOR_DIR}/Albatross"
clone_if_missing "https://github.com/JL-er/RWKV-PEFT.git" "${VENDOR_DIR}/RWKV-PEFT"

echo "Vendor fetch complete."
