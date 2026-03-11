#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"


def default_python() -> str:
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable


def command_display(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def format_duration(seconds: float) -> str:
    whole = int(round(seconds))
    minutes, seconds = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command scheduler for the seed-driven distillation pipeline.")
    parser.add_argument("--dataset-name", default="mmlu_auxiliary_train_seed_run")
    parser.add_argument("--seed-input-path", type=Path, default=Path("data/mmlu_auxiliary_train/questions.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants-per-seed", type=int, default=12)
    return parser.parse_args()


def build_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.run_seed_pipeline",
        "--dataset-name",
        args.dataset_name,
        "--seed-input-path",
        str(args.seed_input_path),
        "--variants-per-seed",
        str(args.variants_per_seed),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def main() -> None:
    args = parse_args()
    command = build_command(args, default_python())
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    print(f"cmd={command_display(command)}", flush=True)
    subprocess.run(command, cwd=ROOT_DIR, env=env, check=True)
    print(f"pipeline_status=completed elapsed={format_duration(time.monotonic() - started)}", flush=True)


if __name__ == "__main__":
    main()
