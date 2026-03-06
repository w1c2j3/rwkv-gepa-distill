#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT_DIR / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    command: list[str]


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


def run_tasks(tasks: list[TaskSpec]) -> None:
    if not tasks:
        raise ValueError("No tasks to run.")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    total = len(tasks)
    started = time.monotonic()
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{total}] {task.name}", flush=True)
        print(f"desc={task.description}", flush=True)
        print(f"cmd={command_display(task.command)}", flush=True)
        task_started = time.monotonic()
        subprocess.run(task.command, cwd=ROOT_DIR, env=env, check=True)
        elapsed = time.monotonic() - task_started
        print(f"status=completed task={task.name} elapsed={format_duration(elapsed)}", flush=True)

    total_elapsed = time.monotonic() - started
    print(f"pipeline_status=completed elapsed={format_duration(total_elapsed)}", flush=True)


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Rebuild pools even if cached JSONL exists.")
    parser.add_argument(
        "--include-community",
        action="store_true",
        help="Also prepare community training pools.",
    )
    parser.add_argument(
        "--only",
        choices=("primary", "eval", "all"),
        default="all",
        help="Pool category selection for prepare_mmlu_pools.py.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap passed to the prepare step.")


def add_optimize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/question_pools/mmlu_dev.jsonl"),
    )
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/question_pools/mmlu_validation.jsonl"),
    )
    parser.add_argument("--train-limit", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=10)
    parser.add_argument("--max-metric-calls", type=int, default=40)
    parser.add_argument(
        "--best-prompt-path",
        type=Path,
        default=Path("artifacts/mmlu_best_prompt.txt"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/mmlu_prompt_optimization_report.json"),
    )
    parser.add_argument(
        "--fresh-run-dir",
        action="store_true",
        default=True,
        help="Reset the GEPA run directory before optimization.",
    )
    parser.add_argument(
        "--no-fresh-run-dir",
        dest="fresh_run_dir",
        action="store_false",
        help="Reuse the existing GEPA run directory if present.",
    )


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--question-pool-path",
        type=Path,
        default=Path("data/question_pools/mmlu_auxiliary_train.jsonl"),
    )
    parser.add_argument(
        "--best-prompt-path",
        type=Path,
        default=Path("artifacts/mmlu_best_prompt.txt"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=99842)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Teacher API concurrency. Keep <= 6 for unstable proxy endpoints.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_correct.jsonl"),
    )
    parser.add_argument("--progress-interval", type=int, default=5000)


def add_sample_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_correct.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("artifacts/mmlu_batch_all_review_40.jsonl"),
    )
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=5000)


def add_export_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_correct.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_sft.jsonl"),
    )
    parser.add_argument("--progress-interval", type=int, default=5000)


def prepare_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    command = [python_bin, "-u", "scripts/prepare_mmlu_pools.py", "--only", args.only]
    if args.force:
        command.append("--force")
    if args.include_community:
        command.append("--include-community")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return command


def optimize_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.optimize_mmlu_prompt",
        "--train-path",
        str(args.train_path),
        "--val-path",
        str(args.val_path),
        "--train-limit",
        str(args.train_limit),
        "--val-limit",
        str(args.val_limit),
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--best-prompt-path",
        str(args.best_prompt_path),
        "--report-path",
        str(args.report_path),
    ]
    if args.fresh_run_dir:
        command.append("--fresh-run-dir")
    return command


def generate_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.generate_mmlu_batch",
        "--question-pool-path",
        str(args.question_pool_path),
        "--best-prompt-path",
        str(args.best_prompt_path),
        "--output-path",
        str(args.output_path),
        "--limit",
        str(args.limit),
        "--max-concurrency",
        str(args.max_concurrency),
        "--progress-interval",
        str(args.progress_interval),
    ]
    if getattr(args, "resume", False):
        command.append("--resume")
    return command


def filter_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.filter_mmlu_batch",
        "--input-path",
        str(args.input_path),
        "--output-path",
        str(args.output_path),
        "--progress-interval",
        str(args.progress_interval),
    ]


def sample_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.sample_review_set",
        "--input-path",
        str(args.input_path),
        "--output-path",
        str(args.output_path),
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
        "--progress-interval",
        str(args.progress_interval),
    ]


def export_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.export_sft_dataset",
        "--input-path",
        str(args.input_path),
        "--output-path",
        str(args.output_path),
        "--progress-interval",
        str(args.progress_interval),
    ]


def batch_100_tasks(args: argparse.Namespace, python_bin: str) -> list[TaskSpec]:
    prepare_args = argparse.Namespace(
        force=args.force_pools,
        include_community=False,
        only="primary",
        limit=None,
    )
    optimize_args = argparse.Namespace(
        train_path=Path("data/question_pools/mmlu_dev.jsonl"),
        val_path=Path("data/question_pools/mmlu_validation.jsonl"),
        train_limit=10,
        val_limit=10,
        max_metric_calls=40,
        best_prompt_path=Path("artifacts/mmlu_best_prompt.txt"),
        report_path=Path("artifacts/mmlu_prompt_optimization_report.json"),
        fresh_run_dir=True,
    )
    generate_args = argparse.Namespace(
        question_pool_path=Path("data/question_pools/mmlu_auxiliary_train.jsonl"),
        best_prompt_path=Path("artifacts/mmlu_best_prompt.txt"),
        output_path=Path("data/distill/mmlu_batch_100.jsonl"),
        limit=100,
        max_concurrency=args.max_concurrency,
        progress_interval=10,
        resume=False,
    )
    sample_args = argparse.Namespace(
        input_path=Path("data/distill/mmlu_batch_100.jsonl"),
        output_path=Path("artifacts/mmlu_batch_review_40.jsonl"),
        sample_size=40,
        seed=42,
        progress_interval=100,
    )
    return [
        TaskSpec(
            name="prepare-pools",
            description="Check or build the primary MMLU question pools.",
            command=prepare_command(prepare_args, python_bin),
        ),
        TaskSpec(
            name="optimize-mmlu-prompt",
            description="Use GEPA to optimize the teacher system prompt on dev/validation slices.",
            command=optimize_command(optimize_args, python_bin),
        ),
        TaskSpec(
            name="generate-100",
            description="Generate the first 100 auxiliary_train teacher responses.",
            command=generate_command(generate_args, python_bin),
        ),
        TaskSpec(
            name="sample-review-40",
            description="Draw a random 40-row review sample from the 100-row batch.",
            command=sample_command(sample_args, python_bin),
        ),
    ]


def full_mmlu_tasks(args: argparse.Namespace, python_bin: str) -> list[TaskSpec]:
    prepare_args = argparse.Namespace(
        force=args.force_pools,
        include_community=False,
        only="primary",
        limit=None,
    )
    optimize_args = argparse.Namespace(
        train_path=Path("data/question_pools/mmlu_dev.jsonl"),
        val_path=Path("data/question_pools/mmlu_validation.jsonl"),
        train_limit=10,
        val_limit=10,
        max_metric_calls=40,
        best_prompt_path=Path("artifacts/mmlu_best_prompt.txt"),
        report_path=Path("artifacts/mmlu_prompt_optimization_report.json"),
        fresh_run_dir=True,
    )
    generate_args = argparse.Namespace(
        question_pool_path=Path("data/question_pools/mmlu_auxiliary_train.jsonl"),
        best_prompt_path=Path("artifacts/mmlu_best_prompt.txt"),
        output_path=Path("data/distill/mmlu_batch_all.jsonl"),
        limit=args.limit,
        max_concurrency=args.max_concurrency,
        progress_interval=args.progress_interval,
        resume=True,
    )
    filter_args = argparse.Namespace(
        input_path=Path("data/distill/mmlu_batch_all.jsonl"),
        output_path=Path("data/distill/mmlu_batch_all_correct.jsonl"),
        progress_interval=args.filter_progress_interval,
    )
    export_args = argparse.Namespace(
        input_path=Path("data/distill/mmlu_batch_all_correct.jsonl"),
        output_path=Path("data/distill/mmlu_batch_all_sft.jsonl"),
        progress_interval=args.export_progress_interval,
    )
    sample_args = argparse.Namespace(
        input_path=Path("data/distill/mmlu_batch_all_correct.jsonl"),
        output_path=Path("artifacts/mmlu_batch_all_review_40.jsonl"),
        sample_size=40,
        seed=42,
        progress_interval=args.sample_progress_interval,
    )
    return [
        TaskSpec(
            name="prepare-pools",
            description="Check or build the primary MMLU question pools.",
            command=prepare_command(prepare_args, python_bin),
        ),
        TaskSpec(
            name="optimize-mmlu-prompt",
            description="Use GEPA to optimize the teacher system prompt on dev/validation slices.",
            command=optimize_command(optimize_args, python_bin),
        ),
        TaskSpec(
            name="generate-full",
            description="Generate teacher responses for the full auxiliary_train pool.",
            command=generate_command(generate_args, python_bin),
        ),
        TaskSpec(
            name="filter-correct",
            description="Keep only rows whose parsed answer matches the gold label.",
            command=filter_command(filter_args, python_bin),
        ),
        TaskSpec(
            name="export-sft",
            description="Export the filtered correct rows into an SFT-ready JSONL dataset.",
            command=export_command(export_args, python_bin),
        ),
        TaskSpec(
            name="sample-review-40",
            description="Draw a random 40-row review sample from the filtered full batch.",
            command=sample_command(sample_args, python_bin),
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scheduler for rwkv-gepa-distill task pipelines.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-pools", help="Prepare Hugging Face question/eval pools.")
    add_prepare_args(prepare_parser)

    optimize_parser = subparsers.add_parser("optimize-mmlu", help="Run GEPA prompt optimization for MMLU.")
    add_optimize_args(optimize_parser)

    generate_parser = subparsers.add_parser("generate-mmlu", help="Generate teacher answers for a question pool.")
    add_generate_args(generate_parser)
    generate_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial output file.",
    )

    filter_parser = subparsers.add_parser("filter-mmlu", help="Keep only correct generated teacher rows.")
    add_filter_args(filter_parser)

    sample_parser = subparsers.add_parser("sample-review", help="Sample a random review subset from JSONL.")
    add_sample_args(sample_parser)

    export_parser = subparsers.add_parser("export-sft", help="Export filtered rows into SFT-ready JSONL.")
    add_export_args(export_parser)

    batch_parser = subparsers.add_parser("batch-100", help="One-command 100-row MMLU pipeline.")
    batch_parser.add_argument(
        "--force-pools",
        action="store_true",
        help="Rebuild primary pools even if the cached JSONL files already exist.",
    )
    batch_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=6,
        help="Teacher API concurrency for the 100-row batch.",
    )

    full_parser = subparsers.add_parser("full-mmlu", help="One-command full auxiliary_train pipeline.")
    full_parser.add_argument(
        "--force-pools",
        action="store_true",
        help="Rebuild primary pools even if the cached JSONL files already exist.",
    )
    full_parser.add_argument("--limit", type=int, default=99842)
    full_parser.add_argument("--max-concurrency", type=int, default=6)
    full_parser.add_argument("--progress-interval", type=int, default=100)
    full_parser.add_argument("--filter-progress-interval", type=int, default=5000)
    full_parser.add_argument("--export-progress-interval", type=int, default=5000)
    full_parser.add_argument("--sample-progress-interval", type=int, default=5000)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    python_bin = default_python()

    if args.command == "prepare-pools":
        tasks = [
            TaskSpec(
                name="prepare-pools",
                description="Prepare normalized MMLU question and evaluation pools.",
                command=prepare_command(args, python_bin),
            )
        ]
    elif args.command == "optimize-mmlu":
        tasks = [
            TaskSpec(
                name="optimize-mmlu-prompt",
                description="Use GEPA to optimize the teacher system prompt.",
                command=optimize_command(args, python_bin),
            )
        ]
    elif args.command == "generate-mmlu":
        tasks = [
            TaskSpec(
                name="generate-mmlu",
                description="Generate teacher responses for the configured question pool.",
                command=generate_command(args, python_bin),
            )
        ]
    elif args.command == "export-sft":
        tasks = [
            TaskSpec(
                name="export-sft",
                description="Export filtered rows into an SFT-ready JSONL dataset.",
                command=export_command(args, python_bin),
            )
        ]
    elif args.command == "filter-mmlu":
        tasks = [
            TaskSpec(
                name="filter-correct",
                description="Filter the generated JSONL down to correct teacher answers.",
                command=filter_command(args, python_bin),
            )
        ]
    elif args.command == "sample-review":
        tasks = [
            TaskSpec(
                name="sample-review",
                description="Sample review rows from an input JSONL.",
                command=sample_command(args, python_bin),
            )
        ]
    elif args.command == "batch-100":
        tasks = batch_100_tasks(args, python_bin)
    elif args.command == "full-mmlu":
        tasks = full_mmlu_tasks(args, python_bin)
    else:
        raise ValueError(f"Unsupported scheduler command: {args.command}")

    run_tasks(tasks)


if __name__ == "__main__":
    main()
