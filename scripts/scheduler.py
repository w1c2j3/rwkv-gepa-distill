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
KNOWN_COMMANDS = {
    "prepare-pools",
    "optimize-mmlu",
    "generate-sft",
    "generate-mmlu",
    "filter-mmlu",
    "export-sft",
    "export-rwkv",
    "sample-review",
    "full-mmlu",
}

DEFAULT_DATASET_VERSION = "v1"
DEFAULT_TRAIN_PATH = Path("data/question_pools/mmlu_dev.jsonl")
DEFAULT_QUESTION_POOL_PATH = Path("data/question_pools/mmlu_auxiliary_train.jsonl")
DEFAULT_GEPA_RUN_DIR = Path("artifacts/mmlu_gepa_run")
DEFAULT_REPORT_PATH = Path("artifacts/mmlu_prompt_optimization_report.json")
DEFAULT_RAW_OUTPUT_PATH = Path("data/distill/mmlu_batch_all.jsonl")
DEFAULT_FAILED_OUTPUT_PATH = Path("artifacts/mmlu_batch_all_failed.jsonl")
DEFAULT_FILTERED_OUTPUT_PATH = Path("data/distill/mmlu_batch_all_strict.jsonl")
DEFAULT_FILTER_STATS_PATH = Path("artifacts/mmlu_batch_all_filter_stats.json")
DEFAULT_SFT_OUTPUT_PATH = Path("data/distill/mmlu_batch_all_sft.jsonl")
DEFAULT_RWKV_OUTPUT_PATH = Path("data/distill/mmlu_batch_all_rwkv.jsonl")
DEFAULT_REVIEW_OUTPUT_PATH = Path("artifacts/mmlu_batch_all_review_40.jsonl")
DEFAULT_MAX_METRIC_CALLS = 40
DEFAULT_LIMIT = 99842
DEFAULT_MAX_CONCURRENCY = 6
DEFAULT_GENERATE_PROGRESS = 100
DEFAULT_STAGE_PROGRESS = 5000
DEFAULT_SAMPLE_SIZE = 40
DEFAULT_SAMPLE_SEED = 42


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    command: list[str]


@dataclass(frozen=True)
class DatasetPaths:
    dataset_version: str
    gepa_run_dir: Path
    prompt_bundle_path: Path
    report_path: Path
    raw_output_path: Path
    failed_output_path: Path
    filtered_output_path: Path
    filter_stats_path: Path
    sft_output_path: Path
    rwkv_output_path: Path
    review_output_path: Path


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


def normalize_dataset_version(dataset_version: str) -> str:
    normalized = dataset_version.strip()
    if not normalized:
        raise ValueError("--dataset-version must be non-empty")
    return normalized


def dataset_paths(dataset_version: str) -> DatasetPaths:
    normalized = normalize_dataset_version(dataset_version)
    artifact_dir = Path("artifacts") / normalized
    distill_dir = Path("data/distill") / normalized
    return DatasetPaths(
        dataset_version=normalized,
        gepa_run_dir=artifact_dir / "mmlu_gepa_run",
        prompt_bundle_path=artifact_dir / "mmlu_prompt_bundle.jsonl",
        report_path=artifact_dir / "mmlu_prompt_optimization_report.json",
        raw_output_path=distill_dir / "mmlu_batch_all.jsonl",
        failed_output_path=artifact_dir / "mmlu_batch_all_failed.jsonl",
        filtered_output_path=distill_dir / "mmlu_batch_all_strict.jsonl",
        filter_stats_path=artifact_dir / "mmlu_batch_all_filter_stats.json",
        sft_output_path=distill_dir / "mmlu_batch_all_sft.jsonl",
        rwkv_output_path=distill_dir / "mmlu_batch_all_rwkv.jsonl",
        review_output_path=artifact_dir / "mmlu_batch_all_review_40.jsonl",
    )


def run_tasks(tasks: list[TaskSpec]) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    started = time.monotonic()
    total = len(tasks)
    for index, task in enumerate(tasks, start=1):
        print(f"[{index}/{total}] {task.name}", flush=True)
        print(f"desc={task.description}", flush=True)
        print(f"cmd={command_display(task.command)}", flush=True)
        task_started = time.monotonic()
        subprocess.run(task.command, cwd=ROOT_DIR, env=env, check=True)
        print(
            f"status=completed task={task.name} elapsed={format_duration(time.monotonic() - task_started)}",
            flush=True,
        )

    print(f"pipeline_status=completed elapsed={format_duration(time.monotonic() - started)}", flush=True)


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Rebuild primary pools even if cached JSONL exists.")
    parser.add_argument("--include-community", action="store_true", help="Also build optional community pools.")
    parser.add_argument("--only", choices=("primary", "eval", "all"), default="primary")
    parser.add_argument("--limit", type=int, default=None)


def add_dataset_version_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-version", default=DEFAULT_DATASET_VERSION)


def add_optimize_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--question-pool-path", type=Path, default=DEFAULT_QUESTION_POOL_PATH)
    parser.add_argument("--train-limit", type=int, default=10)
    parser.add_argument("--target-limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-metric-calls", type=int, default=DEFAULT_MAX_METRIC_CALLS)
    parser.add_argument("--gepa-run-dir", type=Path, default=None)
    parser.add_argument("--prompt-bundle-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--fresh-run-dir", action="store_true", default=True)
    parser.add_argument("--no-fresh-run-dir", dest="fresh_run_dir", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_GENERATE_PROGRESS)


def add_generate_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--question-pool-path", type=Path, default=DEFAULT_QUESTION_POOL_PATH)
    parser.add_argument("--prompt-bundle-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--raw-output-path", type=Path, default=None)
    parser.add_argument("--strict-output-path", type=Path, default=None)
    parser.add_argument("--failed-output-path", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_GENERATE_PROGRESS)
    parser.add_argument("--retry-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")


def add_filter_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--stats-path", type=Path, default=None)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_STAGE_PROGRESS)


def add_export_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_STAGE_PROGRESS)


def add_export_rwkv_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_STAGE_PROGRESS)


def add_sample_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--input-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_STAGE_PROGRESS)


def add_full_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--force-pools", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--progress-interval", type=int, default=DEFAULT_GENERATE_PROGRESS)
    parser.add_argument("--rwkv-progress-interval", type=int, default=DEFAULT_STAGE_PROGRESS)
    parser.add_argument("--sample-progress-interval", type=int, default=DEFAULT_STAGE_PROGRESS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--retry-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")


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
    paths = dataset_paths(args.dataset_version)
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.optimize_mmlu_prompt",
        "--train-path",
        str(args.train_path),
        "--question-pool-path",
        str(args.question_pool_path),
        "--train-limit",
        str(args.train_limit),
        "--target-limit",
        str(args.target_limit),
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--gepa-run-dir",
        str(args.gepa_run_dir or paths.gepa_run_dir),
        "--prompt-bundle-path",
        str(args.prompt_bundle_path or paths.prompt_bundle_path),
        "--report-path",
        str(args.report_path or paths.report_path),
        "--progress-interval",
        str(args.progress_interval),
    ]
    if args.fresh_run_dir:
        command.append("--fresh-run-dir")
    if args.diagnostics:
        command.append("--diagnostics")
    if args.resume:
        command.append("--resume")
    return command


def generate_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.generate_mmlu_batch",
        "--question-pool-path",
        str(args.question_pool_path),
        "--prompt-bundle-path",
        str(args.prompt_bundle_path or paths.prompt_bundle_path),
        "--output-path",
        str(args.output_path or paths.sft_output_path),
        "--limit",
        str(args.limit),
        "--max-concurrency",
        str(args.max_concurrency),
        "--progress-interval",
        str(args.progress_interval),
        "--retry-rounds",
        str(args.retry_rounds),
    ]
    if args.diagnostics:
        command.extend(
            [
                "--diagnostics",
                "--raw-output-path",
                str(args.raw_output_path or paths.raw_output_path),
                "--strict-output-path",
                str(args.strict_output_path or paths.filtered_output_path),
                "--failed-output-path",
                str(args.failed_output_path or paths.failed_output_path),
                "--stats-path",
                str(args.stats_path or paths.filter_stats_path),
            ]
        )
    if args.resume:
        command.append("--resume")
    return command


def filter_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.filter_mmlu_batch",
        "--input-path",
        str(args.input_path or paths.raw_output_path),
        "--output-path",
        str(args.output_path or paths.filtered_output_path),
        "--stats-path",
        str(args.stats_path or paths.filter_stats_path),
        "--progress-interval",
        str(args.progress_interval),
    ]


def export_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.export_sft_dataset",
        "--input-path",
        str(args.input_path or paths.filtered_output_path),
        "--output-path",
        str(args.output_path or paths.sft_output_path),
        "--progress-interval",
        str(args.progress_interval),
    ]


def export_rwkv_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.export_rwkv_dataset",
        "--input-path",
        str(args.input_path or paths.sft_output_path),
        "--output-path",
        str(args.output_path or paths.rwkv_output_path),
        "--progress-interval",
        str(args.progress_interval),
    ]


def sample_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.sample_review_set",
        "--input-path",
        str(args.input_path or paths.sft_output_path),
        "--output-path",
        str(args.output_path or paths.review_output_path),
        "--sample-size",
        str(args.sample_size),
        "--seed",
        str(args.seed),
        "--progress-interval",
        str(args.progress_interval),
    ]


def build_full_tasks(args: argparse.Namespace, python_bin: str) -> list[TaskSpec]:
    paths = dataset_paths(args.dataset_version)
    prepare_args = argparse.Namespace(force=args.force_pools, include_community=False, only="primary", limit=None)
    optimize_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        train_path=DEFAULT_TRAIN_PATH,
        question_pool_path=DEFAULT_QUESTION_POOL_PATH,
        train_limit=10,
        target_limit=args.limit,
        max_metric_calls=DEFAULT_MAX_METRIC_CALLS,
        gepa_run_dir=paths.gepa_run_dir,
        prompt_bundle_path=paths.prompt_bundle_path,
        report_path=paths.report_path,
        diagnostics=args.diagnostics,
        fresh_run_dir=True,
        resume=args.resume,
        progress_interval=args.progress_interval,
    )
    generate_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        question_pool_path=DEFAULT_QUESTION_POOL_PATH,
        prompt_bundle_path=paths.prompt_bundle_path,
        output_path=paths.sft_output_path,
        diagnostics=args.diagnostics,
        raw_output_path=paths.raw_output_path,
        strict_output_path=paths.filtered_output_path,
        failed_output_path=paths.failed_output_path,
        stats_path=paths.filter_stats_path,
        limit=args.limit,
        max_concurrency=args.max_concurrency,
        progress_interval=args.progress_interval,
        retry_rounds=args.retry_rounds,
        resume=args.resume,
    )
    export_rwkv_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        input_path=paths.sft_output_path,
        output_path=paths.rwkv_output_path,
        progress_interval=args.rwkv_progress_interval,
    )
    sample_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        input_path=paths.sft_output_path,
        output_path=paths.review_output_path,
        sample_size=args.sample_size,
        seed=DEFAULT_SAMPLE_SEED,
        progress_interval=args.sample_progress_interval,
    )
    tasks = [
        TaskSpec("prepare-pools", "Check or build the primary MMLU question pools.", prepare_command(prepare_args, python_bin)),
        TaskSpec("optimize-mmlu", "Use GEPA to optimize a per-row teacher prompt bundle.", optimize_command(optimize_args, python_bin)),
        TaskSpec("generate-sft", "Generate teacher responses and write usable rows directly to the SFT JSONL dataset.", generate_command(generate_args, python_bin)),
        TaskSpec("export-rwkv", "Export the strict SFT rows into the final RWKV training JSONL dataset.", export_rwkv_command(export_rwkv_args, python_bin)),
    ]
    if args.diagnostics:
        tasks.append(TaskSpec("sample-review", "Draw a random review sample from the SFT dataset.", sample_command(sample_args, python_bin)))
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified scheduler for the rwkv-gepa-distill MMLU pipeline.")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare-pools", help="Prepare question pools.")
    add_prepare_args(prepare_parser)

    optimize_parser = subparsers.add_parser("optimize-mmlu", help="Optimize a per-row teacher prompt bundle with GEPA.")
    add_optimize_args(optimize_parser)

    generate_parser = subparsers.add_parser(
        "generate-sft",
        aliases=["generate-mmlu"],
        help="Generate teacher answers from the prompt bundle and write usable rows directly to SFT.",
    )
    add_generate_args(generate_parser)

    filter_parser = subparsers.add_parser("filter-mmlu", help="Diagnostics-only: filter raw rows to strict JSON rows usable for SFT.")
    add_filter_args(filter_parser)

    export_parser = subparsers.add_parser("export-sft", help="Diagnostics-only: export filtered rows to SFT JSONL.")
    add_export_args(export_parser)

    export_rwkv_parser = subparsers.add_parser("export-rwkv", help="Export SFT rows to RWKV JSONL.")
    add_export_rwkv_args(export_rwkv_parser)

    sample_parser = subparsers.add_parser("sample-review", help="Diagnostics-only: sample a review subset from the SFT dataset.")
    add_sample_args(sample_parser)

    full_parser = subparsers.add_parser("full-mmlu", help="Run the complete default MMLU pipeline.")
    add_full_args(full_parser)

    return parser


def parse_scheduler_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    argv = sys.argv[1:]
    if not argv:
        return parser.parse_args(["full-mmlu"])
    if argv[0] in {"-h", "--help"}:
        return parser.parse_args(argv)
    if argv[0] in KNOWN_COMMANDS:
        return parser.parse_args(argv)
    if argv[0].startswith("-"):
        return parser.parse_args(["full-mmlu", *argv])
    return parser.parse_args(argv)


def main() -> None:
    parser = build_parser()
    args = parse_scheduler_args(parser)
    python_bin = default_python()

    if args.command == "prepare-pools":
        tasks = [TaskSpec("prepare-pools", "Prepare normalized MMLU pools.", prepare_command(args, python_bin))]
    elif args.command == "optimize-mmlu":
        tasks = [TaskSpec("optimize-mmlu", "Use GEPA to optimize a per-row teacher prompt bundle.", optimize_command(args, python_bin))]
    elif args.command in {"generate-mmlu", "generate-sft"}:
        tasks = [
            TaskSpec(
                "generate-sft",
                "Generate teacher responses with the prompt bundle and write usable rows directly to SFT.",
                generate_command(args, python_bin),
            )
        ]
    elif args.command == "filter-mmlu":
        tasks = [TaskSpec("filter-mmlu", "Diagnostics-only: filter raw rows to strict JSON rows usable for SFT.", filter_command(args, python_bin))]
    elif args.command == "export-sft":
        tasks = [TaskSpec("export-sft", "Diagnostics-only: export filtered rows into the strict SFT dataset.", export_command(args, python_bin))]
    elif args.command == "export-rwkv":
        tasks = [TaskSpec("export-rwkv", "Export strict SFT rows into the final RWKV dataset.", export_rwkv_command(args, python_bin))]
    elif args.command == "sample-review":
        tasks = [TaskSpec("sample-review", "Sample review rows from the SFT dataset.", sample_command(args, python_bin))]
    elif args.command == "full-mmlu":
        tasks = build_full_tasks(args, python_bin)
    else:
        raise ValueError(f"Unsupported scheduler command: {args.command}")

    run_tasks(tasks)


if __name__ == "__main__":
    main()
