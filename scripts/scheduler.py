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
DEFAULT_DATASET_VERSION = "world_knowledge"


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    command: list[str]


@dataclass(frozen=True)
class DatasetPaths:
    dataset_version: str
    dataset_dir: Path
    cache_dir: Path
    question_path: Path
    cache_path: Path
    benchmark_cache_path: Path
    decision_path: Path
    gepa_results_path: Path
    gepa_run_dir: Path
    rewrite_output_path: Path
    merged_sft_path: Path
    rwkv_output_path: Path
    summary_path: Path


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


def dataset_paths(dataset_version: str) -> DatasetPaths:
    dataset_dir = Path("data") / dataset_version
    cache_dir = dataset_dir / "cache"
    return DatasetPaths(
        dataset_version=dataset_version,
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        question_path=dataset_dir / "questions.jsonl",
        cache_path=cache_dir / "request_cache.sqlite",
        benchmark_cache_path=cache_dir / "benchmark_runs.jsonl",
        decision_path=dataset_dir / "question_decisions.jsonl",
        gepa_results_path=dataset_dir / "gepa_results.jsonl",
        gepa_run_dir=cache_dir / "gepa_run",
        rewrite_output_path=dataset_dir / "rewrite_distill.jsonl",
        merged_sft_path=dataset_dir / "distill_sft.jsonl",
        rwkv_output_path=dataset_dir / "distill_rwkv.jsonl",
        summary_path=dataset_dir / "pipeline_summary.json",
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


def add_dataset_version_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-version", "--dataset-name", dest="dataset_version", default=DEFAULT_DATASET_VERSION)


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")


def add_benchmark_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    add_config_arg(parser)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-model", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")


def add_classify_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    add_config_arg(parser)


def add_gepa_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    add_config_arg(parser)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-group-workers", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-metric-calls", type=int, default=60)
    parser.add_argument("--metric-samples", type=int, default=2)
    parser.add_argument("--materialization-samples", type=int, default=8)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fresh-run-dir", action="store_true")


def add_rewrite_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    add_config_arg(parser)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--rewrites-per-example", type=int, default=3)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")


def add_merge_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)


def add_export_rwkv_args(parser: argparse.ArgumentParser) -> None:
    add_dataset_version_arg(parser)
    parser.add_argument("--progress-interval", type=int, default=5000)


def add_full_args(parser: argparse.ArgumentParser) -> None:
    add_prepare_args(parser)
    add_config_arg(parser)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--samples-per-model", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-group-workers", type=int, default=2)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=60)
    parser.add_argument("--metric-samples", type=int, default=2)
    parser.add_argument("--materialization-samples", type=int, default=8)
    parser.add_argument("--rewrites-per-example", type=int, default=3)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fresh-run-dir", action="store_true")
    parser.add_argument("--rwkv-progress-interval", type=int, default=5000)


def resolve_cache_path(args: argparse.Namespace, paths: DatasetPaths) -> Path:
    cache_path = getattr(args, "cache_path", None)
    if cache_path is not None:
        return cache_path
    return paths.cache_path


def prepare_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    command = [
        python_bin,
        "-u",
        "scripts/prepare_world_knowledge_pools.py",
        "--dataset-version",
        args.dataset_version,
        "--output-path",
        str(paths.question_path),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if getattr(args, "force", False):
        command.append("--force")
    return command


def benchmark_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.run_world_benchmark",
        "--config-path",
        str(args.config_path),
        "--question-path",
        str(paths.question_path),
        "--output-path",
        str(paths.benchmark_cache_path),
        "--cache-path",
        str(resolve_cache_path(args, paths)),
        "--samples-per-model",
        str(args.samples_per_model),
        "--max-concurrency",
        str(args.max_concurrency),
        "--progress-interval",
        str(args.progress_interval),
        "--model-attempts",
        str(args.model_attempts),
    ]
    if getattr(args, "clear_cache", False):
        command.append("--clear-cache")
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    command.append("--resume" if args.resume else "--no-resume")
    return command


def classify_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.classify_world_questions",
        "--config-path",
        str(args.config_path),
        "--input-path",
        str(paths.benchmark_cache_path),
        "--output-path",
        str(paths.decision_path),
    ]


def gepa_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.optimize_world_prompts",
        "--config-path",
        str(args.config_path),
        "--decision-path",
        str(paths.decision_path),
        "--output-path",
        str(paths.gepa_results_path),
        "--run-dir",
        str(paths.gepa_run_dir),
        "--cache-path",
        str(resolve_cache_path(args, paths)),
        "--max-group-workers",
        str(args.max_group_workers),
        "--max-concurrency",
        str(args.max_concurrency),
        "--max-metric-calls",
        str(args.max_metric_calls),
        "--metric-samples",
        str(args.metric_samples),
        "--materialization-samples",
        str(args.materialization_samples),
        "--model-attempts",
        str(args.model_attempts),
    ]
    if getattr(args, "clear_cache", False):
        command.append("--clear-cache")
    if args.max_groups is not None:
        command.extend(["--max-groups", str(args.max_groups)])
    if args.fresh_run_dir:
        command.append("--fresh-run-dir")
    command.append("--resume" if args.resume else "--no-resume")
    return command


def rewrite_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    command = [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.rewrite_world_questions",
        "--config-path",
        str(args.config_path),
        "--input-path",
        str(paths.gepa_results_path),
        "--output-path",
        str(paths.rewrite_output_path),
        "--cache-path",
        str(resolve_cache_path(args, paths)),
        "--rewrites-per-example",
        str(args.rewrites_per_example),
        "--validation-samples",
        str(args.validation_samples),
        "--model-attempts",
        str(args.model_attempts),
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    if getattr(args, "clear_cache", False):
        command.append("--clear-cache")
    command.append("--resume" if args.resume else "--no-resume")
    return command


def merge_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.merge_world_distill",
        "--decision-path",
        str(paths.decision_path),
        "--gepa-path",
        str(paths.gepa_results_path),
        "--rewrite-path",
        str(paths.rewrite_output_path),
        "--output-path",
        str(paths.merged_sft_path),
        "--summary-path",
        str(paths.summary_path),
    ]


def export_rwkv_command(args: argparse.Namespace, python_bin: str) -> list[str]:
    paths = dataset_paths(args.dataset_version)
    return [
        python_bin,
        "-u",
        "-m",
        "distill_gepa.export_rwkv_dataset",
        "--input-path",
        str(paths.merged_sft_path),
        "--output-path",
        str(paths.rwkv_output_path),
        "--summary-path",
        str(paths.summary_path),
        "--progress-interval",
        str(args.progress_interval),
    ]


def full_tasks(args: argparse.Namespace, python_bin: str) -> list[TaskSpec]:
    prepare_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        limit=args.limit,
        force=args.force,
    )
    benchmark_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        config_path=args.config_path,
        cache_path=args.cache_path,
        clear_cache=args.clear_cache,
        limit=args.limit,
        samples_per_model=args.samples_per_model,
        max_concurrency=args.max_concurrency,
        model_attempts=args.model_attempts,
        progress_interval=args.progress_interval,
        resume=args.resume,
    )
    classify_args = argparse.Namespace(dataset_version=args.dataset_version, config_path=args.config_path)
    gepa_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        config_path=args.config_path,
        cache_path=args.cache_path,
        clear_cache=False,
        max_groups=args.max_groups,
        max_group_workers=args.max_group_workers,
        max_concurrency=args.max_concurrency,
        max_metric_calls=args.max_metric_calls,
        metric_samples=args.metric_samples,
        materialization_samples=args.materialization_samples,
        model_attempts=args.model_attempts,
        resume=args.resume,
        fresh_run_dir=args.fresh_run_dir,
    )
    rewrite_args = argparse.Namespace(
        dataset_version=args.dataset_version,
        config_path=args.config_path,
        cache_path=args.cache_path,
        clear_cache=False,
        rewrites_per_example=args.rewrites_per_example,
        validation_samples=args.validation_samples,
        model_attempts=args.model_attempts,
        max_concurrency=args.max_concurrency,
        resume=args.resume,
    )
    merge_args = argparse.Namespace(dataset_version=args.dataset_version)
    export_args = argparse.Namespace(dataset_version=args.dataset_version, progress_interval=args.rwkv_progress_interval)
    return [
        TaskSpec("prepare-world", "Prepare unified world-knowledge benchmark pools.", prepare_command(prepare_args, python_bin)),
        TaskSpec("run-benchmark", "Run four-model benchmark evaluation.", benchmark_command(benchmark_args, python_bin)),
        TaskSpec("classify-questions", "Aggregate benchmark samples into per-question decisions.", classify_command(classify_args, python_bin)),
        TaskSpec("optimize-gepa", "Optimize per-model same-domain prompts with GEPA and store grouped results.", gepa_command(gepa_args, python_bin)),
        TaskSpec("rewrite-questions", "Rewrite complex GEPA questions into simpler distillation prompts.", rewrite_command(rewrite_args, python_bin)),
        TaskSpec("merge-distill", "Merge direct and GEPA-derived distillation datasets.", merge_command(merge_args, python_bin)),
        TaskSpec("export-rwkv", "Export the merged SFT dataset into RWKV format.", export_rwkv_command(export_args, python_bin)),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="World-knowledge distillation scheduler.")
    subparsers = parser.add_subparsers(dest="command")

    prepare_parser = subparsers.add_parser("prepare-world", help="Prepare world benchmark pools.")
    add_prepare_args(prepare_parser)

    benchmark_parser = subparsers.add_parser("run-benchmark", help="Run the four-model world benchmark.")
    add_benchmark_args(benchmark_parser)

    classify_parser = subparsers.add_parser("classify-questions", help="Classify benchmark questions.")
    add_classify_args(classify_parser)

    gepa_parser = subparsers.add_parser("optimize-gepa", help="Run grouped GEPA prompt optimization.")
    add_gepa_args(gepa_parser)

    rewrite_parser = subparsers.add_parser("rewrite-questions", help="Rewrite GEPA complex questions.")
    add_rewrite_args(rewrite_parser)

    merge_parser = subparsers.add_parser("merge-distill", help="Merge all distillation datasets.")
    add_merge_args(merge_parser)

    export_parser = subparsers.add_parser("export-rwkv", help="Export merged SFT dataset as RWKV text jsonl.")
    add_export_rwkv_args(export_parser)

    full_parser = subparsers.add_parser("full-world", help="Run the full world-knowledge distillation pipeline.")
    add_full_args(full_parser)
    return parser


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = build_parser()
    if not argv:
        return parser.parse_args(["full-world"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    python_bin = default_python()

    if args.command == "prepare-world":
        tasks = [TaskSpec("prepare-world", "Prepare unified world-knowledge benchmark pools.", prepare_command(args, python_bin))]
    elif args.command == "run-benchmark":
        tasks = [TaskSpec("run-benchmark", "Run four-model world benchmark evaluation.", benchmark_command(args, python_bin))]
    elif args.command == "classify-questions":
        tasks = [TaskSpec("classify-questions", "Classify world benchmark questions.", classify_command(args, python_bin))]
    elif args.command == "optimize-gepa":
        tasks = [TaskSpec("optimize-gepa", "Run grouped GEPA prompt optimization.", gepa_command(args, python_bin))]
    elif args.command == "rewrite-questions":
        tasks = [TaskSpec("rewrite-questions", "Rewrite complex GEPA prompts into simple prompts.", rewrite_command(args, python_bin))]
    elif args.command == "merge-distill":
        tasks = [TaskSpec("merge-distill", "Merge direct and GEPA-derived distillation datasets.", merge_command(args, python_bin))]
    elif args.command == "export-rwkv":
        tasks = [TaskSpec("export-rwkv", "Export merged SFT as RWKV text.", export_rwkv_command(args, python_bin))]
    elif args.command == "full-world":
        tasks = full_tasks(args, python_bin)
    else:
        raise ValueError(f"Unsupported command: {args.command!r}")

    run_tasks(tasks)


if __name__ == "__main__":
    main()
