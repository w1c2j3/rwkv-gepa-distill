#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from distill_gepa.dataset_adapters import build_questions_from_source
from distill_gepa.dataset_toml import DatasetFolderConfig, load_dataset_config
from distill_gepa.world_schema import BenchmarkQuestion, iter_benchmark_questions, write_benchmark_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a benchmark question pool from config/datasets/<dataset-name>.toml."
    )
    parser.add_argument("--dataset-version", "--dataset-name", dest="dataset_version", default="mmlu_auxiliary_train")
    parser.add_argument("--dataset-config-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Final total number of questions to write.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def default_dataset_dir(dataset_version: str) -> Path:
    return Path("data") / dataset_version


def default_dataset_config_path(dataset_version: str) -> Path:
    return Path("config/datasets") / f"{dataset_version}.toml"


def _load_existing_questions(path: Path) -> list[BenchmarkQuestion]:
    return list(iter_benchmark_questions(path))


def _write_source_questions(path: Path, questions: list[BenchmarkQuestion]) -> int:
    return write_benchmark_questions(path, questions)


def prepare_from_dataset_config(
    *,
    dataset_config: DatasetFolderConfig,
    limit: int | None,
    force: bool,
    output_path: Path,
) -> None:
    dataset_config.root_dir.mkdir(parents=True, exist_ok=True)
    merged_questions: list[BenchmarkQuestion] = []
    source_counts: dict[str, int] = {}

    for source in dataset_config.sources:
        if not source.enabled:
            continue
        source_output_path = dataset_config.root_dir / source.output_file
        if source_output_path.exists() and source_output_path.stat().st_size > 0 and not force:
            source_questions = _load_existing_questions(source_output_path)
        else:
            source_questions = build_questions_from_source(dataset_config.root_dir, source, limit)
            _write_source_questions(source_output_path, source_questions)
        source_counts[source.name] = len(source_questions)
        if source.merge_into_world:
            merged_questions.extend(source_questions)

    if limit is not None:
        merged_questions = merged_questions[:limit]
    if not merged_questions:
        raise ValueError(f"No questions were prepared from {dataset_config.config_path}")

    rows_written = write_benchmark_questions(output_path, merged_questions)
    print(f"rows_written={rows_written}", flush=True)
    print(f"output_path={output_path}", flush=True)
    for source_name, source_count in sorted(source_counts.items()):
        print(f"source={source_name} rows={source_count}", flush=True)


def main() -> None:
    args = parse_args()
    dataset_dir = default_dataset_dir(args.dataset_version)
    dataset_config_path = args.dataset_config_path or default_dataset_config_path(args.dataset_version)
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {dataset_config_path}")

    dataset_config = load_dataset_config(dataset_config_path, dataset_dir)
    output_path = args.output_path or dataset_config.merge_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_from_dataset_config(
        dataset_config=dataset_config,
        limit=args.limit,
        force=args.force,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
