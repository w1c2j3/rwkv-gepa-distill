from __future__ import annotations

import argparse
from pathlib import Path

from .dataset_adapters import build_questions_from_source
from .dataset_config import DatasetFolderConfig, load_dataset_config
from .task_schema import TaskItem, load_task_items, write_task_items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a task pool from config/datasets/<dataset-name>.toml or an explicit config path."
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Final total number of tasks to write.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def default_dataset_dir(dataset_name: str) -> Path:
    return Path("data") / dataset_name


def default_dataset_config_path(dataset_name: str) -> Path:
    return Path("config/datasets") / f"{dataset_name}.toml"


def resolve_dataset_inputs(args: argparse.Namespace) -> tuple[DatasetFolderConfig, Path]:
    if args.dataset_config_path is not None:
        dataset_config_path = args.dataset_config_path
        dataset_name = args.dataset_name or dataset_config_path.stem
    else:
        if not args.dataset_name:
            raise ValueError("Provide either --dataset-name or --dataset-config-path.")
        dataset_name = args.dataset_name
        dataset_config_path = default_dataset_config_path(dataset_name)

    dataset_dir = default_dataset_dir(dataset_name)
    dataset_config = load_dataset_config(dataset_config_path, dataset_dir)
    return dataset_config, (args.output_path or dataset_config.merge_output_path)


def _load_existing_tasks(path: Path) -> list[TaskItem]:
    return load_task_items(path)


def prepare_from_dataset_config(
    *,
    dataset_config: DatasetFolderConfig,
    limit: int | None,
    force: bool,
    output_path: Path,
) -> None:
    dataset_config.root_dir.mkdir(parents=True, exist_ok=True)
    merged_tasks: list[TaskItem] = []
    source_counts: dict[str, int] = {}

    for source in dataset_config.sources:
        if not source.enabled:
            continue
        source_output_path = dataset_config.root_dir / source.output_file
        if source_output_path.exists() and source_output_path.stat().st_size > 0 and not force:
            source_tasks = _load_existing_tasks(source_output_path)
        else:
            source_tasks = build_questions_from_source(dataset_config.root_dir, source, limit)
            write_task_items(source_output_path, source_tasks)
        source_counts[source.name] = len(source_tasks)
        if source.merge_into_world:
            merged_tasks.extend(source_tasks)

    if limit is not None:
        merged_tasks = merged_tasks[:limit]
    if not merged_tasks:
        raise ValueError(f"No tasks were prepared from {dataset_config.config_path}")

    rows_written = write_task_items(output_path, merged_tasks)
    print(f"rows_written={rows_written}", flush=True)
    print(f"output_path={output_path}", flush=True)
    for source_name, source_count in sorted(source_counts.items()):
        print(f"source={source_name} rows={source_count}", flush=True)


def main() -> None:
    args = parse_args()
    dataset_config, output_path = resolve_dataset_inputs(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepare_from_dataset_config(
        dataset_config=dataset_config,
        limit=args.limit,
        force=args.force,
        output_path=output_path,
    )


__all__ = [
    "default_dataset_config_path",
    "default_dataset_dir",
    "main",
    "parse_args",
    "prepare_from_dataset_config",
    "resolve_dataset_inputs",
]


if __name__ == "__main__":
    main()
