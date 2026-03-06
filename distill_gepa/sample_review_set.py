from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import orjson

from .common import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a random review subset from a generated JSONL batch.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-interval", type=int, default=5000)
    return parser.parse_args()


def reservoir_sample_jsonl(
    path: Path,
    sample_size: int,
    seed: int,
    progress_interval: int,
) -> list[tuple[int, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {path}")

    rng = random.Random(seed)
    reservoir: list[tuple[int, dict[str, Any]]] = []
    seen = 0

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")

            seen += 1
            item = (seen - 1, payload)
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                replace_index = rng.randint(0, seen - 1)
                if replace_index < sample_size:
                    reservoir[replace_index] = item

            if seen % progress_interval == 0:
                print(f"progress={seen}", flush=True)

    if seen == 0:
        raise ValueError(f"No rows found in {path}")
    if seen < sample_size:
        raise ValueError(
            f"Requested sample_size={sample_size}, but only {seen} rows are available in {path}."
        )

    return sorted(reservoir, key=lambda item: item[0])


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    sampled = reservoir_sample_jsonl(args.input_path, args.sample_size, args.seed, args.progress_interval)

    sampled_rows: list[dict[str, Any]] = []
    for review_index, (source_index, payload) in enumerate(sampled, start=1):
        row = dict(payload)
        meta = dict(row.get("meta", {}))
        meta["review_sample_id"] = review_index
        meta["source_row_index"] = source_index
        row["meta"] = meta
        sampled_rows.append(row)

    write_jsonl(args.output_path, sampled_rows)
    print(f"sample_size={len(sampled_rows)}")
    print(f"seed={args.seed}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
