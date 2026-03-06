from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter generated MMLU teacher rows to keep only correct answers.")
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
    return parser.parse_args()


def is_correct_row(payload: dict[str, Any], source: Path, line_number: int) -> bool:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{source}:{line_number} has invalid 'meta'")

    score = meta.get("score")
    if not isinstance(score, dict):
        raise ValueError(f"{source}:{line_number} has invalid 'meta.score'")

    correct = score.get("correct")
    if not isinstance(correct, bool):
        raise ValueError(f"{source}:{line_number} has invalid 'meta.score.correct'")
    return correct


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {args.input_path}")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    with args.input_path.open("rb") as input_handle, args.output_path.open("wb") as output_handle:
        for line_number, raw_line in enumerate(input_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {args.input_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{args.input_path}:{line_number} must be a JSON object")

            total_rows += 1
            if is_correct_row(payload, args.input_path, line_number):
                output_handle.write(orjson.dumps(payload))
                output_handle.write(b"\n")
                kept_rows += 1

            if total_rows % args.progress_interval == 0:
                print(f"progress={total_rows} kept={kept_rows}", flush=True)

    if total_rows == 0:
        raise ValueError(f"No rows found in {args.input_path}")

    print(f"total_rows={total_rows}", flush=True)
    print(f"kept_rows={kept_rows}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
