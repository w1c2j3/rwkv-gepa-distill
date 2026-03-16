from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset


REQUIRED_FIELDS = ("question", "subject", "choices", "answer")
VALID_ANSWERS = {"A", "B", "C", "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export mmlu_variants.jsonl to parquet with question/subject/choices/answer only."
    )
    parser.add_argument(
        "--input",
        default="data/mmlu_variants.jsonl",
        help="Path to the input variants JSONL file (default: data/mmlu_variants.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/mmlu_variants.parquet",
        help="Path to the output parquet file (default: data/mmlu_variants.parquet)",
    )
    return parser.parse_args()


def validate_record(record: dict[str, Any], line_number: int) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Line {line_number}: missing required fields: {', '.join(missing)}")

    question = record["question"]
    subject = record["subject"]
    choices = record["choices"]
    answer = record["answer"]

    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Line {line_number}: question must be a non-empty string")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError(f"Line {line_number}: subject must be a non-empty string")
    if not isinstance(choices, list) or len(choices) != 4:
        raise ValueError(f"Line {line_number}: choices must be a list of exactly 4 items")
    if any(not isinstance(choice, str) or not choice.strip() for choice in choices):
        raise ValueError(f"Line {line_number}: every choice must be a non-empty string")

    normalized_answer = answer.strip().upper() if isinstance(answer, str) else str(answer).strip().upper()
    if normalized_answer not in VALID_ANSWERS:
        raise ValueError(f"Line {line_number}: answer must be one of A/B/C/D")

    return {
        "question": question.strip(),
        "subject": subject.strip(),
        "choices": [choice.strip() for choice in choices],
        "answer": normalized_answer,
    }


def load_rows(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number}: invalid JSON: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object")
            rows.append(validate_record(payload, line_number))
    return rows


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    try:
        rows = load_rows(input_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset = Dataset.from_list(rows)
        dataset.to_parquet(str(output_path))
        print(f"Exported {len(rows)} rows to {output_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to export parquet: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
