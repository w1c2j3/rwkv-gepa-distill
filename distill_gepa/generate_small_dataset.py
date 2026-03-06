from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import load_seed_examples, prompt_version, write_jsonl
from .teacher_client import TeacherClient


@dataclass(frozen=True)
class DistillRecord:
    instruction: str
    teacher_response: str
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "teacher_response": self.teacher_response,
            "meta": self.meta,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small distilled JSONL dataset.")
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/seeds/train.jsonl"),
        help="JSONL seed set used for small dataset generation.",
    )
    parser.add_argument(
        "--best-prompt-path",
        type=Path,
        default=Path("artifacts/best_prompt.txt"),
        help="Path to the optimized system prompt.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/distill_small.jsonl"),
        help="Where to write the distilled JSONL.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.best_prompt_path.exists():
        raise FileNotFoundError(
            f"Missing optimized prompt at {args.best_prompt_path}. Run optimize_prompt.py first."
        )

    best_prompt = args.best_prompt_path.read_text(encoding="utf-8").strip()
    if not best_prompt:
        raise ValueError(f"Optimized prompt file is empty: {args.best_prompt_path}")

    examples = load_seed_examples(args.train_path)
    teacher = TeacherClient.from_env()
    version = prompt_version(best_prompt)

    records: list[dict[str, Any]] = []
    for example in examples:
        response = teacher.generate(best_prompt, example.instruction, example.expected_keywords)
        record = DistillRecord(
            instruction=example.instruction,
            teacher_response=response.content,
            meta={
                "keywords": list(example.expected_keywords),
                "source": "teacher_api",
                "prompt_version": version,
            },
        )
        records.append(record.to_dict())

    write_jsonl(args.output_path, records)
    print(f"teacher_mode={teacher.mode}")
    print(f"examples_written={len(records)}")
    print(f"output_path={args.output_path}")


if __name__ == "__main__":
    main()
