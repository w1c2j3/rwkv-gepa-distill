from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export filtered MMLU teacher rows into a compact SFT-ready JSONL dataset."
    )
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
    return parser.parse_args()


def build_sft_row(payload: dict[str, Any]) -> dict[str, Any]:
    system_prompt = payload.get("system_prompt")
    instruction = payload.get("instruction")
    teacher_response = payload.get("teacher_response")
    meta = payload.get("meta")
    source_question = payload.get("source_question")

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Missing non-empty 'system_prompt'")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Missing non-empty 'instruction'")
    if not isinstance(teacher_response, str) or not teacher_response.strip():
        raise ValueError("Missing non-empty 'teacher_response'")
    if not isinstance(meta, dict):
        raise ValueError("Missing object 'meta'")
    if not isinstance(source_question, dict):
        raise ValueError("Missing object 'source_question'")

    source_meta = source_question.get("meta")
    if not isinstance(source_meta, dict):
        source_meta = {}

    return {
        "system": system_prompt,
        "user": instruction,
        "assistant": teacher_response,
        "meta": {
            "prompt_version": meta.get("prompt_version"),
            "teacher_model": meta.get("teacher_model"),
            "source_dataset": source_question.get("source_dataset"),
            "source_split": source_question.get("source_split"),
            "subject": source_question.get("subject"),
            "answer_label": source_question.get("answer_label"),
            "answer_index": source_question.get("answer_index"),
            "source_subject_config": source_meta.get("source_subject_config"),
        },
    }


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {args.input_path}")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
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

            sft_row = build_sft_row(payload)
            output_handle.write(orjson.dumps(sft_row))
            output_handle.write(b"\n")
            total_rows += 1

            if total_rows % args.progress_interval == 0:
                print(f"progress={total_rows}", flush=True)

    if total_rows == 0:
        raise ValueError(f"No rows found in {args.input_path}")

    print(f"rows_written={total_rows}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
