from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import orjson

from .common import write_json
from .mmlu_score import score_mcq_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostics-only: filter generated MMLU teacher rows to keep only strict JSON rows usable for SFT."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_strict.jsonl"),
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("artifacts/mmlu_batch_all_filter_stats.json"),
    )
    parser.add_argument("--progress-interval", type=int, default=5000)
    return parser.parse_args()


def recompute_score(payload: dict[str, Any], source: Path, line_number: int) -> dict[str, Any]:
    teacher_response = payload.get("teacher_response")
    source_question = payload.get("source_question")
    if not isinstance(teacher_response, str):
        raise ValueError(f"{source}:{line_number} has invalid 'teacher_response'")
    if not isinstance(source_question, dict):
        raise ValueError(f"{source}:{line_number} has invalid 'source_question'")

    choices = source_question.get("choices")
    gold_answer_index = source_question.get("answer_index")
    pool_subject = source_question.get("subject", "")
    if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
        raise ValueError(f"{source}:{line_number} has invalid 'source_question.choices'")
    if gold_answer_index is not None and not isinstance(gold_answer_index, int):
        raise ValueError(f"{source}:{line_number} has invalid 'source_question.answer_index'")
    if not isinstance(pool_subject, str):
        pool_subject = ""

    return score_mcq_response(
        teacher_response,
        choices,
        gold_answer_index,
        pool_subject=pool_subject,
    ).to_dict()


def rejection_reason(score: dict[str, Any]) -> str:
    if score.get("usable_for_sft") is True:
        return "usable_for_sft"
    if score.get("valid_json") is not True:
        return "invalid_json"
    if score.get("answer_present") is not True:
        return "missing_answer"
    if score.get("answer_consistent") is not True:
        return "inconsistent_answer"
    if score.get("exact_answer_text_match") is not True:
        return "answer_text_mismatch"
    if score.get("reasoning_present") is not True:
        return "missing_reasoning"
    if score.get("correct") is not True:
        return "wrong_answer"
    return "other_rejected"


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {args.input_path}")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    rejection_counts: Counter[str] = Counter()
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
            score = recompute_score(payload, args.input_path, line_number)
            reason = rejection_reason(score)
            rejection_counts[reason] += 1

            if reason == "usable_for_sft":
                row = dict(payload)
                meta = dict(row.get("meta", {}))
                row["teacher_parsed"] = score.get("parsed", row.get("teacher_parsed"))
                meta["score"] = score
                meta["filter_reason"] = reason
                meta["output_contract"] = "strict_json_mcq_v1"
                row["meta"] = meta
                output_handle.write(orjson.dumps(row))
                output_handle.write(b"\n")
                kept_rows += 1

            if total_rows % args.progress_interval == 0:
                print(f"progress={total_rows} kept={kept_rows}", flush=True)

    if total_rows == 0:
        raise ValueError(f"No rows found in {args.input_path}")

    write_json(
        args.stats_path,
        {
            "input_path": str(args.input_path),
            "output_path": str(args.output_path),
            "stats_path": str(args.stats_path),
            "total_rows": total_rows,
            "kept_rows": kept_rows,
            "rejected_rows": total_rows - kept_rows,
            "reason_counts": dict(sorted(rejection_counts.items())),
        },
    )
    print(f"total_rows={total_rows}", flush=True)
    print(f"kept_rows={kept_rows}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"stats_path={args.stats_path}", flush=True)


if __name__ == "__main__":
    main()
