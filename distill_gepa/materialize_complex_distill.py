from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import orjson

from .trajectory_schema import (
    build_complex_trajectory_id,
    build_slot_id,
    build_slot_shuffle_key,
    parse_slot_id,
)
from .world_prompts import WORLD_SEED_SYSTEM_PROMPT
from .world_schema import BenchmarkQuestion, load_benchmark_question_map


GEPA_CHECKPOINT_FILENAME = "checkpoint.json"
SLOT_ALIGNED_MATERIALIZATION_MODE = "slot_aligned_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize slot-aligned complex distillation rows.")
    parser.add_argument("--question-path", type=Path, default=Path("data/world_knowledge/questions.jsonl"))
    parser.add_argument("--benchmark-path", type=Path, default=Path("data/world_knowledge/cache/benchmark_runs.jsonl"))
    parser.add_argument("--gepa-path", type=Path, default=Path("data/world_knowledge/gepa_results.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/complex_distill.jsonl"))
    return parser.parse_args()


def iter_jsonl(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield payload


def build_benchmark_complex_row(
    payload: dict[str, Any],
    questions_by_id: dict[str, BenchmarkQuestion],
) -> dict[str, Any] | None:
    question_id = payload.get("question_id")
    model_name = payload.get("model_name")
    sample_index = payload.get("sample_index")
    choice_permutation = payload.get("choice_permutation")
    correct = payload.get("correct")
    assistant = payload.get("assistant")
    if (
        not isinstance(question_id, str)
        or not isinstance(model_name, str)
        or not isinstance(sample_index, int)
        or not isinstance(choice_permutation, list)
        or not isinstance(correct, bool)
        or not isinstance(assistant, str)
        or not assistant.strip()
        or not correct
    ):
        return None

    question = questions_by_id.get(question_id)
    if question is None:
        return None

    prompted = question.prompted_variant_from_permutation(
        [int(index) for index in choice_permutation],
        shuffle_key=build_slot_shuffle_key(
            question_id=question_id,
            target_model=model_name,
            sample_index=sample_index,
        ),
    )

    slot_id = build_slot_id(question_id=question_id, target_model=model_name, sample_index=sample_index)

    return {
        "source_type": "benchmark_complex",
        "system": WORLD_SEED_SYSTEM_PROMPT,
        "user": prompted.user_message,
        "assistant": assistant,
        "meta": {
            "slot_id": slot_id,
            "trajectory_id": build_complex_trajectory_id(slot_id=slot_id),
        },
    }


def _load_checkpoint_evaluations(run_dir: str) -> list[dict[str, Any]] | None:
    checkpoint_path = Path(run_dir) / GEPA_CHECKPOINT_FILENAME
    if not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0:
        return None
    payload = orjson.loads(checkpoint_path.read_bytes())
    if not isinstance(payload, dict):
        return None
    if payload.get("materialization_mode") != SLOT_ALIGNED_MATERIALIZATION_MODE:
        return None
    evaluations = payload.get("optimized_evaluations")
    if not isinstance(evaluations, list):
        return None
    return [item for item in evaluations if isinstance(item, dict)]


def load_gepa_evaluations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    run_dir = payload.get("run_dir")
    if isinstance(run_dir, str) and run_dir.strip():
        loaded = _load_checkpoint_evaluations(run_dir)
        if loaded is not None:
            return loaded
    return []


def iter_gepa_complex_rows(path: Path):
    for payload in iter_jsonl(path) or []:
        target_model = payload.get("target_model")
        if not isinstance(target_model, str):
            continue

        for evaluation in load_gepa_evaluations(payload):
            question_id = evaluation.get("question_id")
            samples = evaluation.get("samples")
            if not isinstance(question_id, str) or not isinstance(samples, list):
                continue
            for sample in samples:
                if not isinstance(sample, dict):
                    continue
                sample_index = sample.get("sample_index")
                user_prompt = sample.get("user")
                assistant = sample.get("assistant")
                if (
                    sample.get("correct") is not True
                    or not isinstance(sample_index, int)
                    or not isinstance(user_prompt, str)
                    or not user_prompt.strip()
                    or not isinstance(assistant, str)
                    or not assistant.strip()
                ):
                    continue
                slot_id = build_slot_id(question_id=question_id, target_model=target_model, sample_index=sample_index)
                yield {
                    "source_type": "gepa_complex",
                    "system": WORLD_SEED_SYSTEM_PROMPT,
                    "user": user_prompt,
                    "assistant": assistant,
                    "meta": {
                        "slot_id": slot_id,
                        "trajectory_id": build_complex_trajectory_id(slot_id=slot_id),
                    },
                }


def row_sort_key(payload: dict[str, Any]) -> tuple[str, str, int]:
    meta = payload.get("meta", {})
    slot_id = meta.get("slot_id")
    if not isinstance(slot_id, str) or not slot_id:
        raise ValueError("Complex distill rows require meta.slot_id")
    slot = parse_slot_id(slot_id)
    return slot.question_id, slot.target_model, slot.sample_index


def select_complex_rows(
    benchmark_payloads: Iterable[dict[str, Any]],
    gepa_payloads: Iterable[dict[str, Any]],
    questions_by_id: dict[str, BenchmarkQuestion],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    selected_rows: dict[str, dict[str, Any]] = {}
    stats = {
        "benchmark_correct_slots": 0,
        "gepa_correct_slots": 0,
        "gepa_overwrite_slots": 0,
        "benchmark_rows_kept": 0,
        "gepa_rows_kept": 0,
    }

    for payload in benchmark_payloads:
        row = build_benchmark_complex_row(payload, questions_by_id)
        if row is None:
            continue
        stats["benchmark_correct_slots"] += 1
        slot_id = row["meta"]["slot_id"]
        selected_rows[slot_id] = row

    for row in gepa_payloads:
        stats["gepa_correct_slots"] += 1
        slot_id = row["meta"]["slot_id"]
        if slot_id in selected_rows:
            stats["gepa_overwrite_slots"] += 1
        selected_rows[slot_id] = row

    stats["benchmark_rows_kept"] = sum(1 for row in selected_rows.values() if row.get("source_type") == "benchmark_complex")
    stats["gepa_rows_kept"] = sum(1 for row in selected_rows.values() if row.get("source_type") == "gepa_complex")
    return selected_rows, stats


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    questions_by_id = load_benchmark_question_map(args.question_path)

    selected_rows, stats = select_complex_rows(
        benchmark_payloads=iter_jsonl(args.benchmark_path) or [],
        gepa_payloads=iter_gepa_complex_rows(args.gepa_path),
        questions_by_id=questions_by_id,
    )

    with args.output_path.open("wb") as handle:
        for row in sorted(selected_rows.values(), key=row_sort_key):
            handle.write(orjson.dumps(row))
            handle.write(b"\n")

    print(f"benchmark_correct_slots={stats['benchmark_correct_slots']}", flush=True)
    print(f"gepa_correct_slots={stats['gepa_correct_slots']}", flush=True)
    print(f"gepa_overwrite_slots={stats['gepa_overwrite_slots']}", flush=True)
    print(f"benchmark_rows_kept={stats['benchmark_rows_kept']}", flush=True)
    print(f"gepa_rows_kept={stats['gepa_rows_kept']}", flush=True)
    print(f"rows_written={len(selected_rows)}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
