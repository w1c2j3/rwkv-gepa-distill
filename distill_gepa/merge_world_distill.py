from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from .common import write_json
from .trajectory_schema import parse_slot_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge trajectory-first complex and rewrite distillation datasets.")
    parser.add_argument("--decision-path", type=Path, default=Path("data/world_knowledge/question_decisions.jsonl"))
    parser.add_argument("--benchmark-path", type=Path, default=Path("data/world_knowledge/cache/benchmark_runs.jsonl"))
    parser.add_argument("--gepa-path", type=Path, default=Path("data/world_knowledge/gepa_results.jsonl"))
    parser.add_argument("--complex-path", type=Path, default=Path("data/world_knowledge/complex_distill.jsonl"))
    parser.add_argument("--rewrite-path", type=Path, default=Path("data/world_knowledge/rewrite_distill.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/distill_sft.jsonl"))
    parser.add_argument("--summary-path", type=Path, default=Path("data/world_knowledge/pipeline_summary.json"))
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


def row_key(payload: dict[str, Any]) -> str:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Merged rows require 'meta'")
    trajectory_id = meta.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("Merged rows require meta.trajectory_id")
    return trajectory_id


def build_summary(
    *,
    decision_path: Path,
    benchmark_path: Path,
    gepa_path: Path,
    complex_path: Path,
    rewrite_path: Path,
    output_path: Path,
    counts: dict[str, int],
) -> dict[str, Any]:
    decision_counts = {"direct_distill": 0, "needs_optimization": 0, "suspected_anomaly": 0}
    question_count = 0
    for payload in iter_jsonl(decision_path) or []:
        classification = payload.get("classification")
        if isinstance(classification, str) and classification in decision_counts:
            decision_counts[classification] += 1
            question_count += 1

    benchmark_record_count = 0
    benchmark_correct_record_count = 0
    for payload in iter_jsonl(benchmark_path) or []:
        benchmark_record_count += 1
        if payload.get("correct") is True:
            benchmark_correct_record_count += 1

    gepa_group_count = 0
    gepa_group_success_count = 0
    gepa_improved_question_count = 0
    gepa_question_model_pair_count = 0
    for payload in iter_jsonl(gepa_path) or []:
        gepa_group_count += 1
        improved_question_count = payload.get("improved_question_count")
        question_deltas = payload.get("question_deltas")
        if isinstance(improved_question_count, int):
            gepa_improved_question_count += improved_question_count
            gepa_group_success_count += int(improved_question_count > 0)
        if isinstance(question_deltas, list):
            gepa_question_model_pair_count += sum(1 for item in question_deltas if isinstance(item, dict))

    complex_source_counts: dict[str, int] = defaultdict(int)
    per_question_complex_counts: dict[str, int] = defaultdict(int)
    for payload in iter_jsonl(complex_path) or []:
        source_type = payload.get("source_type")
        meta = payload.get("meta")
        slot_id = meta.get("slot_id") if isinstance(meta, dict) else None
        question_id = None
        if isinstance(slot_id, str) and slot_id:
            question_id = parse_slot_id(slot_id).question_id
        if isinstance(source_type, str):
            complex_source_counts[source_type] += 1
        if isinstance(question_id, str):
            per_question_complex_counts[question_id] += 1

    rewrite_source_counts: dict[str, int] = defaultdict(int)
    per_question_rewrite_counts: dict[str, int] = defaultdict(int)
    for payload in iter_jsonl(rewrite_path) or []:
        source_type = payload.get("source_type")
        meta = payload.get("meta")
        slot_id = meta.get("slot_id") if isinstance(meta, dict) else None
        question_id = None
        if isinstance(slot_id, str) and slot_id:
            question_id = parse_slot_id(slot_id).question_id
        if isinstance(source_type, str):
            rewrite_source_counts[source_type] += 1
        if isinstance(question_id, str):
            per_question_rewrite_counts[question_id] += 1

    per_question_final_counts: dict[str, int] = defaultdict(int)
    for question_id, count in per_question_complex_counts.items():
        per_question_final_counts[question_id] += count
    for question_id, count in per_question_rewrite_counts.items():
        per_question_final_counts[question_id] += count

    all_question_ids = set(per_question_complex_counts) | set(per_question_rewrite_counts)
    per_question_complex_row_max = max(per_question_complex_counts.values(), default=0)
    per_question_rewrite_row_max = max(per_question_rewrite_counts.values(), default=0)
    per_question_final_row_max = max(per_question_final_counts.values(), default=0)

    return {
        "dataset_name": output_path.parent.name,
        "question_count": question_count,
        "decision_counts": decision_counts,
        "benchmark_record_count": benchmark_record_count,
        "benchmark_correct_record_count": benchmark_correct_record_count,
        "gepa_group_count": gepa_group_count,
        "gepa_group_success_count": gepa_group_success_count,
        "gepa_question_model_pair_count": gepa_question_model_pair_count,
        "gepa_improved_question_count": gepa_improved_question_count,
        "gepa_success_rate": round(
            gepa_improved_question_count / gepa_question_model_pair_count,
            6,
        )
        if gepa_question_model_pair_count
        else 0.0,
        "complex_row_count": sum(complex_source_counts.values()),
        "complex_source_counts": dict(sorted(complex_source_counts.items())),
        "rewrite_row_count": sum(rewrite_source_counts.values()),
        "rewrite_source_counts": dict(sorted(rewrite_source_counts.items())),
        "final_sft_row_count": sum(counts.values()),
        "sft_counts": counts,
        "per_question_complex_row_max": per_question_complex_row_max,
        "per_question_rewrite_row_max": per_question_rewrite_row_max,
        "per_question_final_row_max": per_question_final_row_max,
        "questions_hit_complex_cap_32": sum(1 for count in per_question_complex_counts.values() if count >= 32),
        "questions_hit_final_cap_128": sum(1 for count in per_question_final_counts.values() if count >= 128),
        "questions_with_any_training_rows": sum(1 for question_id in all_question_ids if per_question_final_counts[question_id] > 0),
    }


def load_existing_summary(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    payload = orjson.loads(path.read_bytes())
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = defaultdict(int)
    seen: set[str] = set()

    with args.output_path.open("wb") as output_handle:
        for payload in iter_jsonl(args.complex_path) or []:
            key = row_key(payload)
            if key in seen:
                continue
            seen.add(key)
            output_handle.write(orjson.dumps(payload))
            output_handle.write(b"\n")
            counts[str(payload.get("source_type") or "unknown")] += 1

        for payload in iter_jsonl(args.rewrite_path) or []:
            key = row_key(payload)
            if key in seen:
                continue
            seen.add(key)
            output_handle.write(orjson.dumps(payload))
            output_handle.write(b"\n")
            counts[str(payload.get("source_type") or "unknown")] += 1

    computed_summary = build_summary(
        decision_path=args.decision_path,
        benchmark_path=args.benchmark_path,
        gepa_path=args.gepa_path,
        complex_path=args.complex_path,
        rewrite_path=args.rewrite_path,
        output_path=args.output_path,
        counts=dict(sorted(counts.items())),
    )
    summary = load_existing_summary(args.summary_path)
    benchmark_storage = summary.get("benchmark_storage")
    summary.update(computed_summary)
    if isinstance(benchmark_storage, dict):
        summary["benchmark_storage"] = benchmark_storage
    write_json(args.summary_path, summary)
    print(f"total_rows={sum(counts.values())}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"summary_path={args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
