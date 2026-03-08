from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson

from .common import prompt_version, write_json


DISTILL_CONTRACT = "distill_row_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge direct and GEPA-derived distillation datasets.")
    parser.add_argument("--decision-path", type=Path, default=Path("data/world_knowledge/question_decisions.jsonl"))
    parser.add_argument("--gepa-path", type=Path, default=Path("data/world_knowledge/gepa_results.jsonl"))
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


def row_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Merged rows require 'meta'")
    question_id = meta.get("question_id")
    source_type = payload.get("source_type")
    user = payload.get("user")
    if not isinstance(question_id, str) or not isinstance(source_type, str) or not isinstance(user, str):
        raise ValueError("Merged rows require source_type, user, and meta.question_id")
    return source_type, question_id, user


def build_direct_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("classification") != "direct_distill":
        return None
    question = payload.get("question")
    preferred_model = payload.get("preferred_model")
    per_model = payload.get("per_model")
    if not isinstance(question, dict) or not isinstance(preferred_model, str) or not isinstance(per_model, dict):
        return None
    preferred = per_model.get(preferred_model)
    if not isinstance(preferred, dict):
        return None
    if preferred.get("usable_for_direct_distill") is not True:
        return None
    system_prompt = preferred.get("preferred_prompt")
    user_prompt = preferred.get("preferred_instruction")
    assistant = preferred.get("preferred_response_text")
    if not isinstance(system_prompt, str) or not isinstance(user_prompt, str) or not isinstance(assistant, str):
        return None
    return {
        "contract": DISTILL_CONTRACT,
        "source_type": "direct_distill",
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant,
        "meta": {
            "question_id": question["question_id"],
            "benchmark_name": question["benchmark_name"],
            "split": question["split"],
            "domain": question["domain"],
            "question_type": question["question_type"],
            "target_model": preferred_model,
            "prompt_version": prompt_version(system_prompt),
            "classification": "direct_distill",
        },
    }


def iter_complex_rows(path: Path):
    for payload in iter_jsonl(path) or []:
        group_id = payload.get("group_id")
        target_model = payload.get("target_model")
        domain = payload.get("domain")
        best_prompt = payload.get("best_prompt")
        question_deltas = payload.get("question_deltas")
        if (
            not isinstance(group_id, str)
            or not isinstance(target_model, str)
            or not isinstance(domain, str)
            or not isinstance(question_deltas, list)
        ):
            continue
        system_prompt = payload.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            continue
        for delta in question_deltas:
            if not isinstance(delta, dict) or delta.get("improved_to_distillable") is not True:
                continue
            question = delta.get("question")
            user_prompt = delta.get("optimized_preferred_instruction")
            assistant = delta.get("optimized_preferred_response")
            question_id = delta.get("question_id")
            if (
                not isinstance(question, dict)
                or not isinstance(user_prompt, str)
                or not isinstance(assistant, str)
                or not isinstance(question_id, str)
            ):
                continue
            yield {
                "contract": DISTILL_CONTRACT,
                "source_type": "gepa_complex",
                "system": system_prompt,
                "user": user_prompt,
                "assistant": assistant,
                "meta": {
                    "question_id": question_id,
                    "benchmark_name": question["benchmark_name"],
                    "split": question["split"],
                    "domain": domain,
                    "question_type": question["question_type"],
                    "target_model": target_model,
                    "group_id": group_id,
                    "system_prompt_version": prompt_version(system_prompt),
                    "best_prompt": best_prompt,
                    "best_prompt_version": prompt_version(best_prompt) if isinstance(best_prompt, str) else None,
                    "baseline_stable_correct": delta.get("baseline_stable_correct"),
                    "optimized_stable_correct": delta.get("optimized_stable_correct"),
                    "baseline_correct_rate": delta.get("baseline_correct_rate"),
                    "optimized_correct_rate": delta.get("optimized_correct_rate"),
                    "baseline_think_tag_rate": delta.get("baseline_think_tag_rate"),
                    "optimized_think_tag_rate": delta.get("optimized_think_tag_rate"),
                },
            }


def build_summary(
    *,
    decision_path: Path,
    gepa_path: Path,
    rewrite_path: Path,
    output_path: Path,
    counts: dict[str, int],
) -> dict[str, Any]:
    decision_counts = {"direct_distill": 0, "needs_optimization": 0, "suspected_anomaly": 0}
    question_count = 0
    base_models: list[str] = []
    for payload in iter_jsonl(decision_path) or []:
        classification = payload.get("classification")
        per_model = payload.get("per_model")
        if not base_models and isinstance(per_model, dict):
            base_models = [key for key in per_model if isinstance(key, str)]
        if isinstance(classification, str) and classification in decision_counts:
            decision_counts[classification] += 1
            question_count += 1

    gepa_group_count = 0
    gepa_improved_question_count = 0
    for payload in iter_jsonl(gepa_path) or []:
        gepa_group_count += 1
        improved_question_count = payload.get("improved_question_count")
        if isinstance(improved_question_count, int):
            gepa_improved_question_count += improved_question_count

    rewrite_row_count = sum(1 for _ in iter_jsonl(rewrite_path) or [])

    return {
        "dataset_name": output_path.parent.name,
        "question_count": question_count,
        "decision_counts": decision_counts,
        "base_models": base_models,
        "gepa_group_count": gepa_group_count,
        "gepa_improved_question_count": gepa_improved_question_count,
        "rewrite_row_count": rewrite_row_count,
        "final_sft_row_count": sum(counts.values()),
        "sft_counts": counts,
    }


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"direct_distill": 0, "gepa_complex": 0, "gepa_rewrite": 0}
    seen: set[tuple[str, str, str]] = set()

    with args.output_path.open("wb") as output_handle:
        for payload in iter_jsonl(args.decision_path) or []:
            row = build_direct_row(payload)
            if row is None:
                continue
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            output_handle.write(orjson.dumps(row))
            output_handle.write(b"\n")
            counts["direct_distill"] += 1

        for row in iter_complex_rows(args.gepa_path):
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            output_handle.write(orjson.dumps(row))
            output_handle.write(b"\n")
            counts["gepa_complex"] += 1

        for payload in iter_jsonl(args.rewrite_path) or []:
            key = row_key(payload)
            if key in seen:
                continue
            seen.add(key)
            output_handle.write(orjson.dumps(payload))
            output_handle.write(b"\n")
            counts["gepa_rewrite"] += 1

    summary = build_summary(
        decision_path=args.decision_path,
        gepa_path=args.gepa_path,
        rewrite_path=args.rewrite_path,
        output_path=args.output_path,
        counts=counts,
    )
    write_json(args.summary_path, summary)
    print(f"total_rows={sum(counts.values())}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"summary_path={args.summary_path}", flush=True)


if __name__ == "__main__":
    main()
