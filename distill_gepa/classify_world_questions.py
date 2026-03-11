from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from .model_registry import load_pipeline_model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark samples into per-question decisions.")
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))
    parser.add_argument("--input-path", type=Path, default=Path("data/world_knowledge/cache/benchmark_runs.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/question_decisions.jsonl"))
    return parser.parse_args()


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")

def build_model_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    correct = sum(1 for item in samples if item.get("correct") is True)
    stable_correct = total > 0 and correct == total
    return {
        "sample_count": total,
        "correct_count": correct,
        "stable_correct": stable_correct,
        "usable_for_gepa": correct > 0 and not stable_correct,
    }


def classification_for_question(per_model: dict[str, dict[str, Any]]) -> str:
    summaries = list(per_model.values())
    if summaries and all(item["stable_correct"] for item in summaries):
        return "direct_distill"
    if summaries and all(item["sample_count"] > 0 and item["correct_count"] == 0 for item in summaries):
        return "suspected_anomaly"
    return "needs_optimization"


def main() -> None:
    args = parse_args()
    config = load_pipeline_model_config(args.config_path)
    ordered_models = [item.name for item in config.base_models]

    grouped_samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    question_ids: set[str] = set()

    with args.input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{args.input_path}:{line_number} must be a JSON object")
            question_id = payload.get("question_id")
            model_name = payload.get("model_name")
            if not isinstance(question_id, str) or not question_id or not isinstance(model_name, str):
                raise ValueError(f"{args.input_path}:{line_number} is missing question_id/model_name")
            question_ids.add(question_id)
            grouped_samples[question_id][model_name].append(payload)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_counts = {"direct_distill": 0, "needs_optimization": 0, "suspected_anomaly": 0}
    with args.output_path.open("wb") as output_handle:
        for question_id in sorted(question_ids):
            per_model_summary = {
                model_name: build_model_summary(grouped_samples[question_id].get(model_name, []))
                for model_name in ordered_models
            }
            classification = classification_for_question(per_model_summary)
            append_jsonl_line(
                output_handle,
                {
                    "question_id": question_id,
                    "classification": classification,
                    "per_model": per_model_summary,
                },
            )
            summary_counts[classification] += 1

    print(f"question_count={sum(summary_counts.values())}", flush=True)
    print(f"direct_distill={summary_counts['direct_distill']}", flush=True)
    print(f"needs_optimization={summary_counts['needs_optimization']}", flush=True)
    print(f"suspected_anomaly={summary_counts['suspected_anomaly']}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
