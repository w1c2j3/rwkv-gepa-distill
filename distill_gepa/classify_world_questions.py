from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

import orjson

from .model_registry import load_pipeline_model_config


QUESTION_DECISION_CONTRACT = "question_decision_v1"
MIN_GEPA_CORRECT_RATE = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark samples into per-question decisions.")
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))
    parser.add_argument("--input-path", type=Path, default=Path("data/world_knowledge/cache/benchmark_runs.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/question_decisions.jsonl"))
    return parser.parse_args()


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")


def _normalize_answer_signature(sample: dict[str, Any]) -> str:
    score = sample.get("score")
    if not isinstance(score, dict):
        return "__missing__"
    parsed = score.get("parsed")
    if not isinstance(parsed, dict):
        return "__missing__"

    answer_index = parsed.get("answer_index")
    shuffled_choices = sample.get("shuffled_choices")
    if isinstance(answer_index, int) and isinstance(shuffled_choices, list) and 0 <= answer_index < len(shuffled_choices):
        choice_text = shuffled_choices[answer_index]
        if isinstance(choice_text, str) and choice_text.strip():
            return choice_text.strip().lower()

    for key in ("answer_text", "final_answer"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return "__missing__"


def build_model_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    correct_and_valid = sum(
        1 for item in samples if item["score"].get("correct") is True and item["score"].get("valid_json") is True
    )
    correct = sum(1 for item in samples if item["score"].get("correct") is True)
    valid_json = sum(1 for item in samples if item["score"].get("valid_json") is True)
    think_tags = sum(1 for item in samples if item["score"].get("think_tags_present") is True)
    correct_rate = round(correct / total, 6) if total else 0.0
    correct_and_valid_rate = round(correct_and_valid / total, 6) if total else 0.0
    valid_json_rate = round(valid_json / total, 6) if total else 0.0
    think_tag_rate = round(think_tags / total, 6) if total else 0.0
    answer_signatures = Counter(_normalize_answer_signature(item) for item in samples)
    answer_signatures.pop("__missing__", None)
    majority_answer_signature = None
    majority_answer_count = 0
    if answer_signatures:
        majority_answer_signature, majority_answer_count = answer_signatures.most_common(1)[0]
    wrong_majority_count = max(
        (
            count
            for signature, count in answer_signatures.items()
            if signature and not any(
                item["score"].get("correct") is True and _normalize_answer_signature(item) == signature for item in samples
            )
        ),
        default=0,
    )
    letter_counter = Counter()
    for item in samples:
        parsed = item.get("score", {}).get("parsed", {})
        letter = parsed.get("answer_letter") if isinstance(parsed, dict) else None
        if isinstance(letter, str) and letter.strip():
            letter_counter[letter.strip().upper()] += 1
    stable_correct = total > 0 and correct_and_valid == total
    stable_wrong = total > 0 and correct == 0
    strict_stable_correct = stable_correct and think_tags == total
    preferred_sample = None
    for item in sorted(samples, key=lambda record: record["sample_index"]):
        if (
            item["score"].get("correct") is True
            and item["score"].get("valid_json") is True
            and item["score"].get("think_tags_present") is True
        ):
            preferred_sample = item
            break
    if preferred_sample is None:
        for item in sorted(samples, key=lambda record: record["sample_index"]):
            if item["score"].get("correct") is True and item["score"].get("valid_json") is True:
                preferred_sample = item
                break
    if preferred_sample is None and samples:
        preferred_sample = sorted(samples, key=lambda record: record["sample_index"])[0]
    return {
        "sample_count": total,
        "correct_count": correct,
        "correct_and_valid_count": correct_and_valid,
        "valid_json_count": valid_json,
        "think_tag_count": think_tags,
        "correct_rate": correct_rate,
        "correct_and_valid_rate": correct_and_valid_rate,
        "valid_json_rate": valid_json_rate,
        "think_tag_rate": think_tag_rate,
        "distinct_answer_count": len(answer_signatures),
        "majority_answer_signature": majority_answer_signature,
        "majority_answer_count": majority_answer_count,
        "stability_rate": round(majority_answer_count / total, 6) if total else 0.0,
        "correct_stability_rate": round(correct_and_valid / total, 6) if total else 0.0,
        "wrong_stability_rate": round(wrong_majority_count / total, 6) if total else 0.0,
        "position_bias_profile": dict(sorted(letter_counter.items())),
        "stable_correct": stable_correct,
        "strict_stable_correct": strict_stable_correct,
        "stable_wrong": stable_wrong,
        "usable_for_direct_distill": strict_stable_correct,
        "usable_for_gepa": correct_rate >= MIN_GEPA_CORRECT_RATE and not stable_correct,
        "usable_for_gepa_threshold": MIN_GEPA_CORRECT_RATE,
        "preferred_response_text": preferred_sample["response_text"] if preferred_sample else None,
        "preferred_instruction": preferred_sample["instruction"] if preferred_sample else None,
        "preferred_prompt": preferred_sample["system_prompt"] if preferred_sample else None,
        "preferred_sample_index": preferred_sample["sample_index"] if preferred_sample else None,
        "preferred_has_think_tags": preferred_sample["score"].get("think_tags_present") is True if preferred_sample else False,
    }


def classification_for_question(per_model: dict[str, dict[str, Any]]) -> str:
    summaries = list(per_model.values())
    if summaries and all(item["stable_correct"] for item in summaries):
        return "direct_distill"
    if summaries and all(item["stable_wrong"] for item in summaries):
        return "suspected_anomaly"
    return "needs_optimization"


def choose_preferred_model(per_model: dict[str, dict[str, Any]], ordered_model_names: list[str]) -> str | None:
    for model_name in ordered_model_names:
        summary = per_model.get(model_name)
        if summary and summary.get("usable_for_direct_distill") is True and summary["preferred_response_text"]:
            return model_name
    for model_name in ordered_model_names:
        summary = per_model.get(model_name)
        if summary and summary["preferred_response_text"]:
            return model_name
    return None


def main() -> None:
    args = parse_args()
    config = load_pipeline_model_config(args.config_path)
    ordered_models = [item.name for item in config.base_models]

    grouped_samples: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    question_payloads: dict[str, dict[str, Any]] = {}

    with args.input_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{args.input_path}:{line_number} must be a JSON object")
            question = payload.get("question")
            model_name = payload.get("model_name")
            if not isinstance(question, dict) or not isinstance(model_name, str):
                raise ValueError(f"{args.input_path}:{line_number} is missing question/model_name")
            question_id = question.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{args.input_path}:{line_number} is missing question.question_id")
            question_payloads[question_id] = question
            grouped_samples[question_id][model_name].append(payload)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_counts = {"direct_distill": 0, "needs_optimization": 0, "suspected_anomaly": 0}
    with args.output_path.open("wb") as output_handle:
        for question_id in sorted(question_payloads):
            question = question_payloads[question_id]
            per_model_summary = {
                model_name: build_model_summary(grouped_samples[question_id].get(model_name, []))
                for model_name in ordered_models
            }
            classification = classification_for_question(per_model_summary)
            preferred_model = choose_preferred_model(per_model_summary, ordered_models)
            append_jsonl_line(
                output_handle,
                {
                    "contract": QUESTION_DECISION_CONTRACT,
                    "question_id": question_id,
                    "classification": classification,
                    "preferred_model": preferred_model,
                    "question": question,
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
