from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson

from .async_request_runner import AsyncRequestRunner
from .common import build_shuffle_key, prompt_version
from .model_registry import PipelineModelConfig, load_pipeline_model_config
from .world_prompts import WORLD_SEED_SYSTEM_PROMPT
from .world_schema import BenchmarkQuestion, iter_benchmark_questions
from .world_scoring import repair_world_response, score_world_response


BENCHMARK_RUN_CONTRACT = "world_benchmark_run_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-model world benchmark evaluation.")
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))
    parser.add_argument("--question-path", type=Path, default=Path("data/world_knowledge/questions.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/cache/benchmark_runs.jsonl"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-model", type=int, default=8)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")
    handle.flush()


def load_processed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    processed: set[tuple[str, str, int]] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            question = payload.get("question")
            model_name = payload.get("model_name")
            sample_index = payload.get("sample_index")
            if not isinstance(question, dict):
                raise ValueError(f"{path}:{line_number} missing object 'question'")
            question_id = question.get("question_id")
            if not isinstance(question_id, str) or not isinstance(model_name, str) or not isinstance(sample_index, int):
                raise ValueError(f"{path}:{line_number} missing resume keys")
            processed.add((question_id, model_name, sample_index))
    return processed


def build_record(
    *,
    question: BenchmarkQuestion,
    model_name: str,
    sample_index: int,
    system_prompt: str,
    user_message: str,
    raw_response_text: str,
    effective_response_text: str,
    score: dict[str, Any],
    choice_permutation: list[int],
    shuffled_choices: list[str],
    shuffled_answer_index: int | None,
    shuffle_key: str,
    attempt_count: int,
    cache_hit: bool,
    generation_errors: list[str],
    json_repair: dict[str, Any],
) -> dict[str, Any]:
    return {
        "contract": BENCHMARK_RUN_CONTRACT,
        "question": question.to_dict(),
        "model_name": model_name,
        "sample_index": sample_index,
        "stage": "benchmark",
        "system_prompt": system_prompt,
        "system_prompt_version": prompt_version(system_prompt),
        "instruction": user_message,
        "response_text": effective_response_text,
        "raw_response_text": raw_response_text,
        "choice_permutation": choice_permutation,
        "shuffled_choices": shuffled_choices,
        "shuffled_answer_index": shuffled_answer_index,
        "shuffle_key": shuffle_key,
        "attempt_count": attempt_count,
        "cache_hit": cache_hit,
        "generation_errors": generation_errors,
        "json_repair": json_repair,
        "score": score,
    }


async def run_single_eval(
    *,
    config: PipelineModelConfig,
    runner: AsyncRequestRunner,
    question: BenchmarkQuestion,
    model_name: str,
    sample_index: int,
    model_attempts: int,
) -> dict[str, Any]:
    prompted = question.prompted_variant(
        sample_index,
        shuffle_key=build_shuffle_key("benchmark", model_name, sample_index),
    )
    scoring_question = question
    if prompted.shuffled_choices:
        scoring_question = replace(
            question,
            choices=prompted.shuffled_choices,
            gold_answer_index=prompted.shuffled_answer_index,
        )

    def ensure_usable_response(raw_response_text: str) -> None:
        initial_score = score_world_response(raw_response_text, scoring_question)
        if initial_score.valid_json and initial_score.think_tags_present:
            return
        repaired_response_text, _ = repair_world_response(raw_response_text, scoring_question)
        if repaired_response_text is None:
            raise ValueError("benchmark response was neither distill-format-valid nor repairable")

    generation = await runner.generate(
        endpoint=config.base_model(model_name),
        system_prompt=WORLD_SEED_SYSTEM_PROMPT,
        user_message=prompted.user_message,
        attempts=model_attempts,
        validator=ensure_usable_response,
        use_cache=True,
    )

    raw_response_text = generation.content
    effective_response_text = raw_response_text
    json_repair = {"status": "not_attempted"}
    score = score_world_response(effective_response_text, scoring_question)
    if not score.valid_json or not score.think_tags_present:
        repaired_response_text, json_repair = repair_world_response(raw_response_text, scoring_question)
        if repaired_response_text is not None:
            effective_response_text = repaired_response_text
            score = score_world_response(effective_response_text, scoring_question)

    return build_record(
        question=question,
        model_name=model_name,
        sample_index=sample_index,
        system_prompt=WORLD_SEED_SYSTEM_PROMPT,
        user_message=prompted.user_message,
        raw_response_text=raw_response_text,
        effective_response_text=effective_response_text,
        score=score.to_dict(),
        choice_permutation=prompted.choice_permutation,
        shuffled_choices=prompted.shuffled_choices,
        shuffled_answer_index=prompted.shuffled_answer_index,
        shuffle_key=prompted.shuffle_key,
        attempt_count=generation.attempt_count,
        cache_hit=generation.cache_hit,
        generation_errors=generation.errors,
        json_repair=json_repair,
    )


def summarize_output(path: Path) -> dict[str, Any]:
    totals_by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "valid_json": 0, "think_tags": 0, "cache_hits": 0}
    )
    totals_by_question: dict[str, int] = defaultdict(int)
    repaired_count = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            model_name = payload.get("model_name")
            score = payload.get("score")
            question = payload.get("question")
            if not isinstance(model_name, str) or not isinstance(score, dict) or not isinstance(question, dict):
                raise ValueError(f"{path}:{line_number} is missing summary fields")
            totals_by_model[model_name]["total"] += 1
            totals_by_model[model_name]["correct"] += int(
                score.get("correct") is True and score.get("valid_json") is True
            )
            totals_by_model[model_name]["valid_json"] += int(score.get("valid_json") is True)
            totals_by_model[model_name]["think_tags"] += int(score.get("think_tags_present") is True)
            totals_by_model[model_name]["cache_hits"] += int(payload.get("cache_hit") is True)
            json_repair = payload.get("json_repair")
            if isinstance(json_repair, dict):
                repaired_count += int(json_repair.get("status") == "repaired")
            question_id = question.get("question_id")
            if isinstance(question_id, str):
                totals_by_question[question_id] += 1

    summary_models: dict[str, Any] = {}
    for model_name, totals in sorted(totals_by_model.items()):
        total = max(1, totals["total"])
        summary_models[model_name] = {
            "total_samples": totals["total"],
            "correct_and_valid_rate": round(totals["correct"] / total, 6),
            "valid_json_rate": round(totals["valid_json"] / total, 6),
            "think_tag_rate": round(totals["think_tags"] / total, 6),
            "cache_hit_rate": round(totals["cache_hits"] / total, 6),
        }

    return {
        "models": summary_models,
        "cache_hit_count": sum(item["cache_hits"] for item in totals_by_model.values()),
        "json_repaired_count": repaired_count,
        "question_count": len(totals_by_question),
        "raw_record_count": sum(item["total_samples"] for item in summary_models.values()),
    }


async def run_jobs(
    *,
    args: argparse.Namespace,
    config: PipelineModelConfig,
    question_list: list[BenchmarkQuestion],
    processed_keys: set[tuple[str, str, int]],
    completion_counts: dict[str, int],
    expected_records_per_question: int,
) -> tuple[int, int]:
    pending_jobs: list[tuple[BenchmarkQuestion, str, int]] = []
    for question in question_list:
        for model in config.base_models:
            for sample_index in range(args.samples_per_model):
                key = (question.question_id, model.name, sample_index)
                if key in processed_keys:
                    continue
                pending_jobs.append((question, model.name, sample_index))

    if not pending_jobs:
        print("benchmark_already_complete=true", flush=True)
        completed_questions = sum(1 for count in completion_counts.values() if count >= expected_records_per_question)
        return 0, completed_questions

    runner = AsyncRequestRunner(
        cache_path=args.cache_path,
        default_max_concurrency=args.max_concurrency,
    )
    total_written = 0
    completed_questions = sum(1 for count in completion_counts.values() if count >= expected_records_per_question)
    target_total_records = len(question_list) * len(config.base_models) * args.samples_per_model
    initial_record_count = len(processed_keys)
    job_iter = iter(pending_jobs)
    in_flight: set[asyncio.Task[dict[str, Any]]] = set()

    async def schedule_next() -> bool:
        try:
            question, model_name, sample_index = next(job_iter)
        except StopIteration:
            return False
        task = asyncio.create_task(
            run_single_eval(
                config=config,
                runner=runner,
                question=question,
                model_name=model_name,
                sample_index=sample_index,
                model_attempts=args.model_attempts,
            )
        )
        in_flight.add(task)
        return True

    try:
        with args.output_path.open(
            "ab" if args.resume and args.output_path.exists() and args.output_path.stat().st_size > 0 else "wb"
        ) as output_handle:
            while len(in_flight) < min(args.max_concurrency, len(pending_jobs)):
                if not await schedule_next():
                    break
            while in_flight:
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    record = await task
                    append_jsonl_line(output_handle, record)
                    total_written += 1
                    question_id = record["question"]["question_id"]
                    previous_count = completion_counts.get(question_id, 0)
                    current_count = previous_count + 1
                    completion_counts[question_id] = current_count
                    if previous_count < expected_records_per_question <= current_count:
                        completed_questions += 1
                    if total_written % args.progress_interval == 0:
                        print(
                            "sample_records_written="
                            f"{initial_record_count + total_written}/{target_total_records} "
                            f"questions_completed={completed_questions}/{len(question_list)}",
                            flush=True,
                        )
                    await schedule_next()
    finally:
        await runner.aclose()

    return total_written, completed_questions


async def async_main(args: argparse.Namespace) -> None:
    if args.samples_per_model <= 0:
        raise ValueError("--samples-per-model must be positive")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    if args.model_attempts <= 0:
        raise ValueError("--model-attempts must be positive")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    config = load_pipeline_model_config(args.config_path)
    question_list = list(iter_benchmark_questions(args.question_path, limit=args.limit))
    if not question_list:
        raise ValueError(f"No questions found in {args.question_path}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_path is None:
        args.cache_path = args.output_path.parent / "request_cache.sqlite"
    if args.clear_cache and args.cache_path.exists():
        args.cache_path.unlink()

    if not args.resume and args.output_path.exists():
        args.output_path.unlink()

    processed_keys = load_processed_keys(args.output_path) if args.resume else set()
    completion_counts: dict[str, int] = defaultdict(int)
    for question_id, _, _ in processed_keys:
        completion_counts[question_id] += 1
    expected_records_per_question = len(config.base_models) * args.samples_per_model
    resumed_questions = sum(1 for count in completion_counts.values() if count >= expected_records_per_question)
    if processed_keys:
        print(f"resume_records={len(processed_keys)}", flush=True)
        print(f"resume_questions_completed={resumed_questions}", flush=True)
    print(f"target_questions={len(question_list)}", flush=True)
    print(
        f"target_sample_records={len(question_list) * len(config.base_models) * args.samples_per_model}",
        flush=True,
    )

    total_written, completed_questions = await run_jobs(
        args=args,
        config=config,
        question_list=question_list,
        processed_keys=processed_keys,
        completion_counts=completion_counts,
        expected_records_per_question=expected_records_per_question,
    )
    summary = summarize_output(args.output_path)
    summary.update(
        {
            "config_path": str(args.config_path),
            "question_path": str(args.question_path),
            "output_path": str(args.output_path),
            "cache_path": str(args.cache_path),
            "base_models": [item.name for item in config.base_models],
            "samples_per_model": args.samples_per_model,
        }
    )
    print(f"records_written_this_run={total_written}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"questions_completed={completed_questions}/{len(question_list)}", flush=True)
    print(f"question_count={summary['question_count']}", flush=True)
    print(f"raw_record_count={summary['raw_record_count']}", flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
