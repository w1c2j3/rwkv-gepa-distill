from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson

from .async_request_runner import AsyncRequestRunner
from .common import write_json
from .model_registry import PipelineModelConfig, load_pipeline_model_config
from .trajectory_schema import build_slot_shuffle_key
from .world_prompts import WORLD_SEED_SYSTEM_PROMPT
from .world_schema import BenchmarkQuestion, iter_benchmark_questions
from .world_scoring import score_with_optional_repair


BENCHMARK_MANIFEST_CONTRACT = "world_benchmark_manifest_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-model world benchmark evaluation.")
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))
    parser.add_argument("--question-path", type=Path, default=Path("data/world_knowledge/questions.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/cache/benchmark_runs.jsonl"))
    parser.add_argument("--summary-path", type=Path, default=None)
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
            question_id = payload.get("question_id")
            model_name = payload.get("model_name")
            sample_index = payload.get("sample_index")
            if not isinstance(question_id, str) or not isinstance(model_name, str) or not isinstance(sample_index, int):
                raise ValueError(f"{path}:{line_number} missing resume keys")
            processed.add((question_id, model_name, sample_index))
    return processed


def build_record(
    *,
    question_id: str,
    model_name: str,
    sample_index: int,
    choice_permutation: list[int],
    effective_response_text: str,
    correct: bool,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "model_name": model_name,
        "sample_index": sample_index,
        "choice_permutation": choice_permutation,
        "assistant": effective_response_text,
        "correct": correct,
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
        shuffle_key=build_slot_shuffle_key(
            question_id=question.question_id,
            target_model=model_name,
            sample_index=sample_index,
        ),
    )
    scoring_question = question
    if prompted.shuffled_choices:
        scoring_question = replace(
            question,
            choices=prompted.shuffled_choices,
            gold_answer_index=prompted.shuffled_answer_index,
        )

    generation = await runner.generate(
        endpoint=config.base_model(model_name),
        system_prompt=WORLD_SEED_SYSTEM_PROMPT,
        user_message=prompted.user_message,
        attempts=model_attempts,
        use_cache=True,
    )

    raw_response_text = generation.content
    effective_response_text, score, _ = score_with_optional_repair(raw_response_text, scoring_question)

    return build_record(
        question_id=question.question_id,
        model_name=model_name,
        sample_index=sample_index,
        choice_permutation=prompted.choice_permutation,
        effective_response_text=effective_response_text,
        correct=score.correct,
    )


def summarize_output(path: Path) -> dict[str, Any]:
    totals_by_model: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    totals_by_question: dict[str, int] = defaultdict(int)
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            model_name = payload.get("model_name")
            correct = payload.get("correct")
            question_id = payload.get("question_id")
            if not isinstance(model_name, str) or not isinstance(correct, bool) or not isinstance(question_id, str):
                raise ValueError(f"{path}:{line_number} is missing summary fields")
            totals_by_model[model_name]["total"] += 1
            totals_by_model[model_name]["correct"] += int(correct)
            totals_by_question[question_id] += 1

    summary_models: dict[str, Any] = {}
    for model_name, totals in sorted(totals_by_model.items()):
        total = max(1, totals["total"])
        summary_models[model_name] = {
            "total_samples": totals["total"],
            "correct_rate": round(totals["correct"] / total, 6),
        }

    return {
        "models": summary_models,
        "question_count": len(totals_by_question),
        "raw_record_count": sum(item["total_samples"] for item in summary_models.values()),
    }


def build_benchmark_storage_summary(
    *,
    question_path: Path,
    question_count: int,
    config: PipelineModelConfig,
    samples_per_model: int,
) -> dict[str, Any]:
    return {
        "contract": BENCHMARK_MANIFEST_CONTRACT,
        "dataset_name": question_path.parent.name,
        "question_path": str(question_path),
        "question_count": question_count,
        "samples_per_model": samples_per_model,
        "models": [item.name for item in config.base_models],
        "benchmark_run_fields": [
            "question_id",
            "model_name",
            "sample_index",
            "choice_permutation",
            "assistant",
            "correct",
        ],
        "question_bank_fields": [
            "benchmark_name",
            "split",
            "domain",
            "question_id",
            "question_type",
            "question_text",
            "choices",
            "gold_answer",
            "gold_answer_index",
            "gold_aliases",
            "metadata",
        ],
        "gold_alignment": {
            "canonical_question_bank": str(question_path),
            "canonical_gold_answer_index_field": "gold_answer_index",
            "choice_permutation_field": "choice_permutation",
            "shuffled_gold_answer_index": "derived_at_runtime_only",
        },
    }


def write_summary_fragment(path: Path, payload: dict[str, Any]) -> None:
    summary: dict[str, Any] = {}
    if path.exists() and path.stat().st_size > 0:
        loaded = orjson.loads(path.read_bytes())
        if isinstance(loaded, dict):
            summary = loaded
    summary.update(payload)
    write_json(path, summary)


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
                    question_id = record["question_id"]
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
    if args.summary_path is None:
        args.summary_path = args.output_path.parent.parent / "pipeline_summary.json"
    if args.clear_cache and args.cache_path.exists():
        args.cache_path.unlink()

    if not args.resume and args.output_path.exists():
        args.output_path.unlink()
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)

    write_summary_fragment(
        args.summary_path,
        {
            "benchmark_storage": build_benchmark_storage_summary(
                question_path=args.question_path,
                question_count=len(question_list),
                config=config,
                samples_per_model=args.samples_per_model,
            )
        },
    )

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
    write_summary_fragment(args.summary_path, summary)
    print(f"records_written_this_run={total_written}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"questions_completed={completed_questions}/{len(question_list)}", flush=True)
    print(f"question_count={summary['question_count']}", flush=True)
    print(f"raw_record_count={summary['raw_record_count']}", flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
