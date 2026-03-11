from __future__ import annotations

import argparse
import asyncio
import re
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson

from .async_request_runner import AsyncRequestRunner
from .common import build_shuffle_key
from .model_registry import load_pipeline_model_config
from .world_prompts import WORLD_REWRITE_SYSTEM_PROMPT, WORLD_SEED_SYSTEM_PROMPT
from .world_schema import BenchmarkQuestion
from .world_scoring import repair_world_response, score_world_response


REWRITE_CONTRACT = "distill_row_v1"
REWRITE_FAILURE_CONTRACT = "rewrite_failure_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite improved GEPA questions into simple distillation prompts.")
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))
    parser.add_argument("--input-path", type=Path, default=Path("data/world_knowledge/gepa_results.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/rewrite_distill.jsonl"))
    parser.add_argument("--failure-path", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--rewrites-per-example", type=int, default=3)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")
    handle.flush()


def load_processed_question_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    processed: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            meta = payload.get("meta")
            if not isinstance(meta, dict):
                raise ValueError(f"{path}:{line_number} missing 'meta'")
            question_id = meta.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                raise ValueError(f"{path}:{line_number} missing meta.question_id")
            processed.add(question_id)
    return processed


def parse_simple_questions(response_text: str) -> list[str]:
    try:
        payload = orjson.loads(response_text)
    except orjson.JSONDecodeError:
        start = response_text.find("{")
        end = response_text.rfind("}")
        payload = None
        if start != -1 and end != -1 and end > start:
            payload = orjson.loads(response_text[start : end + 1])
        if payload is None:
            payload = {
                "simple_questions": [
                    re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
                    for line in response_text.splitlines()
                    if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
                ]
            }
    if not isinstance(payload, dict):
        raise ValueError("Rewrite response must be a JSON object")
    questions = payload.get("simple_questions")
    if not isinstance(questions, list):
        raise ValueError("Rewrite response must contain 'simple_questions'")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in questions:
        if not isinstance(item, str) or not item.strip():
            continue
        normalized = item.strip()
        if normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    if not deduped:
        raise ValueError("Rewrite response did not produce any usable simple questions")
    return deduped


def build_failure_record(payload: dict[str, Any], exc: BaseException) -> dict[str, Any]:
    return {
        "contract": REWRITE_FAILURE_CONTRACT,
        "question_id": payload.get("question_id"),
        "group_id": payload.get("group_id"),
        "target_model": payload.get("target_model"),
        "domain": payload.get("domain"),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def iter_improved_examples(path: Path, processed_question_ids: set[str]):
    with path.open("rb") as input_handle:
        for line_number, raw_line in enumerate(input_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            group_id = payload.get("group_id")
            target_model = payload.get("target_model")
            domain = payload.get("domain")
            best_prompt = payload.get("best_prompt")
            system_prompt = payload.get("system_prompt", WORLD_SEED_SYSTEM_PROMPT)
            question_deltas = payload.get("question_deltas")
            if (
                not isinstance(group_id, str)
                or not isinstance(target_model, str)
                or not isinstance(domain, str)
                or not isinstance(best_prompt, str)
                or not isinstance(system_prompt, str)
                or not isinstance(question_deltas, list)
            ):
                raise ValueError(f"{path}:{line_number} missing grouped GEPA fields")
            for delta in question_deltas:
                if not isinstance(delta, dict) or delta.get("improved_to_distillable") is not True:
                    continue
                question_payload = delta.get("question")
                question_id = delta.get("question_id")
                if (
                    not isinstance(question_payload, dict)
                    or not isinstance(question_id, str)
                    or question_id in processed_question_ids
                ):
                    continue
                instruction = delta.get("optimized_preferred_instruction")
                answer = delta.get("optimized_preferred_response")
                if not isinstance(instruction, str) or not isinstance(answer, str):
                    continue
                yield {
                    "group_id": group_id,
                    "target_model": target_model,
                    "domain": domain,
                    "system_prompt": system_prompt,
                    "best_prompt": best_prompt,
                    "question_id": question_id,
                    "question": question_payload,
                    "instruction": instruction,
                    "answer": answer,
                }


async def validate_simple_question(
    *,
    simple_question: str,
    question: BenchmarkQuestion,
    system_prompt: str,
    target_model_config: Any,
    model_attempts: int,
    validation_samples: int,
    rewrite_variant_index: int,
    runner: AsyncRequestRunner,
) -> tuple[bool, str]:
    rendered_user = ""
    for sample_index in range(validation_samples):
        prompted = question.prompted_variant(
            sample_index,
            shuffle_key=build_shuffle_key(
                "rewrite_validate",
                target_model_config.name,
                rewrite_variant_index,
                sample_index,
            ),
        )
        if prompted.shuffled_choices:
            scoring_question = replace(
                question,
                question_text=simple_question,
                choices=prompted.shuffled_choices,
                gold_answer_index=prompted.shuffled_answer_index,
            )
        else:
            scoring_question = replace(question, question_text=simple_question)
        rendered_user = scoring_question.render_prompt(choices=scoring_question.choices)
        generation = await runner.generate(
            endpoint=target_model_config,
            system_prompt=system_prompt,
            user_message=rendered_user,
            attempts=model_attempts,
            use_cache=True,
        )
        response_text = generation.content
        score = score_world_response(response_text, scoring_question)
        if not score.valid_json or not score.think_tags_present:
            repaired_response_text, _ = repair_world_response(response_text, scoring_question)
            if repaired_response_text is not None:
                response_text = repaired_response_text
                score = score_world_response(response_text, scoring_question)
        if not (score.correct and score.valid_json and score.think_tags_present):
            return False, rendered_user
    return True, rendered_user


async def process_example(
    *,
    payload: dict[str, Any],
    input_path: Path,
    config: Any,
    rewrites_per_example: int,
    validation_samples: int,
    model_attempts: int,
    runner: AsyncRequestRunner,
) -> tuple[str, list[dict[str, Any]], bool]:
    question = BenchmarkQuestion.from_dict(payload["question"], input_path, 1)
    question_id = payload["question_id"]
    system_prompt = payload.get("system_prompt", WORLD_SEED_SYSTEM_PROMPT)
    complex_question = payload["instruction"]
    better_answer = payload["answer"]
    target_model_name = payload["target_model"]
    best_prompt = payload["best_prompt"]

    rewrite_prompt = (
        f"Target Model: {target_model_name}\n"
        f"Question Type: {question.question_type}\n"
        f"Original Prompt:\n{complex_question}\n\n"
        f"Reference Answer:\n{better_answer}\n\n"
        f"Produce up to {rewrites_per_example} simpler equivalent prompts."
    )
    generation = await runner.generate(
        endpoint=config.rewrite_model,
        system_prompt=WORLD_REWRITE_SYSTEM_PROMPT,
        user_message=rewrite_prompt,
        attempts=model_attempts,
        validator=parse_simple_questions,
        use_cache=True,
    )
    simple_questions = parse_simple_questions(generation.content)[:rewrites_per_example]
    target_model_config = config.base_model(target_model_name)

    validations = await asyncio.gather(
        *[
            validate_simple_question(
                simple_question=simple_question,
                question=question,
                system_prompt=system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else WORLD_SEED_SYSTEM_PROMPT,
                target_model_config=target_model_config,
                model_attempts=model_attempts,
                validation_samples=validation_samples,
                rewrite_variant_index=variant_index,
                runner=runner,
            )
            for variant_index, simple_question in enumerate(simple_questions)
        ]
    )

    rows: list[dict[str, Any]] = []
    for variant_index, (simple_question, (is_valid, rendered_user)) in enumerate(zip(simple_questions, validations)):
        if not is_valid:
            continue
        rows.append(
            {
                "contract": REWRITE_CONTRACT,
                "source_type": "gepa_rewrite",
                "system": system_prompt if isinstance(system_prompt, str) and system_prompt.strip() else WORLD_SEED_SYSTEM_PROMPT,
                "user": rendered_user,
                "assistant": better_answer,
                "meta": {
                    "question_id": question_id,
                    "benchmark_name": question.benchmark_name,
                    "split": question.split,
                    "domain": payload["domain"],
                    "question_type": question.question_type,
                    "target_model": target_model_name,
                    "group_id": payload["group_id"],
                    "rewrite_model": config.rewrite_model.name,
                    "rewrite_variant_index": variant_index,
                    "validation_samples": validation_samples,
                    "source_best_prompt": best_prompt,
                    "source_complex_user": complex_question,
                    "simple_question_text": simple_question,
                },
            }
        )
    return question_id, rows, generation.cache_hit


async def async_main(args: argparse.Namespace) -> None:
    if args.rewrites_per_example <= 0:
        raise ValueError("--rewrites-per-example must be positive")
    if args.validation_samples <= 0:
        raise ValueError("--validation-samples must be positive")
    if args.model_attempts <= 0:
        raise ValueError("--model-attempts must be positive")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")

    config = load_pipeline_model_config(args.config_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.failure_path is None:
        args.failure_path = args.output_path.with_name(f"{args.output_path.stem}_failures.jsonl")
    args.failure_path.parent.mkdir(parents=True, exist_ok=True)
    if args.cache_path is None:
        args.cache_path = args.output_path.parent / "cache" / "request_cache.sqlite"
    if args.clear_cache and args.cache_path.exists():
        args.cache_path.unlink()

    if not args.resume and args.output_path.exists():
        args.output_path.unlink()
    if not args.resume and args.failure_path.exists():
        args.failure_path.unlink()

    processed_question_ids = load_processed_question_ids(args.output_path) if args.resume else set()
    output_mode = "ab" if args.resume and args.output_path.exists() and args.output_path.stat().st_size > 0 else "wb"
    failure_mode = "ab" if args.resume and args.failure_path.exists() and args.failure_path.stat().st_size > 0 else "wb"

    pending = list(iter_improved_examples(args.input_path, processed_question_ids))

    total_rows = 0
    processed_examples = 0
    failed_examples = 0
    cache_hit_examples = 0
    runner = AsyncRequestRunner(cache_path=args.cache_path, default_max_concurrency=args.max_concurrency)
    job_iter = iter(pending)
    in_flight: set[asyncio.Task[tuple[str, list[dict[str, Any]], bool]]] = set()
    task_payloads: dict[asyncio.Task[tuple[str, list[dict[str, Any]], bool]], dict[str, Any]] = {}

    async def schedule_next() -> bool:
        try:
            payload = next(job_iter)
        except StopIteration:
            return False
        task = asyncio.create_task(
            process_example(
                payload=payload,
                input_path=args.input_path,
                config=config,
                rewrites_per_example=args.rewrites_per_example,
                validation_samples=args.validation_samples,
                model_attempts=args.model_attempts,
                runner=runner,
            )
        )
        in_flight.add(task)
        task_payloads[task] = payload
        return True

    try:
        with args.output_path.open(output_mode) as output_handle, args.failure_path.open(failure_mode) as failure_handle:
            while len(in_flight) < min(args.max_concurrency, len(pending)):
                if not await schedule_next():
                    break
            while in_flight:
                done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    payload = task_payloads.pop(task)
                    try:
                        _, rows, cache_hit = await task
                    except Exception as exc:
                        failed_examples += 1
                        append_jsonl_line(failure_handle, build_failure_record(payload, exc))
                        print(
                            f"question_id={payload.get('question_id')} status=failed "
                            f"error_type={type(exc).__name__} error={str(exc)!r}",
                            flush=True,
                        )
                        await schedule_next()
                        continue

                    processed_examples += 1
                    cache_hit_examples += int(cache_hit)
                    for row in rows:
                        append_jsonl_line(output_handle, row)
                        total_rows += 1
                    await schedule_next()
    finally:
        await runner.aclose()

    print(f"processed_examples={processed_examples}", flush=True)
    print(f"failed_examples={failed_examples}", flush=True)
    print(f"cache_hit_examples={cache_hit_examples}", flush=True)
    print(f"rows_written={total_rows}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"failure_path={args.failure_path}", flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
