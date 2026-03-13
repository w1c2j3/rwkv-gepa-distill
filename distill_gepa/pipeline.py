from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path
from typing import Any, BinaryIO

import orjson

from .request_runner import AsyncRequestRunner
from .common import prompt_version, write_json
from .constants import QUESTION_TYPE_MULTIPLE_CHOICE
from .dataset_adapters import build_questions_from_source
from .dataset_config import load_dataset_config
from .model_registry import load_pipeline_model_config
from .prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
    PROMPT_OPTIMIZER_SYSTEM_PROMPT,
    VARIANT_GENERATION_SYSTEM_PROMPT,
)
from .task_schema import TaskItem, iter_task_items, write_task_items
from .world_scoring import extract_json_object, score_with_optional_repair


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the task distillation pipeline.")
    parser.add_argument("--dataset-name", dest="dataset_name", default="seed_run")
    parser.add_argument("--dataset-config-path", type=Path, default=None)
    parser.add_argument("--task-input-path", type=Path, default=Path("data/seed_run/tasks.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants-per-task", type=int, default=12)
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _write_jsonl_record(handle: BinaryIO, record: dict[str, Any]) -> None:
    handle.write(orjson.dumps(record))
    handle.write(b"\n")


def _permuted_variant_for_model(variant: TaskItem, model_name: str) -> tuple[TaskItem, list[int]]:
    if variant.question_type != QUESTION_TYPE_MULTIPLE_CHOICE or len(variant.choices) <= 1:
        return variant, list(range(len(variant.choices)))

    permutation = list(range(len(variant.choices)))
    rng = random.Random(f"{variant.question_id}::{model_name}")
    rng.shuffle(permutation)
    shuffled_choices = [variant.choices[index] for index in permutation]
    shuffled_answer_index = (
        permutation.index(variant.reference_answer_index)
        if variant.reference_answer_index is not None
        else None
    )
    return (
        TaskItem(
            question_id=variant.question_id,
            data_split=variant.data_split,
            domain=variant.domain,
            question_type=variant.question_type,
            question_text=variant.question_text,
            reference_answer=variant.reference_answer,
            choices=shuffled_choices,
            reference_answer_index=shuffled_answer_index,
            reference_aliases=list(variant.reference_aliases),
            metadata=dict(variant.metadata),
        ),
        permutation,
    )


def _task_seed_payload(seed: TaskItem) -> dict[str, Any]:
    return {
        "question_id": seed.question_id,
        "data_split": seed.data_split,
        "question_text": seed.question_text,
        "choices": list(seed.choices),
        "reference_answer": seed.reference_answer,
        "reference_answer_index": seed.reference_answer_index,
    }


def _variant_question_payload(variant: TaskItem) -> dict[str, Any]:
    return {
        "id": variant.question_id,
        "question_text": variant.question_text,
        "choices": list(variant.choices),
        "reference_answer": variant.reference_answer,
        "reference_answer_index": variant.reference_answer_index,
    }


def _question_variants_row(seed: TaskItem, variants: list[TaskItem]) -> dict[str, Any]:
    row = _task_seed_payload(seed)
    row["variants"] = [_variant_question_payload(variant) for variant in variants]
    return row


def _variant_result_row(
    *,
    seed: TaskItem,
    variant: TaskItem,
    status: str,
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": variant.question_id,
        "question_id": seed.question_id,
        "data_split": seed.data_split,
        "question_text": variant.question_text,
        "choices": list(variant.choices),
        "reference_answer": variant.reference_answer,
        "reference_answer_index": variant.reference_answer_index,
        "status": status,
        "models": models,
    }


def load_task_inputs(path: Path, *, limit: int | None) -> list[TaskItem]:
    task_items = list(iter_task_items(path, limit=limit))
    if not task_items:
        raise ValueError(f"No task inputs found in {path}")
    return task_items


def default_dataset_config_path(dataset_version: str) -> Path:
    return Path("config/datasets") / f"{dataset_version}.toml"


def ensure_task_input_path(
    task_input_path: Path,
    dataset_config_path: Path | None = None,
    *,
    limit: int | None = None,
) -> None:
    if task_input_path.exists() and task_input_path.stat().st_size > 0:
        return
    if task_input_path.name != "tasks.jsonl":
        raise FileNotFoundError(f"Missing task input file: {task_input_path}")

    resolved_dataset_config_path = dataset_config_path
    if resolved_dataset_config_path is None:
        dataset_version = task_input_path.parent.name
        resolved_dataset_config_path = default_dataset_config_path(dataset_version)
    if not resolved_dataset_config_path.exists():
        raise FileNotFoundError(
            f"Missing task input file: {task_input_path} and no dataset config found at {resolved_dataset_config_path}"
        )

    dataset_dir = task_input_path.parent
    dataset_config = load_dataset_config(resolved_dataset_config_path, dataset_dir)
    merged_tasks: list[TaskItem] = []

    for source in dataset_config.sources:
        if not source.enabled:
            continue
        source_output_path = dataset_config.root_dir / source.output_file
        if source_output_path.exists() and source_output_path.stat().st_size > 0:
            source_tasks = load_task_inputs(source_output_path, limit=limit)
        else:
            source_tasks = build_questions_from_source(dataset_config.root_dir, source, limit)
            write_task_items(source_output_path, source_tasks)
        if source.merge_into_world:
            merged_tasks.extend(source_tasks)
            if limit is not None and len(merged_tasks) >= limit:
                merged_tasks = merged_tasks[:limit]
                break

    if not merged_tasks:
        raise ValueError(f"No tasks were prepared from {resolved_dataset_config_path}")
    write_task_items(task_input_path, merged_tasks)


def parse_generated_variants(response_text: str, *, seed: TaskItem, variants_per_task: int) -> list[dict[str, Any]]:
    try:
        payload = orjson.loads(response_text)
    except orjson.JSONDecodeError:
        payload = extract_json_object(response_text)
    if not isinstance(payload, dict):
        raise ValueError("Variant generation response must be a JSON object")
    raw_variants = payload.get("variants")
    if not isinstance(raw_variants, list):
        raise ValueError("Variant generation response must contain 'variants'")

    variants: list[dict[str, Any]] = []
    for index, item in enumerate(raw_variants[:variants_per_task]):
        if not isinstance(item, dict):
            continue
        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        if not question or not answer:
            continue
        variants.append(
            {
                "variant_id": f"{seed.question_id}::variant::{index}",
                "parent_question_id": seed.question_id,
                "variant_index": index,
                "data_split": seed.data_split,
                "domain": seed.domain,
                "question_type": seed.question_type,
                "question_text": question,
                "reference_answer": answer,
                "choices": item.get("choices", []),
                "reference_answer_index": item.get("answer_index"),
                "reference_aliases": item.get("answer_aliases", []),
            }
        )
    if not variants:
        raise ValueError("Variant generation did not produce any usable variants")
    return variants


def render_generation_user_message(seed: TaskItem, variants_per_task: int) -> str:
    lines = [
        f"请仔细思考，对于下面的原题，生成{variants_per_task}个与它相关的变种问题，适合用于模型训练。",
        "你可以修改数字和语义等等。返回json格式的问题列表。",
        f"题目类型：{seed.question_type}",
        f"领域：{seed.domain}",
        "",
        "原题：",
        seed.question_text,
    ]
    if seed.question_type == "multiple_choice":
        lines.extend(["", "原题选项："])
        for choice in seed.choices:
            lines.append(f"- {choice}")
        lines.extend(
            [
                "",
                f"原题正确答案：{seed.reference_answer}",
                "请为每个变种题同时返回 choices、answer、answer_index。",
            ]
        )
    else:
        lines.extend(["", f"原题正确答案：{seed.reference_answer}", "请为每个变种题同时返回 answer。"])
    return "\n".join(lines).strip()


def parse_optimized_prompt(response_text: str) -> str:
    try:
        payload = orjson.loads(response_text)
    except orjson.JSONDecodeError:
        payload = extract_json_object(response_text)
    if not isinstance(payload, dict):
        raise ValueError("Prompt optimizer response must be a JSON object")
    optimized_prompt = payload.get("optimized_system_prompt")
    if not isinstance(optimized_prompt, str) or not optimized_prompt.strip():
        raise ValueError("Missing optimized_system_prompt")
    return optimized_prompt.strip()


def render_optimizer_user_message(
    *,
    seed: TaskItem,
    variant: TaskItem,
    model_name: str,
    wrong_answer: str,
) -> str:
    lines = [
        f"Target Model: {model_name}",
        "Improve the system prompt so this model answers the generated question correctly.",
        "",
        "Current System Prompt:",
        DIRECT_ANSWER_SYSTEM_PROMPT,
        "",
        "Original Seed Question:",
        seed.render_prompt(),
        "",
        "Generated Variant Question:",
        variant.render_prompt(),
        "",
        f"Gold Answer: {variant.reference_answer}",
        f"Model Wrong Answer: {wrong_answer}",
    ]
    return "\n".join(lines).strip()


def canonical_model_answer(score: Any, question: TaskItem) -> tuple[str, int | None]:
    parsed = score.parsed
    answer_index = parsed.answer_index
    answer_text = _clean_text(parsed.answer_text or parsed.final_answer)
    if (
        question.question_type == "multiple_choice"
        and not answer_text
        and answer_index is not None
        and 0 <= answer_index < len(question.choices)
    ):
        answer_text = question.choices[answer_index]
    return answer_text, answer_index


async def generate_variants_for_seed(
    *,
    seed: TaskItem,
    variants_per_task: int,
    runner: AsyncRequestRunner,
    config: Any,
) -> tuple[list[TaskItem], dict[str, Any]]:
    user_message = render_generation_user_message(seed, variants_per_task)
    generation = await runner.generate(
        endpoint=config.variant_generator_model,
        system_prompt=VARIANT_GENERATION_SYSTEM_PROMPT,
        user_message=user_message,
        attempts=2,
        validator=lambda text: parse_generated_variants(text, seed=seed, variants_per_task=variants_per_task),
        use_cache=True,
    )
    parsed = parse_generated_variants(generation.content, seed=seed, variants_per_task=variants_per_task)
    generation_prompt_id = prompt_version(VARIANT_GENERATION_SYSTEM_PROMPT)
    variants: list[TaskItem] = []
    for index, payload in enumerate(parsed, start=1):
        variant_seed = TaskItem.from_dict(
            {
                "question_id": payload["variant_id"],
                "data_split": payload["data_split"],
                "domain": payload["domain"],
                "question_type": payload["question_type"],
                "question_text": payload["question_text"],
                "choices": payload["choices"],
                "reference_answer": payload["reference_answer"],
                "reference_answer_index": payload["reference_answer_index"],
                "reference_aliases": payload["reference_aliases"],
                "metadata": {
                    "parent_question_id": seed.question_id,
                    "generation_prompt_version": generation_prompt_id,
                    "generator_model": generation.model_name,
                    "variant_index": index - 1,
                },
            },
            Path("<generated-variant>"),
            index,
        )
        variants.append(variant_seed)

    trace_row = {
        "stage": "variant_generation",
        "question_id": seed.question_id,
        "variant_id": None,
        "model_name": generation.model_name,
        "prompt_text": VARIANT_GENERATION_SYSTEM_PROMPT,
        "user_message": user_message,
        "response_text": generation.content,
        "cache_hit": generation.cache_hit,
        "attempt_count": generation.attempt_count,
        "errors": generation.errors,
    }
    return variants, trace_row


async def answer_with_prompt_safe(
    *,
    variant: TaskItem,
    model_name: str,
    system_prompt: str,
    runner: AsyncRequestRunner,
    config: Any,
) -> dict[str, Any]:
    prompted_variant, choice_permutation = _permuted_variant_for_model(variant, model_name)
    user_message = prompted_variant.render_prompt()
    try:
        generation = await runner.generate(
            endpoint=config.base_model(model_name),
            system_prompt=system_prompt,
            user_message=user_message,
            attempts=2,
            use_cache=True,
        )
        _, score, _ = score_with_optional_repair(generation.content, prompted_variant)
        answer_text, answer_index = canonical_model_answer(score, prompted_variant)
        return {
            "model_name": model_name,
            "prompt_text": system_prompt,
            "user_message": user_message,
            "raw_response": generation.content,
            "answer_text": answer_text,
            "answer_index": answer_index,
            "correct": bool(score.correct),
            "cache_hit": generation.cache_hit,
            "attempt_count": generation.attempt_count,
            "errors": generation.errors,
            "choice_permutation": choice_permutation,
            "error": None,
        }
    except Exception as exc:
        return {
            "model_name": model_name,
            "prompt_text": system_prompt,
            "user_message": user_message,
            "raw_response": None,
            "answer_text": None,
            "answer_index": None,
            "correct": False,
            "cache_hit": False,
            "attempt_count": 0,
            "errors": [],
            "choice_permutation": choice_permutation,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def maybe_recover_wrong_answer_safe(
    *,
    seed: TaskItem,
    variant: TaskItem,
    model_name: str,
    wrong_answer: str,
    runner: AsyncRequestRunner,
    config: Any,
) -> dict[str, Any]:
    prompted_variant, choice_permutation = _permuted_variant_for_model(variant, model_name)
    optimizer_user_message = render_optimizer_user_message(
        seed=seed,
        variant=prompted_variant,
        model_name=model_name,
        wrong_answer=wrong_answer,
    )
    result: dict[str, Any] = {
        "model_name": model_name,
        "optimizer_model_name": config.prompt_optimizer_model.name,
        "optimizer_prompt_text": PROMPT_OPTIMIZER_SYSTEM_PROMPT,
        "optimizer_user_message": optimizer_user_message,
        "optimizer_raw_response": None,
        "optimizer_cache_hit": False,
        "optimizer_attempt_count": 0,
        "optimizer_errors": [],
        "optimized_prompt": None,
        "optimized_raw_response": None,
        "optimized_cache_hit": False,
        "optimized_attempt_count": 0,
        "optimized_errors": [],
        "optimized_answer": None,
        "optimized_answer_index": None,
        "optimized_correct": False,
        "optimized_user_message": prompted_variant.render_prompt(),
        "choice_permutation": choice_permutation,
        "error": None,
    }

    try:
        optimizer_generation = await runner.generate(
            endpoint=config.prompt_optimizer_model,
            system_prompt=PROMPT_OPTIMIZER_SYSTEM_PROMPT,
            user_message=optimizer_user_message,
            attempts=2,
            validator=parse_optimized_prompt,
            use_cache=True,
        )
        optimized_prompt = parse_optimized_prompt(optimizer_generation.content)
        result.update(
            {
                "optimizer_model_name": optimizer_generation.model_name,
                "optimizer_raw_response": optimizer_generation.content,
                "optimizer_cache_hit": optimizer_generation.cache_hit,
                "optimizer_attempt_count": optimizer_generation.attempt_count,
                "optimizer_errors": optimizer_generation.errors,
                "optimized_prompt": optimized_prompt,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    optimized_answer_result = await answer_with_prompt_safe(
        variant=variant,
        model_name=model_name,
        system_prompt=str(result["optimized_prompt"]),
        runner=runner,
        config=config,
    )
    result.update(
        {
            "optimized_raw_response": optimized_answer_result["raw_response"],
            "optimized_cache_hit": optimized_answer_result["cache_hit"],
            "optimized_attempt_count": optimized_answer_result["attempt_count"],
            "optimized_errors": optimized_answer_result["errors"],
            "optimized_answer": optimized_answer_result["answer_text"],
            "optimized_answer_index": optimized_answer_result["answer_index"],
            "optimized_correct": optimized_answer_result["correct"],
            "optimized_user_message": optimized_answer_result["user_message"],
        }
    )
    if optimized_answer_result["error"] is not None:
        result["error"] = optimized_answer_result["error"]
    return result


def _write_direct_trace(
    handle: BinaryIO,
    *,
    question_id: str,
    variant_id: str,
    result: dict[str, Any],
) -> int:
    _write_jsonl_record(
        handle,
        {
            "stage": "direct_answer",
            "question_id": question_id,
            "variant_id": variant_id,
            "model_name": result["model_name"],
            "prompt_text": result["prompt_text"],
            "user_message": result["user_message"],
            "response_text": result["raw_response"],
            "cache_hit": result["cache_hit"],
            "attempt_count": result["attempt_count"],
            "errors": result["errors"],
            "choice_permutation": result["choice_permutation"],
            "error": result["error"],
        },
    )
    return 1


def _write_gepa_traces(
    handle: BinaryIO,
    *,
    question_id: str,
    variant_id: str,
    result: dict[str, Any],
) -> int:
    written = 1
    _write_jsonl_record(
        handle,
        {
            "stage": "prompt_optimizer",
            "question_id": question_id,
            "variant_id": variant_id,
            "model_name": result["optimizer_model_name"],
            "target_model_name": result["model_name"],
            "prompt_text": result["optimizer_prompt_text"],
            "user_message": result["optimizer_user_message"],
            "response_text": result["optimizer_raw_response"],
            "cache_hit": result["optimizer_cache_hit"],
            "attempt_count": result["optimizer_attempt_count"],
            "errors": result["optimizer_errors"],
            "error": result["error"] if result["optimizer_raw_response"] is None else None,
        },
    )
    if (
        result["optimized_prompt"] is not None
        or result["optimized_raw_response"] is not None
        or result["optimized_attempt_count"] > 0
        or result["optimized_errors"]
    ):
        _write_jsonl_record(
            handle,
            {
                "stage": "gepa_answer",
                "question_id": question_id,
                "variant_id": variant_id,
                "model_name": result["model_name"],
                "prompt_text": result["optimized_prompt"],
                "user_message": result["optimized_user_message"],
                "response_text": result["optimized_raw_response"],
                "cache_hit": result["optimized_cache_hit"],
                "attempt_count": result["optimized_attempt_count"],
                "errors": result["optimized_errors"],
                "choice_permutation": result["choice_permutation"],
                "error": result["error"] if result["optimized_raw_response"] is None else None,
            },
        )
        written += 1
    return written


async def async_main(args: argparse.Namespace) -> None:
    ensure_task_input_path(args.task_input_path, args.dataset_config_path, limit=args.limit)
    config = load_pipeline_model_config(Path("config/world_pipeline.yaml"))

    dataset_dir = Path("data") / args.dataset_name
    cache_dir = dataset_dir / "cache"
    question_variants_path = dataset_dir / "question_variants.jsonl"
    variant_results_path = dataset_dir / "variant_results.jsonl"
    failures_path = dataset_dir / "failures.jsonl"
    summary_path = dataset_dir / "summary.json"
    api_trace_path = cache_dir / "api_trace.jsonl"
    cache_path = cache_dir / "request_cache.sqlite"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_task_inputs(args.task_input_path, limit=args.limit)
    runner = AsyncRequestRunner(
        cache_path=cache_path,
        default_max_concurrency=8,
    )

    task_count = 0
    variant_count = 0
    all_direct_correct_count = 0
    resolved_by_gepa_count = 0
    failure_count = 0
    task_generation_failure_count = 0
    api_trace_count = 0
    direct_error_count = 0
    gepa_error_count = 0

    try:
        with (
            question_variants_path.open("wb") as question_variants_handle,
            variant_results_path.open("wb") as variant_results_handle,
            failures_path.open("wb") as failures_handle,
            api_trace_path.open("wb") as api_trace_handle,
        ):
            for seed in seeds:
                generation_user_message = render_generation_user_message(seed, args.variants_per_task)
                try:
                    variants, generation_trace = await generate_variants_for_seed(
                        seed=seed,
                        variants_per_task=args.variants_per_task,
                        runner=runner,
                        config=config,
                    )
                except Exception as exc:
                    task_generation_failure_count += 1
                    api_trace_count += 1
                    _write_jsonl_record(
                        api_trace_handle,
                        {
                            "stage": "variant_generation",
                            "question_id": seed.question_id,
                            "variant_id": None,
                            "model_name": config.variant_generator_model.name,
                            "prompt_text": VARIANT_GENERATION_SYSTEM_PROMPT,
                            "user_message": generation_user_message,
                            "response_text": None,
                            "cache_hit": False,
                            "attempt_count": 0,
                            "errors": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                    continue

                api_trace_count += 1
                _write_jsonl_record(api_trace_handle, generation_trace)

                task_count += 1
                variant_count += len(variants)
                _write_jsonl_record(question_variants_handle, _question_variants_row(seed, variants))
                print(f"task_generated={seed.question_id} variants={len(variants)}", flush=True)

                for variant in variants:
                    direct_results = await asyncio.gather(
                        *[
                            answer_with_prompt_safe(
                                variant=variant,
                                model_name=endpoint.name,
                                system_prompt=DIRECT_ANSWER_SYSTEM_PROMPT,
                                runner=runner,
                                config=config,
                            )
                            for endpoint in config.base_models
                        ]
                    )

                    for result in direct_results:
                        api_trace_count += _write_direct_trace(
                            api_trace_handle,
                            question_id=seed.question_id,
                            variant_id=variant.question_id,
                            result=result,
                        )

                    final_model_rows: dict[str, dict[str, Any]] = {}
                    optimization_inputs: list[tuple[str, str]] = []
                    variant_failed = False

                    for result in direct_results:
                        model_name = str(result["model_name"])
                        if result["error"] is not None:
                            direct_error_count += 1
                            variant_failed = True
                            continue
                        if result["correct"]:
                            final_model_rows[model_name] = {
                                "model_name": model_name,
                                "source": "direct",
                                "prompt_text": result["prompt_text"],
                                "response_text": result["raw_response"],
                            }
                            continue
                        optimization_inputs.append(
                            (
                                model_name,
                                _clean_text(result["answer_text"]) or _clean_text(result["raw_response"]) or "[empty response]",
                            )
                        )

                    used_gepa = False
                    if not variant_failed and optimization_inputs:
                        recovery_results = await asyncio.gather(
                            *[
                                maybe_recover_wrong_answer_safe(
                                    seed=seed,
                                    variant=variant,
                                    model_name=model_name,
                                    wrong_answer=wrong_answer,
                                    runner=runner,
                                    config=config,
                                )
                                for model_name, wrong_answer in optimization_inputs
                            ]
                        )
                        for recovery_result in recovery_results:
                            api_trace_count += _write_gepa_traces(
                                api_trace_handle,
                                question_id=seed.question_id,
                                variant_id=variant.question_id,
                                result=recovery_result,
                            )
                            if recovery_result["error"] is not None or not recovery_result["optimized_correct"]:
                                gepa_error_count += 1 if recovery_result["error"] is not None else 0
                                variant_failed = True
                                continue
                            used_gepa = True
                            final_model_rows[str(recovery_result["model_name"])] = {
                                "model_name": str(recovery_result["model_name"]),
                                "source": "gepa",
                                "prompt_text": recovery_result["optimized_prompt"],
                                "response_text": recovery_result["optimized_raw_response"],
                            }

                    if variant_failed or len(final_model_rows) != len(config.base_models):
                        failure_count += 1
                        _write_jsonl_record(
                            failures_handle,
                            {
                                "question_text": variant.question_text,
                            },
                        )
                        continue

                    status = "resolved_by_gepa" if used_gepa else "all_direct_correct"
                    if used_gepa:
                        resolved_by_gepa_count += 1
                    else:
                        all_direct_correct_count += 1

                    ordered_models = [
                        final_model_rows[endpoint.name]
                        for endpoint in config.base_models
                        if endpoint.name in final_model_rows
                    ]
                    _write_jsonl_record(
                        variant_results_handle,
                        _variant_result_row(
                            seed=seed,
                            variant=variant,
                            status=status,
                            models=ordered_models,
                        ),
                    )
                print(f"task_completed={seed.question_id}", flush=True)
    finally:
        await runner.aclose()

    summary_payload = {
        "dataset_name": args.dataset_name,
        "task_input_path": str(args.task_input_path),
        "task_count": task_count,
        "variant_count": variant_count,
        "all_direct_correct_count": all_direct_correct_count,
        "resolved_by_gepa_count": resolved_by_gepa_count,
        "failure_count": failure_count,
        "task_generation_failure_count": task_generation_failure_count,
        "direct_error_count": direct_error_count,
        "gepa_error_count": gepa_error_count,
        "api_trace_count": api_trace_count,
        "model_names": [endpoint.name for endpoint in config.base_models],
        "variant_generator_model": config.variant_generator_model.name,
        "prompt_optimizer_model": config.prompt_optimizer_model.name,
        "base_prompt_text": DIRECT_ANSWER_SYSTEM_PROMPT,
        "outputs": {
            "question_variants": str(question_variants_path),
            "variant_results": str(variant_results_path),
            "failures": str(failures_path),
            "summary": str(summary_path),
        },
        "cache": {
            "request_cache": str(cache_path),
            "api_trace": str(api_trace_path),
        },
    }
    write_json(summary_path, summary_payload)

    print(f"question_variants_path={question_variants_path}", flush=True)
    print(f"variant_results_path={variant_results_path}", flush=True)
    print(f"failures_path={failures_path}", flush=True)
    print(f"summary_path={summary_path}", flush=True)
    print(f"api_trace_path={api_trace_path}", flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
