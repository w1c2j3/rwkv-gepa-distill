from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import orjson

from .async_request_runner import AsyncRequestRunner
from .common import prompt_version, write_json
from .dataset_adapters import build_questions_from_source
from .dataset_toml import load_dataset_config
from .model_registry import load_pipeline_model_config
from .seed_pipeline_schema import SeedInput
from .world_prompts import (
    DIRECT_ANSWER_SYSTEM_PROMPT,
    PROMPT_OPTIMIZER_SYSTEM_PROMPT,
    VARIANT_GENERATION_SYSTEM_PROMPT,
)
from .world_scoring import extract_json_object, score_with_optional_repair
from .world_schema import BenchmarkQuestion, iter_benchmark_questions, write_benchmark_questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the seed-driven distillation pipeline.")
    parser.add_argument("--dataset-name", "--dataset-version", dest="dataset_name", default="mmlu_auxiliary_train_seed_run")
    parser.add_argument("--seed-input-path", type=Path, default=Path("data/mmlu_auxiliary_train/questions.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants-per-seed", type=int, default=12)
    return parser.parse_args()


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def normalize_seed_payload(payload: dict[str, Any], source: Path, line_number: int) -> dict[str, Any]:
    contract = payload.get("contract")
    if contract == "world_question_v1":
        question = BenchmarkQuestion.from_dict(payload, source, line_number)
        return {
            "seed_id": question.question_id,
            "domain": question.domain,
            "question_type": question.question_type,
            "question": question.question_text,
            "choices": question.choices,
            "answer": question.gold_answer,
            "answer_index": question.gold_answer_index,
            "answer_aliases": question.gold_aliases,
            "metadata": question.metadata,
        }
    return payload


def load_seed_inputs(path: Path, *, limit: int | None) -> list[SeedInput]:
    seeds: list[SeedInput] = []
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            seeds.append(
                SeedInput.from_dict(
                    normalize_seed_payload(payload, path, line_number),
                    path,
                    line_number,
                )
            )
            if limit is not None and len(seeds) >= limit:
                break
    if not seeds:
        raise ValueError(f"No seed inputs found in {path}")
    return seeds


def default_dataset_config_path(dataset_version: str) -> Path:
    return Path("config/datasets") / f"{dataset_version}.toml"


def ensure_seed_input_path(seed_input_path: Path) -> None:
    if seed_input_path.exists() and seed_input_path.stat().st_size > 0:
        return
    if seed_input_path.name != "questions.jsonl":
        raise FileNotFoundError(f"Missing seed input file: {seed_input_path}")

    dataset_version = seed_input_path.parent.name
    dataset_config_path = default_dataset_config_path(dataset_version)
    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"Missing seed input file: {seed_input_path} and no dataset config found at {dataset_config_path}"
        )

    dataset_dir = seed_input_path.parent
    dataset_config = load_dataset_config(dataset_config_path, dataset_dir)
    merged_questions: list[BenchmarkQuestion] = []

    for source in dataset_config.sources:
        if not source.enabled:
            continue
        source_output_path = dataset_config.root_dir / source.output_file
        if source_output_path.exists() and source_output_path.stat().st_size > 0:
            source_questions = list(iter_benchmark_questions(source_output_path))
        else:
            source_questions = build_questions_from_source(dataset_config.root_dir, source, None)
            write_benchmark_questions(source_output_path, source_questions)
        if source.merge_into_world:
            merged_questions.extend(source_questions)

    if not merged_questions:
        raise ValueError(f"No questions were prepared from {dataset_config_path}")
    write_benchmark_questions(seed_input_path, merged_questions)


def parse_generated_variants(response_text: str, *, seed: SeedInput, variants_per_seed: int) -> list[dict[str, Any]]:
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
    for index, item in enumerate(raw_variants[:variants_per_seed]):
        if not isinstance(item, dict):
            continue
        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("answer"))
        if not question or not answer:
            continue
        variants.append(
            {
                "variant_id": f"{seed.seed_id}::variant::{index}",
                "seed_id": seed.seed_id,
                "variant_index": index,
                "domain": seed.domain,
                "question_type": seed.question_type,
                "question": question,
                "answer": answer,
                "choices": item.get("choices", []),
                "answer_index": item.get("answer_index"),
                "answer_aliases": item.get("answer_aliases", []),
                "generator_model": "",
                "generation_prompt_version": "",
                "metadata": {},
            }
        )
    if not variants:
        raise ValueError("Variant generation did not produce any usable variants")
    return variants


def render_generation_user_message(seed: SeedInput, variants_per_seed: int) -> str:
    lines = [
        f"请仔细思考，对于下面的原题，生成{variants_per_seed}个与它相关的变种问题，适合用于模型训练。",
        "你可以修改数字和语义等等。返回json格式的问题列表。",
        f"题目类型：{seed.question_type}",
        f"领域：{seed.domain}",
        "",
        "原题：",
        seed.question,
    ]
    if seed.question_type == "multiple_choice":
        lines.extend(["", "原题选项："])
        for choice in seed.choices:
            lines.append(f"- {choice}")
        lines.extend(
            [
                "",
                f"原题正确答案：{seed.answer}",
                "请为每个变种题同时返回 choices、answer、answer_index。",
            ]
        )
    else:
        lines.extend(["", f"原题正确答案：{seed.answer}", "请为每个变种题同时返回 answer。"])
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
    seed: SeedInput,
    variant: SeedInput,
    model_name: str,
    wrong_answer: str,
) -> str:
    lines = [
        f"Target Model: {model_name}",
        "Improve the system prompt so this model answers the generated question correctly.",
        "",
        "Original Seed Question:",
        seed.render_prompt(),
        "",
        "Generated Variant Question:",
        variant.render_prompt(),
        "",
        f"Gold Answer: {variant.answer}",
        f"Model Wrong Answer: {wrong_answer}",
    ]
    return "\n".join(lines).strip()


def build_distill_row(
    *,
    variant: SeedInput,
    model_name: str,
    model_answer: str,
    model_answer_index: int | None,
    source_type: str,
    optimized_system_prompt: str | None = None,
    direct_model_answer: str | None = None,
    direct_model_answer_index: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_type": source_type,
        "seed_id": variant.metadata.get("source_seed_id", variant.seed_id),
        "variant_id": variant.seed_id,
        "model_name": model_name,
        "question_type": variant.question_type,
        "question": variant.question,
        "gold_answer": variant.answer,
        "model_answer": model_answer,
    }
    if variant.choices:
        row["choices"] = list(variant.choices)
    if variant.answer_index is not None:
        row["gold_answer_index"] = variant.answer_index
    if model_answer_index is not None:
        row["model_answer_index"] = model_answer_index
    if optimized_system_prompt:
        row["optimized_system_prompt"] = optimized_system_prompt
        row["optimized_prompt_version"] = prompt_version(optimized_system_prompt)
    if direct_model_answer is not None:
        row["direct_model_answer"] = direct_model_answer
    if direct_model_answer_index is not None:
        row["direct_model_answer_index"] = direct_model_answer_index
    return row


def canonical_model_answer(score: Any, question: BenchmarkQuestion) -> tuple[str, int | None]:
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
    seed: SeedInput,
    variants_per_seed: int,
    runner: AsyncRequestRunner,
    config: Any,
) -> list[SeedInput]:
    generation = await runner.generate(
        endpoint=config.variant_generator_model,
        system_prompt=VARIANT_GENERATION_SYSTEM_PROMPT,
        user_message=render_generation_user_message(seed, variants_per_seed),
        attempts=2,
        validator=lambda text: parse_generated_variants(text, seed=seed, variants_per_seed=variants_per_seed),
        use_cache=True,
    )
    parsed = parse_generated_variants(generation.content, seed=seed, variants_per_seed=variants_per_seed)
    prompt_id = prompt_version(VARIANT_GENERATION_SYSTEM_PROMPT)
    variants: list[SeedInput] = []
    for index, payload in enumerate(parsed, start=1):
        variant_seed = SeedInput.from_dict(
            {
                "seed_id": payload["variant_id"],
                "domain": payload["domain"],
                "question_type": payload["question_type"],
                "question": payload["question"],
                "choices": payload["choices"],
                "answer": payload["answer"],
                "answer_index": payload["answer_index"],
                "answer_aliases": payload["answer_aliases"],
                "metadata": {
                    "source_seed_id": seed.seed_id,
                    "generator_model": generation.model_name,
                    "generation_prompt_version": prompt_id,
                    "variant_index": index - 1,
                },
            },
            Path("<generated-variant>"),
            index,
        )
        variants.append(variant_seed)
    return variants


async def answer_with_prompt(
    *,
    variant: SeedInput,
    model_name: str,
    system_prompt: str,
    runner: AsyncRequestRunner,
    config: Any,
) -> dict[str, Any]:
    benchmark_question = variant.to_benchmark_question()
    generation = await runner.generate(
        endpoint=config.base_model(model_name),
        system_prompt=system_prompt,
        user_message=variant.render_prompt(),
        attempts=2,
        use_cache=True,
    )
    _, score, _ = score_with_optional_repair(generation.content, benchmark_question)
    answer_text, answer_index = canonical_model_answer(score, benchmark_question)
    return {
        "raw_response": generation.content,
        "answer_text": answer_text,
        "answer_index": answer_index,
        "correct": score.correct,
    }


async def answer_with_prompt_safe(
    *,
    variant: SeedInput,
    model_name: str,
    system_prompt: str,
    runner: AsyncRequestRunner,
    config: Any,
) -> dict[str, Any]:
    try:
        result = await answer_with_prompt(
            variant=variant,
            model_name=model_name,
            system_prompt=system_prompt,
            runner=runner,
            config=config,
        )
        return {
            "model_name": model_name,
            "raw_response": result["raw_response"],
            "answer_text": result["answer_text"],
            "answer_index": result["answer_index"],
            "correct": result["correct"],
            "error": None,
        }
    except Exception as exc:
        return {
            "model_name": model_name,
            "raw_response": None,
            "answer_text": None,
            "answer_index": None,
            "correct": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def maybe_recover_wrong_answer(
    *,
    seed: SeedInput,
    variant: SeedInput,
    model_name: str,
    wrong_answer: str,
    runner: AsyncRequestRunner,
    config: Any,
) -> dict[str, Any]:
    optimizer_generation = await runner.generate(
        endpoint=config.prompt_optimizer_model,
        system_prompt=PROMPT_OPTIMIZER_SYSTEM_PROMPT,
        user_message=render_optimizer_user_message(
            seed=seed,
            variant=variant,
            model_name=model_name,
            wrong_answer=wrong_answer,
        ),
        attempts=2,
        validator=parse_optimized_prompt,
        use_cache=True,
    )
    optimized_prompt = parse_optimized_prompt(optimizer_generation.content)
    optimized_result = await answer_with_prompt(
        variant=variant,
        model_name=model_name,
        system_prompt=optimized_prompt,
        runner=runner,
        config=config,
    )
    return {
        "optimized_prompt": optimized_prompt,
        "optimized_answer": optimized_result["answer_text"],
        "optimized_answer_index": optimized_result["answer_index"],
        "optimized_correct": optimized_result["correct"],
        "optimized_raw_response": optimized_result["raw_response"],
    }


async def maybe_recover_wrong_answer_safe(
    *,
    seed: SeedInput,
    variant: SeedInput,
    model_name: str,
    wrong_answer: str,
    runner: AsyncRequestRunner,
    config: Any,
) -> dict[str, Any]:
    try:
        result = await maybe_recover_wrong_answer(
            seed=seed,
            variant=variant,
            model_name=model_name,
            wrong_answer=wrong_answer,
            runner=runner,
            config=config,
        )
        return {
            "model_name": model_name,
            "optimized_prompt": result["optimized_prompt"],
            "optimized_answer": result["optimized_answer"],
            "optimized_answer_index": result["optimized_answer_index"],
            "optimized_correct": result["optimized_correct"],
            "optimized_raw_response": result["optimized_raw_response"],
            "error": None,
        }
    except Exception as exc:
        return {
            "model_name": model_name,
            "optimized_prompt": None,
            "optimized_answer": None,
            "optimized_answer_index": None,
            "optimized_correct": False,
            "optimized_raw_response": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def async_main(args: argparse.Namespace) -> None:
    ensure_seed_input_path(args.seed_input_path)
    config = load_pipeline_model_config(Path("config/world_pipeline.yaml"))

    dataset_dir = Path("data") / args.dataset_name
    cache_dir = dataset_dir / "cache"
    generated_record_path = dataset_dir / "generated_questions.jsonl"
    distill_path = dataset_dir / "distill_data.jsonl"
    unresolved_path = dataset_dir / "unresolved_failures.jsonl"
    summary_path = cache_dir / "pipeline_summary.json"
    cache_path = cache_dir / "request_cache.sqlite"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = load_seed_inputs(args.seed_input_path, limit=args.limit)

    runner = AsyncRequestRunner(
        cache_path=cache_path,
        default_max_concurrency=8,
    )

    per_model_direct: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    per_model_optimized: dict[str, dict[str, int]] = defaultdict(lambda: {"attempted": 0, "recovered": 0})
    generated_variant_count = 0
    variant_counts_by_seed: list[int] = []
    direct_kept_count = 0
    optimized_kept_count = 0
    unresolved_count = 0
    seed_generation_failure_count = 0
    answer_failure_count = 0
    optimization_failure_count = 0

    try:
        with (
            generated_record_path.open("wb") as generated_handle,
            distill_path.open("wb") as distill_handle,
            unresolved_path.open("wb") as unresolved_handle,
        ):
            for seed in seeds:
                try:
                    variants = await generate_variants_for_seed(
                        seed=seed,
                        variants_per_seed=args.variants_per_seed,
                        runner=runner,
                        config=config,
                    )
                except Exception:
                    seed_generation_failure_count += 1
                    continue
                generated_variant_count += len(variants)
                variant_counts_by_seed.append(len(variants))
                generated_handle.write(
                    orjson.dumps(
                        {
                            "seed_id": seed.seed_id,
                            "domain": seed.domain,
                            "question_type": seed.question_type,
                            "question": seed.question,
                            "answer": seed.answer,
                            "answer_index": seed.answer_index,
                            "choices": seed.choices,
                            "generated_questions": [
                                {
                                    "variant_id": variant.seed_id,
                                    "question": variant.question,
                                    "answer": variant.answer,
                                    "answer_index": variant.answer_index,
                                    "choices": variant.choices,
                                    "generator_model": variant.metadata.get("generator_model"),
                                    "generation_prompt_version": variant.metadata.get("generation_prompt_version"),
                                }
                                for variant in variants
                            ],
                        }
                    )
                )
                generated_handle.write(b"\n")
                generated_handle.flush()
                print(f"seed_generated={seed.seed_id} variants={len(variants)}", flush=True)

                for variant in variants:
                    any_kept = False
                    failed_models: list[str] = []
                    optimized_failed_models: list[str] = []
                    attempt_records: dict[str, dict[str, Any]] = {}
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

                    optimization_inputs: list[tuple[str, str]] = []
                    for result in direct_results:
                        model_name = str(result["model_name"])
                        attempt_records[model_name] = {
                            "model_name": model_name,
                            "direct_model_answer": result["answer_text"],
                            "direct_model_answer_index": result["answer_index"],
                            "direct_error": result["error"],
                        }
                        per_model_direct[model_name]["total"] += 1
                        if result["error"] is not None:
                            answer_failure_count += 1
                            failed_models.append(model_name)
                            optimized_failed_models.append(model_name)
                            continue
                        answer_text = _clean_text(result["answer_text"])
                        is_correct = bool(result["correct"])
                        if is_correct:
                            per_model_direct[model_name]["correct"] += 1
                            distill_handle.write(
                                orjson.dumps(
                                    build_distill_row(
                                        variant=variant,
                                        model_name=model_name,
                                        model_answer=answer_text,
                                        model_answer_index=result["answer_index"],
                                        source_type="direct_answer",
                                    )
                                )
                            )
                            distill_handle.write(b"\n")
                            any_kept = True
                            direct_kept_count += 1
                            continue

                        failed_models.append(model_name)
                        per_model_optimized[model_name]["attempted"] += 1
                        optimization_inputs.append(
                            (
                                model_name,
                                answer_text or _clean_text(result["raw_response"]) or "[empty response]",
                            )
                        )

                    optimization_results = await asyncio.gather(
                        *[
                            maybe_recover_wrong_answer_safe(
                                seed=seed,
                                variant=variant,
                                model_name=model_name,
                                wrong_answer=answer_text,
                                runner=runner,
                                config=config,
                            )
                            for model_name, answer_text in optimization_inputs
                        ]
                    )
                    for result in optimization_results:
                        model_name = str(result["model_name"])
                        record = attempt_records.setdefault(model_name, {"model_name": model_name})
                        record["optimized_model_answer"] = result["optimized_answer"]
                        record["optimized_model_answer_index"] = result["optimized_answer_index"]
                        record["optimized_error"] = result["error"]
                        if result["error"] is not None:
                            optimization_failure_count += 1
                            optimized_failed_models.append(model_name)
                            continue
                        optimized_correct = bool(result["optimized_correct"])
                        if optimized_correct:
                            per_model_optimized[model_name]["recovered"] += 1
                            distill_handle.write(
                                orjson.dumps(
                                    build_distill_row(
                                        variant=variant,
                                        model_name=model_name,
                                        model_answer=_clean_text(result["optimized_answer"]),
                                        model_answer_index=result["optimized_answer_index"],
                                        source_type="optimized_answer",
                                        optimized_system_prompt=str(result["optimized_prompt"]),
                                        direct_model_answer=record.get("direct_model_answer"),
                                        direct_model_answer_index=record.get("direct_model_answer_index"),
                                    )
                                )
                            )
                            distill_handle.write(b"\n")
                            any_kept = True
                            optimized_kept_count += 1
                        else:
                            optimized_failed_models.append(model_name)

                    if not any_kept:
                        unresolved_handle.write(
                            orjson.dumps(
                                {
                                    "seed_id": seed.seed_id,
                                    "variant_id": variant.seed_id,
                                    "question": variant.question,
                                    "choices": variant.choices,
                                    "gold_answer": variant.answer,
                                    "gold_answer_index": variant.answer_index,
                                    "failed_models": failed_models,
                                    "optimization_attempted_models": optimized_failed_models,
                                    "model_attempts": [
                                        attempt_records[key]
                                        for key in sorted(attempt_records)
                                    ],
                                }
                            )
                        )
                        unresolved_handle.write(b"\n")
                        unresolved_count += 1
                    distill_handle.flush()
                    unresolved_handle.flush()
                print(f"seed_completed={seed.seed_id}", flush=True)
    finally:
        await runner.aclose()

    summary_payload = {
        "seed_input_path": str(args.seed_input_path),
        "seed_input_count": len(seeds),
        "generated_variant_count": generated_variant_count,
        "average_variants_per_seed": round(mean(variant_counts_by_seed), 6) if variant_counts_by_seed else 0.0,
        "direct_kept_count": direct_kept_count,
        "optimized_kept_count": optimized_kept_count,
        "distill_data_count": direct_kept_count + optimized_kept_count,
        "unresolved_failure_count": unresolved_count,
        "seed_generation_failure_count": seed_generation_failure_count,
        "answer_failure_count": answer_failure_count,
        "optimization_failure_count": optimization_failure_count,
        "per_model_direct": {
            model_name: {
                "total": counts["total"],
                "correct": counts["correct"],
                "correct_rate": round(counts["correct"] / max(1, counts["total"]), 6),
            }
            for model_name, counts in sorted(per_model_direct.items())
        },
        "per_model_optimized": {
            model_name: {
                "attempted": counts["attempted"],
                "recovered": counts["recovered"],
                "recovery_rate": round(counts["recovered"] / max(1, counts["attempted"]), 6),
            }
            for model_name, counts in sorted(per_model_optimized.items())
        },
        "outputs": {
            "generated_questions": str(generated_record_path),
            "distill_data": str(distill_path),
            "unresolved_failures": str(unresolved_path),
        },
    }
    write_json(summary_path, summary_payload)

    print(f"generated_questions_path={generated_record_path}", flush=True)
    print(f"distill_data_path={distill_path}", flush=True)
    print(f"unresolved_failures_path={unresolved_path}", flush=True)
    print(f"summary_path={summary_path}", flush=True)


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
