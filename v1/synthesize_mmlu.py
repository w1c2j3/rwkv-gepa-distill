from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tomllib
from datasets import load_dataset
from openai import AsyncOpenAI


@dataclass(frozen=True)
class MMLUConfig:
    dataset_name: str
    subset: str
    split: str
    limit: int
    start_index: int


@dataclass(frozen=True)
class RewriteConfig:
    model: ModelConfig
    variant_count: int
    max_concurrency: int
    parse_retry_times: int


@dataclass(frozen=True)
class AnswerConfig:
    max_concurrency: int


@dataclass(frozen=True)
class ModelConfig:
    name: str
    base_url: str
    api_key: str
    enable_thinking: bool
    reasoning_effort: str | None


@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    original_jsonl: Path
    variants_jsonl: Path
    responses_jsonl: Path


@dataclass(frozen=True)
class RunConfig:
    resume: bool
    request_timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class AppConfig:
    mmlu: MMLUConfig
    rewrite: RewriteConfig
    answer: AnswerConfig
    answer_models: list[ModelConfig]
    output: OutputConfig
    run: RunConfig


@dataclass
class AppendLocks:
    original: asyncio.Lock
    variants: asyncio.Lock
    responses: asyncio.Lock


@dataclass
class FailureTracker:
    rewrite_failures: list[str]
    answer_failures: list[str]


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str | None


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize MMLU training data.")
    parser.add_argument("--config", default="config.toml")
    return parser.parse_args()


def require_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid table: [{key}]")
    return value


def require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid string: {key}")
    return value.strip()


def require_number(
    data: dict[str, Any], key: str, kind: type[int] | type[float]
) -> int | float:
    value = data.get(key)
    if kind is int:
        if not isinstance(value, int):
            raise ValueError(f"Missing or invalid integer: {key}")
        return value
    if not isinstance(value, (int, float)):
        raise ValueError(f"Missing or invalid number: {key}")
    return float(value)


def optional_bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Invalid boolean: {key}")
    return value


def optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid string: {key}")
    return value.strip()


def resolve_path(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_config(config_path: Path) -> AppConfig:
    data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    config_dir = config_path.parent.resolve()

    mmlu_raw = require_table(data, "mmlu")
    rewrite_raw = require_table(data, "rewrite")
    answer_raw = require_table(data, "answer")
    output_raw = require_table(data, "output")
    run_raw = require_table(data, "run")

    answer_models_raw = data.get("answer_models")
    if not isinstance(answer_models_raw, list) or not answer_models_raw:
        raise ValueError("Missing or invalid [[answer_models]] table array")

    mmlu = MMLUConfig(
        dataset_name=require_string(mmlu_raw, "dataset_name"),
        subset=require_string(mmlu_raw, "subset"),
        split=require_string(mmlu_raw, "split"),
        limit=int(require_number(mmlu_raw, "limit", int)),
        start_index=int(require_number(mmlu_raw, "start_index", int)),
    )
    rewrite_model = ModelConfig(
        name=require_string(rewrite_raw, "model"),
        base_url=require_string(rewrite_raw, "base_url"),
        api_key=require_string(rewrite_raw, "api_key"),
        enable_thinking=optional_bool(rewrite_raw, "enable_thinking", False),
        reasoning_effort=optional_string(rewrite_raw, "reasoning_effort"),
    )
    rewrite = RewriteConfig(
        model=rewrite_model,
        variant_count=int(require_number(rewrite_raw, "variant_count", int)),
        max_concurrency=int(require_number(rewrite_raw, "max_concurrency", int)),
        parse_retry_times=int(require_number(rewrite_raw, "parse_retry_times", int)),
    )
    answer = AnswerConfig(
        max_concurrency=int(require_number(answer_raw, "max_concurrency", int)),
    )

    answer_models: list[ModelConfig] = []
    for index, model_raw in enumerate(answer_models_raw):
        if not isinstance(model_raw, dict):
            raise ValueError(f"Invalid answer model entry at index {index}")
        answer_models.append(
            ModelConfig(
                name=require_string(model_raw, "name"),
                base_url=require_string(model_raw, "base_url"),
                api_key=require_string(model_raw, "api_key"),
                enable_thinking=optional_bool(model_raw, "enable_thinking", False),
                reasoning_effort=optional_string(model_raw, "reasoning_effort"),
            )
        )

    output_dir = resolve_path(config_dir, require_string(output_raw, "dir"))
    output = OutputConfig(
        dir=output_dir,
        original_jsonl=resolve_path(
            config_dir, require_string(output_raw, "original_jsonl")
        ),
        variants_jsonl=resolve_path(
            config_dir, require_string(output_raw, "variants_jsonl")
        ),
        responses_jsonl=resolve_path(
            config_dir, require_string(output_raw, "responses_jsonl")
        ),
    )
    resume_value = run_raw.get("resume")
    if not isinstance(resume_value, bool):
        raise ValueError("Missing or invalid boolean: resume")
    run = RunConfig(
        resume=resume_value,
        request_timeout_seconds=float(
            require_number(run_raw, "request_timeout_seconds", float)
        ),
        max_retries=int(require_number(run_raw, "max_retries", int)),
        retry_backoff_seconds=float(
            require_number(run_raw, "retry_backoff_seconds", float)
        ),
    )

    config = AppConfig(
        mmlu=mmlu,
        rewrite=rewrite,
        answer=answer,
        answer_models=answer_models,
        output=output,
        run=run,
    )
    validate_config(config)
    return config


def validate_config(config: AppConfig) -> None:
    allowed_reasoning_efforts = {"none", "low", "medium", "high", "xhigh"}
    if config.mmlu.limit < 0:
        raise ValueError("mmlu.limit must be non-negative")
    if config.mmlu.start_index < 0:
        raise ValueError("mmlu.start_index must be non-negative")
    if config.rewrite.variant_count <= 0:
        raise ValueError("rewrite.variant_count must be greater than 0")
    if config.rewrite.max_concurrency <= 0:
        raise ValueError("rewrite.max_concurrency must be greater than 0")
    if config.rewrite.parse_retry_times < 0:
        raise ValueError("rewrite.parse_retry_times must be non-negative")
    if config.answer.max_concurrency <= 0:
        raise ValueError("answer.max_concurrency must be greater than 0")
    if config.run.request_timeout_seconds <= 0:
        raise ValueError("run.request_timeout_seconds must be greater than 0")
    if config.run.max_retries < 0:
        raise ValueError("run.max_retries must be non-negative")
    if config.run.retry_backoff_seconds < 0:
        raise ValueError("run.retry_backoff_seconds must be non-negative")
    if not config.answer_models:
        raise ValueError("At least one answer model is required")
    if config.rewrite.variant_count % len(config.answer_models) != 0:
        raise ValueError(
            "rewrite.variant_count must be divisible by len(answer_models)"
        )
    all_models = [config.rewrite.model, *config.answer_models]
    for model in all_models:
        if (
            model.reasoning_effort is not None
            and model.reasoning_effort not in allowed_reasoning_efforts
        ):
            raise ValueError(
                f"{model.name}.reasoning_effort must be one of none/low/medium/high/xhigh"
            )


def ensure_output_paths(config: AppConfig) -> None:
    config.output.dir.mkdir(parents=True, exist_ok=True)
    for path in (
        config.output.original_jsonl,
        config.output.variants_jsonl,
        config.output.responses_jsonl,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not config.run.resume:
            path.write_text("", encoding="utf-8")
        elif not path.exists():
            path.touch()


def index_to_option_label(index: int) -> str:
    label = ""
    current = index
    while True:
        label = chr(ord("A") + (current % 26)) + label
        current = current // 26 - 1
        if current < 0:
            return label


def normalize_answer(answer: Any, choices: list[str]) -> str:
    if isinstance(answer, int):
        if 0 <= answer < len(choices):
            return index_to_option_label(answer)
        return str(answer)
    if isinstance(answer, str):
        stripped = answer.strip()
        upper = stripped.upper()
        if upper in {index_to_option_label(i) for i in range(len(choices))}:
            return upper
        if stripped.isdigit():
            numeric = int(stripped)
            if 0 <= numeric < len(choices):
                return index_to_option_label(numeric)
        for index, choice in enumerate(choices):
            if stripped == choice:
                return index_to_option_label(index)
        return stripped
    return str(answer)


def build_prompt(question: str, choices: list[str]) -> str:
    lines = [f"Question: {question.strip()}"]
    for index, choice in enumerate(choices):
        lines.append(f"{index_to_option_label(index)}. {str(choice).strip()}")
    return "\n".join(lines)


def build_rewrite_prompt(
    question: str, choices: list[str], answer: str, variant_count: int
) -> str:
    original_payload = json.dumps(
        {
            "question": question,
            "choices": choices,
            "answer": answer,
        },
        ensure_ascii=False,
        indent=2,
    )
    return (
        "You are generating training data variants.\n\n"
        "Carefully read the original multiple-choice question below.\n"
        f"Generate {variant_count} related multiple-choice variants suitable for model training.\n\n"
        "Requirements:\n"
        "1. Keep them semantically related to the original question.\n"
        "2. You may change numbers, entities, wording, scenario, or reasoning path.\n"
        "3. Preserve roughly similar difficulty.\n"
        "4. Each variant must be self-contained and unambiguous.\n"
        "5. Every variant must have exactly 4 choices.\n"
        "6. The answer must be a single uppercase letter: A, B, C, or D.\n"
        "7. Return only valid JSON. Do not use markdown fences. Do not include explanations.\n"
        "8. Output format:\n"
        "[\n"
        '  {"question": "...", "choices": ["...", "...", "...", "..."], "answer": "A"},\n'
        '  {"question": "...", "choices": ["...", "...", "...", "..."], "answer": "B"}\n'
        "]\n\n"
        "Original sample:\n"
        f"{original_payload}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )
            records.append(record)
    return records


def build_resume_state(
    config: AppConfig,
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    if not config.run.resume:
        return set(), set(), set()

    existing_originals = load_jsonl(config.output.original_jsonl)
    existing_variants = load_jsonl(config.output.variants_jsonl)
    existing_responses = load_jsonl(config.output.responses_jsonl)

    sample_ids = {
        str(record["sample_id"])
        for record in existing_originals
        if "sample_id" in record
    }
    variant_ids = {
        str(record["variant_id"])
        for record in existing_variants
        if is_valid_variant_record(record)
    }
    response_keys = {
        (str(record["variant_id"]), str(record["answer_model"]))
        for record in existing_responses
        if is_valid_response_record(record)
    }
    return sample_ids, variant_ids, response_keys


async def append_jsonl(path: Path, record: dict[str, Any], lock: asyncio.Lock) -> None:
    serialized = json.dumps(record, ensure_ascii=False)
    async with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")


def normalize_variant_record(item: dict[str, Any]) -> dict[str, Any]:
    question = item.get("question")
    raw_choices = item.get("choices")
    raw_answer = item.get("answer")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Variant is missing a non-empty string question")
    if not isinstance(raw_choices, list) or len(raw_choices) != 4:
        raise ValueError("Variant must contain exactly 4 choices")

    choices = [str(choice).strip() for choice in raw_choices]
    if any(not choice for choice in choices):
        raise ValueError("Variant choices must be non-empty strings")

    answer = normalize_answer(raw_answer, choices)
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError("Variant answer must normalize to one of A/B/C/D")

    return {
        "question": question.strip(),
        "choices": choices,
        "answer": answer,
    }


def is_valid_variant_record(record: dict[str, Any]) -> bool:
    try:
        normalize_variant_record(record)
    except Exception:  # noqa: BLE001
        return False
    return (
        isinstance(record.get("sample_id"), str)
        and isinstance(record.get("variant_id"), str)
        and isinstance(record.get("subject"), str)
        and isinstance(record.get("rewrite_model"), str)
    )


def is_valid_response_record(record: dict[str, Any]) -> bool:
    return (
        isinstance(record.get("variant_id"), str)
        and isinstance(record.get("answer_model"), str)
        and isinstance(record.get("prompt"), str)
        and isinstance(record.get("answer"), str)
        and isinstance(record.get("model_response"), str)
    )


def parse_variants(text: str, expected_count: int) -> list[dict[str, Any]]:
    payload: Any
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response does not contain a JSON array")
        payload = json.loads(text[start : end + 1])

    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Model response is not a JSON array")
    if len(payload) != expected_count:
        raise ValueError(f"Expected {expected_count} variants, got {len(payload)}")

    variants: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(
                f"Unsupported variant type at index {index}: {type(item).__name__}"
            )
        try:
            variants.append(normalize_variant_record(item))
        except ValueError as exc:
            raise ValueError(f"Variant at index {index} is invalid: {exc}") from exc

    return variants


async def call_chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    run_config: RunConfig,
    enable_thinking: bool = False,
    reasoning_effort: str | None = None,
) -> ChatResult:
    last_error: Exception | None = None
    for attempt in range(run_config.max_retries + 1):
        try:
            request_kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if enable_thinking:
                request_kwargs["extra_body"] = {"enable_thinking": True}
            if reasoning_effort is not None:
                request_kwargs["reasoning_effort"] = reasoning_effort

            response = await client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Model response content is empty")
            reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            if reasoning is not None and not isinstance(reasoning, str):
                reasoning = str(reasoning)
            return ChatResult(content=content, reasoning=reasoning)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= run_config.max_retries:
                break
            delay = run_config.retry_backoff_seconds * (2**attempt)
            await asyncio.sleep(delay)

    if last_error is None:
        raise RuntimeError("Chat completion failed without an exception")
    raise last_error


def choose_answer_model(
    rewrite_index: int,
    answer_models: list[ModelConfig],
    variants_per_model: int,
) -> ModelConfig:
    model_index = rewrite_index // variants_per_model
    return answer_models[model_index]


def parse_variant_index(variant_id: str) -> int:
    prefix, separator, suffix = variant_id.rpartition("_v")
    if not separator or not prefix or not suffix.isdigit():
        raise ValueError(f"Invalid variant_id format: {variant_id}")
    return int(suffix)


def load_mmlu_samples(config: AppConfig) -> list[dict[str, Any]]:
    log(
        "Loading dataset "
        f"{config.mmlu.dataset_name} subset={config.mmlu.subset} split={config.mmlu.split}"
    )
    dataset = load_dataset(
        config.mmlu.dataset_name,
        config.mmlu.subset,
        split=config.mmlu.split,
    )

    total_rows = len(dataset)
    start_index = config.mmlu.start_index
    end_index = min(start_index + config.mmlu.limit, total_rows)
    if start_index >= total_rows:
        return []

    records: list[dict[str, Any]] = []
    for global_index in range(start_index, end_index):
        row = dataset[global_index]
        question = str(row["question"]).strip()
        raw_choices = row["choices"]
        if not isinstance(raw_choices, list):
            raw_choices = list(raw_choices)
        choices = [str(choice).strip() for choice in raw_choices]
        answer = normalize_answer(row["answer"], choices)
        subject = str(row.get("subject", config.mmlu.subset)).strip()
        record = {
            "sample_id": f"mmlu_{global_index:06d}",
            "subject": subject,
            "question": question,
            "choices": choices,
            "answer": answer,
        }
        records.append(record)

    log(f"Loaded {len(records)} MMLU samples")
    return records


def build_openai_client(
    base_url: str, api_key: str, timeout_seconds: float
) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
        max_retries=0,
    )


async def write_original_rows(
    samples: list[dict[str, Any]],
    config: AppConfig,
    locks: AppendLocks,
    existing_sample_ids: set[str],
) -> None:
    written = 0
    for sample in samples:
        sample_id = str(sample["sample_id"])
        if sample_id in existing_sample_ids:
            continue
        await append_jsonl(config.output.original_jsonl, sample, locks.original)
        existing_sample_ids.add(sample_id)
        written += 1
    log(f"Wrote {written} new original samples to {config.output.original_jsonl}")


async def rewrite_single_sample(
    sample: dict[str, Any],
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppConfig,
    locks: AppendLocks,
    existing_variant_ids: set[str],
    failure_tracker: FailureTracker,
) -> None:
    sample_id = str(sample["sample_id"])
    missing_variant_ids = [
        f"{sample_id}_v{rewrite_index:03d}"
        for rewrite_index in range(config.rewrite.variant_count)
        if f"{sample_id}_v{rewrite_index:03d}" not in existing_variant_ids
    ]
    if not missing_variant_ids:
        return

    prompt = build_rewrite_prompt(
        sample["question"],
        sample["choices"],
        sample["answer"],
        config.rewrite.variant_count,
    )

    variants: list[dict[str, Any]] | None = None
    last_error: Exception | None = None
    for attempt in range(config.rewrite.parse_retry_times + 1):
        try:
            async with semaphore:
                result = await call_chat_completion(
                    client,
                    model=config.rewrite.model.name,
                    prompt=prompt,
                    run_config=config.run,
                    enable_thinking=config.rewrite.model.enable_thinking,
                    reasoning_effort=config.rewrite.model.reasoning_effort,
                )
            variants = parse_variants(result.content, config.rewrite.variant_count)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= config.rewrite.parse_retry_times:
                log(f"Rewrite failed for {sample_id}: {exc}")
                failure_tracker.rewrite_failures.append(sample_id)
                return
            log(
                f"Rewrite parse failed for {sample_id}, retrying "
                f"({attempt + 1}/{config.rewrite.parse_retry_times}): {exc}"
            )

    if variants is None:
        if last_error is not None:
            log(f"Rewrite failed for {sample_id}: {last_error}")
        failure_tracker.rewrite_failures.append(sample_id)
        return

    for rewrite_index, variant in enumerate(variants):
        variant_id = f"{sample_id}_v{rewrite_index:03d}"
        if variant_id in existing_variant_ids:
            continue
        record = {
            "sample_id": sample_id,
            "variant_id": variant_id,
            "subject": sample["subject"],
            "rewrite_model": config.rewrite.model.name,
            "question": variant["question"],
            "choices": variant["choices"],
            "answer": variant["answer"],
        }
        await append_jsonl(config.output.variants_jsonl, record, locks.variants)
        existing_variant_ids.add(variant_id)


async def generate_variants(
    samples: list[dict[str, Any]],
    *,
    config: AppConfig,
    locks: AppendLocks,
    existing_variant_ids: set[str],
    failure_tracker: FailureTracker,
) -> None:
    semaphore = asyncio.Semaphore(config.rewrite.max_concurrency)
    client = build_openai_client(
        base_url=config.rewrite.model.base_url,
        api_key=config.rewrite.model.api_key,
        timeout_seconds=config.run.request_timeout_seconds,
    )
    tasks = [
        rewrite_single_sample(
            sample,
            client=client,
            semaphore=semaphore,
            config=config,
            locks=locks,
            existing_variant_ids=existing_variant_ids,
            failure_tracker=failure_tracker,
        )
        for sample in samples
    ]
    await asyncio.gather(*tasks)


def load_variant_records(
    path: Path, allowed_sample_ids: set[str]
) -> list[dict[str, Any]]:
    raw_records = load_jsonl(path)
    deduped: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        if not is_valid_variant_record(record):
            continue
        variant_id = record.get("variant_id")
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in allowed_sample_ids:
            continue
        if isinstance(variant_id, str) and variant_id not in deduped:
            deduped[variant_id] = record
    records = list(deduped.values())
    records.sort(key=lambda record: str(record["variant_id"]))
    return records


async def answer_single_variant(
    variant_record: dict[str, Any],
    *,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    config: AppConfig,
    locks: AppendLocks,
    response_keys: set[tuple[str, str]],
    variants_per_model: int,
    failure_tracker: FailureTracker,
) -> None:
    rewrite_index = parse_variant_index(str(variant_record["variant_id"]))
    model_config = choose_answer_model(
        rewrite_index, config.answer_models, variants_per_model
    )
    response_key = (str(variant_record["variant_id"]), model_config.name)
    if response_key in response_keys:
        return

    prompt = build_prompt(
        str(variant_record["question"]),
        list(variant_record["choices"]),
    )

    try:
        async with semaphore:
            result = await call_chat_completion(
                client,
                model=model_config.name,
                prompt=prompt,
                run_config=config.run,
                enable_thinking=model_config.enable_thinking,
                reasoning_effort=model_config.reasoning_effort,
            )
    except Exception as exc:  # noqa: BLE001
        log(
            f"Answer failed for {variant_record['variant_id']} with {model_config.name}: {exc}"
        )
        failure_tracker.answer_failures.append(
            f"{variant_record['variant_id']}::{model_config.name}"
        )
        return

    response_record = {
        "sample_id": variant_record["sample_id"],
        "variant_id": variant_record["variant_id"],
        "subject": variant_record["subject"],
        "rewrite_model": variant_record["rewrite_model"],
        "answer_model": model_config.name,
        "prompt": prompt,
        "answer": variant_record["answer"],
        "model_reasoning": result.reasoning,
        "model_response": result.content,
    }
    await append_jsonl(config.output.responses_jsonl, response_record, locks.responses)
    response_keys.add(response_key)


async def generate_responses(
    variants: list[dict[str, Any]],
    *,
    config: AppConfig,
    locks: AppendLocks,
    response_keys: set[tuple[str, str]],
    failure_tracker: FailureTracker,
) -> None:
    semaphore = asyncio.Semaphore(config.answer.max_concurrency)
    variants_per_model = config.rewrite.variant_count // len(config.answer_models)
    clients = {
        model.name: build_openai_client(
            base_url=model.base_url,
            api_key=model.api_key,
            timeout_seconds=config.run.request_timeout_seconds,
        )
        for model in config.answer_models
    }
    tasks = [
        answer_single_variant(
            variant_record,
            client=clients[
                choose_answer_model(
                    parse_variant_index(str(variant_record["variant_id"])),
                    config.answer_models,
                    variants_per_model,
                ).name
            ],
            semaphore=semaphore,
            config=config,
            locks=locks,
            response_keys=response_keys,
            variants_per_model=variants_per_model,
            failure_tracker=failure_tracker,
        )
        for variant_record in variants
    ]
    await asyncio.gather(*tasks)


def summarize_distribution(
    variants: list[dict[str, Any]], config: AppConfig
) -> dict[str, int]:
    counts = defaultdict(int)
    variants_per_model = config.rewrite.variant_count // len(config.answer_models)
    for variant_record in variants:
        rewrite_index = parse_variant_index(str(variant_record["variant_id"]))
        model = choose_answer_model(
            rewrite_index, config.answer_models, variants_per_model
        )
        counts[model.name] += 1
    return dict(counts)


async def async_main(config: AppConfig) -> int:
    ensure_output_paths(config)
    existing_sample_ids, existing_variant_ids, response_keys = build_resume_state(
        config
    )
    locks = AppendLocks(
        original=asyncio.Lock(),
        variants=asyncio.Lock(),
        responses=asyncio.Lock(),
    )
    failure_tracker = FailureTracker(rewrite_failures=[], answer_failures=[])

    samples = load_mmlu_samples(config)
    if not samples:
        log("No samples selected. Nothing to do.")
        return 0

    await write_original_rows(samples, config, locks, existing_sample_ids)

    log("Starting rewrite stage")
    await generate_variants(
        samples,
        config=config,
        locks=locks,
        existing_variant_ids=existing_variant_ids,
        failure_tracker=failure_tracker,
    )

    allowed_sample_ids = {str(sample["sample_id"]) for sample in samples}
    variants = load_variant_records(config.output.variants_jsonl, allowed_sample_ids)
    if not variants:
        log("No variants available after rewrite stage. Skipping answer stage.")
        return 0

    distribution = summarize_distribution(variants, config)
    log(
        f"Variant distribution across answer models: {json.dumps(distribution, ensure_ascii=False)}"
    )

    log("Starting answer stage")
    await generate_responses(
        variants,
        config=config,
        locks=locks,
        response_keys=response_keys,
        failure_tracker=failure_tracker,
    )

    log(
        "Run finished with "
        f"{len(failure_tracker.rewrite_failures)} rewrite failures and "
        f"{len(failure_tracker.answer_failures)} answer failures"
    )
    if failure_tracker.rewrite_failures:
        log(f"Rewrite failures: {', '.join(failure_tracker.rewrite_failures)}")
    if failure_tracker.answer_failures:
        log(f"Answer failures: {', '.join(failure_tracker.answer_failures)}")

    return 0


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
        return asyncio.run(async_main(config))
    except Exception as exc:  # noqa: BLE001
        log(f"Fatal error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
