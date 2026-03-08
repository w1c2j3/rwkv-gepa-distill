from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable

import orjson

try:
    from datasets import get_dataset_config_names, load_dataset
except ImportError:  # pragma: no cover
    get_dataset_config_names = None
    load_dataset = None

from .mmlu_adapter_shared import (
    build_record,
    clean_text,
    format_subject_name,
    infer_cais_auxiliary_subject,
    interleave_row_groups,
    normalize_choices,
    progress_iter,
    tokenize_subject_text,
)

from .dataset_toml import DatasetSourceConfig
from .question_pools import QuestionPoolRecord, iter_question_pool
from .world_schema import (
    BenchmarkQuestion,
    QUESTION_TYPE_MULTIPLE_CHOICE,
    QUESTION_TYPE_OPEN_QA,
)


def _require_dataset_apis() -> tuple[Callable[..., Any], Callable[..., Any]]:
    if load_dataset is None or get_dataset_config_names is None:
        raise RuntimeError("Missing dependency: datasets. Run `bash scripts/bootstrap.sh` first.")
    return load_dataset, get_dataset_config_names


def _normalize_domain(value: Any, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip().replace("_", " ")
    return fallback


def _normalize_question_pool_record(
    record: QuestionPoolRecord,
    *,
    benchmark_name: str,
    question_id: str,
) -> BenchmarkQuestion:
    return BenchmarkQuestion(
        benchmark_name=benchmark_name,
        split=record.source_split.strip() or "train",
        domain=_normalize_domain(record.subject, fallback=benchmark_name),
        question_id=question_id,
        question_type=QUESTION_TYPE_MULTIPLE_CHOICE,
        question_text=record.question.strip(),
        choices=list(record.choices),
        gold_answer=record.answer.strip(),
        gold_answer_index=record.answer_index,
        gold_aliases=[record.answer.strip()] if record.answer.strip() else [],
        metadata={
            "source_dataset": record.source_dataset,
            "source_split": record.source_split,
            "prompt_text": record.prompt_text,
            "source_meta": record.meta,
        },
    )


def _normalize_built_mcq_row(
    row: dict[str, Any],
    *,
    benchmark_name: str,
    question_id: str,
) -> BenchmarkQuestion:
    answer = str(row.get("answer", "")).strip()
    return BenchmarkQuestion(
        benchmark_name=benchmark_name,
        split=str(row.get("source_split", "train")).strip() or "train",
        domain=_normalize_domain(row.get("subject"), fallback=benchmark_name),
        question_id=question_id,
        question_type=QUESTION_TYPE_MULTIPLE_CHOICE,
        question_text=str(row.get("question", "")).strip(),
        choices=[str(item).strip() for item in row.get("choices", [])],
        gold_answer=answer,
        gold_answer_index=row.get("answer_index"),
        gold_aliases=[answer] if answer else [],
        metadata={
            "source_dataset": row.get("source_dataset"),
            "source_split": row.get("source_split"),
            "source_meta": row.get("meta", {}),
        },
    )


def _trivia_answer_aliases(answer_payload: Any) -> tuple[str, list[str]]:
    if not isinstance(answer_payload, dict):
        return "", []
    value = answer_payload.get("value")
    aliases = answer_payload.get("aliases", [])
    normalized_aliases = answer_payload.get("normalized_aliases", [])

    parsed_aliases: list[str] = []
    for bucket in (aliases, normalized_aliases):
        if isinstance(bucket, list):
            parsed_aliases.extend(str(item).strip() for item in bucket if str(item).strip())

    canonical = str(value).strip() if value is not None else ""
    if canonical and canonical not in parsed_aliases:
        parsed_aliases.insert(0, canonical)
    return canonical, parsed_aliases


def _require_huggingface(source: DatasetSourceConfig) -> tuple[str, str | None, str | None]:
    if source.huggingface is None:
        raise ValueError(f"Source {source.name!r} is missing huggingface configuration")
    return (
        source.huggingface.repo_id,
        source.huggingface.config_name or None,
        source.huggingface.revision or None,
    )


def _load_huggingface_split(
    source: DatasetSourceConfig,
    *,
    split: str,
    config_name: str | None = None,
) -> Any:
    dataset_loader, _ = _require_dataset_apis()
    repo_id, default_config_name, revision = _require_huggingface(source)
    resolved_config_name = config_name if config_name is not None else default_config_name

    split_attempts = [split]
    if split == "auxiliary_train":
        split_attempts = ["auxiliary_train", "train"]
    elif split == "validation":
        split_attempts = ["validation", "val"]

    errors: list[str] = []
    for split_name in split_attempts:
        try:
            return dataset_loader(repo_id, resolved_config_name, split=split_name, revision=revision)
        except Exception as exc:  # pragma: no cover
            errors.append(f"config={resolved_config_name!r}, split={split_name!r}: {exc}")

    joined = " | ".join(errors)
    raise RuntimeError(f"Unable to load {repo_id!r} for source {source.name!r}. {joined}")


def _load_subject_configs(source: DatasetSourceConfig) -> list[str]:
    _, config_name_loader = _require_dataset_apis()
    repo_id, _, revision = _require_huggingface(source)
    config_names = [
        name
        for name in config_name_loader(repo_id, revision=revision)
        if name not in {"all", "auxiliary_train"}
    ]
    if not config_names:
        raise RuntimeError(f"No subject configs found for {repo_id!r}")
    return sorted(config_names)


def _subject_index_cache_path(dataset_root: Path, source: DatasetSourceConfig) -> Path:
    return dataset_root / "cache" / f"{source.name}_subject_index.json"


def _load_subject_index_cache(path: Path) -> dict[str, Counter[str]] | None:
    if not path.exists():
        return None
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid subject index cache at {path}")
    return {
        clean_text(subject): Counter({clean_text(token): int(count) for token, count in token_counts.items()})
        for subject, token_counts in payload.items()
        if isinstance(token_counts, dict)
    }


def _save_subject_index_cache(path: Path, subject_token_index: dict[str, Counter[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        subject: dict(counter.most_common())
        for subject, counter in subject_token_index.items()
    }
    path.write_bytes(orjson.dumps(payload))


def _build_subject_index(dataset_root: Path, source: DatasetSourceConfig) -> dict[str, Counter[str]]:
    cache_path = _subject_index_cache_path(dataset_root, source)
    cached = _load_subject_index_cache(cache_path)
    if cached:
        return cached

    subject_token_index: dict[str, Counter[str]] = {}
    for config_name in progress_iter(
        _load_subject_configs(source),
        desc=f"Building subject index for {source.name}",
        total=None,
    ):
        counter: Counter[str] = Counter()
        for split_name in ("dev", "validation"):
            dataset = _load_huggingface_split(source, split=split_name, config_name=config_name)
            for record in progress_iter(
                dataset,
                desc=f"Index {config_name}:{split_name}",
                total=len(dataset),
            ):
                text = " ".join(
                    [
                        clean_text(record.get("question")),
                        *normalize_choices(record.get("choices")),
                    ]
                )
                counter.update(tokenize_subject_text(text))
        if not counter:
            raise RuntimeError(f"No labeled vocabulary found for subject config {config_name!r}")
        subject_token_index[config_name] = counter

    _save_subject_index_cache(cache_path, subject_token_index)
    return subject_token_index


def _build_cais_labeled_pool(
    dataset_root: Path,
    source: DatasetSourceConfig,
    *,
    split: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    row_groups: list[list[dict[str, Any]]] = []
    for config_name in progress_iter(
        _load_subject_configs(source),
        desc=f"Preparing {source.name} {split}",
        total=None,
    ):
        dataset = _load_huggingface_split(source, split=split, config_name=config_name)
        rows_for_subject: list[dict[str, Any]] = []
        for record in progress_iter(
            dataset,
            desc=f"{split}:{config_name}",
            total=len(dataset),
        ):
            question = clean_text(record.get("question"))
            choices = normalize_choices(record.get("choices"))
            subject = format_subject_name(clean_text(record.get("subject"))) or format_subject_name(config_name)
            rows_for_subject.append(
                build_record(
                    source_dataset=source.huggingface.repo_id if source.huggingface else source.name,
                    source_split=split,
                    subject=subject,
                    question=question,
                    choices=choices,
                    raw_answer=record.get("answer"),
                    meta={
                        "source_subset": config_name,
                        "source_subject_config": config_name,
                    },
                )
            )
        if not rows_for_subject:
            raise RuntimeError(f"Loaded split {split!r} config {config_name!r} but no rows were returned.")
        row_groups.append(rows_for_subject)

    rows = interleave_row_groups(row_groups, limit)
    if not rows:
        raise RuntimeError(f"Loaded split {split!r} but no rows were returned.")
    return rows


def _build_cais_auxiliary_train_pool(
    dataset_root: Path,
    source: DatasetSourceConfig,
    limit: int | None,
) -> list[dict[str, Any]]:
    dataset = _load_huggingface_split(source, split="auxiliary_train", config_name=source.huggingface.config_name or "all")
    subject_token_index = _build_subject_index(dataset_root, source)

    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in progress_iter(
        dataset,
        desc=f"Preparing {source.name} auxiliary_train",
        total=len(dataset),
    ):
        question = clean_text(record.get("question"))
        choices = normalize_choices(record.get("choices"))
        subject, subject_meta = infer_cais_auxiliary_subject(question, choices, subject_token_index)
        normalized = build_record(
            source_dataset=source.huggingface.repo_id if source.huggingface else source.name,
            source_split="auxiliary_train",
            subject=subject,
            question=question,
            choices=choices,
            raw_answer=record.get("answer"),
            meta={
                "source_subset": source.huggingface.config_name or "all" if source.huggingface else "all",
                **subject_meta,
            },
        )
        subject_key = clean_text(str(normalized["meta"].get("source_subject_config", subject)))
        if not subject_key:
            subject_key = subject
        rows_by_subject.setdefault(subject_key, []).append(normalized)

    rows = interleave_row_groups([rows_by_subject[key] for key in sorted(rows_by_subject)], limit)
    if not rows:
        raise RuntimeError("Loaded auxiliary_train split but no rows were returned.")
    return rows


def _build_cais_mmlu(dataset_root: Path, source: DatasetSourceConfig, limit: int | None) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    for split in ("auxiliary_train", "dev", "validation"):
        if split == "auxiliary_train":
            rows = _build_cais_auxiliary_train_pool(dataset_root, source, limit)
        else:
            rows = _build_cais_labeled_pool(dataset_root, source, split=split, limit=limit)
        for index, row in enumerate(rows):
            questions.append(
                _normalize_built_mcq_row(
                    row,
                    benchmark_name=source.name,
                    question_id=f"{source.name}::{split}::{index}",
                )
            )
    return questions


def _build_cais_mmlu_auxiliary_train(
    dataset_root: Path,
    source: DatasetSourceConfig,
    limit: int | None,
) -> list[BenchmarkQuestion]:
    rows = _build_cais_auxiliary_train_pool(dataset_root, source, limit)
    return [
        _normalize_built_mcq_row(
            row,
            benchmark_name=source.name,
            question_id=f"{source.name}::auxiliary_train::{index}",
        )
        for index, row in enumerate(rows)
    ]


def _build_trivia_qa_rc(
    dataset_root: Path,
    source: DatasetSourceConfig,
    limit: int | None,
) -> list[BenchmarkQuestion]:
    del dataset_root
    if source.huggingface is None:
        raise ValueError(f"Source {source.name!r} is missing huggingface configuration")

    questions: list[BenchmarkQuestion] = []
    for split in ("train", "validation"):
        dataset = _load_huggingface_split(source, split=split)
        split_count = 0
        for index, row in enumerate(dataset):
            canonical_answer, aliases = _trivia_answer_aliases(row.get("answer"))
            if not canonical_answer:
                continue
            question_id = str(row.get("question_id") or f"{source.name}::{split}::{index}").strip()
            questions.append(
                BenchmarkQuestion(
                    benchmark_name=source.name,
                    split=split,
                    domain=source.name,
                    question_id=question_id,
                    question_type=QUESTION_TYPE_OPEN_QA,
                    question_text=str(row.get("question", "")).strip(),
                    choices=[],
                    gold_answer=canonical_answer,
                    gold_answer_index=None,
                    gold_aliases=aliases,
                    metadata={
                        "source_dataset": source.huggingface.repo_id,
                        "source_split": split,
                    },
                )
            )
            split_count += 1
            if limit is not None and split_count >= limit:
                break
    return questions


def _resolve_local_jsonl_path(dataset_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    dataset_relative = dataset_root / candidate
    if dataset_relative.exists():
        return dataset_relative
    return candidate


def _build_local_question_pool_mcq(
    dataset_root: Path,
    source: DatasetSourceConfig,
    limit: int | None,
) -> list[BenchmarkQuestion]:
    if source.local_jsonl is None:
        raise ValueError(f"Source {source.name!r} is missing local_jsonl configuration")
    source_path = _resolve_local_jsonl_path(dataset_root, source.local_jsonl.path)
    records = iter_question_pool(source_path, limit=limit)
    return [
        _normalize_question_pool_record(
            record,
            benchmark_name=source.name,
            question_id=f"{source.name}::{record.source_split}::{index}",
        )
        for index, record in enumerate(records)
    ]


ADAPTERS: dict[str, Callable[[Path, DatasetSourceConfig, int | None], list[BenchmarkQuestion]]] = {
    "mmlu_cais": _build_cais_mmlu,
    "cais_mmlu_auxiliary_train": _build_cais_mmlu_auxiliary_train,
    "trivia_qa_rc": _build_trivia_qa_rc,
    "question_pool_mcq": _build_local_question_pool_mcq,
}


def build_questions_from_source(
    dataset_root: Path,
    source: DatasetSourceConfig,
    limit: int | None,
) -> list[BenchmarkQuestion]:
    builder = ADAPTERS.get(source.adapter)
    if builder is None:
        raise ValueError(f"Unsupported adapter {source.adapter!r} for source {source.name!r}")
    return builder(dataset_root, source, limit)
