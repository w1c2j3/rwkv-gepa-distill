#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import orjson

try:
    from datasets import get_dataset_config_names, load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Run `bash scripts/bootstrap.sh` first."
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


ROOT_DIR = Path(__file__).resolve().parents[1]
QUESTION_POOLS_DIR = ROOT_DIR / "data" / "question_pools"
EVAL_POOLS_DIR = ROOT_DIR / "data" / "eval_pools"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CAIS_MMLU_SUBJECT_INDEX_PATH = ARTIFACTS_DIR / "cais_mmlu_subject_token_index.json"
ANSWER_LABELS = list(string.ascii_uppercase)
SUBJECT_INFERENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "to",
    "what",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class PoolSpec:
    pool_name: str
    category: str
    target_path: Path
    builder: Callable[[argparse.Namespace], list[dict[str, Any]]]


@dataclass(frozen=True)
class PoolSummary:
    pool_name: str
    target_path: Path
    rows_written: int
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare normalized MMLU question and evaluation pools as JSONL."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild target JSONL files even if they already exist and are non-empty.",
    )
    parser.add_argument(
        "--include-community",
        action="store_true",
        help="Also prepare optional community train pools.",
    )
    parser.add_argument(
        "--only",
        choices=("primary", "eval", "all"),
        default="all",
        help="Choose which pool groups to prepare.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to write per output file.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def file_is_non_empty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("wb") as handle:
        for row in rows:
            handle.write(orjson.dumps(row))
            handle.write(b"\n")
            count += 1
    return count


def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


def apply_limit(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if limit is None:
        return rows
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    return rows[:limit]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def format_subject_name(value: str) -> str:
    return clean_text(value.replace("_", " "))


def normalize_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value]
    if isinstance(value, tuple):
        return [clean_text(item) for item in value]
    return [clean_text(value)]


def answer_label(answer_index: int | None) -> str | None:
    if answer_index is None or answer_index < 0 or answer_index >= len(ANSWER_LABELS):
        return None
    return ANSWER_LABELS[answer_index]


def infer_answer_index(raw_answer: Any, choices: list[str]) -> int | None:
    if raw_answer is None:
        return None
    if isinstance(raw_answer, bool):
        return int(raw_answer)
    if isinstance(raw_answer, int):
        return raw_answer
    if isinstance(raw_answer, float) and raw_answer.is_integer():
        return int(raw_answer)

    answer_str = clean_text(raw_answer)
    if not answer_str:
        return None

    if answer_str.isdigit():
        return int(answer_str)

    upper = answer_str.upper()
    if len(upper) == 1 and upper in ANSWER_LABELS:
        return ANSWER_LABELS.index(upper)

    for index, choice in enumerate(choices):
        if choice == answer_str:
            return index
    return None


def canonical_answer(raw_answer: Any, answer_index: int | None, choices: list[str]) -> str:
    if answer_index is not None and 0 <= answer_index < len(choices):
        return choices[answer_index]
    return clean_text(raw_answer)


def build_prompt_text(subject: str, question: str, choices: list[str]) -> str:
    if not choices:
        return question

    lines = [
        f"Subject: {subject}",
        "",
        "Question:",
        question,
        "",
        "Options:",
    ]
    for index, choice in enumerate(choices):
        label = ANSWER_LABELS[index] if index < len(ANSWER_LABELS) else f"Option {index + 1}"
        lines.append(f"{label}. {choice}")
    return "\n".join(lines).strip()


def build_record(
    *,
    source_dataset: str,
    source_split: str,
    subject: str,
    question: str,
    choices: list[str],
    raw_answer: Any,
    meta: dict[str, Any],
) -> dict[str, Any]:
    answer_index = infer_answer_index(raw_answer, choices)
    return {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "subject": subject,
        "question": question,
        "choices": choices,
        "answer": canonical_answer(raw_answer, answer_index, choices),
        "answer_index": answer_index,
        "prompt_text": build_prompt_text(subject, question, choices),
        "meta": {
            **meta,
            "original_answer": raw_answer,
            "answer_label": answer_label(answer_index),
        },
    }


def cais_split_attempts(requested_split: str) -> list[str]:
    if requested_split == "auxiliary_train":
        return ["auxiliary_train", "train"]
    if requested_split == "validation":
        return ["validation", "val"]
    return [requested_split]


def load_cais_mmlu_split(requested_split: str, config_name: str = "all") -> tuple[Any, str, str]:
    attempts = cais_split_attempts(requested_split)

    errors: list[str] = []
    for split_name in attempts:
        try:
            dataset = load_dataset("cais/mmlu", name=config_name, split=split_name)
            return dataset, config_name, split_name
        except Exception as exc:
            errors.append(f"name={config_name!r}, split={split_name!r}: {exc}")

    raise RuntimeError(
        "Unable to load cais/mmlu with the expected config/split combinations. "
        + " | ".join(errors)
    )


def normalize_cais_mmlu_record(
    record: dict[str, Any],
    *,
    source_split: str,
    source_subset: str,
    default_subject: str,
) -> dict[str, Any]:
    question = clean_text(record.get("question"))
    choices = normalize_choices(record.get("choices"))
    subject = format_subject_name(clean_text(record.get("subject"))) or format_subject_name(default_subject)
    return build_record(
        source_dataset="cais/mmlu",
        source_split=source_split,
        subject=subject,
        question=question,
        choices=choices,
        raw_answer=record.get("answer"),
        meta={
            "source_subset": source_subset,
            "source_subject_config": default_subject,
        },
    )


def load_cais_mmlu_subject_configs() -> list[str]:
    config_names = [
        name
        for name in get_dataset_config_names("cais/mmlu")
        if name not in {"all", "auxiliary_train"}
    ]
    if not config_names:
        raise RuntimeError("No subject configs found for cais/mmlu.")
    return sorted(config_names)


def load_cached_cais_subject_token_index() -> dict[str, Counter[str]] | None:
    if not CAIS_MMLU_SUBJECT_INDEX_PATH.exists():
        return None
    payload = orjson.loads(CAIS_MMLU_SUBJECT_INDEX_PATH.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid subject index cache at {CAIS_MMLU_SUBJECT_INDEX_PATH}")
    return {
        clean_text(subject): Counter({clean_text(token): int(count) for token, count in token_counts.items()})
        for subject, token_counts in payload.items()
        if isinstance(token_counts, dict)
    }


def save_cais_subject_token_index(subject_token_index: dict[str, Counter[str]]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {
        subject: dict(counter.most_common())
        for subject, counter in subject_token_index.items()
    }
    CAIS_MMLU_SUBJECT_INDEX_PATH.write_bytes(orjson.dumps(serializable))


def tokenize_subject_text(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower())
        if token not in SUBJECT_INFERENCE_STOPWORDS
    ]


def build_cais_subject_token_index(config_names: list[str]) -> dict[str, Counter[str]]:
    cached = load_cached_cais_subject_token_index()
    if cached:
        return cached

    subject_token_index: dict[str, Counter[str]] = {}
    for config_name in progress_iter(
        config_names,
        desc="Building cais/mmlu subject index",
        total=len(config_names),
    ):
        counter: Counter[str] = Counter()
        for split_name in ("dev", "validation"):
            dataset, _, _ = load_cais_mmlu_split(split_name, config_name=config_name)
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
            raise RuntimeError(f"No labeled vocabulary found for cais/mmlu subject config {config_name!r}")
        subject_token_index[config_name] = counter
    save_cais_subject_token_index(subject_token_index)
    return subject_token_index


def infer_cais_auxiliary_subject(
    question: str,
    choices: list[str],
    subject_token_index: dict[str, Counter[str]],
) -> tuple[str, dict[str, Any]]:
    tokens = tokenize_subject_text(" ".join([question, *choices]))
    best_config = ""
    best_score = -1

    for config_name, counter in subject_token_index.items():
        score = sum(counter.get(token, 0) for token in set(tokens))
        if score > best_score:
            best_config = config_name
            best_score = score

    if not best_config or best_score <= 0:
        return "general knowledge", {
            "subject_source": "fallback_default",
            "subject_inference_score": 0,
        }

    return format_subject_name(best_config), {
        "subject_source": "inferred_from_labeled_splits",
        "subject_inference_score": best_score,
        "source_subject_config": best_config,
    }


def interleave_row_groups(
    row_groups: list[list[dict[str, Any]]],
    limit: int | None,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    max_group_size = max((len(group) for group in row_groups), default=0)
    for row_index in range(max_group_size):
        for group in row_groups:
            if row_index >= len(group):
                continue
            merged.append(group[row_index])
            if limit is not None and len(merged) >= limit:
                return merged
    return merged


def build_cais_mmlu_labeled_pool(requested_split: str, limit: int | None) -> list[dict[str, Any]]:
    row_groups: list[list[dict[str, Any]]] = []
    config_names = load_cais_mmlu_subject_configs()
    for config_name in progress_iter(
        config_names,
        desc=f"Preparing cais/mmlu {requested_split}",
        total=len(config_names),
    ):
        dataset, loaded_config, loaded_split = load_cais_mmlu_split(requested_split, config_name=config_name)
        rows_for_subject: list[dict[str, Any]] = []
        for record in progress_iter(
            dataset,
            desc=f"{requested_split}:{config_name}",
            total=len(dataset),
        ):
            rows_for_subject.append(
                normalize_cais_mmlu_record(
                    record,
                    source_split=requested_split,
                    source_subset=loaded_config,
                    default_subject=config_name,
                )
            )
        if not rows_for_subject:
            raise RuntimeError(
                f"Loaded cais/mmlu config {config_name!r} split {loaded_split!r} but no rows were returned."
            )
        row_groups.append(rows_for_subject)

    rows = interleave_row_groups(row_groups, limit)
    if not rows:
        raise RuntimeError(f"Loaded cais/mmlu split {requested_split!r} but no rows were returned.")
    return rows


def build_cais_mmlu_auxiliary_train_pool(limit: int | None) -> list[dict[str, Any]]:
    dataset, config_name, loaded_split = load_cais_mmlu_split("auxiliary_train", config_name="all")
    subject_token_index = build_cais_subject_token_index(load_cais_mmlu_subject_configs())

    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for record in progress_iter(
        dataset,
        desc="Preparing cais/mmlu auxiliary_train",
        total=len(dataset),
    ):
        question = clean_text(record.get("question"))
        choices = normalize_choices(record.get("choices"))
        subject, subject_meta = infer_cais_auxiliary_subject(question, choices, subject_token_index)
        normalized = build_record(
            source_dataset="cais/mmlu",
            source_split="auxiliary_train",
            subject=subject,
            question=question,
            choices=choices,
            raw_answer=record.get("answer"),
            meta={
                "source_subset": config_name,
                **subject_meta,
            },
        )
        subject_key = clean_text(str(normalized["meta"].get("source_subject_config", subject)))
        if not subject_key:
            subject_key = subject
        rows_by_subject.setdefault(subject_key, []).append(normalized)

    row_groups = [rows_by_subject[key] for key in sorted(rows_by_subject)]
    rows = interleave_row_groups(row_groups, limit)

    if not rows:
        raise RuntimeError(f"Loaded cais/mmlu split {loaded_split!r} but no rows were returned.")
    return rows


def build_cais_mmlu_pool(requested_split: str, limit: int | None) -> list[dict[str, Any]]:
    if requested_split == "auxiliary_train":
        return build_cais_mmlu_auxiliary_train_pool(limit)
    return build_cais_mmlu_labeled_pool(requested_split, limit)


def normalize_mmlu_pro_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    question = clean_text(record.get("question"))
    choices = normalize_choices(record.get("options"))
    subject = clean_text(record.get("subject") or record.get("category"))
    return build_record(
        source_dataset="TIGER-Lab/MMLU-Pro",
        source_split=split,
        subject=subject,
        question=question,
        choices=choices,
        raw_answer=record.get("answer"),
        meta={
            "category": record.get("category"),
            "source": record.get("src"),
            "question_id": record.get("question_id"),
            "provided_answer_index": record.get("answer_index"),
        },
    )


def build_mmlu_pro_pool(split: str, limit: int | None) -> list[dict[str, Any]]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    rows: list[dict[str, Any]] = []
    for record in progress_iter(
        dataset,
        desc=f"Preparing MMLU-Pro {split}",
        total=len(dataset),
    ):
        rows.append(normalize_mmlu_pro_record(record, split))
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise RuntimeError(f"Loaded TIGER-Lab/MMLU-Pro split {split!r} but no rows were returned.")
    return rows


def normalize_mmlu_redux_record(
    record: dict[str, Any],
    *,
    subject: str,
    split: str,
) -> dict[str, Any]:
    question = clean_text(record.get("question"))
    choices = normalize_choices(record.get("choices"))
    return build_record(
        source_dataset="edinburgh-dawg/mmlu-redux",
        source_split=split,
        subject=subject,
        question=question,
        choices=choices,
        raw_answer=record.get("answer"),
        meta={
            "source_subset": subject,
            "error_type": record.get("error_type"),
            "source": record.get("source"),
            "correct_answer": record.get("correct_answer"),
            "potential_reason": record.get("potential_reason"),
        },
    )


def build_mmlu_redux_pool(limit: int | None) -> list[dict[str, Any]]:
    config_names = [name for name in get_dataset_config_names("edinburgh-dawg/mmlu-redux") if name != "default"]
    if not config_names:
        raise RuntimeError("No config names found for edinburgh-dawg/mmlu-redux.")

    rows: list[dict[str, Any]] = []
    for config_name in progress_iter(
        config_names,
        desc="Preparing mmlu-redux test",
        total=len(config_names),
    ):
        dataset = load_dataset("edinburgh-dawg/mmlu-redux", name=config_name, split="test")
        for record in progress_iter(
            dataset,
            desc=f"mmlu-redux:{config_name}",
            total=len(dataset),
        ):
            rows.append(
                normalize_mmlu_redux_record(
                    record,
                    subject=config_name,
                    split="test",
                )
            )
            if limit is not None and len(rows) >= limit:
                return rows
    if not rows:
        raise RuntimeError("Loaded edinburgh-dawg/mmlu-redux but no rows were returned.")
    return rows


def infer_subject_from_prompt(prompt: str) -> str:
    match = re.search(r"about ([A-Za-z_ -]+?)\.", prompt)
    if match:
        return clean_text(match.group(1))
    return ""


def normalize_linggm_record(record: dict[str, Any]) -> dict[str, Any]:
    prompt = clean_text(record.get("prompt"))
    return {
        "source_dataset": "linggm/mmlu-train",
        "source_split": "train",
        "subject": infer_subject_from_prompt(prompt),
        "question": prompt,
        "choices": [],
        "answer": "",
        "answer_index": None,
        "prompt_text": prompt,
        "meta": {
            "community_format": "packed_prompt_with_label_sequence",
            "raw": record,
        },
    }


def build_linggm_pool(limit: int | None) -> list[dict[str, Any]]:
    dataset = load_dataset("linggm/mmlu-train", split="train")
    rows: list[dict[str, Any]] = []
    for record in progress_iter(
        dataset,
        desc="Preparing linggm/mmlu-train",
        total=len(dataset),
    ):
        rows.append(normalize_linggm_record(record))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def normalize_manu_record(record: dict[str, Any]) -> dict[str, Any]:
    text = clean_text(record.get("text"))
    return {
        "source_dataset": "manu/mmlu_auxiliary_train_unformatted",
        "source_split": "train",
        "subject": "",
        "question": text,
        "choices": [],
        "answer": "",
        "answer_index": None,
        "prompt_text": text,
        "meta": {
            "community_format": "unformatted_text_blob",
            "raw": record,
        },
    }


def build_manu_pool(limit: int | None) -> list[dict[str, Any]]:
    dataset = load_dataset("manu/mmlu_auxiliary_train_unformatted", split="train")
    rows: list[dict[str, Any]] = []
    for record in progress_iter(
        dataset,
        desc="Preparing manu/mmlu_auxiliary_train_unformatted",
        total=len(dataset),
    ):
        rows.append(normalize_manu_record(record))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def build_specs(args: argparse.Namespace) -> list[PoolSpec]:
    specs: list[PoolSpec] = []

    if args.only in ("primary", "all"):
        specs.extend(
            [
                PoolSpec(
                    pool_name="mmlu_auxiliary_train",
                    category="primary",
                    target_path=QUESTION_POOLS_DIR / "mmlu_auxiliary_train.jsonl",
                    builder=lambda current_args: build_cais_mmlu_pool("auxiliary_train", current_args.limit),
                ),
                PoolSpec(
                    pool_name="mmlu_dev",
                    category="primary",
                    target_path=QUESTION_POOLS_DIR / "mmlu_dev.jsonl",
                    builder=lambda current_args: build_cais_mmlu_pool("dev", current_args.limit),
                ),
                PoolSpec(
                    pool_name="mmlu_validation",
                    category="primary",
                    target_path=QUESTION_POOLS_DIR / "mmlu_validation.jsonl",
                    builder=lambda current_args: build_cais_mmlu_pool("validation", current_args.limit),
                ),
            ]
        )
        if args.include_community:
            specs.extend(
                [
                    PoolSpec(
                        pool_name="linggm_mmlu_train",
                        category="primary",
                        target_path=QUESTION_POOLS_DIR / "linggm_mmlu_train.jsonl",
                        builder=lambda current_args: build_linggm_pool(current_args.limit),
                    ),
                    PoolSpec(
                        pool_name="manu_mmlu_auxiliary_train_unformatted",
                        category="primary",
                        target_path=QUESTION_POOLS_DIR / "manu_mmlu_auxiliary_train_unformatted.jsonl",
                        builder=lambda current_args: build_manu_pool(current_args.limit),
                    ),
                ]
            )

    if args.only in ("eval", "all"):
        specs.extend(
            [
                PoolSpec(
                    pool_name="mmlu_test",
                    category="eval",
                    target_path=EVAL_POOLS_DIR / "mmlu_test.jsonl",
                    builder=lambda current_args: build_cais_mmlu_pool("test", current_args.limit),
                ),
                PoolSpec(
                    pool_name="mmlu_pro_validation",
                    category="eval",
                    target_path=EVAL_POOLS_DIR / "mmlu_pro_validation.jsonl",
                    builder=lambda current_args: build_mmlu_pro_pool("validation", current_args.limit),
                ),
                PoolSpec(
                    pool_name="mmlu_pro_test",
                    category="eval",
                    target_path=EVAL_POOLS_DIR / "mmlu_pro_test.jsonl",
                    builder=lambda current_args: build_mmlu_pro_pool("test", current_args.limit),
                ),
                PoolSpec(
                    pool_name="mmlu_redux_test",
                    category="eval",
                    target_path=EVAL_POOLS_DIR / "mmlu_redux_test.jsonl",
                    builder=lambda current_args: build_mmlu_redux_pool(current_args.limit),
                ),
            ]
        )

    return specs


def prepare_pool(spec: PoolSpec, args: argparse.Namespace) -> PoolSummary:
    if file_is_non_empty(spec.target_path) and not args.force:
        with spec.target_path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        print(
            f"pool={spec.pool_name} status=existing rows={line_count} path={spec.target_path}",
            flush=True,
        )
        return PoolSummary(
            pool_name=spec.pool_name,
            target_path=spec.target_path,
            rows_written=line_count,
            status="existing",
        )

    print(
        f"pool={spec.pool_name} status=preparing path={spec.target_path}",
        flush=True,
    )
    rows = apply_limit(spec.builder(args), args.limit)
    if not rows:
        raise RuntimeError(f"No rows produced for pool {spec.pool_name}")

    written = write_jsonl(spec.target_path, rows)
    print(
        f"pool={spec.pool_name} status=prepared rows={written} path={spec.target_path}",
        flush=True,
    )
    return PoolSummary(
        pool_name=spec.pool_name,
        target_path=spec.target_path,
        rows_written=written,
        status="prepared",
    )


def print_summary(summaries: list[PoolSummary]) -> None:
    existing = [item for item in summaries if item.status == "existing"]
    prepared = [item for item in summaries if item.status == "prepared"]

    print("Existing pools skipped:")
    if existing:
        for item in existing:
            print(f"- {item.pool_name}: {item.rows_written} rows -> {item.target_path}")
    else:
        print("- none")

    print("Downloaded / prepared pools:")
    if prepared:
        for item in prepared:
            print(f"- {item.pool_name}: {item.rows_written} rows -> {item.target_path}")
    else:
        print("- none")


def main() -> None:
    args = parse_args()
    ensure_dir(QUESTION_POOLS_DIR)
    ensure_dir(EVAL_POOLS_DIR)

    summaries = [prepare_pool(spec, args) for spec in build_specs(args)]
    print_summary(summaries)


if __name__ == "__main__":
    main()
