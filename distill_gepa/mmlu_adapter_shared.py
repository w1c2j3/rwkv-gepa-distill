from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Iterable

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


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


def progress_iter(
    iterable: Iterable[Any],
    *,
    desc: str,
    total: int | None = None,
) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)


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


def tokenize_subject_text(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower())
        if token not in SUBJECT_INFERENCE_STOPWORDS
    ]


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
