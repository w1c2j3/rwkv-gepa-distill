from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson

from .constants import (
    ANSWER_LABELS,
    QUESTION_TYPE_MULTIPLE_CHOICE,
    QUESTION_TYPE_OPEN_QA,
)


TASK_ITEM_CONTRACT = "task_v1"
_LEGACY_TASK_ITEM_CONTRACTS = {
    TASK_ITEM_CONTRACT,
    "task_item_v1",
    "seed_input_v1",
    "world_question_v1",
}


def _clean_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()


def _stable_id(prefix: str, *parts: str) -> str:
    joined = "||".join(part.strip() for part in parts if part.strip())
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}::{digest}"


def _normalize_question_type(value: Any, *, choices: list[str]) -> str:
    cleaned = _clean_text(value).lower()
    if cleaned:
        if cleaned not in {QUESTION_TYPE_MULTIPLE_CHOICE, QUESTION_TYPE_OPEN_QA}:
            raise ValueError(f"Unsupported question_type {cleaned!r}")
        return cleaned
    return QUESTION_TYPE_MULTIPLE_CHOICE if choices else QUESTION_TYPE_OPEN_QA


def _clean_choices(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("'choices' must be a list when provided")
    return [_clean_text(item) for item in value if _clean_text(item)]


def _infer_answer_index(answer_text: str, answer_index: Any, choices: list[str]) -> int | None:
    if not choices:
        return None
    if isinstance(answer_index, int):
        return answer_index
    if isinstance(answer_index, str) and answer_index.strip():
        normalized = answer_index.strip().upper()
        if normalized in ANSWER_LABELS:
            return ANSWER_LABELS.index(normalized)
        if normalized.isdigit():
            return int(normalized)
    if answer_text:
        for index, choice in enumerate(choices):
            if choice == answer_text:
                return index
    return None


def _clean_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("'reference_aliases' must be a list when provided")
    return [_clean_text(item) for item in value if _clean_text(item)]


@dataclass(frozen=True)
class TaskItem:
    question_id: str
    data_split: str
    domain: str
    question_type: str
    question_text: str
    reference_answer: str
    choices: list[str]
    reference_answer_index: int | None
    reference_aliases: list[str]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path, line_number: int) -> "TaskItem":
        contract = payload.get("contract", TASK_ITEM_CONTRACT)
        if contract not in _LEGACY_TASK_ITEM_CONTRACTS:
            raise ValueError(f"{source}:{line_number} has unsupported contract {contract!r}")

        question_text = _clean_text(payload.get("question_text", payload.get("question")))
        reference_answer = _clean_text(
            payload.get("reference_answer", payload.get("answer", payload.get("gold_answer")))
        )
        if not question_text:
            raise ValueError(f"{source}:{line_number} missing non-empty question_text")
        if not reference_answer:
            raise ValueError(f"{source}:{line_number} missing non-empty reference_answer")

        choices = _clean_choices(payload.get("choices"))
        question_type = _normalize_question_type(payload.get("question_type"), choices=choices)
        reference_answer_index = _infer_answer_index(
            reference_answer,
            payload.get(
                "reference_answer_index",
                payload.get("answer_index", payload.get("gold_answer_index")),
            ),
            choices,
        )
        if question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            if not choices:
                raise ValueError(f"{source}:{line_number} multiple_choice tasks require choices")
            if reference_answer_index is None or reference_answer_index < 0 or reference_answer_index >= len(choices):
                raise ValueError(
                    f"{source}:{line_number} multiple_choice task is missing a valid reference_answer_index"
                )

        question_id = (
            _clean_text(payload.get("question_id"))
            or _clean_text(payload.get("seed_id"))
            or _clean_text(payload.get("id"))
            or _stable_id("task", question_text, reference_answer)
        )
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}:{line_number} metadata must be an object")
        if contract == "world_question_v1":
            metadata = {
                **metadata,
                "benchmark_name": _clean_text(payload.get("benchmark_name")) or metadata.get("benchmark_name"),
            }

        return cls(
            question_id=question_id,
            data_split=_clean_text(payload.get("data_split", payload.get("split"))) or "train",
            domain=_clean_text(payload.get("domain")) or "unknown",
            question_type=question_type,
            question_text=question_text,
            reference_answer=reference_answer,
            choices=choices,
            reference_answer_index=reference_answer_index,
            reference_aliases=_clean_aliases(
                payload.get(
                    "reference_aliases",
                    payload.get("answer_aliases", payload.get("gold_aliases")),
                )
            ),
            metadata=metadata,
        )

    @property
    def seed_id(self) -> str:
        return self.question_id

    @property
    def question(self) -> str:
        return self.question_text

    @property
    def answer(self) -> str:
        return self.reference_answer

    @property
    def answer_index(self) -> int | None:
        return self.reference_answer_index

    @property
    def answer_aliases(self) -> list[str]:
        return self.reference_aliases

    def render_prompt(self) -> str:
        lines = [
            f"Domain: {self.domain}",
            f"Question Type: {self.question_type}",
            "",
            "Question:",
            self.question_text,
        ]
        if self.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            lines.extend(["", "Options:"])
            for index, choice in enumerate(self.choices):
                label = ANSWER_LABELS[index] if index < len(ANSWER_LABELS) else f"Option {index + 1}"
                lines.append(f"{label}. {choice}")
        return "\n".join(lines).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "data_split": self.data_split,
            "domain": self.domain,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "choices": self.choices,
            "reference_answer": self.reference_answer,
            "reference_answer_index": self.reference_answer_index,
            "reference_aliases": self.reference_aliases,
            "metadata": self.metadata,
        }


def iter_task_items(path: Path, limit: int | None = None) -> Iterable[TaskItem]:
    if not path.exists():
        raise FileNotFoundError(f"Missing task input file: {path}")
    yielded = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield TaskItem.from_dict(payload, path, line_number)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def load_task_items(path: Path, limit: int | None = None) -> list[TaskItem]:
    task_items = list(iter_task_items(path, limit=limit))
    if not task_items:
        raise ValueError(f"No task inputs found in {path}")
    return task_items


def write_task_items(path: Path, task_items: Iterable[TaskItem]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("wb") as handle:
        for task_item in task_items:
            handle.write(orjson.dumps(task_item.to_dict()))
            handle.write(b"\n")
            total += 1
    return total


__all__ = [
    "TASK_ITEM_CONTRACT",
    "TaskItem",
    "iter_task_items",
    "load_task_items",
    "write_task_items",
]
