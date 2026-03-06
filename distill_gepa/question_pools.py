from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson


@dataclass(frozen=True)
class QuestionPoolRecord:
    source_dataset: str
    source_split: str
    subject: str
    question: str
    choices: list[str]
    answer: str
    answer_index: int | None
    prompt_text: str
    meta: dict[str, Any]

    @property
    def answer_label(self) -> str | None:
        label = self.meta.get("answer_label")
        return label if isinstance(label, str) and label else None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path, line_number: int) -> "QuestionPoolRecord":
        required_string_fields = (
            "source_dataset",
            "source_split",
            "subject",
            "question",
            "answer",
            "prompt_text",
        )
        for field_name in required_string_fields:
            value = payload.get(field_name)
            if not isinstance(value, str):
                raise ValueError(f"{source}:{line_number} missing string field {field_name!r}")

        choices = payload.get("choices")
        if not isinstance(choices, list) or not all(isinstance(item, str) for item in choices):
            raise ValueError(f"{source}:{line_number} has invalid 'choices'")

        answer_index = payload.get("answer_index")
        if answer_index is not None and not isinstance(answer_index, int):
            raise ValueError(f"{source}:{line_number} has invalid 'answer_index'")

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"{source}:{line_number} has invalid 'meta'")

        return cls(
            source_dataset=payload["source_dataset"],
            source_split=payload["source_split"],
            subject=payload["subject"],
            question=payload["question"],
            choices=choices,
            answer=payload["answer"],
            answer_index=answer_index,
            prompt_text=payload["prompt_text"],
            meta=meta,
        )


def iter_question_pool(path: Path, limit: int | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Missing question pool: {path}")

    yielded = 0
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            payload = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")

        yield QuestionPoolRecord.from_dict(payload, path, line_number)
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def load_question_pool(path: Path, limit: int | None = None) -> list[QuestionPoolRecord]:
    records = list(iter_question_pool(path, limit=limit))

    if not records:
        raise ValueError(f"No question-pool rows found in {path}")

    return records
