from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson

from .constants import ANSWER_LABELS


WORLD_QUESTION_CONTRACT = "world_question_v1"
QUESTION_TYPE_MULTIPLE_CHOICE = "multiple_choice"
QUESTION_TYPE_OPEN_QA = "open_qa"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


@dataclass(frozen=True)
class PromptedQuestion:
    user_message: str
    shuffled_choices: list[str]
    shuffled_answer_index: int | None
    choice_permutation: list[int]
    shuffle_key: str


@dataclass(frozen=True)
class BenchmarkQuestion:
    benchmark_name: str
    split: str
    domain: str
    question_id: str
    question_type: str
    question_text: str
    choices: list[str]
    gold_answer: str
    gold_answer_index: int | None
    gold_aliases: list[str]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path, line_number: int) -> "BenchmarkQuestion":
        contract = payload.get("contract", WORLD_QUESTION_CONTRACT)
        if contract != WORLD_QUESTION_CONTRACT:
            raise ValueError(f"{source}:{line_number} has unsupported contract {contract!r}")

        required_string_fields = (
            "benchmark_name",
            "split",
            "domain",
            "question_id",
            "question_type",
            "question_text",
            "gold_answer",
        )
        for field_name in required_string_fields:
            value = payload.get(field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{source}:{line_number} missing non-empty string field {field_name!r}")

        question_type = payload["question_type"].strip()
        if question_type not in {QUESTION_TYPE_MULTIPLE_CHOICE, QUESTION_TYPE_OPEN_QA}:
            raise ValueError(f"{source}:{line_number} has unsupported question_type {question_type!r}")

        choices = payload.get("choices", [])
        if choices is None:
            choices = []
        if not isinstance(choices, list) or not all(isinstance(item, str) and item.strip() for item in choices):
            raise ValueError(f"{source}:{line_number} has invalid 'choices'")

        gold_answer_index = payload.get("gold_answer_index")
        if gold_answer_index is not None and not isinstance(gold_answer_index, int):
            raise ValueError(f"{source}:{line_number} has invalid 'gold_answer_index'")

        gold_aliases = payload.get("gold_aliases", [])
        if gold_aliases is None:
            gold_aliases = []
        if not isinstance(gold_aliases, list) or not all(isinstance(item, str) for item in gold_aliases):
            raise ValueError(f"{source}:{line_number} has invalid 'gold_aliases'")

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}:{line_number} has invalid 'metadata'")

        return cls(
            benchmark_name=payload["benchmark_name"].strip(),
            split=payload["split"].strip(),
            domain=payload["domain"].strip(),
            question_id=payload["question_id"].strip(),
            question_type=question_type,
            question_text=payload["question_text"].strip(),
            choices=[item.strip() for item in choices],
            gold_answer=payload["gold_answer"].strip(),
            gold_answer_index=gold_answer_index,
            gold_aliases=[item.strip() for item in gold_aliases if item.strip()],
            metadata=metadata,
        )

    @property
    def prompt_text(self) -> str:
        return self.render_prompt()

    def render_prompt(self, *, choices: list[str] | None = None, gold_answer_index: int | None = None) -> str:
        rendered_choices = self.choices if choices is None else choices
        lines = [
            f"Benchmark: {self.benchmark_name}",
            f"Domain: {self.domain}",
            f"Question Type: {self.question_type}",
            "",
            "Question:",
            self.question_text,
        ]
        if self.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            lines.extend(["", "Options:"])
            for index, choice in enumerate(rendered_choices):
                label = ANSWER_LABELS[index] if index < len(ANSWER_LABELS) else f"Option {index + 1}"
                lines.append(f"{label}. {choice}")
        return "\n".join(lines).strip()

    def prompted_variant(self, sample_index: int, *, shuffle_key: str | None = None) -> PromptedQuestion:
        if self.question_type != QUESTION_TYPE_MULTIPLE_CHOICE:
            return PromptedQuestion(
                user_message=self.render_prompt(),
                shuffled_choices=[],
                shuffled_answer_index=None,
                choice_permutation=[],
                shuffle_key=shuffle_key or f"{self.question_id}:{sample_index}",
            )

        indices = list(range(len(self.choices)))
        effective_shuffle_key = shuffle_key or f"{self.question_id}:{sample_index}"
        rng = random.Random(effective_shuffle_key)
        rng.shuffle(indices)
        shuffled_choices = [self.choices[index] for index in indices]
        shuffled_answer_index = None
        if self.gold_answer_index is not None:
            shuffled_answer_index = indices.index(self.gold_answer_index)

        return PromptedQuestion(
            user_message=self.render_prompt(
                choices=shuffled_choices,
                gold_answer_index=shuffled_answer_index,
            ),
            shuffled_choices=shuffled_choices,
            shuffled_answer_index=shuffled_answer_index,
            choice_permutation=indices,
            shuffle_key=effective_shuffle_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": WORLD_QUESTION_CONTRACT,
            "benchmark_name": self.benchmark_name,
            "split": self.split,
            "domain": self.domain,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question_text": self.question_text,
            "choices": self.choices,
            "gold_answer": self.gold_answer,
            "gold_answer_index": self.gold_answer_index,
            "gold_aliases": self.gold_aliases,
            "metadata": self.metadata,
        }


def iter_benchmark_questions(path: Path, limit: int | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark question pool: {path}")

    yielded = 0
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield BenchmarkQuestion.from_dict(payload, path, line_number)
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def load_benchmark_questions(path: Path, limit: int | None = None) -> list[BenchmarkQuestion]:
    questions = list(iter_benchmark_questions(path, limit=limit))
    if not questions:
        raise ValueError(f"No benchmark questions found in {path}")
    return questions


def write_benchmark_questions(path: Path, questions: Iterable[BenchmarkQuestion]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with path.open("wb") as handle:
        for question in questions:
            handle.write(orjson.dumps(question.to_dict()))
            handle.write(b"\n")
            total += 1
    return total
