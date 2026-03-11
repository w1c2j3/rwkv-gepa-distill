from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson

from .constants import ANSWER_LABELS
from .world_schema import (
    QUESTION_TYPE_MULTIPLE_CHOICE,
    QUESTION_TYPE_OPEN_QA,
    BenchmarkQuestion,
)


SEED_INPUT_CONTRACT = "seed_input_v1"
GENERATED_VARIANT_CONTRACT = "generated_variant_v1"


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
    cleaned = [_clean_text(item) for item in value if _clean_text(item)]
    return cleaned


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
        raise ValueError("'answer_aliases' must be a list when provided")
    return [_clean_text(item) for item in value if _clean_text(item)]


@dataclass(frozen=True)
class SeedInput:
    seed_id: str
    domain: str
    question_type: str
    question: str
    answer: str
    choices: list[str]
    answer_index: int | None
    answer_aliases: list[str]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path, line_number: int) -> "SeedInput":
        contract = payload.get("contract", SEED_INPUT_CONTRACT)
        if contract != SEED_INPUT_CONTRACT:
            raise ValueError(f"{source}:{line_number} has unsupported contract {contract!r}")

        question = _clean_text(payload.get("question", payload.get("question_text")))
        answer = _clean_text(payload.get("answer", payload.get("gold_answer")))
        if not question:
            raise ValueError(f"{source}:{line_number} missing non-empty question")
        if not answer:
            raise ValueError(f"{source}:{line_number} missing non-empty answer")

        choices = _clean_choices(payload.get("choices"))
        question_type = _normalize_question_type(payload.get("question_type"), choices=choices)
        answer_index = _infer_answer_index(
            answer,
            payload.get("answer_index", payload.get("gold_answer_index")),
            choices,
        )
        if question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            if not choices:
                raise ValueError(f"{source}:{line_number} multiple_choice seeds require choices")
            if answer_index is None or answer_index < 0 or answer_index >= len(choices):
                raise ValueError(f"{source}:{line_number} multiple_choice seed is missing a valid answer_index")

        seed_id = _clean_text(payload.get("seed_id")) or _stable_id("seed", question, answer)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}:{line_number} metadata must be an object")

        return cls(
            seed_id=seed_id,
            domain=_clean_text(payload.get("domain")) or "unknown",
            question_type=question_type,
            question=question,
            answer=answer,
            choices=choices,
            answer_index=answer_index,
            answer_aliases=_clean_aliases(payload.get("answer_aliases", payload.get("gold_aliases"))),
            metadata=metadata,
        )

    def render_prompt(self) -> str:
        lines = [
            f"Domain: {self.domain}",
            f"Question Type: {self.question_type}",
            "",
            "Question:",
            self.question,
        ]
        if self.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            lines.extend(["", "Options:"])
            for index, choice in enumerate(self.choices):
                label = ANSWER_LABELS[index] if index < len(ANSWER_LABELS) else f"Option {index + 1}"
                lines.append(f"{label}. {choice}")
        return "\n".join(lines).strip()

    def to_benchmark_question(self, *, question_id: str | None = None) -> BenchmarkQuestion:
        return BenchmarkQuestion(
            benchmark_name="seed_variants",
            split="generated",
            domain=self.domain,
            question_id=question_id or self.seed_id,
            question_type=self.question_type,
            question_text=self.question,
            choices=list(self.choices),
            gold_answer=self.answer,
            gold_answer_index=self.answer_index,
            gold_aliases=list(self.answer_aliases),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": SEED_INPUT_CONTRACT,
            "seed_id": self.seed_id,
            "domain": self.domain,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "choices": self.choices,
            "answer_index": self.answer_index,
            "answer_aliases": self.answer_aliases,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class GeneratedVariant:
    variant_id: str
    seed_id: str
    variant_index: int
    domain: str
    question_type: str
    question: str
    answer: str
    choices: list[str]
    answer_index: int | None
    answer_aliases: list[str]
    generator_model: str
    generation_prompt_version: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path, line_number: int) -> "GeneratedVariant":
        contract = payload.get("contract", GENERATED_VARIANT_CONTRACT)
        if contract != GENERATED_VARIANT_CONTRACT:
            raise ValueError(f"{source}:{line_number} has unsupported contract {contract!r}")

        variant_id = _clean_text(payload.get("variant_id"))
        seed_id = _clean_text(payload.get("seed_id"))
        question = _clean_text(payload.get("question"))
        answer = _clean_text(payload.get("answer"))
        if not variant_id or not seed_id or not question or not answer:
            raise ValueError(f"{source}:{line_number} missing required variant fields")

        choices = _clean_choices(payload.get("choices"))
        question_type = _normalize_question_type(payload.get("question_type"), choices=choices)
        answer_index = _infer_answer_index(answer, payload.get("answer_index"), choices)
        if question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            if not choices:
                raise ValueError(f"{source}:{line_number} multiple_choice variants require choices")
            if answer_index is None or answer_index < 0 or answer_index >= len(choices):
                raise ValueError(f"{source}:{line_number} multiple_choice variant is missing a valid answer_index")

        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{source}:{line_number} metadata must be an object")

        variant_index = payload.get("variant_index")
        if not isinstance(variant_index, int):
            raise ValueError(f"{source}:{line_number} variant_index must be an integer")

        return cls(
            variant_id=variant_id,
            seed_id=seed_id,
            variant_index=variant_index,
            domain=_clean_text(payload.get("domain")) or "unknown",
            question_type=question_type,
            question=question,
            answer=answer,
            choices=choices,
            answer_index=answer_index,
            answer_aliases=_clean_aliases(payload.get("answer_aliases")),
            generator_model=_clean_text(payload.get("generator_model")) or "unknown",
            generation_prompt_version=_clean_text(payload.get("generation_prompt_version")) or "unknown",
            metadata=metadata,
        )

    def render_prompt(self) -> str:
        lines = [
            f"Domain: {self.domain}",
            f"Question Type: {self.question_type}",
            "",
            "Question:",
            self.question,
        ]
        if self.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
            lines.extend(["", "Options:"])
            for index, choice in enumerate(self.choices):
                label = ANSWER_LABELS[index] if index < len(ANSWER_LABELS) else f"Option {index + 1}"
                lines.append(f"{label}. {choice}")
        return "\n".join(lines).strip()

    def to_benchmark_question(self) -> BenchmarkQuestion:
        return BenchmarkQuestion(
            benchmark_name="generated_variant",
            split="generated",
            domain=self.domain,
            question_id=self.variant_id,
            question_type=self.question_type,
            question_text=self.question,
            choices=list(self.choices),
            gold_answer=self.answer,
            gold_answer_index=self.answer_index,
            gold_aliases=list(self.answer_aliases),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": GENERATED_VARIANT_CONTRACT,
            "variant_id": self.variant_id,
            "seed_id": self.seed_id,
            "variant_index": self.variant_index,
            "domain": self.domain,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "choices": self.choices,
            "answer_index": self.answer_index,
            "answer_aliases": self.answer_aliases,
            "generator_model": self.generator_model,
            "generation_prompt_version": self.generation_prompt_version,
            "metadata": self.metadata,
        }


def iter_seed_inputs(path: Path) -> Iterable[SeedInput]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed input file: {path}")
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield SeedInput.from_dict(payload, path, line_number)


def load_seed_inputs(path: Path) -> list[SeedInput]:
    seeds = list(iter_seed_inputs(path))
    if not seeds:
        raise ValueError(f"No seed inputs found in {path}")
    return seeds


def iter_generated_variants(path: Path) -> Iterable[GeneratedVariant]:
    if not path.exists():
        raise FileNotFoundError(f"Missing generated variants file: {path}")
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield GeneratedVariant.from_dict(payload, path, line_number)


def load_generated_variant_map(path: Path) -> dict[str, GeneratedVariant]:
    return {variant.variant_id: variant for variant in iter_generated_variants(path)}

