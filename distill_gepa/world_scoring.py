from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import orjson

from .constants import ANSWER_LABELS
from .world_schema import BenchmarkQuestion, QUESTION_TYPE_MULTIPLE_CHOICE


def normalize_answer_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


THINK_TAG_PATTERN = re.compile(r"<think>\s*(?P<content>.*?)\s*</think>", re.IGNORECASE | re.DOTALL)


def has_think_tags(value: str) -> bool:
    return bool(THINK_TAG_PATTERN.search(value))


def ensure_think_tags(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return "<think></think>"
    if has_think_tags(stripped):
        return stripped
    return f"<think>{stripped}</think>"


def extract_json_object(raw_text: str) -> dict[str, Any] | None:
    start = raw_text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for index in range(start, len(raw_text)):
        char = raw_text[index]

        if in_string:
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                snippet = raw_text[start : index + 1]
                try:
                    payload = orjson.loads(snippet)
                except orjson.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _dict_get(payload: Any, key: str) -> Any:
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _normalize_choice_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return str(value).strip() if value is not None else ""


def infer_answer_index(value: Any, choices: list[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)

    text = _normalize_choice_text(value)
    if not text:
        return None
    if text.isdigit():
        return int(text)

    upper = text.upper()
    if len(upper) == 1 and upper in ANSWER_LABELS:
        return ANSWER_LABELS.index(upper)

    for index, choice in enumerate(choices):
        if choice == text:
            return index
    return None


@dataclass(frozen=True)
class ParsedWorldResponse:
    raw_response: str
    valid_json: bool
    parser_recovered: bool
    final_answer: str
    answer_letter: str | None
    answer_index: int | None
    answer_text: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "valid_json": self.valid_json,
            "parser_recovered": self.parser_recovered,
            "final_answer": self.final_answer,
            "answer_letter": self.answer_letter,
            "answer_index": self.answer_index,
            "answer_text": self.answer_text,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class WorldScoreResult:
    total: float
    valid_json: bool
    parser_recovered: bool
    answer_present: bool
    reasoning_present: bool
    think_tags_present: bool
    correct: bool
    usable_for_distill: bool
    exact_answer_text_match: bool
    parsed: ParsedWorldResponse
    question_type: str
    gold_answer: str
    gold_answer_index: int | None
    gold_aliases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid_json": self.valid_json,
            "parser_recovered": self.parser_recovered,
            "answer_present": self.answer_present,
            "reasoning_present": self.reasoning_present,
            "think_tags_present": self.think_tags_present,
            "correct": self.correct,
            "usable_for_distill": self.usable_for_distill,
            "exact_answer_text_match": self.exact_answer_text_match,
            "question_type": self.question_type,
            "gold_answer": self.gold_answer,
            "gold_answer_index": self.gold_answer_index,
            "gold_aliases": self.gold_aliases,
            "parsed": self.parsed.to_dict(),
        }


def parse_world_response(raw_response: str, question: BenchmarkQuestion) -> ParsedWorldResponse:
    try:
        payload = orjson.loads(raw_response)
    except orjson.JSONDecodeError:
        payload = extract_json_object(raw_response)

    if isinstance(payload, dict):
        final_answer = _normalize_choice_text(
            payload.get("final_answer", payload.get("answer", payload.get("answer_text")))
        )
        answer_text = _normalize_choice_text(payload.get("answer_text", final_answer))
        answer_letter_value = payload.get("answer_letter")
        answer_letter = answer_letter_value.strip().upper() if isinstance(answer_letter_value, str) else None
        answer_index = infer_answer_index(
            payload.get("answer_index", answer_letter or answer_text or final_answer),
            question.choices,
        )
        reasoning_value = payload.get("reasoning", payload.get("explanation", payload.get("rationale", "")))
        reasoning = reasoning_value.strip() if isinstance(reasoning_value, str) else ""
        return ParsedWorldResponse(
            raw_response=raw_response,
            valid_json=True,
            parser_recovered=False,
            final_answer=final_answer,
            answer_letter=answer_letter,
            answer_index=answer_index,
            answer_text=answer_text,
            reasoning=reasoning,
        )

    fallback_answer = raw_response.strip()
    answer_letter = None
    answer_index = None
    if question.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
        match = re.search(r"\b([A-J])\b", raw_response, re.IGNORECASE)
        if match:
            answer_letter = match.group(1).upper()
            answer_index = infer_answer_index(answer_letter, question.choices)

    return ParsedWorldResponse(
        raw_response=raw_response,
        valid_json=False,
        parser_recovered=bool(fallback_answer),
        final_answer=fallback_answer,
        answer_letter=answer_letter,
        answer_index=answer_index,
        answer_text="",
        reasoning=fallback_answer,
    )


def _truncate_reasoning(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped[:400]


def _extract_short_open_answer(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    first_line = stripped.splitlines()[0].strip()
    normalized = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", first_line, flags=re.IGNORECASE)
    return normalized[:160].strip()


def repair_world_response(raw_response: str, question: BenchmarkQuestion) -> tuple[str | None, dict[str, Any]]:
    parsed = parse_world_response(raw_response, question)
    if parsed.valid_json and has_think_tags(parsed.reasoning):
        return raw_response, {"status": "not_needed"}
    if parsed.valid_json:
        try:
            payload = orjson.loads(raw_response)
        except orjson.JSONDecodeError:
            payload = extract_json_object(raw_response)
        if not isinstance(payload, dict):
            return None, {"status": "failed", "reason": "valid_json_payload_not_extractable"}
        reasoning_text = _truncate_reasoning(parsed.reasoning)
        payload["reasoning"] = ensure_think_tags(reasoning_text or "Brief reasoning unavailable.")
        return orjson.dumps(payload).decode("utf-8"), {
            "status": "repaired",
            "strategy": "reasoning_think_tag_wrap",
        }

    if question.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
        answer_index = parsed.answer_index
        if answer_index is None:
            answer_index = infer_answer_index(
                parsed.final_answer or parsed.answer_text or parsed.answer_letter,
                question.choices,
            )
        if answer_index is None or answer_index < 0 or answer_index >= len(question.choices):
            return None, {"status": "failed", "reason": "unable_to_infer_choice"}
        repaired_payload = {
            "final_answer": question.choices[answer_index],
            "answer_letter": ANSWER_LABELS[answer_index],
            "answer_index": answer_index,
            "answer_text": question.choices[answer_index],
            "reasoning": ensure_think_tags(_truncate_reasoning(parsed.reasoning or raw_response)),
        }
        return orjson.dumps(repaired_payload).decode("utf-8"), {
            "status": "repaired",
            "strategy": "mcq_structural_repair",
        }

    short_answer = _extract_short_open_answer(parsed.final_answer or raw_response)
    if not short_answer:
        return None, {"status": "failed", "reason": "unable_to_extract_short_answer"}
    repaired_payload = {
        "final_answer": short_answer,
        "reasoning": ensure_think_tags(_truncate_reasoning(parsed.reasoning or raw_response)),
    }
    return orjson.dumps(repaired_payload).decode("utf-8"), {
        "status": "repaired",
        "strategy": "open_qa_structural_repair",
    }


def _mcq_correct(parsed: ParsedWorldResponse, question: BenchmarkQuestion) -> tuple[bool, bool]:
    gold_index = question.gold_answer_index
    exact_text_match = parsed.answer_text == question.gold_answer if bool(parsed.answer_text) else False

    if gold_index is not None and parsed.answer_index is not None:
        return parsed.answer_index == gold_index, exact_text_match

    accepted = {normalize_answer_text(question.gold_answer)}
    accepted.update(normalize_answer_text(alias) for alias in question.gold_aliases if alias.strip())

    for candidate in (parsed.answer_text, parsed.final_answer):
        normalized = normalize_answer_text(candidate)
        if normalized:
            return normalized in accepted, exact_text_match

    return False, exact_text_match


def _open_qa_correct(parsed: ParsedWorldResponse, question: BenchmarkQuestion) -> bool:
    final_answer = normalize_answer_text(parsed.final_answer)
    if not final_answer:
        return False
    accepted = {normalize_answer_text(question.gold_answer)}
    accepted.update(normalize_answer_text(alias) for alias in question.gold_aliases if alias.strip())
    return final_answer in accepted


def score_world_response(raw_response: str, question: BenchmarkQuestion) -> WorldScoreResult:
    parsed = parse_world_response(raw_response, question)
    answer_present = bool(parsed.final_answer or parsed.answer_text or parsed.answer_index is not None)
    reasoning_present = bool(parsed.reasoning.strip())
    think_tags_present = has_think_tags(parsed.reasoning)

    if question.question_type == QUESTION_TYPE_MULTIPLE_CHOICE:
        correct, exact_answer_text_match = _mcq_correct(parsed, question)
    else:
        correct = _open_qa_correct(parsed, question)
        exact_answer_text_match = normalize_answer_text(parsed.final_answer) == normalize_answer_text(question.gold_answer)

    usable_for_distill = parsed.valid_json and correct and reasoning_present and think_tags_present
    total = (
        0.25 * float(parsed.valid_json)
        + 0.15 * float(answer_present)
        + 0.10 * float(reasoning_present)
        + 0.15 * float(think_tags_present)
        + 0.35 * float(correct)
    )

    return WorldScoreResult(
        total=round(total, 6),
        valid_json=parsed.valid_json,
        parser_recovered=parsed.parser_recovered,
        answer_present=answer_present,
        reasoning_present=reasoning_present,
        think_tags_present=think_tags_present,
        correct=correct,
        usable_for_distill=usable_for_distill,
        exact_answer_text_match=exact_answer_text_match,
        parsed=parsed,
        question_type=question.question_type,
        gold_answer=question.gold_answer,
        gold_answer_index=question.gold_answer_index,
        gold_aliases=question.gold_aliases,
    )
