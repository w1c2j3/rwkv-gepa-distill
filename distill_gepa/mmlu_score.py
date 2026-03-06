from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

import orjson


ANSWER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class ParsedMCQResponse:
    raw_response: str
    valid_json: bool
    answer_letter: str | None
    answer_index: int | None
    answer_text: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "valid_json": self.valid_json,
            "answer_letter": self.answer_letter,
            "answer_index": self.answer_index,
            "answer_text": self.answer_text,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class MCQScoreResult:
    total: float
    valid_json: bool
    answer_present: bool
    correct: bool
    parsed: ParsedMCQResponse
    gold_answer_index: int | None
    gold_answer_label: str | None
    gold_answer_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid_json": self.valid_json,
            "answer_present": self.answer_present,
            "correct": self.correct,
            "gold_answer_index": self.gold_answer_index,
            "gold_answer_label": self.gold_answer_label,
            "gold_answer_text": self.gold_answer_text,
            "parsed": self.parsed.to_dict(),
        }


def answer_label(answer_index: int | None) -> str | None:
    if answer_index is None or answer_index < 0 or answer_index >= len(ANSWER_LABELS):
        return None
    return ANSWER_LABELS[answer_index]


def infer_answer_index(value: Any, choices: Sequence[str]) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)

    answer_str = str(value).strip()
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


def _json_parse(raw_response: str, choices: Sequence[str]) -> ParsedMCQResponse | None:
    try:
        payload = orjson.loads(raw_response)
    except orjson.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    answer_index = infer_answer_index(
        payload.get("answer_index", payload.get("answer_letter", payload.get("answer"))),
        choices,
    )
    answer_letter_value = payload.get("answer_letter")
    answer_letter_result = None
    if isinstance(answer_letter_value, str) and answer_letter_value.strip():
        answer_letter_result = answer_letter_value.strip().upper()
    elif answer_index is not None:
        answer_letter_result = answer_label(answer_index)

    answer_text = ""
    answer_text_value = payload.get("answer_text")
    if isinstance(answer_text_value, str) and answer_text_value.strip():
        answer_text = answer_text_value.strip()
    elif answer_index is not None and 0 <= answer_index < len(choices):
        answer_text = choices[answer_index]

    reasoning_value = payload.get("reasoning", payload.get("explanation", payload.get("rationale", "")))
    reasoning = reasoning_value.strip() if isinstance(reasoning_value, str) else ""

    if answer_index is None and answer_letter_result is not None:
        answer_index = infer_answer_index(answer_letter_result, choices)

    return ParsedMCQResponse(
        raw_response=raw_response,
        valid_json=True,
        answer_letter=answer_letter_result,
        answer_index=answer_index,
        answer_text=answer_text,
        reasoning=reasoning,
    )


def _regex_parse(raw_response: str, choices: Sequence[str]) -> ParsedMCQResponse:
    pattern = re.compile(r"(?:answer|choice|option)\s*[:=]?\s*([A-J])\b", re.IGNORECASE)
    match = pattern.search(raw_response)
    answer_letter_result = match.group(1).upper() if match else None

    if answer_letter_result is None:
        bare_pattern = re.compile(r"\b([A-J])\b")
        bare_match = bare_pattern.search(raw_response)
        answer_letter_result = bare_match.group(1).upper() if bare_match else None

    answer_index = infer_answer_index(answer_letter_result, choices)
    answer_text = choices[answer_index] if answer_index is not None and 0 <= answer_index < len(choices) else ""

    return ParsedMCQResponse(
        raw_response=raw_response,
        valid_json=False,
        answer_letter=answer_letter_result,
        answer_index=answer_index,
        answer_text=answer_text,
        reasoning=raw_response.strip(),
    )


def parse_mcq_response(raw_response: str, choices: Sequence[str]) -> ParsedMCQResponse:
    parsed = _json_parse(raw_response, choices)
    if parsed is not None:
        return parsed
    return _regex_parse(raw_response, choices)


def score_mcq_response(
    raw_response: str,
    choices: Sequence[str],
    gold_answer_index: int | None,
) -> MCQScoreResult:
    parsed = parse_mcq_response(raw_response, choices)
    answer_present = parsed.answer_index is not None or parsed.answer_letter is not None
    correct = gold_answer_index is not None and parsed.answer_index == gold_answer_index

    total = (
        0.20 * float(parsed.valid_json)
        + 0.20 * float(answer_present)
        + 0.60 * float(correct)
    )

    gold_label = answer_label(gold_answer_index)
    gold_text = (
        choices[gold_answer_index]
        if gold_answer_index is not None and 0 <= gold_answer_index < len(choices)
        else ""
    )

    return MCQScoreResult(
        total=round(total, 6),
        valid_json=parsed.valid_json,
        answer_present=answer_present,
        correct=correct,
        parsed=parsed,
        gold_answer_index=gold_answer_index,
        gold_answer_label=gold_label,
        gold_answer_text=gold_text,
    )
