from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import orjson


@dataclass(frozen=True)
class ScoreResult:
    total: float
    valid_json: bool
    non_empty_answer: bool
    keyword_coverage: float
    missing_keywords: list[str]
    answer_text: str
    char_count: int
    too_long: bool
    length_component: float
    parsed_payload: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid_json": self.valid_json,
            "non_empty_answer": self.non_empty_answer,
            "keyword_coverage": self.keyword_coverage,
            "missing_keywords": self.missing_keywords,
            "answer_text": self.answer_text,
            "char_count": self.char_count,
            "too_long": self.too_long,
            "length_component": self.length_component,
            "parsed_payload": self.parsed_payload,
        }


def score_response(
    raw_response: str,
    expected_keywords: Sequence[str],
    max_answer_chars: int = 280,
) -> ScoreResult:
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be a string")

    parsed_payload: dict[str, Any] | None = None
    valid_json = False
    answer_text = raw_response.strip()

    try:
        parsed_candidate = orjson.loads(raw_response)
        if isinstance(parsed_candidate, dict):
            parsed_payload = parsed_candidate
            valid_json = True
            json_answer = parsed_candidate.get("answer")
            if isinstance(json_answer, str) and json_answer.strip():
                answer_text = json_answer.strip()
    except orjson.JSONDecodeError:
        parsed_payload = None

    non_empty_answer = bool(answer_text.strip())
    searchable_text = f"{answer_text}\n{raw_response}".lower()

    missing_keywords = [
        keyword
        for keyword in expected_keywords
        if keyword.lower() not in searchable_text
    ]
    keyword_coverage = (
        (len(expected_keywords) - len(missing_keywords)) / len(expected_keywords)
        if expected_keywords
        else 0.0
    )

    char_count = len(answer_text)
    too_long = char_count > max_answer_chars
    if too_long:
        overage = char_count - max_answer_chars
        length_component = max(0.0, 1.0 - (overage / max_answer_chars))
    else:
        length_component = 1.0

    total = (
        0.40 * float(valid_json)
        + 0.20 * float(non_empty_answer)
        + 0.30 * keyword_coverage
        + 0.10 * length_component
    )

    return ScoreResult(
        total=round(total, 6),
        valid_json=valid_json,
        non_empty_answer=non_empty_answer,
        keyword_coverage=round(keyword_coverage, 6),
        missing_keywords=missing_keywords,
        answer_text=answer_text,
        char_count=char_count,
        too_long=too_long,
        length_component=round(length_component, 6),
        parsed_payload=parsed_payload,
    )
