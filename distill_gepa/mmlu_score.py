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
    parser_recovered: bool
    subject: str
    answer_letter: str | None
    answer_index: int | None
    answer_text: str
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_response": self.raw_response,
            "valid_json": self.valid_json,
            "parser_recovered": self.parser_recovered,
            "subject": self.subject,
            "answer_letter": self.answer_letter,
            "answer_index": self.answer_index,
            "answer_text": self.answer_text,
            "reasoning": self.reasoning,
        }


@dataclass(frozen=True)
class MCQScoreResult:
    total: float
    valid_json: bool
    parser_recovered: bool
    answer_present: bool
    answer_consistent: bool
    exact_answer_text_match: bool
    reasoning_present: bool
    subject_present: bool
    subject_matches_pool: bool
    correct: bool
    usable_for_sft: bool
    parsed: ParsedMCQResponse
    pool_subject: str
    gold_answer_index: int | None
    gold_answer_label: str | None
    gold_answer_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "valid_json": self.valid_json,
            "parser_recovered": self.parser_recovered,
            "answer_present": self.answer_present,
            "answer_consistent": self.answer_consistent,
            "exact_answer_text_match": self.exact_answer_text_match,
            "reasoning_present": self.reasoning_present,
            "subject_present": self.subject_present,
            "subject_matches_pool": self.subject_matches_pool,
            "correct": self.correct,
            "usable_for_sft": self.usable_for_sft,
            "pool_subject": self.pool_subject,
            "gold_answer_index": self.gold_answer_index,
            "gold_answer_label": self.gold_answer_label,
            "gold_answer_text": self.gold_answer_text,
            "parsed": self.parsed.to_dict(),
        }


def answer_label(answer_index: int | None) -> str | None:
    if answer_index is None or answer_index < 0 or answer_index >= len(ANSWER_LABELS):
        return None
    return ANSWER_LABELS[answer_index]


def normalize_subject(subject: str) -> str:
    return re.sub(r"\s+", " ", subject.replace("_", " ").strip().lower())


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


def _extract_subject(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _json_parse(raw_response: str, choices: Sequence[str]) -> ParsedMCQResponse | None:
    try:
        payload = orjson.loads(raw_response)
    except orjson.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    subject = _extract_subject(payload.get("subject"))

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

    if answer_index is None and answer_text:
        answer_index = infer_answer_index(answer_text, choices)
    if answer_index is None and answer_letter_result is not None:
        answer_index = infer_answer_index(answer_letter_result, choices)

    reasoning_value = payload.get("reasoning", payload.get("explanation", payload.get("rationale", "")))
    reasoning = reasoning_value.strip() if isinstance(reasoning_value, str) else ""

    return ParsedMCQResponse(
        raw_response=raw_response,
        valid_json=True,
        parser_recovered=False,
        subject=subject,
        answer_letter=answer_letter_result,
        answer_index=answer_index,
        answer_text=answer_text,
        reasoning=reasoning,
    )


def _regex_subject(raw_response: str) -> str:
    match = re.search(r"(?:subject|topic|domain)\s*[:=]\s*([A-Za-z0-9 _-]{2,80})", raw_response, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _regex_parse(raw_response: str, choices: Sequence[str]) -> ParsedMCQResponse:
    pattern = re.compile(r"(?:answer|choice|option)\s*[:=]?\s*([A-J])\b", re.IGNORECASE)
    match = pattern.search(raw_response)
    answer_letter_result = match.group(1).upper() if match else None

    if answer_letter_result is None:
        bare_pattern = re.compile(r"\b([A-J])\b")
        bare_match = bare_pattern.search(raw_response)
        answer_letter_result = bare_match.group(1).upper() if bare_match else None

    answer_index = infer_answer_index(answer_letter_result, choices)
    return ParsedMCQResponse(
        raw_response=raw_response,
        valid_json=False,
        parser_recovered=answer_index is not None or answer_letter_result is not None,
        subject=_regex_subject(raw_response),
        answer_letter=answer_letter_result,
        answer_index=answer_index,
        answer_text="",
        reasoning=raw_response.strip(),
    )


def parse_mcq_response(raw_response: str, choices: Sequence[str]) -> ParsedMCQResponse:
    parsed = _json_parse(raw_response, choices)
    if parsed is not None:
        return parsed
    return _regex_parse(raw_response, choices)


def subject_matches_pool(parsed_subject: str, pool_subject: str) -> bool:
    parsed_norm = normalize_subject(parsed_subject)
    pool_norm = normalize_subject(pool_subject)
    if not parsed_norm or not pool_norm:
        return False
    if parsed_norm == pool_norm:
        return True

    parsed_tokens = set(re.findall(r"[a-z0-9]+", parsed_norm))
    pool_tokens = set(re.findall(r"[a-z0-9]+", pool_norm))
    if not parsed_tokens or not pool_tokens:
        return False

    overlap = parsed_tokens & pool_tokens
    if not overlap:
        return False
    return (
        parsed_tokens.issubset(pool_tokens)
        or pool_tokens.issubset(parsed_tokens)
        or (len(overlap) / max(len(parsed_tokens), len(pool_tokens))) >= 0.5
    )


def answer_consistent(parsed: ParsedMCQResponse, choices: Sequence[str]) -> bool:
    indices: list[int] = []
    components_present = 0

    if parsed.answer_index is not None:
        indices.append(parsed.answer_index)
        components_present += 1

    if parsed.answer_letter:
        components_present += 1
        inferred = infer_answer_index(parsed.answer_letter, choices)
        if inferred is None:
            return False
        indices.append(inferred)

    if parsed.answer_text:
        components_present += 1
        inferred = infer_answer_index(parsed.answer_text, choices)
        if inferred is None:
            return False
        indices.append(inferred)

    if components_present == 0 or not indices:
        return False
    return len(set(indices)) == 1


def score_mcq_response(
    raw_response: str,
    choices: Sequence[str],
    gold_answer_index: int | None,
    pool_subject: str = "",
) -> MCQScoreResult:
    parsed = parse_mcq_response(raw_response, choices)
    answer_present = bool(parsed.answer_letter or parsed.answer_text or parsed.answer_index is not None)
    consistent = answer_consistent(parsed, choices)
    gold_label = answer_label(gold_answer_index)
    gold_text = (
        choices[gold_answer_index]
        if gold_answer_index is not None and 0 <= gold_answer_index < len(choices)
        else ""
    )
    exact_text_match = bool(gold_text) and parsed.answer_text == gold_text
    reasoning_present = bool(parsed.reasoning.strip())
    correct = gold_answer_index is not None and parsed.answer_index == gold_answer_index
    has_subject = bool(parsed.subject.strip())
    subject_match = subject_matches_pool(parsed.subject, pool_subject)
    usable_for_sft = parsed.valid_json and correct and consistent and exact_text_match and reasoning_present

    total = (
        0.25 * float(parsed.valid_json)
        + 0.10 * float(answer_present)
        + 0.10 * float(consistent)
        + 0.10 * float(exact_text_match)
        + 0.05 * float(reasoning_present)
        + 0.40 * float(correct)
    )

    return MCQScoreResult(
        total=round(total, 6),
        valid_json=parsed.valid_json,
        parser_recovered=parsed.parser_recovered,
        answer_present=answer_present,
        answer_consistent=consistent,
        exact_answer_text_match=exact_text_match,
        reasoning_present=reasoning_present,
        subject_present=has_subject,
        subject_matches_pool=subject_match,
        correct=correct,
        usable_for_sft=usable_for_sft,
        parsed=parsed,
        pool_subject=pool_subject,
        gold_answer_index=gold_answer_index,
        gold_answer_label=gold_label,
        gold_answer_text=gold_text,
    )
