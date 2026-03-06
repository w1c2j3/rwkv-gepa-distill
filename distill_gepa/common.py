from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import orjson


@dataclass(frozen=True)
class SeedExample:
    instruction: str
    expected_keywords: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source: Path, line_number: int) -> "SeedExample":
        instruction = payload.get("instruction")
        expected_keywords = payload.get("expected_keywords")

        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{source}:{line_number} is missing a non-empty 'instruction' string")
        if not isinstance(expected_keywords, list) or not expected_keywords:
            raise ValueError(f"{source}:{line_number} must contain a non-empty 'expected_keywords' list")
        if not all(isinstance(item, str) and item.strip() for item in expected_keywords):
            raise ValueError(f"{source}:{line_number} contains invalid expected keywords")

        return cls(
            instruction=instruction.strip(),
            expected_keywords=[item.strip() for item in expected_keywords],
        )


def load_seed_examples(path: Path) -> list[SeedExample]:
    if not path.exists():
        raise FileNotFoundError(f"Missing seed file: {path}")

    examples: list[SeedExample] = []
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

        examples.append(SeedExample.from_dict(payload, path, line_number))

    if not examples:
        raise ValueError(f"No seed examples found in {path}")

    return examples


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for record in records:
            handle.write(orjson.dumps(record))
            handle.write(b"\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def prompt_version(prompt: str) -> str:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"prompt-{digest[:12]}"
