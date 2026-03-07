from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import orjson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the strict SFT dataset into RWKV jsonl training format."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_sft.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_rwkv.jsonl"),
    )
    parser.add_argument("--progress-interval", type=int, default=5000)
    return parser.parse_args()


def build_rwkv_row(payload: dict[str, Any]) -> dict[str, str]:
    system_prompt = payload.get("system")
    user_prompt = payload.get("user")
    assistant_response = payload.get("assistant")

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("Missing non-empty 'system'")
    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("Missing non-empty 'user'")
    if not isinstance(assistant_response, str) or not assistant_response.strip():
        raise ValueError("Missing non-empty 'assistant'")

    text = (
        f"System: {system_prompt.strip()}\n\n"
        f"User: {user_prompt.strip()}\n\n"
        f"Assistant: {assistant_response.strip()}"
    )
    return {"text": text}


def main() -> None:
    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Missing input JSONL: {args.input_path}")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    with args.input_path.open("rb") as input_handle, args.output_path.open("wb") as output_handle:
        for line_number, raw_line in enumerate(input_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {args.input_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{args.input_path}:{line_number} must be a JSON object")

            output_handle.write(orjson.dumps(build_rwkv_row(payload)))
            output_handle.write(b"\n")
            total_rows += 1

            if total_rows % args.progress_interval == 0:
                print(f"progress={total_rows}", flush=True)

    if total_rows == 0:
        raise ValueError(f"No rows found in {args.input_path}")

    print(f"rows_written={total_rows}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
