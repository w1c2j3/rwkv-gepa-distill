from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from .common import prompt_version
from .mmlu_score import score_mcq_response
from .question_pools import QuestionPoolRecord, iter_question_pool
from .teacher_client import TeacherClient


@dataclass(frozen=True)
class DistillBatchRecord:
    system_prompt: str
    instruction: str
    teacher_response: str
    teacher_parsed: dict[str, Any]
    source_question: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "instruction": self.instruction,
            "teacher_response": self.teacher_response,
            "teacher_parsed": self.teacher_parsed,
            "source_question": self.source_question,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class GeneratedExample:
    index: int
    record: dict[str, Any]
    correct: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an MMLU teacher batch from a question pool.")
    parser.add_argument(
        "--question-pool-path",
        type=Path,
        default=Path("data/question_pools/mmlu_auxiliary_train.jsonl"),
    )
    parser.add_argument(
        "--best-prompt-path",
        type=Path,
        default=Path("artifacts/mmlu_best_prompt.txt"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_100.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing partial JSONL output instead of restarting from scratch.",
    )
    return parser.parse_args()


def source_question_payload(example: QuestionPoolRecord) -> dict[str, Any]:
    return {
        "source_dataset": example.source_dataset,
        "source_split": example.source_split,
        "subject": example.subject,
        "question": example.question,
        "choices": example.choices,
        "answer": example.answer,
        "answer_index": example.answer_index,
        "answer_label": example.answer_label,
        "prompt_text": example.prompt_text,
        "meta": example.meta,
    }


def build_generated_example(
    index: int,
    example: QuestionPoolRecord,
    *,
    best_prompt: str,
    version: str,
    teacher: TeacherClient,
) -> GeneratedExample:
    response = teacher.generate_from_user_message(best_prompt, example.prompt_text)
    score = score_mcq_response(response.content, example.choices, example.answer_index)

    record = DistillBatchRecord(
        system_prompt=best_prompt,
        instruction=example.prompt_text,
        teacher_response=response.content,
        teacher_parsed=score.parsed.to_dict(),
        source_question=source_question_payload(example),
        meta={
            "prompt_version": version,
            "teacher_mode": teacher.mode,
            "teacher_model": teacher.config.model,
            "score": score.to_dict(),
        },
    )
    return GeneratedExample(
        index=index,
        record=record.to_dict(),
        correct=score.correct,
    )


def load_existing_progress(output_path: Path) -> tuple[int, int]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return 0, 0

    completed = 0
    correct_count = 0
    with output_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in existing output {output_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{output_path}:{line_number} must be a JSON object")
            meta = payload.get("meta")
            if not isinstance(meta, dict):
                raise ValueError(f"{output_path}:{line_number} has invalid 'meta'")
            score = meta.get("score")
            if not isinstance(score, dict):
                raise ValueError(f"{output_path}:{line_number} has invalid 'meta.score'")
            correct = score.get("correct")
            if not isinstance(correct, bool):
                raise ValueError(f"{output_path}:{line_number} has invalid 'meta.score.correct'")
            completed += 1
            correct_count += int(correct)
    return completed, correct_count


def skip_examples(example_iter: Any, count: int) -> None:
    for index in range(count):
        try:
            next(example_iter)
        except StopIteration as exc:
            raise RuntimeError(
                f"Cannot resume from row {count}: question pool ended early at row {index}."
            ) from exc


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    if args.max_concurrency > 6:
        raise ValueError("--max-concurrency must be <= 6 to avoid overloading unstable proxy endpoints")
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if not args.best_prompt_path.exists():
        raise FileNotFoundError(
            f"Missing optimized prompt at {args.best_prompt_path}. Run optimize_mmlu_prompt.py first."
        )

    best_prompt = args.best_prompt_path.read_text(encoding="utf-8").strip()
    if not best_prompt:
        raise ValueError(f"Optimized prompt file is empty: {args.best_prompt_path}")

    teacher = TeacherClient.from_env()
    if teacher.mode != "api":
        raise RuntimeError(
            "Teacher API is not fully configured. Set TEACHER_MODEL and TEACHER_API_KEY before batch generation."
        )

    version = prompt_version(best_prompt)
    completed = 0
    correct_count = 0
    submitted = 0
    next_write_index = 0
    pending_results: dict[int, GeneratedExample] = {}
    example_iter = iter_question_pool(args.question_pool_path, limit=args.limit)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        completed, correct_count = load_existing_progress(args.output_path)
        if completed > args.limit:
            raise RuntimeError(
                f"Existing output has {completed} rows, which exceeds --limit={args.limit}. "
                "Delete the output file or raise --limit."
            )
        if completed > 0:
            skip_examples(example_iter, completed)
            submitted = completed
            next_write_index = completed
            print(f"resume_from={completed}", flush=True)

    print(f"max_concurrency={args.max_concurrency}", flush=True)
    file_mode = "ab" if args.resume and completed > 0 else "wb"
    with args.output_path.open(file_mode) as output_handle:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            future_to_index: dict[Any, int] = {}

            def submit_next() -> bool:
                nonlocal submitted
                try:
                    example = next(example_iter)
                except StopIteration:
                    return False

                future = executor.submit(
                    build_generated_example,
                    submitted,
                    example,
                    best_prompt=best_prompt,
                    version=version,
                    teacher=teacher,
                )
                future_to_index[future] = submitted
                submitted += 1
                return True

            for _ in range(args.max_concurrency):
                if not submit_next():
                    break

            if completed == args.limit and not future_to_index:
                print("generation_already_complete=true", flush=True)

            while future_to_index:
                done, _ = wait(future_to_index.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    source_index = future_to_index.pop(future)
                    try:
                        generated = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"Teacher generation failed for example index {source_index}") from exc

                    pending_results[generated.index] = generated
                    correct_count += int(generated.correct)
                    completed += 1

                    while next_write_index in pending_results:
                        ready = pending_results.pop(next_write_index)
                        output_handle.write(orjson.dumps(ready.record))
                        output_handle.write(b"\n")
                        next_write_index += 1

                    if completed % args.progress_interval == 0:
                        print(f"progress={completed}/{args.limit}", flush=True)

                    submit_next()

    if completed != args.limit:
        raise RuntimeError(
            f"Expected to generate {args.limit} examples, but only completed {completed}. "
            "Aborting because the output would be incomplete."
        )

    accuracy = round(correct_count / completed, 6)

    print(f"teacher_mode={teacher.mode}", flush=True)
    print(f"examples_written={completed}", flush=True)
    print(f"accuracy_vs_gold={accuracy}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
