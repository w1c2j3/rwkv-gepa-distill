from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from .mmlu_score import score_mcq_response
from .question_pools import QuestionPoolRecord, iter_question_pool
from .teacher_client import TeacherClient


PROMPT_BUNDLE_CONTRACT = "mmlu_prompt_trace_v1"


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


@dataclass
class GenerationJob:
    index: int
    example: QuestionPoolRecord
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an MMLU teacher batch from a question pool.")
    parser.add_argument(
        "--question-pool-path",
        type=Path,
        default=Path("data/question_pools/mmlu_auxiliary_train.jsonl"),
    )
    parser.add_argument(
        "--prompt-bundle-path",
        type=Path,
        default=Path("artifacts/mmlu_prompt_bundle.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all.jsonl"),
    )
    parser.add_argument(
        "--failed-output-path",
        type=Path,
        default=Path("artifacts/mmlu_batch_all_failed.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=99842)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=3,
        help="How many extra full retry rounds to run for failed questions after the first pass.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Append to an existing partial JSONL output instead of restarting from scratch.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Restart generation from scratch and overwrite the output and fail files.",
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
    optimized_prompt: str,
    version: str,
    teacher: TeacherClient,
) -> GeneratedExample:
    response = teacher.generate_from_user_message(optimized_prompt, example.prompt_text)
    score = score_mcq_response(
        response.content,
        example.choices,
        example.answer_index,
        pool_subject=example.subject,
    )

    record = DistillBatchRecord(
        system_prompt=optimized_prompt,
        instruction=example.prompt_text,
        teacher_response=response.content,
        teacher_parsed=score.parsed.to_dict(),
        source_question=source_question_payload(example),
        meta={
            "source_row_index": index,
            "prompt_version": version,
            "teacher_mode": teacher.mode,
            "teacher_model": teacher.config.model,
            "output_contract": "strict_json_mcq_v1",
            "pool_subject": example.subject,
            "teacher_subject": score.parsed.subject,
            "subject_match_pool": score.subject_matches_pool,
            "score": score.to_dict(),
        },
    )
    return GeneratedExample(
        index=index,
        record=record.to_dict(),
        correct=score.correct,
    )


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")
    handle.flush()


def load_prompt_bundle(prompt_bundle_path: Path) -> dict[int, tuple[str, str]]:
    if not prompt_bundle_path.exists():
        raise FileNotFoundError(f"Missing prompt bundle: {prompt_bundle_path}")

    prompt_map: dict[int, tuple[str, str]] = {}
    with prompt_bundle_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {prompt_bundle_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{prompt_bundle_path}:{line_number} must be a JSON object")

            bundle_contract = payload.get("bundle_contract")
            source_row_index = payload.get("source_row_index")
            question = payload.get("question")
            prompt = payload.get("prompt")
            prompt_teacher_response = payload.get("prompt_teacher_response")
            best_prompt = payload.get("best_prompt")
            prompt_version_value = payload.get("best_prompt_version")
            best_prompt_teacher_response = payload.get("best_prompt_teacher_response")
            optimizer = payload.get("optimizer")
            comparison = payload.get("comparison")
            if bundle_contract is not None and bundle_contract != PROMPT_BUNDLE_CONTRACT:
                raise ValueError(
                    f"{prompt_bundle_path}:{line_number} has unsupported 'bundle_contract': {bundle_contract!r}"
                )
            if bundle_contract != PROMPT_BUNDLE_CONTRACT:
                raise ValueError(
                    f"{prompt_bundle_path}:{line_number} is not using the current prompt bundle contract"
                )
            if not isinstance(source_row_index, int) or source_row_index < 0:
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'source_row_index'")
            if not isinstance(question, dict):
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'question'")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'prompt'")
            if not isinstance(prompt_teacher_response, str) or not prompt_teacher_response.strip():
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'prompt_teacher_response'")
            if not isinstance(best_prompt, str) or not best_prompt.strip():
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'best_prompt'")
            if not isinstance(prompt_version_value, str) or not prompt_version_value.strip():
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'best_prompt_version'")
            if not isinstance(best_prompt_teacher_response, str) or not best_prompt_teacher_response.strip():
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'best_prompt_teacher_response'")
            if not isinstance(optimizer, dict):
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'optimizer'")
            if not isinstance(comparison, dict):
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'comparison'")

            prompt_map[source_row_index] = (best_prompt.strip(), prompt_version_value.strip())

    if not prompt_map:
        raise ValueError(f"No prompt rows found in {prompt_bundle_path}")
    return prompt_map


def processed_index_from_success(payload: dict[str, Any], line_number: int) -> int:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid success payload at line {line_number}: missing 'meta'")
    source_row_index = meta.get("source_row_index")
    if isinstance(source_row_index, int) and source_row_index >= 0:
        return source_row_index
    return line_number - 1


def load_success_state(output_path: Path) -> tuple[set[int], int, int]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set(), 0, 0

    processed_indices: set[int] = set()
    success_count = 0
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

            processed_indices.add(processed_index_from_success(payload, line_number))
            success_count += 1
            correct_count += int(correct)

    return processed_indices, success_count, correct_count


def processed_index_from_failure(payload: dict[str, Any], line_number: int) -> int:
    source_row_index = payload.get("source_row_index")
    if isinstance(source_row_index, int) and source_row_index >= 0:
        return source_row_index
    raise ValueError(f"Invalid failure payload at line {line_number}: missing 'source_row_index'")


def load_failure_indices(failed_output_path: Path) -> set[int]:
    if not failed_output_path.exists() or failed_output_path.stat().st_size == 0:
        return set()

    processed_indices: set[int] = set()
    with failed_output_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = orjson.loads(line)
            except orjson.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in existing fail output {failed_output_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{failed_output_path}:{line_number} must be a JSON object")
            processed_indices.add(processed_index_from_failure(payload, line_number))
    return processed_indices


def format_job_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def write_success_record(
    output_handle: Any,
    generated: GeneratedExample,
    *,
    processed_indices: set[int],
) -> None:
    append_jsonl_line(output_handle, generated.record)
    processed_indices.add(generated.index)


def make_failed_record(
    job: GenerationJob,
    *,
    prompt_version_value: str,
    teacher: TeacherClient,
) -> dict[str, Any]:
    return {
        "source_row_index": job.index,
        "attempt_count": job.attempts,
        "error_count": len(job.errors),
        "errors": job.errors,
        "prompt_version": prompt_version_value,
        "teacher_model": teacher.config.model,
        "source_question": source_question_payload(job.example),
    }


def prompt_version_for_job(job_index: int, *, prompt_map: dict[int, tuple[str, str]]) -> str:
    prompt_payload = prompt_map.get(job_index)
    if prompt_payload is None:
        return "missing-prompt"
    return prompt_payload[1]


def submit_job(
    executor: ThreadPoolExecutor,
    job: GenerationJob,
    *,
    prompt_map: dict[int, tuple[str, str]],
    teacher: TeacherClient,
) -> Any:
    job.attempts += 1
    prompt_payload = prompt_map.get(job.index)
    if prompt_payload is None:
        raise ValueError(f"Missing optimized prompt for source_row_index={job.index}")
    job_prompt, job_version = prompt_payload
    return executor.submit(
        build_generated_example,
        job.index,
        job.example,
        optimized_prompt=job_prompt,
        version=job_version,
        teacher=teacher,
    )


def iter_pending_jobs(path: Path, limit: int, processed_indices: set[int]):
    for index, example in enumerate(iter_question_pool(path, limit=limit)):
        if index in processed_indices:
            continue
        yield GenerationJob(index=index, example=example)


def run_retry_round(
    jobs: list[GenerationJob],
    *,
    output_handle: Any,
    processed_indices: set[int],
    prompt_map: dict[int, tuple[str, str]],
    teacher: TeacherClient,
    max_concurrency: int,
) -> tuple[list[GenerationJob], int]:
    if not jobs:
        return [], 0

    resolved_correct = 0
    remaining_jobs: list[GenerationJob] = []
    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(jobs))) as executor:
        future_to_job = {
            submit_job(
                executor,
                job,
                prompt_map=prompt_map,
                teacher=teacher,
            ): job
            for job in jobs
        }
        while future_to_job:
            done, _ = wait(future_to_job.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                job = future_to_job.pop(future)
                try:
                    generated = future.result()
                except Exception as exc:
                    job.errors.append(format_job_error(exc))
                    remaining_jobs.append(job)
                    continue

                write_success_record(output_handle, generated, processed_indices=processed_indices)
                resolved_correct += int(generated.correct)

    return remaining_jobs, resolved_correct


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
    if args.retry_rounds < 0:
        raise ValueError("--retry-rounds must be non-negative")

    prompt_map = load_prompt_bundle(args.prompt_bundle_path)
    print("prompt_source=prompt_bundle", flush=True)
    print(f"prompt_bundle_path={args.prompt_bundle_path}", flush=True)
    print(f"prompt_bundle_rows={len(prompt_map)}", flush=True)

    teacher = TeacherClient.from_env()
    if teacher.mode != "api":
        raise RuntimeError(
            "Teacher API is not fully configured. Set TEACHER_MODEL and TEACHER_API_KEY before batch generation."
        )

    if args.resume:
        success_indices, success_count, correct_count = load_success_state(args.output_path)
        failed_indices = load_failure_indices(args.failed_output_path)
        processed_indices = success_indices | failed_indices
        if success_count or failed_indices:
            print(f"resume_from={success_count}", flush=True)
            print(f"resume_failed={len(failed_indices)}", flush=True)
    else:
        processed_indices = set()
        success_count = 0
        correct_count = 0
        if args.output_path.exists():
            args.output_path.unlink()
        if args.failed_output_path.exists():
            args.failed_output_path.unlink()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output_path.parent.mkdir(parents=True, exist_ok=True)

    pending_jobs = args.limit - len(processed_indices)
    if pending_jobs <= 0:
        print("generation_already_complete=true", flush=True)
        print(f"examples_written={success_count}", flush=True)
        print(f"failed_examples={len(load_failure_indices(args.failed_output_path))}", flush=True)
        print(f"output_path={args.output_path}", flush=True)
        print(f"failed_output_path={args.failed_output_path}", flush=True)
        return

    jobs = iter_pending_jobs(args.question_pool_path, args.limit, processed_indices)
    print(f"max_concurrency={args.max_concurrency}", flush=True)
    print(f"pending_jobs={pending_jobs}", flush=True)

    failed_jobs: list[GenerationJob] = []
    existing_failed_count = len(load_failure_indices(args.failed_output_path))
    file_mode = "ab" if args.resume and args.output_path.exists() and args.output_path.stat().st_size > 0 else "wb"
    with args.output_path.open(file_mode) as output_handle:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            future_to_job: dict[Any, GenerationJob] = {}

            def submit_next() -> bool:
                try:
                    job = next(jobs)
                except StopIteration:
                    return False
                future = submit_job(
                    executor,
                    job,
                    prompt_map=prompt_map,
                    teacher=teacher,
                )
                future_to_job[future] = job
                return True

            for _ in range(min(args.max_concurrency, pending_jobs)):
                if not submit_next():
                    break

            while future_to_job:
                done, _ = wait(future_to_job.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    job = future_to_job.pop(future)
                    try:
                        generated = future.result()
                    except Exception as exc:
                        job.errors.append(format_job_error(exc))
                        failed_jobs.append(job)
                    else:
                        write_success_record(output_handle, generated, processed_indices=processed_indices)
                        success_count += 1
                        correct_count += int(generated.correct)

                        if (success_count + existing_failed_count) % args.progress_interval == 0:
                            print(f"progress={success_count + existing_failed_count}/{args.limit}", flush=True)

                    submit_next()

    if failed_jobs:
        print(f"initial_failures={len(failed_jobs)}", flush=True)

    for retry_round in range(1, args.retry_rounds + 1):
        if not failed_jobs:
            break
        print(f"retry_round={retry_round} pending_failures={len(failed_jobs)}", flush=True)
        with args.output_path.open("ab") as output_handle:
            failed_jobs, _ = run_retry_round(
                failed_jobs,
                output_handle=output_handle,
                processed_indices=processed_indices,
                prompt_map=prompt_map,
                teacher=teacher,
                max_concurrency=args.max_concurrency,
            )
        _, success_count, correct_count = load_success_state(args.output_path)
        print(
            f"retry_round_complete={retry_round} remaining_failures={len(failed_jobs)}",
            flush=True,
        )

    new_failures_written = 0
    if failed_jobs:
        fail_file_mode = (
            "ab" if args.resume and args.failed_output_path.exists() and args.failed_output_path.stat().st_size > 0 else "wb"
        )
        with args.failed_output_path.open(fail_file_mode) as fail_handle:
            for job in sorted(failed_jobs, key=lambda item: item.index):
                append_jsonl_line(
                    fail_handle,
                    make_failed_record(
                        job,
                        prompt_version_value=prompt_version_for_job(
                            job.index,
                            prompt_map=prompt_map,
                        ),
                        teacher=teacher,
                    ),
                )
                processed_indices.add(job.index)
                new_failures_written += 1

    all_failed_indices = load_failure_indices(args.failed_output_path)
    success_indices, success_count, correct_count = load_success_state(args.output_path)
    processed_total = len(success_indices | all_failed_indices)
    if processed_total != args.limit:
        print(
            f"warning=processed_total_mismatch processed_total={processed_total} expected_limit={args.limit}",
            flush=True,
        )

    accuracy = round(correct_count / success_count, 6) if success_count else 0.0

    print(f"teacher_mode={teacher.mode}", flush=True)
    print(f"examples_written={success_count}", flush=True)
    print(f"failed_examples={len(all_failed_indices)}", flush=True)
    print(f"new_failures_written={new_failures_written}", flush=True)
    print(f"accuracy_vs_gold={accuracy}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"failed_output_path={args.failed_output_path}", flush=True)


if __name__ == "__main__":
    main()
