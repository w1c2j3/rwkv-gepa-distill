from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from .common import write_json
from .export_sft_dataset import build_sft_row
from .filter_mmlu_batch import rejection_reason
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
    raw_record: dict[str, Any]
    sft_record: dict[str, Any] | None
    rejection_reason: str


@dataclass
class GenerationJob:
    index: int
    example: QuestionPoolRecord
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MMLU teacher answers from the prompt bundle and write usable rows directly to SFT."
    )
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
        default=Path("data/distill/mmlu_batch_all_sft.jsonl"),
        help="Primary SFT JSONL output path.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Also write raw/strict/failure/stats diagnostics alongside the primary SFT output.",
    )
    parser.add_argument(
        "--raw-output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all.jsonl"),
        help="Diagnostics only: append raw teacher responses here.",
    )
    parser.add_argument(
        "--strict-output-path",
        type=Path,
        default=Path("data/distill/mmlu_batch_all_strict.jsonl"),
        help="Diagnostics only: append strict-filtered rows here.",
    )
    parser.add_argument(
        "--failed-output-path",
        type=Path,
        default=Path("artifacts/mmlu_batch_all_failed.jsonl"),
        help="Diagnostics only: append exhausted failures here.",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("artifacts/mmlu_batch_all_filter_stats.json"),
        help="Diagnostics only: write filter stats here.",
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
        help="Restart generation from scratch and overwrite the output and diagnostics files.",
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
    score_payload = score.to_dict()
    reason = rejection_reason(score_payload)

    raw_record = DistillBatchRecord(
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
            "score": score_payload,
        },
    ).to_dict()

    sft_record = build_sft_row(raw_record) if reason == "usable_for_sft" else None
    return GeneratedExample(
        index=index,
        raw_record=raw_record,
        sft_record=sft_record,
        rejection_reason=reason,
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


def iter_jsonl_records(path: Path):
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
            yield line_number, payload


def meta_source_row_index(payload: dict[str, Any], *, source: Path, line_number: int) -> int:
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{source}:{line_number} has invalid 'meta'")
    source_row_index = meta.get("source_row_index")
    if not isinstance(source_row_index, int) or source_row_index < 0:
        raise ValueError(f"{source}:{line_number} has invalid 'meta.source_row_index'")
    return source_row_index


def load_meta_index_file(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    processed_indices: set[int] = set()
    for line_number, payload in iter_jsonl_records(path):
        processed_indices.add(meta_source_row_index(payload, source=path, line_number=line_number))
    return processed_indices


def processed_index_from_failure(payload: dict[str, Any], line_number: int) -> int:
    source_row_index = payload.get("source_row_index")
    if isinstance(source_row_index, int) and source_row_index >= 0:
        return source_row_index
    raise ValueError(f"Invalid failure payload at line {line_number}: missing 'source_row_index'")


def load_failure_indices(failed_output_path: Path) -> set[int]:
    if not failed_output_path.exists() or failed_output_path.stat().st_size == 0:
        return set()

    processed_indices: set[int] = set()
    for line_number, payload in iter_jsonl_records(failed_output_path):
        processed_indices.add(processed_index_from_failure(payload, line_number))
    return processed_indices


def build_strict_record(raw_record: dict[str, Any], *, reason: str) -> dict[str, Any]:
    if reason != "usable_for_sft":
        raise ValueError("Strict rows can only be built from usable_for_sft examples")

    meta = raw_record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Raw record is missing object 'meta'")

    score = meta.get("score")
    if not isinstance(score, dict):
        raise ValueError("Raw record is missing object 'meta.score'")

    row = dict(raw_record)
    row["teacher_parsed"] = score.get("parsed", row.get("teacher_parsed"))
    strict_meta = dict(meta)
    strict_meta["filter_reason"] = reason
    strict_meta["output_contract"] = "strict_json_mcq_v1"
    row["meta"] = strict_meta
    return row


def summarize_raw_diagnostics(raw_output_path: Path) -> tuple[int, int, dict[str, int]]:
    if not raw_output_path.exists() or raw_output_path.stat().st_size == 0:
        return 0, 0, {}

    total_rows = 0
    kept_rows = 0
    rejection_counts: Counter[str] = Counter()
    for line_number, payload in iter_jsonl_records(raw_output_path):
        total_rows += 1
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"{raw_output_path}:{line_number} has invalid 'meta'")
        score = meta.get("score")
        if not isinstance(score, dict):
            raise ValueError(f"{raw_output_path}:{line_number} has invalid 'meta.score'")
        reason = rejection_reason(score)
        rejection_counts[reason] += 1
        kept_rows += int(reason == "usable_for_sft")

    return total_rows, kept_rows, dict(sorted(rejection_counts.items()))


def write_diagnostic_stats(
    *,
    raw_output_path: Path,
    strict_output_path: Path,
    stats_path: Path,
) -> None:
    total_rows, kept_rows, reason_counts = summarize_raw_diagnostics(raw_output_path)
    write_json(
        stats_path,
        {
            "input_path": str(raw_output_path),
            "output_path": str(strict_output_path),
            "stats_path": str(stats_path),
            "total_rows": total_rows,
            "kept_rows": kept_rows,
            "rejected_rows": total_rows - kept_rows,
            "reason_counts": reason_counts,
        },
    )


def reconcile_diagnostic_outputs(
    *,
    raw_output_path: Path,
    strict_output_path: Path,
    sft_output_path: Path,
) -> tuple[int, int]:
    if not raw_output_path.exists() or raw_output_path.stat().st_size == 0:
        return 0, 0

    repaired_strict_rows = 0
    repaired_sft_rows = 0
    strict_indices = load_meta_index_file(strict_output_path)
    sft_indices = load_meta_index_file(sft_output_path)

    strict_handle: Any | None = None
    sft_handle: Any | None = None
    with ExitStack() as stack:
        for _, payload in iter_jsonl_records(raw_output_path):
            meta = payload.get("meta")
            if not isinstance(meta, dict):
                raise ValueError(f"Invalid diagnostics raw payload in {raw_output_path}")
            source_row_index = meta.get("source_row_index")
            if not isinstance(source_row_index, int) or source_row_index < 0:
                raise ValueError(f"Invalid diagnostics raw payload in {raw_output_path}: missing meta.source_row_index")
            score = meta.get("score")
            if not isinstance(score, dict):
                raise ValueError(f"Invalid diagnostics raw payload in {raw_output_path}: missing meta.score")
            reason = rejection_reason(score)
            if reason != "usable_for_sft":
                continue

            if source_row_index not in strict_indices:
                if strict_handle is None:
                    strict_output_path.parent.mkdir(parents=True, exist_ok=True)
                    strict_mode = "ab" if strict_output_path.exists() and strict_output_path.stat().st_size > 0 else "wb"
                    strict_handle = stack.enter_context(strict_output_path.open(strict_mode))
                append_jsonl_line(strict_handle, build_strict_record(payload, reason=reason))
                strict_indices.add(source_row_index)
                repaired_strict_rows += 1

            if source_row_index not in sft_indices:
                if sft_handle is None:
                    sft_output_path.parent.mkdir(parents=True, exist_ok=True)
                    sft_mode = "ab" if sft_output_path.exists() and sft_output_path.stat().st_size > 0 else "wb"
                    sft_handle = stack.enter_context(sft_output_path.open(sft_mode))
                append_jsonl_line(sft_handle, build_sft_row(payload))
                sft_indices.add(source_row_index)
                repaired_sft_rows += 1

    return repaired_strict_rows, repaired_sft_rows


def format_job_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


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


def jsonl_file_mode(path: Path, *, resume: bool) -> str:
    return "ab" if resume and path.exists() and path.stat().st_size > 0 else "wb"


def write_generated_records(
    *,
    generated: GeneratedExample,
    sft_handle: Any,
    diagnostics: bool,
    raw_handle: Any | None,
    strict_handle: Any | None,
) -> bool:
    if diagnostics:
        if raw_handle is None or strict_handle is None:
            raise ValueError("Diagnostics mode requires raw and strict output handles")
        append_jsonl_line(raw_handle, generated.raw_record)
    if generated.sft_record is None:
        return False
    if diagnostics:
        append_jsonl_line(
            strict_handle,
            build_strict_record(generated.raw_record, reason=generated.rejection_reason),
        )
    append_jsonl_line(sft_handle, generated.sft_record)
    return True


def run_retry_round(
    jobs: list[GenerationJob],
    *,
    sft_handle: Any,
    diagnostics: bool,
    raw_handle: Any | None,
    strict_handle: Any | None,
    prompt_map: dict[int, tuple[str, str]],
    teacher: TeacherClient,
    max_concurrency: int,
) -> tuple[list[GenerationJob], int, int]:
    if not jobs:
        return [], 0, 0

    resolved_accepted = 0
    resolved_attempted = 0
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

                resolved_attempted += 1
                resolved_accepted += int(
                    write_generated_records(
                        generated=generated,
                        sft_handle=sft_handle,
                        diagnostics=diagnostics,
                        raw_handle=raw_handle,
                        strict_handle=strict_handle,
                    )
                )

    return remaining_jobs, resolved_accepted, resolved_attempted


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

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.diagnostics:
        args.raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.strict_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.failed_output_path.parent.mkdir(parents=True, exist_ok=True)
        args.stats_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        if args.output_path.exists():
            args.output_path.unlink()
        if args.diagnostics:
            for path in (
                args.raw_output_path,
                args.strict_output_path,
                args.failed_output_path,
                args.stats_path,
            ):
                if path.exists():
                    path.unlink()

    repaired_strict_rows = 0
    repaired_sft_rows = 0
    if args.resume and args.diagnostics:
        repaired_strict_rows, repaired_sft_rows = reconcile_diagnostic_outputs(
            raw_output_path=args.raw_output_path,
            strict_output_path=args.strict_output_path,
            sft_output_path=args.output_path,
        )

    success_indices = load_meta_index_file(args.output_path) if args.resume else set()
    success_count = len(success_indices)
    if args.resume and args.diagnostics:
        raw_indices = load_meta_index_file(args.raw_output_path)
        failed_indices = load_failure_indices(args.failed_output_path)
        processed_indices = raw_indices | failed_indices
        if success_count or raw_indices or failed_indices or repaired_strict_rows or repaired_sft_rows:
            print(f"resume_sft_rows={success_count}", flush=True)
            print(f"resume_processed_rows={len(processed_indices)}", flush=True)
            print(f"resume_failed_rows={len(failed_indices)}", flush=True)
            if repaired_strict_rows or repaired_sft_rows:
                print(
                    f"resume_repaired_strict_rows={repaired_strict_rows} resume_repaired_sft_rows={repaired_sft_rows}",
                    flush=True,
                )
    elif args.resume:
        processed_indices = set(success_indices)
        if success_count:
            print(f"resume_sft_rows={success_count}", flush=True)
    else:
        processed_indices = set()

    jobs = iter_pending_jobs(args.question_pool_path, args.limit, processed_indices)
    print(f"max_concurrency={args.max_concurrency}", flush=True)
    print(f"diagnostics={args.diagnostics}", flush=True)

    failed_jobs: list[GenerationJob] = []
    attempted_this_run = 0
    accepted_this_run = 0
    with ExitStack() as stack:
        sft_handle = stack.enter_context(args.output_path.open(jsonl_file_mode(args.output_path, resume=args.resume)))
        raw_handle: Any | None = None
        strict_handle: Any | None = None
        if args.diagnostics:
            raw_handle = stack.enter_context(
                args.raw_output_path.open(jsonl_file_mode(args.raw_output_path, resume=args.resume))
            )
            strict_handle = stack.enter_context(
                args.strict_output_path.open(jsonl_file_mode(args.strict_output_path, resume=args.resume))
            )

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

            for _ in range(args.max_concurrency):
                if not submit_next():
                    break

            if not future_to_job:
                print("generation_already_complete=true", flush=True)
                print(f"examples_written={success_count}", flush=True)
                if args.diagnostics:
                    write_diagnostic_stats(
                        raw_output_path=args.raw_output_path,
                        strict_output_path=args.strict_output_path,
                        stats_path=args.stats_path,
                    )
                    print(f"raw_output_path={args.raw_output_path}", flush=True)
                    print(f"strict_output_path={args.strict_output_path}", flush=True)
                    print(f"failed_output_path={args.failed_output_path}", flush=True)
                    print(f"stats_path={args.stats_path}", flush=True)
                print(f"output_path={args.output_path}", flush=True)
                return

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
                        attempted_this_run += 1
                        accepted = write_generated_records(
                            generated=generated,
                            sft_handle=sft_handle,
                            diagnostics=args.diagnostics,
                            raw_handle=raw_handle,
                            strict_handle=strict_handle,
                        )
                        accepted_this_run += int(accepted)
                        success_count += int(accepted)

                        if attempted_this_run % args.progress_interval == 0:
                            print(
                                f"progress_attempted={attempted_this_run} accepted_total={success_count}",
                                flush=True,
                            )

                    submit_next()

        if failed_jobs:
            print(f"initial_failures={len(failed_jobs)}", flush=True)

        for retry_round in range(1, args.retry_rounds + 1):
            if not failed_jobs:
                break
            print(f"retry_round={retry_round} pending_failures={len(failed_jobs)}", flush=True)
            failed_jobs, retry_accepted, retry_attempted = run_retry_round(
                failed_jobs,
                sft_handle=sft_handle,
                diagnostics=args.diagnostics,
                raw_handle=raw_handle,
                strict_handle=strict_handle,
                prompt_map=prompt_map,
                teacher=teacher,
                max_concurrency=args.max_concurrency,
            )
            accepted_this_run += retry_accepted
            attempted_this_run += retry_attempted
            success_count += retry_accepted
            print(
                f"retry_round_complete={retry_round} remaining_failures={len(failed_jobs)} accepted_total={success_count}",
                flush=True,
            )

    new_failures_written = 0
    if failed_jobs and args.diagnostics:
        with args.failed_output_path.open(jsonl_file_mode(args.failed_output_path, resume=args.resume)) as fail_handle:
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
                new_failures_written += 1

    failed_count = len(load_failure_indices(args.failed_output_path)) if args.diagnostics else len(failed_jobs)
    if args.diagnostics:
        write_diagnostic_stats(
            raw_output_path=args.raw_output_path,
            strict_output_path=args.strict_output_path,
            stats_path=args.stats_path,
        )
        total_rows, kept_rows, reason_counts = summarize_raw_diagnostics(args.raw_output_path)
        print(f"diagnostic_total_rows={total_rows}", flush=True)
        print(f"diagnostic_kept_rows={kept_rows}", flush=True)
        print(f"diagnostic_reason_counts={orjson.dumps(reason_counts).decode('utf-8')}", flush=True)

    print(f"teacher_mode={teacher.mode}", flush=True)
    print(f"examples_written={success_count}", flush=True)
    print(f"attempted_this_run={attempted_this_run}", flush=True)
    print(f"accepted_this_run={accepted_this_run}", flush=True)
    print(f"failed_examples={failed_count}", flush=True)
    if args.diagnostics:
        print(f"new_failures_written={new_failures_written}", flush=True)
        print(f"raw_output_path={args.raw_output_path}", flush=True)
        print(f"strict_output_path={args.strict_output_path}", flush=True)
        print(f"failed_output_path={args.failed_output_path}", flush=True)
        print(f"stats_path={args.stats_path}", flush=True)
    print(f"output_path={args.output_path}", flush=True)


if __name__ == "__main__":
    main()
