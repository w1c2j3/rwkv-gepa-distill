from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import orjson

from .common import prompt_version, write_json
from .mmlu_score import MCQScoreResult, score_mcq_response
from .question_pools import QuestionPoolRecord, iter_question_pool, load_question_pool
from .reflection_lm import make_openai_lm, resolve_reflection_runtime
from .teacher_client import TeacherClient

try:
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
except ImportError:
    oa = None
    EngineConfig = GEPAConfig = ReflectionConfig = None


SEED_SYSTEM_PROMPT = """You are a precise multiple-choice world-knowledge teacher.
You will receive one multiple-choice question.
Return exactly one JSON object with keys "answer_letter", "answer_index", "answer_text", and "reasoning".
Rules:
- Use the provided Subject context when it helps disambiguate the question.
- Choose the single best option.
- Keep reasoning brief: 1-2 plain sentences.
- Use subject-appropriate terminology when helpful, but stay concise.
- Ensure answer_letter, answer_index, and answer_text all refer to the same option.
- answer_text must exactly match the chosen option text.
- Output ONLY valid JSON. No markdown, headings, bullets, bold text, or extra commentary.
- Start with { and end with }.
- Use double quotes for every JSON key and every string value.
- Before answering, check that the response is a single valid JSON object and nothing else."""

PROMPT_BUNDLE_CONTRACT = "mmlu_prompt_trace_v1"


HEURISTIC_CANDIDATES = [
    SEED_SYSTEM_PROMPT,
    """You are a precise multiple-choice world-knowledge teacher.
Return exactly one valid JSON object with keys "answer_letter", "answer_index", "answer_text", and "reasoning".
Use the Subject context when helpful, select the best option, and keep reasoning concise.
Output JSON only, with no markdown or extra text.""",
    """You are a precise multiple-choice world-knowledge teacher.
Return exactly one valid JSON object and nothing else.
Schema:
{"answer_letter": "A-J", "answer_index": 0, "answer_text": "<full option text>", "reasoning": "<brief explanation>"}
Rules:
- Pick exactly one best answer.
- The answer_letter, answer_index, and answer_text must agree.
- answer_text must match the chosen option exactly.
- Reasoning should be brief, factual, and use subject-appropriate terminology when useful.
- Do not add markdown fences, bold text, or extra commentary.""",
    """You are a precise multiple-choice world-knowledge teacher.
You will receive one question formatted with Subject, Question, and Options.
Return exactly one valid JSON object and nothing else.
Schema:
{"answer_letter":"A-J","answer_index":0,"answer_text":"<exact option text>","reasoning":"<1-2 sentence subject-aware explanation>"}
Rules:
- Use the Subject field as context when deciding and explaining.
- Choose exactly one best option.
- answer_letter, answer_index, and answer_text must refer to the same option.
- answer_text must copy the chosen option text exactly.
- reasoning must be plain text, concise, and domain-appropriate.
- Do not write phrases like "Correct choice", markdown fences, bullets, or any text outside the JSON object.""",
]


@dataclass(frozen=True)
class ExampleEvaluation:
    prompt_text: str
    teacher_response: str
    score: MCQScoreResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text": self.prompt_text,
            "teacher_response": self.teacher_response,
            "score": self.score.to_dict(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-row MMLU prompt bundle with GEPA.")
    parser.add_argument("--train-path", type=Path, default=Path("data/question_pools/mmlu_dev.jsonl"))
    parser.add_argument("--question-pool-path", type=Path, default=Path("data/question_pools/mmlu_auxiliary_train.jsonl"))
    parser.add_argument("--train-limit", type=int, default=10)
    parser.add_argument("--target-limit", type=int, default=None)
    parser.add_argument("--max-metric-calls", type=int, default=40)
    parser.add_argument("--gepa-run-dir", type=Path, default=Path("artifacts/mmlu_gepa_run"))
    parser.add_argument("--fresh-run-dir", action="store_true")
    parser.add_argument("--prompt-bundle-path", type=Path, default=Path("artifacts/mmlu_prompt_bundle.jsonl"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/mmlu_prompt_optimization_report.json"),
    )
    parser.add_argument("--force-heuristic", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def safe_oa_log(message: str) -> None:
    if oa is None or not hasattr(oa, "log"):
        return
    oa.log(message)


def evaluate_example(
    candidate_prompt: str,
    example: QuestionPoolRecord,
    teacher: TeacherClient,
    emit_oa_logs: bool,
) -> ExampleEvaluation:
    response = teacher.generate_from_user_message(candidate_prompt, example.prompt_text)
    score = score_mcq_response(
        response.content,
        example.choices,
        example.answer_index,
        pool_subject=example.subject,
    )

    if emit_oa_logs:
        if not score.valid_json:
            safe_oa_log(f"invalid_json :: subject={example.subject} :: question={example.question[:120]}")
        if score.parser_recovered:
            safe_oa_log(f"parser_recovered :: subject={example.subject} :: question={example.question[:120]}")
        if not score.answer_present:
            safe_oa_log(f"missing_answer :: subject={example.subject} :: question={example.question[:120]}")
        if not score.answer_consistent:
            safe_oa_log(
                "inconsistent_answer :: "
                f"subject={example.subject} :: "
                f"pred_letter={score.parsed.answer_letter} :: "
                f"pred_index={score.parsed.answer_index} :: "
                f"pred_text={score.parsed.answer_text[:80]}"
            )
        if not score.exact_answer_text_match:
            safe_oa_log(
                "answer_text_mismatch :: "
                f"subject={example.subject} :: "
                f"gold_text={score.gold_answer_text[:80]} :: "
                f"pred_text={score.parsed.answer_text[:80]}"
            )
        if not score.reasoning_present:
            safe_oa_log(f"missing_reasoning :: subject={example.subject} :: question={example.question[:120]}")
        if not score.correct:
            safe_oa_log(
                "wrong_answer :: "
                f"gold={score.gold_answer_label} :: "
                f"pred={score.parsed.answer_letter} :: "
                f"subject={example.subject}"
            )
        safe_oa_log(f"example_output :: {response.content[:400]}")

    return ExampleEvaluation(
        prompt_text=example.prompt_text,
        teacher_response=response.content,
        score=score,
    )


def evaluate_prompt(
    candidate_prompt: str,
    examples: Sequence[QuestionPoolRecord],
    teacher: TeacherClient,
    emit_oa_logs: bool,
) -> dict[str, Any]:
    per_example: list[ExampleEvaluation] = []
    total_score = 0.0

    for example in examples:
        evaluation = evaluate_example(candidate_prompt, example, teacher, emit_oa_logs=emit_oa_logs)
        per_example.append(evaluation)
        total_score += evaluation.score.total

    average_score = round(total_score / len(examples), 6)
    def rate(predicate: str) -> float:
        return round(sum(1 for item in per_example if getattr(item.score, predicate)) / len(per_example), 6)

    return {
        "average_score": average_score,
        "accuracy": rate("correct"),
        "strict_json_rate": rate("valid_json"),
        "parser_recovered_rate": rate("parser_recovered"),
        "answer_consistency_rate": rate("answer_consistent"),
        "exact_answer_text_match_rate": rate("exact_answer_text_match"),
        "reasoning_present_rate": rate("reasoning_present"),
        "usable_for_sft_rate": rate("usable_for_sft"),
        "examples": [item.to_dict() for item in per_example],
    }


def normalize_candidate(candidate: Any) -> str:
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if isinstance(candidate, dict):
        for key in ("system_prompt", "prompt", "text", "candidate"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise TypeError(f"Unsupported best candidate shape: {type(candidate)!r}")


def prepare_gepa_run_dir(run_dir: Path, fresh_run_dir: bool) -> None:
    if fresh_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "gepa_state.bin"
    if state_path.exists() and state_path.stat().st_size == 0:
        state_path.unlink()


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")
    handle.flush()


def question_payload(example: QuestionPoolRecord) -> dict[str, Any]:
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


def load_processed_prompt_indices(prompt_bundle_path: Path) -> set[int]:
    if not prompt_bundle_path.exists() or prompt_bundle_path.stat().st_size == 0:
        return set()

    processed_indices: set[int] = set()
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
            if payload.get("bundle_contract") != PROMPT_BUNDLE_CONTRACT:
                raise ValueError(
                    f"{prompt_bundle_path}:{line_number} has unsupported 'bundle_contract'. "
                    "Delete the old bundle or rerun with --no-resume."
                )

            source_row_index = payload.get("source_row_index")
            if not isinstance(source_row_index, int) or source_row_index < 0:
                raise ValueError(f"{prompt_bundle_path}:{line_number} has invalid 'source_row_index'")
            processed_indices.add(source_row_index)

    return processed_indices


def run_gepa(
    train_examples: Sequence[QuestionPoolRecord],
    val_examples: Sequence[QuestionPoolRecord],
    teacher: TeacherClient,
    max_metric_calls: int,
    run_dir: Path,
    fresh_run_dir: bool,
) -> tuple[str, dict[str, Any]]:
    if oa is None or GEPAConfig is None or EngineConfig is None or ReflectionConfig is None:
        raise RuntimeError("gepa is not installed. Run `bash scripts/bootstrap.sh` first.")

    (
        reflection_model_name,
        reflection_api_key,
        reflection_api_base,
        reflection_api_protocol,
    ) = resolve_reflection_runtime(teacher)
    if not reflection_model_name:
        raise RuntimeError(
            "No reflection model available for GEPA. Set GEPA_MODEL / GEPA_REFLECTION_MODEL or TEACHER_* variables."
        )

    prepare_gepa_run_dir(run_dir, fresh_run_dir=fresh_run_dir)
    reflection_lm = make_openai_lm(
        model_name=reflection_model_name,
        api_key=reflection_api_key,
        api_base=reflection_api_base,
        api_protocol=reflection_api_protocol,
        timeout_seconds=teacher.config.timeout_seconds,
        num_retries=teacher.config.num_retries,
    )

    def evaluator(candidate_prompt: str, example: QuestionPoolRecord) -> tuple[float, dict[str, Any]]:
        evaluation = evaluate_example(candidate_prompt, example, teacher, emit_oa_logs=True)
        safe_oa_log(
            "candidate_example_summary :: "
            f"prompt_version={prompt_version(candidate_prompt)} :: "
            f"score={evaluation.score.total} :: "
            f"gold={evaluation.score.gold_answer_label} :: "
            f"pred={evaluation.score.parsed.answer_letter}"
        )
        return evaluation.score.total, {
            "scores": {
                "valid_json": 1.0 if evaluation.score.valid_json else 0.0,
                "parser_recovered": 1.0 if evaluation.score.parser_recovered else 0.0,
                "answer_present": 1.0 if evaluation.score.answer_present else 0.0,
                "answer_consistent": 1.0 if evaluation.score.answer_consistent else 0.0,
                "exact_answer_text_match": 1.0 if evaluation.score.exact_answer_text_match else 0.0,
                "reasoning_present": 1.0 if evaluation.score.reasoning_present else 0.0,
                "correct": 1.0 if evaluation.score.correct else 0.0,
                "usable_for_sft": 1.0 if evaluation.score.usable_for_sft else 0.0,
            },
            "prompt_text": example.prompt_text,
            "teacher_response": evaluation.teacher_response,
            "parsed": evaluation.score.parsed.to_dict(),
            "pool_subject": example.subject,
            "gold_answer_label": evaluation.score.gold_answer_label,
            "gold_answer_text": evaluation.score.gold_answer_text,
        }

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            max_metric_calls=max_metric_calls,
            cache_evaluation=True,
            use_cloudpickle=False,
        ),
        reflection=ReflectionConfig(reflection_lm=reflection_lm),
    )

    try:
        result = oa.optimize_anything(
            seed_candidate=SEED_SYSTEM_PROMPT,
            evaluator=evaluator,
            dataset=list(train_examples),
            valset=list(val_examples),
            objective=(
                "Optimize a system prompt for answering MMLU multiple-choice world-knowledge questions."
                " The teacher must return strict JSON with answer_letter, answer_index, answer_text,"
                " and concise reasoning."
            ),
            background=(
                "The evaluator rewards strict raw JSON, answer consistency, non-empty reasoning,"
                " exact answer_text matching the chosen option, and selecting the correct choice."
                " Do not rely on parser recovery or any non-JSON fallback."
            ),
            config=config,
        )
    except EOFError:
        state_path = run_dir / "gepa_state.bin"
        if state_path.exists():
            state_path.unlink()
        result = oa.optimize_anything(
            seed_candidate=SEED_SYSTEM_PROMPT,
            evaluator=evaluator,
            dataset=list(train_examples),
            valset=list(val_examples),
            objective=(
                "Optimize a system prompt for answering MMLU multiple-choice world-knowledge questions."
                " The teacher must return strict JSON with answer_letter, answer_index, answer_text,"
                " and concise reasoning."
            ),
            background=(
                "The evaluator rewards strict raw JSON, answer consistency, non-empty reasoning,"
                " exact answer_text matching the chosen option, and selecting the correct choice."
                " Do not rely on parser recovery or any non-JSON fallback."
            ),
            config=config,
        )

    best_prompt = normalize_candidate(result.best_candidate)
    best_summary = evaluate_prompt(best_prompt, val_examples, teacher, emit_oa_logs=False)
    best_score = result.val_aggregate_scores[result.best_idx]

    report = {
        "optimizer_mode": "gepa",
        "teacher_mode": teacher.mode,
        "reflection_model": reflection_model_name,
        "reflection_api_base": reflection_api_base,
        "reflection_api_protocol": reflection_api_protocol,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_score": best_score,
        "best_prompt_version": prompt_version(best_prompt),
        "seed_prompt": SEED_SYSTEM_PROMPT,
        "best_prompt": best_prompt,
        "best_prompt_eval": best_summary,
        "gepa_run_dir": str(run_dir),
        "gepa_best_idx": result.best_idx,
        "gepa_num_candidates": result.num_candidates,
        "gepa_total_metric_calls": result.total_metric_calls,
        "gepa_num_full_val_evals": result.num_full_val_evals,
    }
    return best_prompt, report


def run_heuristic_search(
    train_examples: Sequence[QuestionPoolRecord],
    val_examples: Sequence[QuestionPoolRecord],
    teacher: TeacherClient,
    fallback_reason: str | None,
) -> tuple[str, dict[str, Any]]:
    candidate_reports: list[dict[str, Any]] = []
    best_prompt = HEURISTIC_CANDIDATES[0]
    best_score = -1.0

    for index, candidate_prompt in enumerate(HEURISTIC_CANDIDATES, start=1):
        train_summary = evaluate_prompt(candidate_prompt, train_examples, teacher, emit_oa_logs=False)
        val_summary = evaluate_prompt(candidate_prompt, val_examples, teacher, emit_oa_logs=False)
        report = {
            "candidate_index": index,
            "prompt": candidate_prompt,
            "prompt_version": prompt_version(candidate_prompt),
            "train_average_score": train_summary["average_score"],
            "train_accuracy": train_summary["accuracy"],
            "val_average_score": val_summary["average_score"],
            "val_accuracy": val_summary["accuracy"],
            "val_examples": val_summary["examples"],
        }
        candidate_reports.append(report)
        if val_summary["average_score"] > best_score:
            best_prompt = candidate_prompt
            best_score = val_summary["average_score"]

    best_summary = next(item for item in candidate_reports if item["prompt"] == best_prompt)
    return best_prompt, {
        "optimizer_mode": "heuristic_search",
        "teacher_mode": teacher.mode,
        "fallback_reason": fallback_reason,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "best_score": best_score,
        "best_prompt_version": prompt_version(best_prompt),
        "seed_prompt": SEED_SYSTEM_PROMPT,
        "best_prompt": best_prompt,
        "candidates": candidate_reports,
        "best_prompt_eval": {
            "average_score": best_summary["val_average_score"],
            "accuracy": best_summary["val_accuracy"],
            "examples": best_summary["val_examples"],
        },
    }


def optimize_prompt_for_target(
    *,
    target_index: int,
    target_example: QuestionPoolRecord,
    train_examples: Sequence[QuestionPoolRecord],
    teacher: TeacherClient,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    row_run_dir = args.gepa_run_dir / f"row_{target_index:06d}"
    fallback_reason: str | None = None

    if args.force_heuristic:
        best_prompt, report = run_heuristic_search(
            train_examples=train_examples,
            val_examples=[target_example],
            teacher=teacher,
            fallback_reason="Forced by --force-heuristic.",
        )
    else:
        try:
            best_prompt, report = run_gepa(
                train_examples=train_examples,
                val_examples=[target_example],
                teacher=teacher,
                max_metric_calls=args.max_metric_calls,
                run_dir=row_run_dir,
                fresh_run_dir=args.fresh_run_dir,
            )
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            best_prompt, report = run_heuristic_search(
                train_examples=train_examples,
                val_examples=[target_example],
                teacher=teacher,
                fallback_reason=fallback_reason,
            )

    report.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "target_row_index": target_index,
            "target_subject": target_example.subject,
            "target_question": target_example.question,
            "target_prompt_text": target_example.prompt_text,
            "prompt_bundle_path": str(args.prompt_bundle_path),
        }
    )
    return best_prompt, report


def comparison_payload(
    *,
    prompt_text: str,
    prompt_eval: ExampleEvaluation,
    best_prompt: str,
    best_eval: ExampleEvaluation,
) -> dict[str, Any]:
    return {
        "prompt_changed": prompt_text != best_prompt,
        "prompt_version": prompt_version(prompt_text),
        "best_prompt_version": prompt_version(best_prompt),
        "score_delta": round(best_eval.score.total - prompt_eval.score.total, 6),
        "strict_json_delta": int(best_eval.score.valid_json) - int(prompt_eval.score.valid_json),
        "correct_delta": int(best_eval.score.correct) - int(prompt_eval.score.correct),
        "usable_for_sft_delta": int(best_eval.score.usable_for_sft) - int(prompt_eval.score.usable_for_sft),
    }


def run_per_row_optimization(
    *,
    args: argparse.Namespace,
    train_examples: Sequence[QuestionPoolRecord],
    teacher: TeacherClient,
) -> None:
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.question_pool_path is None:
        raise ValueError("--question-pool-path is required for per-row optimization")

    processed_indices = load_processed_prompt_indices(args.prompt_bundle_path) if args.resume else set()
    resumed_rows = len(processed_indices)

    if not args.resume and args.prompt_bundle_path.exists():
        args.prompt_bundle_path.unlink()

    args.prompt_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    file_mode = "ab" if args.resume and args.prompt_bundle_path.exists() and args.prompt_bundle_path.stat().st_size > 0 else "wb"
    written_rows = 0
    best_scores: list[float] = []
    total_rows_seen = 0
    with args.prompt_bundle_path.open(file_mode) as output_handle:
        for target_index, target_example in enumerate(iter_question_pool(args.question_pool_path, limit=args.target_limit)):
            total_rows_seen += 1
            if target_index in processed_indices:
                continue

            prompt_eval = evaluate_example(
                SEED_SYSTEM_PROMPT,
                target_example,
                teacher,
                emit_oa_logs=False,
            )
            best_prompt, report = optimize_prompt_for_target(
                target_index=target_index,
                target_example=target_example,
                train_examples=train_examples,
                teacher=teacher,
                args=args,
            )
            best_eval = evaluate_example(
                best_prompt,
                target_example,
                teacher,
                emit_oa_logs=False,
            )
            append_jsonl_line(
                output_handle,
                {
                    "bundle_contract": PROMPT_BUNDLE_CONTRACT,
                    "source_row_index": target_index,
                    "question": question_payload(target_example),
                    "prompt": SEED_SYSTEM_PROMPT,
                    "prompt_version": prompt_version(SEED_SYSTEM_PROMPT),
                    "prompt_teacher_response": prompt_eval.teacher_response,
                    "prompt_score": prompt_eval.score.to_dict(),
                    "best_prompt": best_prompt,
                    "best_prompt_version": prompt_version(best_prompt),
                    "best_prompt_teacher_response": best_eval.teacher_response,
                    "best_prompt_score": best_eval.score.to_dict(),
                    "optimizer": {
                        "mode": report.get("optimizer_mode"),
                        "best_score": report.get("best_score"),
                        "generated_at_utc": report.get("generated_at_utc"),
                    },
                    "comparison": comparison_payload(
                        prompt_text=SEED_SYSTEM_PROMPT,
                        prompt_eval=prompt_eval,
                        best_prompt=best_prompt,
                        best_eval=best_eval,
                    ),
                    "teacher": {
                        "mode": teacher.mode,
                        "model": teacher.config.model,
                    },
                },
            )
            processed_indices.add(target_index)
            written_rows += 1

            best_score = best_eval.score.total
            if isinstance(best_score, (int, float)):
                best_scores.append(float(best_score))

            if (resumed_rows + written_rows) % args.progress_interval == 0:
                print(f"progress={resumed_rows + written_rows}", flush=True)

    report_payload = {
        "optimizer_mode": "per_row_prompt_bundle",
        "bundle_contract": PROMPT_BUNDLE_CONTRACT,
        "teacher_mode": teacher.mode,
        "train_examples": len(train_examples),
        "question_pool_path": str(args.question_pool_path),
        "target_limit": args.target_limit,
        "prompt_bundle_path": str(args.prompt_bundle_path),
        "report_path": str(args.report_path),
        "resumed_rows": resumed_rows,
        "written_rows": written_rows,
        "total_rows_seen": total_rows_seen,
        "average_best_score": round(sum(best_scores) / len(best_scores), 6) if best_scores else None,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.report_path, report_payload)
    print("optimizer_mode=per_row_prompt_bundle")
    print(f"teacher_mode={teacher.mode}")
    print(f"written_rows={written_rows}")
    print(f"prompt_bundle_path={args.prompt_bundle_path}")
    print(f"report_path={args.report_path}")


def main() -> None:
    args = parse_args()
    train_examples = load_question_pool(args.train_path, limit=args.train_limit)
    teacher = TeacherClient.from_env()

    if teacher.mode != "api":
        raise RuntimeError(
            "Teacher API is not fully configured. Set TEACHER_MODEL and TEACHER_API_KEY before optimizing."
        )

    run_per_row_optimization(args=args, train_examples=train_examples, teacher=teacher)


if __name__ == "__main__":
    main()
