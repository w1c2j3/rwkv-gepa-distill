from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .common import prompt_version, write_json
from .mmlu_score import MCQScoreResult, score_mcq_response
from .question_pools import QuestionPoolRecord, load_question_pool
from .reflection_lm import make_streaming_litellm_lm, resolve_reflection_runtime
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
- Output ONLY valid JSON. No markdown, headings, bullets, bold text, or extra commentary."""


HEURISTIC_CANDIDATES = [
    SEED_SYSTEM_PROMPT,
    """You are a precise multiple-choice world-knowledge teacher.
Return exactly one JSON object with keys "answer_letter", "answer_index", "answer_text", and "reasoning".
Use the Subject context when helpful. Select the best option and keep reasoning concise.
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
    parser = argparse.ArgumentParser(description="Optimize an MMLU teacher system prompt with GEPA.")
    parser.add_argument("--train-path", type=Path, default=Path("data/question_pools/mmlu_dev.jsonl"))
    parser.add_argument("--val-path", type=Path, default=Path("data/question_pools/mmlu_validation.jsonl"))
    parser.add_argument("--train-limit", type=int, default=10)
    parser.add_argument("--val-limit", type=int, default=10)
    parser.add_argument("--max-metric-calls", type=int, default=12)
    parser.add_argument("--gepa-run-dir", type=Path, default=Path("artifacts/mmlu_gepa_run"))
    parser.add_argument("--fresh-run-dir", action="store_true")
    parser.add_argument("--best-prompt-path", type=Path, default=Path("artifacts/mmlu_best_prompt.txt"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/mmlu_prompt_optimization_report.json"),
    )
    parser.add_argument("--force-heuristic", action="store_true")
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
    score = score_mcq_response(response.content, example.choices, example.answer_index)

    if emit_oa_logs:
        if not score.valid_json:
            safe_oa_log(f"invalid_json :: subject={example.subject} :: question={example.question[:120]}")
        if not score.answer_present:
            safe_oa_log(f"missing_answer :: subject={example.subject} :: question={example.question[:120]}")
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
    accuracy = round(
        sum(1 for item in per_example if item.score.correct) / len(per_example),
        6,
    )
    return {
        "average_score": average_score,
        "accuracy": accuracy,
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

    reflection_model_name, reflection_api_key, reflection_api_base = resolve_reflection_runtime(teacher)
    if not reflection_model_name:
        raise RuntimeError(
            "No reflection model available for GEPA. Set GEPA_MODEL / GEPA_REFLECTION_MODEL or TEACHER_* variables."
        )

    prepare_gepa_run_dir(run_dir, fresh_run_dir=fresh_run_dir)
    reflection_lm = make_streaming_litellm_lm(
        model_name=reflection_model_name,
        api_key=reflection_api_key,
        api_base=reflection_api_base,
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
                "answer_present": 1.0 if evaluation.score.answer_present else 0.0,
                "correct": 1.0 if evaluation.score.correct else 0.0,
            },
            "prompt_text": example.prompt_text,
            "teacher_response": evaluation.teacher_response,
            "parsed": evaluation.score.parsed.to_dict(),
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
                " The teacher should return concise JSON with the correct answer."
            ),
            background=(
                "The evaluator rewards valid JSON, a parseable answer, and selecting the correct choice."
                " Ensure answer_letter, answer_index, and answer_text agree."
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
                " The teacher should return concise JSON with the correct answer."
            ),
            background=(
                "The evaluator rewards valid JSON, a parseable answer, and selecting the correct choice."
                " Ensure answer_letter, answer_index, and answer_text agree."
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


def main() -> None:
    args = parse_args()
    train_examples = load_question_pool(args.train_path, limit=args.train_limit)
    val_examples = load_question_pool(args.val_path, limit=args.val_limit)
    teacher = TeacherClient.from_env()

    if teacher.mode != "api":
        raise RuntimeError(
            "Teacher API is not fully configured. Set TEACHER_MODEL and TEACHER_API_KEY before optimizing."
        )

    fallback_reason: str | None = None
    if args.force_heuristic:
        best_prompt, report = run_heuristic_search(
            train_examples=train_examples,
            val_examples=val_examples,
            teacher=teacher,
            fallback_reason="Forced by --force-heuristic.",
        )
    else:
        try:
            best_prompt, report = run_gepa(
                train_examples=train_examples,
                val_examples=val_examples,
                teacher=teacher,
                max_metric_calls=args.max_metric_calls,
                run_dir=args.gepa_run_dir,
                fresh_run_dir=args.fresh_run_dir,
            )
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            best_prompt, report = run_heuristic_search(
                train_examples=train_examples,
                val_examples=val_examples,
                teacher=teacher,
                fallback_reason=fallback_reason,
            )

    args.best_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_prompt_path.write_text(best_prompt.strip() + "\n", encoding="utf-8")
    report.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "best_prompt_path": str(args.best_prompt_path),
            "report_path": str(args.report_path),
        }
    )
    write_json(args.report_path, report)

    print(f"optimizer_mode={report['optimizer_mode']}")
    print(f"teacher_mode={teacher.mode}")
    print(f"best_score={report['best_score']}")
    print(f"best_prompt_path={args.best_prompt_path}")
    print(f"report_path={args.report_path}")


if __name__ == "__main__":
    main()
