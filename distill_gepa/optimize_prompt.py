from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .common import SeedExample, load_seed_examples, prompt_version, write_json
from .reflection_lm import make_streaming_litellm_lm, resolve_reflection_runtime
from .score import ScoreResult, score_response
from .teacher_client import TeacherClient

try:
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
except ImportError:
    oa = None
    EngineConfig = GEPAConfig = ReflectionConfig = None


SEED_SYSTEM_PROMPT = """You are a teacher model for dataset distillation.
Answer the user request helpfully."""


HEURISTIC_CANDIDATES = [
    SEED_SYSTEM_PROMPT,
    """You are a teacher model for dataset distillation.
Return a JSON response with an answer to the instruction.""",
    """You are a teacher model for dataset distillation.
Return exactly one JSON object with keys "answer" and "keywords_used".
Make sure the answer includes the required keywords.""",
    """You are a teacher model for dataset distillation.
Return exactly one JSON object with keys "answer" and "keywords_used".
The answer must be concise, non-empty, and include every required keyword.
Do not emit markdown or extra text.""",
    """You are the teacher model for a small distillation pipeline.
Return exactly one valid JSON object and nothing else.
Schema:
{"answer": "<concise 1-2 sentence answer>", "keywords_used": ["<keyword>", "..."]}
Rules:
- The answer must be non-empty.
- Include every required keyword in the answer text.
- Keep the answer concise and easy to parse.
- Do not add markdown fences, explanations, or trailing commentary.""",
]


@dataclass(frozen=True)
class ExampleEvaluation:
    instruction: str
    expected_keywords: list[str]
    teacher_response: str
    score: ScoreResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "expected_keywords": self.expected_keywords,
            "teacher_response": self.teacher_response,
            "score": self.score.to_dict(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a teacher system prompt for smoke-test distillation.")
    parser.add_argument(
        "--val-path",
        type=Path,
        default=Path("data/seeds/val.jsonl"),
        help="JSONL seed set used for prompt selection.",
    )
    parser.add_argument(
        "--best-prompt-path",
        type=Path,
        default=Path("artifacts/best_prompt.txt"),
        help="Where to write the best prompt.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("artifacts/optimization_report.json"),
        help="Where to write the optimization report.",
    )
    parser.add_argument(
        "--max-metric-calls",
        type=int,
        default=12,
        help="GEPA evaluation budget when a reflection model is available.",
    )
    parser.add_argument(
        "--force-heuristic",
        action="store_true",
        help="Skip GEPA and use the built-in deterministic prompt sweep.",
    )
    return parser.parse_args()


def safe_oa_log(message: str) -> None:
    if oa is None or not hasattr(oa, "log"):
        return
    oa.log(message)


def evaluate_example(
    candidate_prompt: str,
    example: SeedExample,
    teacher: TeacherClient,
    emit_oa_logs: bool,
) -> ExampleEvaluation:
    response = teacher.generate(candidate_prompt, example.instruction, example.expected_keywords)
    score = score_response(response.content, example.expected_keywords)

    if emit_oa_logs:
        if not score.valid_json:
            safe_oa_log(f"invalid_json :: instruction={example.instruction}")
        if score.too_long:
            safe_oa_log(
                f"too_long :: instruction={example.instruction} :: chars={score.char_count}"
            )
        if score.missing_keywords:
            safe_oa_log(
                f"missing_keywords :: instruction={example.instruction} :: missing={', '.join(score.missing_keywords)}"
            )
        safe_oa_log(
            f"example_output :: instruction={example.instruction} :: output={response.content[:240]}"
        )

    return ExampleEvaluation(
        instruction=example.instruction,
        expected_keywords=list(example.expected_keywords),
        teacher_response=response.content,
        score=score,
    )


def gepa_side_info(example_evaluation: ExampleEvaluation) -> dict[str, Any]:
    score = example_evaluation.score
    return {
        "scores": {
            "valid_json": 1.0 if score.valid_json else 0.0,
            "non_empty_answer": 1.0 if score.non_empty_answer else 0.0,
            "keyword_coverage": score.keyword_coverage,
            "length_component": score.length_component,
        },
        "instruction": example_evaluation.instruction,
        "expected_keywords": example_evaluation.expected_keywords,
        "teacher_response": example_evaluation.teacher_response,
        "answer_text": score.answer_text,
        "invalid_json": not score.valid_json,
        "too_long": score.too_long,
        "missing_keywords": score.missing_keywords,
    }


def evaluate_prompt(
    candidate_prompt: str,
    examples: Sequence[SeedExample],
    teacher: TeacherClient,
    emit_oa_logs: bool,
) -> dict[str, Any]:
    per_example: list[ExampleEvaluation] = []
    total_score = 0.0

    for example in examples:
        evaluation = evaluate_example(
            candidate_prompt=candidate_prompt,
            example=example,
            teacher=teacher,
            emit_oa_logs=emit_oa_logs,
        )
        per_example.append(evaluation)
        total_score += evaluation.score.total

    average_score = round(total_score / len(examples), 6)
    return {
        "average_score": average_score,
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


def run_gepa(
    examples: Sequence[SeedExample],
    teacher: TeacherClient,
    max_metric_calls: int,
) -> tuple[str, dict[str, Any]]:
    if oa is None or GEPAConfig is None or EngineConfig is None or ReflectionConfig is None:
        raise RuntimeError("gepa is not installed. Run `bash scripts/bootstrap.sh` first.")

    reflection_model_name, reflection_api_key, reflection_api_base = resolve_reflection_runtime(teacher)
    if not reflection_model_name:
        raise RuntimeError(
            "No reflection model available for GEPA. Set GEPA_MODEL / GEPA_REFLECTION_MODEL or TEACHER_* variables."
        )

    run_dir = Path("artifacts/gepa_run")
    run_dir.mkdir(parents=True, exist_ok=True)
    reflection_lm = make_streaming_litellm_lm(
        model_name=reflection_model_name,
        api_key=reflection_api_key,
        api_base=reflection_api_base,
        timeout_seconds=teacher.config.timeout_seconds,
        num_retries=teacher.config.num_retries,
    )

    def evaluator(candidate_prompt: str, example: SeedExample) -> tuple[float, dict[str, Any]]:
        evaluation = evaluate_example(
            candidate_prompt=candidate_prompt,
            example=example,
            teacher=teacher,
            emit_oa_logs=True,
        )
        safe_oa_log(
            "candidate_example_summary :: "
            f"prompt_version={prompt_version(candidate_prompt)} :: "
            f"score={evaluation.score.total} :: "
            f"instruction={example.instruction}"
        )
        return evaluation.score.total, gepa_side_info(evaluation)

    result = oa.optimize_anything(
        seed_candidate=SEED_SYSTEM_PROMPT,
        evaluator=evaluator,
        dataset=list(examples),
        valset=list(examples),
        objective=(
            "Optimize a teacher system prompt for small distillation examples."
            " The prompt must produce concise JSON responses that include required keywords."
        ),
        background=(
            "The evaluator rewards: valid JSON output, a non-empty answer, inclusion of required keywords,"
            " and short responses. Avoid markdown, explanations, and trailing commentary."
        ),
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(run_dir),
                max_metric_calls=max_metric_calls,
                cache_evaluation=True,
                use_cloudpickle=False,
            ),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
        ),
    )

    best_prompt = normalize_candidate(getattr(result, "best_candidate", result))
    best_summary = evaluate_prompt(best_prompt, examples, teacher, emit_oa_logs=False)
    best_score = result.val_aggregate_scores[result.best_idx]

    report = {
        "optimizer_mode": "gepa",
        "teacher_mode": teacher.mode,
        "reflection_model": reflection_model_name,
        "reflection_api_base": reflection_api_base,
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
        "gepa_val_aggregate_scores": result.val_aggregate_scores,
    }
    return best_prompt, report


def run_heuristic_search(
    examples: Sequence[SeedExample],
    teacher: TeacherClient,
    fallback_reason: str | None,
) -> tuple[str, dict[str, Any]]:
    candidate_reports: list[dict[str, Any]] = []
    best_prompt = HEURISTIC_CANDIDATES[0]
    best_score = -1.0

    for index, candidate_prompt in enumerate(HEURISTIC_CANDIDATES, start=1):
        summary = evaluate_prompt(candidate_prompt, examples, teacher, emit_oa_logs=False)
        report = {
            "candidate_index": index,
            "prompt": candidate_prompt,
            "prompt_version": prompt_version(candidate_prompt),
            "average_score": summary["average_score"],
            "examples": summary["examples"],
        }
        candidate_reports.append(report)
        if summary["average_score"] > best_score:
            best_prompt = candidate_prompt
            best_score = summary["average_score"]

    best_summary = next(
        item for item in candidate_reports if item["prompt"] == best_prompt
    )
    report = {
        "optimizer_mode": "heuristic_search",
        "teacher_mode": teacher.mode,
        "fallback_reason": fallback_reason,
        "best_score": best_score,
        "best_prompt_version": prompt_version(best_prompt),
        "seed_prompt": SEED_SYSTEM_PROMPT,
        "best_prompt": best_prompt,
        "candidates": candidate_reports,
        "best_prompt_eval": {
            "average_score": best_summary["average_score"],
            "examples": best_summary["examples"],
        },
    }
    return best_prompt, report


def main() -> None:
    args = parse_args()
    examples = load_seed_examples(args.val_path)
    teacher = TeacherClient.from_env()

    fallback_reason: str | None = None
    if args.force_heuristic:
        best_prompt, report = run_heuristic_search(
            examples=examples,
            teacher=teacher,
            fallback_reason="Forced by --force-heuristic.",
        )
    else:
        try:
            best_prompt, report = run_gepa(
                examples=examples,
                teacher=teacher,
                max_metric_calls=args.max_metric_calls,
            )
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            best_prompt, report = run_heuristic_search(
                examples=examples,
                teacher=teacher,
                fallback_reason=fallback_reason,
            )

    args.best_prompt_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_prompt_path.write_text(best_prompt.strip() + "\n", encoding="utf-8")

    report.update(
        {
            "evaluated_examples": len(examples),
            "best_prompt_path": str(args.best_prompt_path),
            "report_path": str(args.report_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
