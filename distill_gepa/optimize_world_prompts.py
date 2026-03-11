from __future__ import annotations

import argparse
import asyncio
import re
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import orjson

from .async_request_runner import AsyncRequestRunner
from .common import build_shuffle_key, prompt_version, write_json
from .model_registry import ModelEndpointConfig, load_pipeline_model_config
from .reflection_lm import make_openai_lm
from .world_prompts import WORLD_GEPA_USER_SEED_PROMPT, WORLD_SEED_SYSTEM_PROMPT
from .world_schema import BenchmarkQuestion
from .world_scoring import WorldScoreResult, repair_world_response, score_world_response

try:
    import gepa.optimize_anything as oa
    from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig
except ImportError:
    oa = None
    EngineConfig = GEPAConfig = ReflectionConfig = None


GEPA_RESULT_CONTRACT = "gepa_result_v1"
GEPA_FAILURE_CONTRACT = "gepa_failure_v1"
GEPA_CHECKPOINT_CONTRACT = "gepa_checkpoint_v1"
GEPA_CHECKPOINT_FILENAME = "checkpoint.json"


@dataclass(frozen=True)
class SampleOutcome:
    sample_index: int
    instruction: str
    response_text: str
    score: WorldScoreResult
    attempt_count: int
    cache_hit: bool


@dataclass(frozen=True)
class QuestionPromptEvaluation:
    question: BenchmarkQuestion
    stable_correct: bool
    correct_rate: float
    valid_json_rate: float
    think_tag_rate: float
    average_score: float
    usable_for_distill_rate: float
    preferred_sample_index: int
    preferred_instruction: str
    preferred_response_text: str
    samples: list[SampleOutcome]

    @property
    def composite_score(self) -> float:
        return round(
            0.55 * float(self.stable_correct)
            + 0.25 * self.correct_rate
            + 0.10 * self.think_tag_rate
            + 0.10 * self.valid_json_rate,
            6,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question.to_dict(),
            "stable_correct": self.stable_correct,
            "correct_rate": self.correct_rate,
            "valid_json_rate": self.valid_json_rate,
            "think_tag_rate": self.think_tag_rate,
            "average_score": self.average_score,
            "usable_for_distill_rate": self.usable_for_distill_rate,
            "preferred_sample_index": self.preferred_sample_index,
            "preferred_instruction": self.preferred_instruction,
            "preferred_response_text": self.preferred_response_text,
            "samples": [
                {
                    "sample_index": item.sample_index,
                    "instruction": item.instruction,
                    "response_text": item.response_text,
                    "attempt_count": item.attempt_count,
                    "cache_hit": item.cache_hit,
                    "score": item.score.to_dict(),
                }
                for item in self.samples
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize grouped world prompts with GEPA.")
    parser.add_argument("--config-path", type=Path, default=Path("config/world_pipeline.yaml"))
    parser.add_argument("--decision-path", type=Path, default=Path("data/world_knowledge/question_decisions.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("data/world_knowledge/gepa_results.jsonl"))
    parser.add_argument("--failure-path", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=Path("data/world_knowledge/cache/gepa_run"))
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-group-workers", type=int, default=2)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-metric-calls", type=int, default=60)
    parser.add_argument("--metric-samples", type=int, default=2)
    parser.add_argument("--materialization-samples", type=int, default=8)
    parser.add_argument("--model-attempts", type=int, default=2)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--fresh-run-dir", action="store_true", default=False)
    return parser.parse_args()


def append_jsonl_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(orjson.dumps(payload))
    handle.write(b"\n")
    handle.flush()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "group"


def load_needs_groups(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            classification = payload.get("classification")
            question_payload = payload.get("question")
            per_model = payload.get("per_model")
            if (
                classification != "needs_optimization"
                or not isinstance(question_payload, dict)
                or not isinstance(per_model, dict)
            ):
                continue
            for model_name, model_summary in per_model.items():
                if not isinstance(model_summary, dict):
                    continue
                if model_summary.get("stable_correct") is True:
                    continue
                usable_for_gepa = model_summary.get("usable_for_gepa")
                if usable_for_gepa is False:
                    continue
                if usable_for_gepa is None and int(model_summary.get("correct_count", 0)) <= 0:
                    continue
                group_key = (model_name, str(question_payload.get("domain") or "unknown"))
                groups.setdefault(group_key, []).append(question_payload)
    return groups


def load_processed_group_ids(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    processed: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = orjson.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            group_id = payload.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"{path}:{line_number} missing group_id")
            processed.add(group_id)
    return processed


def checkpoint_path_for_run_dir(run_dir: Path) -> Path:
    return run_dir / GEPA_CHECKPOINT_FILENAME


def load_group_checkpoint(path: Path, *, group_id: str) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object checkpoint")
    contract = payload.get("contract")
    checkpoint_group_id = payload.get("group_id")
    if contract != GEPA_CHECKPOINT_CONTRACT:
        raise ValueError(f"{path} has unsupported contract: {contract!r}")
    if checkpoint_group_id != group_id:
        raise ValueError(f"{path} belongs to {checkpoint_group_id!r}, expected {group_id!r}")
    return payload


def write_group_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def build_group_checkpoint(
    *,
    group_id: str,
    target_model_name: str,
    domain: str,
    questions: Sequence[BenchmarkQuestion],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "contract": GEPA_CHECKPOINT_CONTRACT,
        "group_id": group_id,
        "target_model": target_model_name,
        "domain": domain,
        "question_count": len(questions),
        "question_ids": [question.question_id for question in questions],
        "run_dir": str(run_dir),
        "stage": "initialized",
        "system_prompt": WORLD_SEED_SYSTEM_PROMPT,
        "system_prompt_version": prompt_version(WORLD_SEED_SYSTEM_PROMPT),
        "seed_prompt": WORLD_GEPA_USER_SEED_PROMPT,
        "seed_prompt_version": prompt_version(WORLD_GEPA_USER_SEED_PROMPT),
        "best_prompt": None,
        "best_prompt_version": None,
        "gepa_report": None,
        "baseline_metrics": None,
        "baseline_evaluations": [],
        "optimized_metrics": None,
        "optimized_evaluations": [],
        "optimized_progress": {"completed": 0, "total": len(questions)},
        "question_deltas": [],
        "result_record": None,
        "last_error": None,
    }


def split_train_val(questions: Sequence[BenchmarkQuestion]) -> tuple[list[BenchmarkQuestion], list[BenchmarkQuestion]]:
    if len(questions) <= 1:
        question_list = list(questions)
        return question_list, question_list
    split_index = max(1, int(round(len(questions) * 0.8)))
    if split_index >= len(questions):
        split_index = len(questions) - 1
    train_questions = list(questions[:split_index])
    val_questions = list(questions[split_index:])
    if not val_questions:
        val_questions = list(train_questions[-1:])
    return train_questions, val_questions


def normalize_candidate(candidate: Any) -> str:
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if isinstance(candidate, dict):
        for key in ("system_prompt", "prompt", "text", "candidate"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise TypeError(f"Unsupported GEPA candidate shape: {type(candidate)!r}")


def render_candidate_user_message(candidate_prompt: str, base_question_prompt: str) -> str:
    prefix = candidate_prompt.strip()
    question_prompt = base_question_prompt.strip()
    if not prefix:
        return question_prompt
    return f"{prefix}\n\n{question_prompt}"


def _rehydrate_questions(question_payloads: Sequence[dict[str, Any]]) -> list[BenchmarkQuestion]:
    return [
        BenchmarkQuestion.from_dict(question_payload, Path("<gepa-group>"), line_number)
        for line_number, question_payload in enumerate(question_payloads, start=1)
    ]


def evaluation_question_id(payload: dict[str, Any]) -> str:
    question = payload.get("question")
    question_id = question.get("question_id") if isinstance(question, dict) else None
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("Evaluation payload is missing question.question_id")
    return question_id


def order_evaluations_for_questions(
    questions: Sequence[BenchmarkQuestion],
    evaluations_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for question in questions:
        payload = evaluations_by_id.get(question.question_id)
        if payload is not None:
            ordered.append(payload)
    return ordered


def has_full_evaluation_coverage(
    questions: Sequence[BenchmarkQuestion],
    evaluations: Sequence[dict[str, Any]],
) -> bool:
    if len(evaluations) != len(questions):
        return False
    expected = {question.question_id for question in questions}
    actual = {evaluation_question_id(payload) for payload in evaluations}
    return actual == expected


async def evaluate_prompt_for_question_async(
    *,
    prompt_text: str,
    question: BenchmarkQuestion,
    endpoint: ModelEndpointConfig,
    request_runner: AsyncRequestRunner,
    repeats: int,
    model_attempts: int,
    shuffle_namespace: str,
) -> QuestionPromptEvaluation:
    async def evaluate_sample(sample_index: int) -> SampleOutcome:
        prompted = question.prompted_variant(
            sample_index,
            shuffle_key=build_shuffle_key(shuffle_namespace, sample_index),
        )
        effective_user_message = render_candidate_user_message(prompt_text, prompted.user_message)
        scoring_question = question
        if prompted.shuffled_choices:
            scoring_question = BenchmarkQuestion(
                benchmark_name=question.benchmark_name,
                split=question.split,
                domain=question.domain,
                question_id=question.question_id,
                question_type=question.question_type,
                question_text=question.question_text,
                choices=prompted.shuffled_choices,
                gold_answer=question.gold_answer,
                gold_answer_index=prompted.shuffled_answer_index,
                gold_aliases=question.gold_aliases,
                metadata=question.metadata,
            )

        def ensure_usable_response(raw_response_text: str) -> None:
            initial_score = score_world_response(raw_response_text, scoring_question)
            if initial_score.valid_json and initial_score.think_tags_present:
                return
            repaired_response_text, _ = repair_world_response(raw_response_text, scoring_question)
            if repaired_response_text is None:
                raise ValueError("GEPA evaluation response was neither distill-format-valid nor repairable")

        generation = await request_runner.generate(
            endpoint=endpoint,
            system_prompt=WORLD_SEED_SYSTEM_PROMPT,
            user_message=effective_user_message,
            attempts=model_attempts,
            validator=ensure_usable_response,
            use_cache=True,
        )
        response_text = generation.content
        score = score_world_response(response_text, scoring_question)
        if not score.valid_json or not score.think_tags_present:
            repaired_response_text, _ = repair_world_response(response_text, scoring_question)
            if repaired_response_text is not None:
                response_text = repaired_response_text
                score = score_world_response(response_text, scoring_question)

        return SampleOutcome(
            sample_index=sample_index,
            instruction=effective_user_message,
            response_text=response_text,
            score=score,
            attempt_count=generation.attempt_count,
            cache_hit=generation.cache_hit,
        )

    samples = list(await asyncio.gather(*[evaluate_sample(sample_index) for sample_index in range(repeats)]))
    correct_flags = [item.score.correct and item.score.valid_json for item in samples]
    valid_json_flags = [item.score.valid_json for item in samples]
    think_tag_flags = [item.score.think_tags_present for item in samples]
    usable_flags = [item.score.usable_for_distill for item in samples]
    preferred_sample = next(
        (item for item in samples if item.score.correct and item.score.valid_json and item.score.think_tags_present),
        None,
    )
    if preferred_sample is None:
        preferred_sample = next((item for item in samples if item.score.correct and item.score.valid_json), samples[0])
    return QuestionPromptEvaluation(
        question=question,
        stable_correct=all(correct_flags),
        correct_rate=round(sum(correct_flags) / len(samples), 6),
        valid_json_rate=round(sum(valid_json_flags) / len(samples), 6),
        think_tag_rate=round(sum(think_tag_flags) / len(samples), 6),
        average_score=round(mean(item.score.total for item in samples), 6),
        usable_for_distill_rate=round(sum(usable_flags) / len(samples), 6),
        preferred_sample_index=preferred_sample.sample_index,
        preferred_instruction=preferred_sample.instruction,
        preferred_response_text=preferred_sample.response_text,
        samples=samples,
    )


def aggregate_group_metrics(evaluations: Sequence[QuestionPromptEvaluation]) -> dict[str, Any]:
    if not evaluations:
        return {
            "question_count": 0,
            "stable_correct_rate": 0.0,
            "avg_correct_rate": 0.0,
            "valid_json_rate": 0.0,
            "think_tag_rate": 0.0,
            "average_score": 0.0,
            "composite_score": 0.0,
        }
    stable_correct_rate = round(mean(float(item.stable_correct) for item in evaluations), 6)
    avg_correct_rate = round(mean(item.correct_rate for item in evaluations), 6)
    valid_json_rate = round(mean(item.valid_json_rate for item in evaluations), 6)
    think_tag_rate = round(mean(item.think_tag_rate for item in evaluations), 6)
    average_score = round(mean(item.average_score for item in evaluations), 6)
    composite_score = round(
        0.55 * stable_correct_rate
        + 0.25 * avg_correct_rate
        + 0.10 * think_tag_rate
        + 0.10 * valid_json_rate,
        6,
    )
    return {
        "question_count": len(evaluations),
        "stable_correct_rate": stable_correct_rate,
        "avg_correct_rate": avg_correct_rate,
        "valid_json_rate": valid_json_rate,
        "think_tag_rate": think_tag_rate,
        "average_score": average_score,
        "composite_score": composite_score,
    }


def aggregate_group_metrics_from_payloads(evaluations: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not evaluations:
        return {
            "question_count": 0,
            "stable_correct_rate": 0.0,
            "avg_correct_rate": 0.0,
            "valid_json_rate": 0.0,
            "think_tag_rate": 0.0,
            "average_score": 0.0,
            "composite_score": 0.0,
        }
    stable_correct_rate = round(mean(float(payload["stable_correct"]) for payload in evaluations), 6)
    avg_correct_rate = round(mean(float(payload["correct_rate"]) for payload in evaluations), 6)
    valid_json_rate = round(mean(float(payload["valid_json_rate"]) for payload in evaluations), 6)
    think_tag_rate = round(mean(float(payload["think_tag_rate"]) for payload in evaluations), 6)
    average_score = round(mean(float(payload["average_score"]) for payload in evaluations), 6)
    composite_score = round(
        0.55 * stable_correct_rate
        + 0.25 * avg_correct_rate
        + 0.10 * think_tag_rate
        + 0.10 * valid_json_rate,
        6,
    )
    return {
        "question_count": len(evaluations),
        "stable_correct_rate": stable_correct_rate,
        "avg_correct_rate": avg_correct_rate,
        "valid_json_rate": valid_json_rate,
        "think_tag_rate": think_tag_rate,
        "average_score": average_score,
        "composite_score": composite_score,
    }


async def evaluate_prompt_on_group_async(
    *,
    prompt_text: str,
    questions: Sequence[BenchmarkQuestion],
    endpoint: ModelEndpointConfig,
    request_runner: AsyncRequestRunner,
    repeats: int,
    model_attempts: int,
    shuffle_namespace: str,
    completed_evaluations: Sequence[dict[str, Any]] | None = None,
    per_question_callback: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluations_by_id: dict[str, dict[str, Any]] = {}
    if completed_evaluations is not None:
        evaluations_by_id = {
            evaluation_question_id(payload): payload
            for payload in completed_evaluations
            if isinstance(payload, dict)
        }

    pending_questions = [question for question in questions if question.question_id not in evaluations_by_id]
    tasks = [
        asyncio.create_task(
            evaluate_prompt_for_question_async(
                prompt_text=prompt_text,
                question=question,
                endpoint=endpoint,
                request_runner=request_runner,
                repeats=repeats,
                model_attempts=model_attempts,
                shuffle_namespace=shuffle_namespace,
            )
        )
        for question in pending_questions
    ]

    for task in asyncio.as_completed(tasks):
        evaluation = await task
        payload = evaluation.to_dict()
        evaluations_by_id[evaluation.question.question_id] = payload
        if per_question_callback is not None:
            per_question_callback(payload)

    ordered_evaluations = order_evaluations_for_questions(questions, evaluations_by_id)
    return aggregate_group_metrics_from_payloads(ordered_evaluations), ordered_evaluations


def run_gepa_for_group(
    *,
    group_id: str,
    target_model_name: str,
    train_questions: Sequence[BenchmarkQuestion],
    val_questions: Sequence[BenchmarkQuestion],
    target_endpoint: ModelEndpointConfig,
    reflection_config: Any,
    request_runner: AsyncRequestRunner,
    event_loop_runner: asyncio.Runner,
    run_dir: Path,
    max_metric_calls: int,
    metric_samples: int,
    model_attempts: int,
    fresh_run_dir: bool,
) -> tuple[str, dict[str, Any]]:
    if oa is None or GEPAConfig is None or EngineConfig is None or ReflectionConfig is None:
        raise RuntimeError("gepa is not installed. Run `bash scripts/bootstrap.sh` first.")

    if fresh_run_dir and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if reflection_config.force_mock:
        reflection_lm = lambda prompt: WORLD_GEPA_USER_SEED_PROMPT
    else:
        reflection_lm = make_openai_lm(
            model_name=reflection_config.model,
            api_key=reflection_config.api_key,
            api_base=reflection_config.api_base,
            api_protocol=reflection_config.api_protocol,
            timeout_seconds=reflection_config.timeout_seconds,
            num_retries=reflection_config.num_retries,
        )

    def evaluator(candidate_prompt: str, example: BenchmarkQuestion) -> tuple[float, dict[str, Any]]:
        evaluation = event_loop_runner.run(
            evaluate_prompt_for_question_async(
                prompt_text=candidate_prompt,
                question=example,
                endpoint=target_endpoint,
                request_runner=request_runner,
                repeats=metric_samples,
                model_attempts=model_attempts,
                shuffle_namespace=build_shuffle_key("gepa_search", target_model_name, group_id),
            )
        )
        return evaluation.composite_score, evaluation.to_dict()

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            max_metric_calls=max_metric_calls,
            cache_evaluation=True,
            use_cloudpickle=False,
        ),
        reflection=ReflectionConfig(reflection_lm=reflection_lm),
    )
    result = oa.optimize_anything(
        seed_candidate=WORLD_GEPA_USER_SEED_PROMPT,
        evaluator=evaluator,
        dataset=list(train_questions),
        valset=list(val_questions),
        objective=(
            "Optimize a world-knowledge user-side prompt template so the target model answers a same-domain "
            "question group with stable correctness across repeated shuffled samples."
        ),
        background=(
            "Reward stable repeated correctness under shuffled options first, then average correctness, then "
            "presence of <think>...</think>, then strict JSON compliance. Keep reasoning concise and valid for "
            "both multiple_choice and open_qa."
        ),
        config=config,
    )
    best_prompt = normalize_candidate(result.best_candidate)
    report = {
        "group_id": group_id,
        "target_model": target_model_name,
        "best_idx": result.best_idx,
        "num_candidates": result.num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "num_full_val_evals": result.num_full_val_evals,
        "best_score": result.val_aggregate_scores[result.best_idx],
        "run_dir": str(run_dir),
    }
    return best_prompt, report


def build_question_delta(
    *,
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    improved = (not bool(baseline["stable_correct"])) and bool(optimized["stable_correct"])
    improved_to_distillable = improved and float(optimized["usable_for_distill_rate"]) >= 1.0 and float(
        optimized["think_tag_rate"]
    ) >= 1.0
    question = optimized["question"]
    return {
        "question_id": question["question_id"],
        "question": question,
        "improved_to_stable_correct": improved,
        "improved_to_distillable": improved_to_distillable,
        "baseline_stable_correct": baseline["stable_correct"],
        "optimized_stable_correct": optimized["stable_correct"],
        "baseline_correct_rate": baseline["correct_rate"],
        "optimized_correct_rate": optimized["correct_rate"],
        "baseline_valid_json_rate": baseline["valid_json_rate"],
        "optimized_valid_json_rate": optimized["valid_json_rate"],
        "baseline_think_tag_rate": baseline["think_tag_rate"],
        "optimized_think_tag_rate": optimized["think_tag_rate"],
        "baseline_usable_for_distill_rate": baseline["usable_for_distill_rate"],
        "optimized_usable_for_distill_rate": optimized["usable_for_distill_rate"],
        "baseline_preferred_response": baseline["preferred_response_text"],
        "optimized_preferred_response": optimized["preferred_response_text"],
        "optimized_preferred_instruction": optimized["preferred_instruction"],
        "preferred_sample_index": optimized["preferred_sample_index"],
    }


def build_question_deltas(
    *,
    questions: Sequence[BenchmarkQuestion],
    baseline_evaluations: Sequence[dict[str, Any]],
    optimized_evaluations: Sequence[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    baseline_by_id = {evaluation_question_id(item): item for item in baseline_evaluations}
    optimized_by_id = {evaluation_question_id(item): item for item in optimized_evaluations}
    question_deltas: list[dict[str, Any]] = []
    improved_question_count = 0
    for question in questions:
        optimized = optimized_by_id.get(question.question_id)
        if optimized is None:
            continue
        baseline = baseline_by_id[question.question_id]
        delta = build_question_delta(baseline=baseline, optimized=optimized)
        improved_question_count += int(delta["improved_to_stable_correct"])
        question_deltas.append(delta)
    return improved_question_count, question_deltas


def build_result_record(
    *,
    group_id: str,
    target_model_name: str,
    domain: str,
    questions: Sequence[BenchmarkQuestion],
    best_prompt: str,
    baseline_metrics: dict[str, Any],
    optimized_metrics: dict[str, Any],
    baseline_evaluations: Sequence[dict[str, Any]],
    optimized_evaluations: Sequence[dict[str, Any]],
    gepa_report: dict[str, Any],
) -> dict[str, Any]:
    improved_question_count, question_deltas = build_question_deltas(
        questions=questions,
        baseline_evaluations=baseline_evaluations,
        optimized_evaluations=optimized_evaluations,
    )
    return {
        "contract": GEPA_RESULT_CONTRACT,
        "group_id": group_id,
        "target_model": target_model_name,
        "domain": domain,
        "question_count": len(questions),
        "improved_question_count": improved_question_count,
        "prompt_mode": "user_prefix",
        "system_prompt": WORLD_SEED_SYSTEM_PROMPT,
        "system_prompt_version": prompt_version(WORLD_SEED_SYSTEM_PROMPT),
        "seed_prompt": WORLD_GEPA_USER_SEED_PROMPT,
        "seed_prompt_version": prompt_version(WORLD_GEPA_USER_SEED_PROMPT),
        "best_prompt": best_prompt,
        "best_prompt_version": prompt_version(best_prompt),
        "baseline_metrics": baseline_metrics,
        "optimized_metrics": optimized_metrics,
        "question_deltas": question_deltas,
        "gepa_report": gepa_report,
    }


def build_group_job_result(result_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "group_id": result_record["group_id"],
        "target_model": result_record["target_model"],
        "domain": result_record["domain"],
        "question_count": result_record["question_count"],
        "improved_question_count": result_record["improved_question_count"],
        "result_record": result_record,
        "baseline_metrics": result_record["baseline_metrics"],
        "optimized_metrics": result_record["optimized_metrics"],
    }


def build_failure_record(
    *,
    group_id: str,
    target_model_name: str,
    domain: str,
    question_payloads: Sequence[dict[str, Any]],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "contract": GEPA_FAILURE_CONTRACT,
        "group_id": group_id,
        "target_model": target_model_name,
        "domain": domain,
        "question_count": len(question_payloads),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def optimize_group_job(
    *,
    config_path: str,
    cache_path: str,
    group_id: str,
    target_model_name: str,
    domain: str,
    question_payloads: list[dict[str, Any]],
    run_dir: str,
    max_metric_calls: int,
    metric_samples: int,
    materialization_samples: int,
    model_attempts: int,
    max_concurrency: int,
    resume: bool,
    fresh_run_dir: bool,
) -> dict[str, Any]:
    config = load_pipeline_model_config(Path(config_path))
    questions = _rehydrate_questions(question_payloads)
    run_dir_path = Path(run_dir)
    checkpoint_path = checkpoint_path_for_run_dir(run_dir_path)
    checkpoint = load_group_checkpoint(checkpoint_path, group_id=group_id) if resume else None
    if checkpoint is None:
        checkpoint = build_group_checkpoint(
            group_id=group_id,
            target_model_name=target_model_name,
            domain=domain,
            questions=questions,
            run_dir=run_dir_path,
        )
        write_group_checkpoint(checkpoint_path, checkpoint)

    completed_result = checkpoint.get("result_record")
    if checkpoint.get("stage") == "complete" and isinstance(completed_result, dict):
        return build_group_job_result(completed_result)

    train_questions, val_questions = split_train_val(questions)
    target_endpoint = config.base_model(target_model_name)
    request_runner = AsyncRequestRunner(
        cache_path=Path(cache_path),
        default_max_concurrency=max_concurrency,
    )

    materialize_shuffle_namespace = build_shuffle_key("gepa_materialize", target_model_name, group_id)

    try:
        with asyncio.Runner() as event_loop_runner:
            try:
                best_prompt = checkpoint.get("best_prompt")
                gepa_report = checkpoint.get("gepa_report")
                if not isinstance(best_prompt, str) or not best_prompt.strip() or not isinstance(gepa_report, dict):
                    best_prompt, gepa_report = run_gepa_for_group(
                        group_id=group_id,
                        target_model_name=target_model_name,
                        train_questions=train_questions,
                        val_questions=val_questions,
                        target_endpoint=target_endpoint,
                        reflection_config=config.gepa_reflection_model,
                        request_runner=request_runner,
                        event_loop_runner=event_loop_runner,
                        run_dir=run_dir_path,
                        max_metric_calls=max_metric_calls,
                        metric_samples=metric_samples,
                        model_attempts=model_attempts,
                        fresh_run_dir=fresh_run_dir,
                    )
                    checkpoint["stage"] = "best_prompt_complete"
                    checkpoint["best_prompt"] = best_prompt
                    checkpoint["best_prompt_version"] = prompt_version(best_prompt)
                    checkpoint["gepa_report"] = gepa_report
                    checkpoint["last_error"] = None
                    write_group_checkpoint(checkpoint_path, checkpoint)

                baseline_metrics = checkpoint.get("baseline_metrics")
                baseline_evaluations = checkpoint.get("baseline_evaluations")
                if (
                    not isinstance(baseline_metrics, dict)
                    or not isinstance(baseline_evaluations, list)
                    or not has_full_evaluation_coverage(questions, baseline_evaluations)
                ):
                    baseline_metrics, baseline_evaluations = event_loop_runner.run(
                        evaluate_prompt_on_group_async(
                            prompt_text=WORLD_GEPA_USER_SEED_PROMPT,
                            questions=questions,
                            endpoint=target_endpoint,
                            request_runner=request_runner,
                            repeats=materialization_samples,
                            model_attempts=model_attempts,
                            shuffle_namespace=materialize_shuffle_namespace,
                        )
                    )
                    checkpoint["stage"] = "baseline_complete"
                    checkpoint["baseline_metrics"] = baseline_metrics
                    checkpoint["baseline_evaluations"] = baseline_evaluations
                    checkpoint["last_error"] = None
                    write_group_checkpoint(checkpoint_path, checkpoint)

                existing_optimized_evaluations = checkpoint.get("optimized_evaluations")
                if not isinstance(existing_optimized_evaluations, list):
                    existing_optimized_evaluations = []
                optimized_by_id = {
                    evaluation_question_id(payload): payload
                    for payload in existing_optimized_evaluations
                    if isinstance(payload, dict)
                }

                def persist_optimized_question(payload: dict[str, Any]) -> None:
                    optimized_by_id[evaluation_question_id(payload)] = payload
                    ordered_evaluations = order_evaluations_for_questions(questions, optimized_by_id)
                    _, question_deltas = build_question_deltas(
                        questions=questions,
                        baseline_evaluations=baseline_evaluations,
                        optimized_evaluations=ordered_evaluations,
                    )
                    checkpoint["stage"] = "optimized_partial"
                    checkpoint["optimized_evaluations"] = ordered_evaluations
                    checkpoint["optimized_progress"] = {
                        "completed": len(ordered_evaluations),
                        "total": len(questions),
                    }
                    checkpoint["question_deltas"] = question_deltas
                    checkpoint["last_error"] = None
                    write_group_checkpoint(checkpoint_path, checkpoint)

                optimized_metrics, optimized_evaluations = event_loop_runner.run(
                    evaluate_prompt_on_group_async(
                        prompt_text=best_prompt,
                        questions=questions,
                        endpoint=target_endpoint,
                        request_runner=request_runner,
                        repeats=materialization_samples,
                        model_attempts=model_attempts,
                        shuffle_namespace=materialize_shuffle_namespace,
                        completed_evaluations=existing_optimized_evaluations,
                        per_question_callback=persist_optimized_question,
                    )
                )
            finally:
                event_loop_runner.run(request_runner.aclose())

        result_record = build_result_record(
            group_id=group_id,
            target_model_name=target_model_name,
            domain=domain,
            questions=questions,
            best_prompt=best_prompt,
            baseline_metrics=baseline_metrics,
            optimized_metrics=optimized_metrics,
            baseline_evaluations=baseline_evaluations,
            optimized_evaluations=optimized_evaluations,
            gepa_report=gepa_report,
        )
        checkpoint["stage"] = "complete"
        checkpoint["optimized_metrics"] = optimized_metrics
        checkpoint["optimized_evaluations"] = optimized_evaluations
        checkpoint["optimized_progress"] = {
            "completed": len(optimized_evaluations),
            "total": len(questions),
        }
        checkpoint["question_deltas"] = result_record["question_deltas"]
        checkpoint["result_record"] = result_record
        checkpoint["last_error"] = None
        write_group_checkpoint(checkpoint_path, checkpoint)
        return build_group_job_result(result_record)
    except Exception as exc:
        checkpoint["last_error"] = build_failure_record(
            group_id=group_id,
            target_model_name=target_model_name,
            domain=domain,
            question_payloads=question_payloads,
            exc=exc,
        )
        write_group_checkpoint(checkpoint_path, checkpoint)
        raise


def main() -> None:
    args = parse_args()
    if args.metric_samples <= 0:
        raise ValueError("--metric-samples must be positive")
    if args.materialization_samples <= 0:
        raise ValueError("--materialization-samples must be positive")
    if args.max_metric_calls <= 0:
        raise ValueError("--max-metric-calls must be positive")
    if args.max_group_workers <= 0:
        raise ValueError("--max-group-workers must be positive")
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    if args.model_attempts <= 0:
        raise ValueError("--model-attempts must be positive")

    load_pipeline_model_config(args.config_path)
    groups = load_needs_groups(args.decision_path)
    if not groups:
        raise ValueError(f"No GEPA groups found in {args.decision_path}")

    if args.failure_path is None:
        args.failure_path = args.output_path.with_name(f"{args.output_path.stem}_failures.jsonl")
    if args.cache_path is None:
        args.cache_path = args.output_path.parent / "cache" / "request_cache.sqlite"
    if args.clear_cache and args.cache_path.exists():
        args.cache_path.unlink()

    processed_group_ids = load_processed_group_ids(args.output_path) if args.resume else set()
    if not args.resume:
        for path in (args.output_path, args.failure_path):
            if path.exists():
                path.unlink()
    elif args.failure_path.exists():
        args.failure_path.unlink()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.failure_path.parent.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.cache_path.parent.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for target_model_name, domain in sorted(groups):
        group_id = f"{slugify(target_model_name)}__{slugify(domain)}"
        if group_id in processed_group_ids:
            continue
        jobs.append(
            {
                "config_path": str(args.config_path),
                "cache_path": str(args.cache_path),
                "group_id": group_id,
                "target_model_name": target_model_name,
                "domain": domain,
                "question_payloads": groups[(target_model_name, domain)],
                "run_dir": str(args.run_dir / group_id),
                "max_metric_calls": args.max_metric_calls,
                "metric_samples": args.metric_samples,
                "materialization_samples": args.materialization_samples,
                "model_attempts": args.model_attempts,
                "max_concurrency": args.max_concurrency,
                "resume": args.resume,
                "fresh_run_dir": args.fresh_run_dir,
            }
        )

    if args.max_groups is not None:
        jobs = jobs[: args.max_groups]

    processed = 0
    failed = 0

    with (
        args.output_path.open("ab" if args.resume and args.output_path.exists() and args.output_path.stat().st_size > 0 else "wb") as output_handle,
        args.failure_path.open("wb") as failure_handle,
        ProcessPoolExecutor(max_workers=min(args.max_group_workers, max(1, len(jobs)))) as executor,
    ):
        future_to_job = {executor.submit(optimize_group_job, **job): job for job in jobs}
        futures = list(future_to_job)
        for group_index, future in enumerate(as_completed(futures), start=1):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:
                failed += 1
                append_jsonl_line(
                    failure_handle,
                    build_failure_record(
                        group_id=job["group_id"],
                        target_model_name=job["target_model_name"],
                        domain=job["domain"],
                        question_payloads=job["question_payloads"],
                        exc=exc,
                    ),
                )
                print(
                    f"group={group_index} group_id={job['group_id']} questions={len(job['question_payloads'])} "
                    f"status=failed error_type={type(exc).__name__} error={str(exc)!r}",
                    flush=True,
                )
                continue

            append_jsonl_line(output_handle, result["result_record"])
            processed += 1
            print(
                f"group={group_index} group_id={result['group_id']} questions={result['question_count']} "
                f"improved_questions={result['improved_question_count']}",
                flush=True,
            )
    print(f"groups_processed={processed}", flush=True)
    print(f"groups_failed={failed}", flush=True)
    print(f"output_path={args.output_path}", flush=True)
    print(f"failure_path={args.failure_path}", flush=True)


if __name__ == "__main__":
    main()
