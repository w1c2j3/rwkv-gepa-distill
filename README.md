# RWKV GEPA Distill

Minimal MMLU-focused data pipeline for:

- preparing MMLU question pools from Hugging Face `datasets`
- optimizing a teacher system prompt with GEPA for each target row
- generating teacher answers through an OpenAI-compatible API
- filtering to strict JSON-only SFT rows
- exporting a final RWKV training `jsonl`

This repository is intentionally scoped to data generation and export.
Training and evaluation are expected to happen in separate projects.

The official prompt-optimization artifact is only `artifacts/<version>/mmlu_prompt_bundle.jsonl`.
The old standalone `mmlu_best_prompt.txt` output is no longer part of the pipeline.

## Output Contract

The teacher is optimized to return exactly one valid JSON object with these keys:

```json
{
  "answer_letter": "A",
  "answer_index": 0,
  "answer_text": "exact option text",
  "reasoning": "brief explanation"
}
```

Rules enforced by the pipeline:

- only raw teacher responses that are already valid JSON can enter SFT
- parser recovery is allowed for diagnostics only
- non-JSON rows are kept in raw outputs, but never enter strict SFT or RWKV exports
- the final RWKV dataset keeps the original teacher JSON string as the assistant target

## Prompt Bundle Contract

Each line of `artifacts/v1/mmlu_prompt_bundle.jsonl` records one full prompt-optimization trace for one question:

```json
{
  "bundle_contract": "mmlu_prompt_trace_v1",
  "source_row_index": 17,
  "question": {
    "source_dataset": "cais/mmlu",
    "source_split": "auxiliary_train",
    "subject": "high_school_biology",
    "question": "Which organelle is primarily responsible for ATP production?",
    "choices": [
      "Ribosome",
      "Mitochondrion",
      "Golgi apparatus",
      "Lysosome"
    ],
    "answer": "B",
    "answer_index": 1,
    "answer_label": "B",
    "prompt_text": "Subject: high_school_biology\nQuestion: Which organelle is primarily responsible for ATP production?\nOptions:\nA. Ribosome\nB. Mitochondrion\nC. Golgi apparatus\nD. Lysosome",
    "meta": {}
  },
  "prompt": "original system prompt text",
  "prompt_version": "sha256:...",
  "prompt_teacher_response": "{\"answer_letter\":\"B\",\"answer_index\":1,\"answer_text\":\"Mitochondrion\",\"reasoning\":\"Mitochondria are the primary site of aerobic ATP production in eukaryotic cells.\"}",
  "prompt_score": {
    "total": 1.0,
    "valid_json": true,
    "correct": true,
    "usable_for_sft": true
  },
  "best_prompt": "optimized system prompt text",
  "best_prompt_version": "sha256:...",
  "best_prompt_teacher_response": "{\"answer_letter\":\"B\",\"answer_index\":1,\"answer_text\":\"Mitochondrion\",\"reasoning\":\"The mitochondrion generates most ATP through oxidative phosphorylation.\"}",
  "best_prompt_score": {
    "total": 1.0,
    "valid_json": true,
    "correct": true,
    "usable_for_sft": true
  },
  "optimizer": {
    "mode": "gepa",
    "best_score": 1.0,
    "generated_at_utc": "2026-03-07T12:34:56+00:00"
  },
  "comparison": {
    "prompt_changed": true,
    "prompt_version": "sha256:...",
    "best_prompt_version": "sha256:...",
    "score_delta": 0.0,
    "strict_json_delta": 0,
    "correct_delta": 0,
    "usable_for_sft_delta": 0
  },
  "teacher": {
    "mode": "api",
    "model": "your-teacher-model"
  }
}
```

The intended read order is:

1. inspect `question`
2. inspect the baseline `prompt`
3. inspect `prompt_teacher_response`
4. inspect the optimized `best_prompt`
5. inspect `best_prompt_teacher_response`
6. inspect `comparison`

## Bootstrap

```bash
bash scripts/bootstrap.sh
```

## Unified Entry Point

The official entrypoint is:

```bash
python scripts/scheduler.py
```

If run with no arguments, it executes the default `full-mmlu` pipeline for `v1`:

1. prepare the primary MMLU pools
2. optimize a per-row prompt bundle with GEPA
3. generate teacher responses for `auxiliary_train` using the per-row prompt bundle
4. filter to strict JSON rows usable for SFT
5. export a structured SFT dataset
6. export the final RWKV training `jsonl`
7. sample a review subset

To target another dataset version:

```bash
python scripts/scheduler.py --dataset-version v2
```

## Optional Subcommands

```bash
python scripts/scheduler.py optimize-mmlu --dataset-version v1
python scripts/scheduler.py generate-mmlu --dataset-version v1
python scripts/scheduler.py filter-mmlu --dataset-version v1
python scripts/scheduler.py export-sft --dataset-version v1
python scripts/scheduler.py export-rwkv --dataset-version v1
python scripts/scheduler.py sample-review --dataset-version v1
```

Useful overrides:

```bash
python scripts/scheduler.py --dataset-version v1 --max-concurrency 6
python scripts/scheduler.py generate-mmlu --dataset-version v1 --no-resume
python scripts/scheduler.py full-mmlu --dataset-version v2 --limit 5000
```

## Default Versioned Outputs

For `--dataset-version v1`, the core outputs are:

- per-row prompt bundle: `artifacts/v1/mmlu_prompt_bundle.jsonl`
- raw teacher outputs generated with `best_prompt`: `data/distill/v1/mmlu_batch_all.jsonl`
- strict filtered rows: `data/distill/v1/mmlu_batch_all_strict.jsonl`
- structured SFT dataset: `data/distill/v1/mmlu_batch_all_sft.jsonl`
- final RWKV training dataset: `data/distill/v1/mmlu_batch_all_rwkv.jsonl`

Diagnostics and audit files:

- failed generation rows: `artifacts/v1/mmlu_batch_all_failed.jsonl`
- filter stats: `artifacts/v1/mmlu_batch_all_filter_stats.json`
- review sample: `artifacts/v1/mmlu_batch_all_review_40.jsonl`
- prompt report: `artifacts/v1/mmlu_prompt_optimization_report.json`

## Current Plan

The current pipeline is:

1. run MMLU question pools to build synthetic teacher-answer data
2. record a baseline teacher answer with the ordinary prompt
3. run GEPA to optimize that prompt for each target row
4. record the optimized `best_prompt` and the teacher answer produced with it
5. keep only rows that are both correct and strict-JSON-valid
6. export the surviving rows as SFT and RWKV training datasets
7. fine-tune RWKV in a separate training project
8. analyze weak domains from the new score report and synthesize the next dataset version

The current implementation uses per-row optimization (`scope = 1`).
If later you want block optimization such as 100 rows per GEPA run, the main code changes will be:

- add an optimization-scope parameter to the optimizer
- let the prompt bundle store `scope_size` / `scope_id`
- let the optimizer report compare block-level metrics in addition to row-level metrics
- keep row-level `best_prompt` materialization so generation stays simple

## Current Gaps

- The pipeline can measure correctness, JSON validity, and SFT usability, but it still cannot quantify how much reasoning quality improved.
- The pipeline also cannot tell whether a correct answer was produced with the shortest valid reasoning path.

## Teacher / GEPA API Configuration

The code reads `.env` automatically. Example variables:

```bash
TEACHER_API_KEY=...
TEACHER_BASE_URL=https://your-openai-compatible-endpoint/v1
TEACHER_MODEL=gpt-5.2

GEPA_API_KEY=...
GEPA_BASE_URL=https://your-openai-compatible-endpoint/v1
GEPA_MODEL=Kimi-K2.5
```

Teacher generation and GEPA reflection use the official `openai` Python SDK.
The code automatically selects `responses` or `chat.completions` based on the configured model, and you can
override that with `TEACHER_API_PROTOCOL` or `GEPA_API_PROTOCOL` if needed.

## GitHub Upload Checklist

Upload:

- `distill_gepa/`
- `scripts/`
- `data/seeds/`
- `README.md`
- `pyproject.toml`
- `.python-version`
- `.gitignore`

Do not upload:

- `.env`
- `.venv/`
- `vendor/`
- `artifacts/`
- `data/question_pools/`
- `data/eval_pools/`
- `data/distill/`
