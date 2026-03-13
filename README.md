# Task-Driven Distillation

This repo runs one pipeline:

1. read task seeds from a local file
2. optionally auto-prepare a dataset from `config/datasets/*.toml`
3. use one generator model to generate related variants
4. let all configured answer models answer every generated variant
5. retry failed answers with model-specific optimized prompts
6. keep correct rows as distillation data
7. keep fully unresolved rows in a separate failure file

## Quick Start

### 1. Install dependencies

```bash
bash scripts/bootstrap.sh
```

### 2. Fill model and API config

Copy `.env.example` to `.env` and fill the key:

```bash
cp .env.example .env
```

The pipeline reads model name, base URL, API key, and protocol directly from `.env`.
You must fill three sections:

```bash
GENERATOR_MODEL_*
OPTIMIZER_MODEL_*
ANSWER_MODEL_*
```

The config file:

```bash
config/world_pipeline.yaml
```

does not lock specific models anymore. It only tells the loader which `.env` prefixes to read.

Minimal example:

```bash
OPENAI_TIMEOUT=120
OPENAI_MAX_RETRIES=3
OPENAI_MAX_TOKENS=512

GENERATOR_MODEL_NAME=gpt-5.4
GENERATOR_MODEL_BASE_URL=https://reelxai.com/v1
GENERATOR_MODEL_API_KEY=your_key
GENERATOR_MODEL_API_PROTOCOL=responses

OPTIMIZER_MODEL_NAME=gpt-5.4
OPTIMIZER_MODEL_BASE_URL=https://reelxai.com/v1
OPTIMIZER_MODEL_API_KEY=your_key
OPTIMIZER_MODEL_API_PROTOCOL=responses

ANSWER_MODEL_IDS=1,2,3,4
ANSWER_MODEL_SHARED_BASE_URL=https://reelxai.com/v1
ANSWER_MODEL_SHARED_API_KEY=your_key
ANSWER_MODEL_1_NAME=gemini-3-flash-preview
ANSWER_MODEL_2_NAME=gpt-5-2025-08-07
ANSWER_MODEL_2_API_PROTOCOL=responses
ANSWER_MODEL_3_NAME=claude-sonnet-4-5-20250929
ANSWER_MODEL_4_NAME=grok-4.2
```

If different answer models use different providers, set per-model fields directly:

```bash
ANSWER_MODEL_1_BASE_URL=...
ANSWER_MODEL_1_API_KEY=...
ANSWER_MODEL_2_BASE_URL=...
ANSWER_MODEL_2_API_KEY=...
```

## How To Change The Number Of Answer Models

The number of answer models is controlled only by:

```bash
ANSWER_MODEL_IDS=...
```

The pipeline reads every id in that list, then loads:

```bash
ANSWER_MODEL_<id>_NAME
ANSWER_MODEL_<id>_BASE_URL
ANSWER_MODEL_<id>_API_KEY
ANSWER_MODEL_<id>_API_PROTOCOL
```

If you use shared provider settings, you only need:

```bash
ANSWER_MODEL_SHARED_BASE_URL=...
ANSWER_MODEL_SHARED_API_KEY=...
```

and then each model only needs a name, plus protocol when necessary.

### Example: use 2 answer models

```bash
ANSWER_MODEL_IDS=1,2
ANSWER_MODEL_SHARED_BASE_URL=https://reelxai.com/v1
ANSWER_MODEL_SHARED_API_KEY=your_key

ANSWER_MODEL_1_NAME=gemini-3-flash-preview
ANSWER_MODEL_1_API_PROTOCOL=chat_completions

ANSWER_MODEL_2_NAME=gpt-5-2025-08-07
ANSWER_MODEL_2_API_PROTOCOL=responses
```

### Example: use 4 answer models

```bash
ANSWER_MODEL_IDS=1,2,3,4
ANSWER_MODEL_SHARED_BASE_URL=https://reelxai.com/v1
ANSWER_MODEL_SHARED_API_KEY=your_key

ANSWER_MODEL_1_NAME=gemini-3-flash-preview
ANSWER_MODEL_2_NAME=gpt-5-2025-08-07
ANSWER_MODEL_2_API_PROTOCOL=responses
ANSWER_MODEL_3_NAME=claude-sonnet-4-5-20250929
ANSWER_MODEL_4_NAME=grok-4.2
```

### Example: add a 5th answer model

```bash
ANSWER_MODEL_IDS=1,2,3,4,5
ANSWER_MODEL_5_NAME=gemini-3.1-pro-preview
ANSWER_MODEL_5_API_PROTOCOL=chat_completions
```

### Example: remove one answer model

If you no longer want model 4:

```bash
ANSWER_MODEL_IDS=1,2,3
```

You can leave old `ANSWER_MODEL_4_*` lines in `.env`, but they will be ignored once `4` is removed from `ANSWER_MODEL_IDS`.

In short:

- add a model: add a new id to `ANSWER_MODEL_IDS`, then fill `ANSWER_MODEL_<id>_*`
- remove a model: delete its id from `ANSWER_MODEL_IDS`
- reorder models: change the order inside `ANSWER_MODEL_IDS`
- use different providers per model: set per-model `BASE_URL` and `API_KEY`

## Recommended Commands

Prepare an example MMLU task pool:

```bash
python scripts/prepare_dataset.py --dataset-config-path config/datasets/mmlu.toml --dataset-name mmlu --limit 100
```

Run the pipeline from a prepared task file:

```bash
python scripts/run_pipeline.py --dataset-name demo_run --task-input-path data/mmlu/tasks.jsonl --limit 100
```

Or let the pipeline auto-prepare the task file from a dataset config:

```bash
python scripts/run_pipeline.py --dataset-name demo_run --task-input-path data/demo_run/tasks.jsonl --dataset-config-path config/datasets/mmlu.toml --limit 100
```

## Useful Commands

Run a small test:

```bash
python scripts/run_pipeline.py --dataset-name smoke --task-input-path data/smoke/tasks.jsonl --dataset-config-path config/datasets/mmlu.toml --limit 5
```

Run from a custom task file:

```bash
python scripts/run_pipeline.py --task-input-path /path/to/tasks.jsonl --dataset-name my_run --variants-per-task 24
```

## Final Outputs

The pipeline writes these public outputs:

- `data/<dataset-name>/question_variants.jsonl`
- `data/<dataset-name>/variant_results.jsonl`
- `data/<dataset-name>/failures.jsonl`
- `data/<dataset-name>/summary.json`

Internal files stay under:

- `data/<dataset-name>/cache/`
- notably `data/<dataset-name>/cache/api_trace.jsonl`, which stores every model prompt/response pair seen by the pipeline

## Output Schemas

`question_variants.jsonl`: one row per original task seed, with all generated variants.

Example:

```json
{
  "question_id": "mmlu::train::0",
  "data_split": "train",
  "question_text": "Which of these is an element?",
  "choices": ["KBr", "O_{2}", "2KCl", "FeO_{2}"],
  "reference_answer": "O_{2}",
  "reference_answer_index": 1,
  "variants": [
    {
      "id": "mmlu::train::0::variant::0",
      "question_text": "Which of the following represents an element rather than a compound or mixture?",
      "choices": ["NaCl", "N_{2}", "3MgO", "CaCO_{3}"],
      "reference_answer": "N_{2}",
      "reference_answer_index": 1
    }
  ]
}
```

`variant_results.jsonl`: one row per successfully resolved variant. If a model initially failed but GEPA later fixed it, the final row stores only the final surviving prompt/response pair for that model.

Example:

```json
{
  "id": "mmlu::train::0::variant::0",
  "question_id": "mmlu::train::0",
  "data_split": "train",
  "question_text": "Which of the following represents an element rather than a compound or mixture?",
  "choices": ["NaCl", "N_{2}", "3MgO", "CaCO_{3}"],
  "reference_answer": "N_{2}",
  "reference_answer_index": 1,
  "status": "resolved_by_gepa",
  "models": [
    {
      "model_name": "gemini-3-flash-preview",
      "source": "direct",
      "prompt_text": "You answer questions accurately and concisely.",
      "response_text": "N_{2}"
    },
    {
      "model_name": "grok-4.1",
      "source": "gepa",
      "prompt_text": "improved system prompt text",
      "response_text": "N_{2}"
    }
  ]
}
```

`failures.jsonl`: one row per variant that still could not be fully resolved. The public failure file is intentionally minimal.

Example:

```json
{
  "question_text": "Which option is a pure element?"
}
```

`summary.json`: aggregate counts and output locations for the run.

## Upload Notes

Before pushing to GitHub:

- do not commit `.env`
- commit `.env.example`
- commit `config/world_pipeline.yaml` because it now only contains generic env-prefix wiring
- large output folders under `data/` are usually better left out unless you intentionally want to publish artifacts
