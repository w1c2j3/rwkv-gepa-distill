# Seed-Driven Distillation

This repo runs one pipeline:

1. read seed questions from a local file
2. auto-download and prepare the dataset if the local seed file is missing
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

## One-Command Reproduction

Run the README demo with a 100-question cap:

```bash
python scripts/scheduler.py --limit 100
```

This is the recommended command for other people cloning the repo.

## Default Behavior

`python scripts/scheduler.py` uses these defaults:

- seed input path: `data/mmlu_auxiliary_train/questions.jsonl`
- dataset name: `mmlu_auxiliary_train_seed_run`
- variants per seed: `12`
- limit: no limit
- answer models: all ids listed in `ANSWER_MODEL_IDS`

If `data/mmlu_auxiliary_train/questions.jsonl` does not exist, the scheduler will automatically check:

```bash
config/datasets/mmlu_auxiliary_train.toml
```

and then auto-download/prepare the dataset from Hugging Face before running.

So on a fresh clone, the following is enough:

```bash
python scripts/scheduler.py --limit 100
```

## Useful Commands

Run all available seed questions:

```bash
python scripts/scheduler.py
```

Run a small test:

```bash
python scripts/scheduler.py --limit 5
```

Run with a custom seed file:

```bash
python scripts/scheduler.py --seed-input-path /path/to/wrong_questions.jsonl --dataset-name my_run --variants-per-seed 24
```

## Final Outputs

The pipeline writes these public outputs:

- `data/<dataset-name>/generated_questions.jsonl`
- `data/<dataset-name>/distill_data.jsonl`
- `data/<dataset-name>/unresolved_failures.jsonl`

Internal files stay under:

- `data/<dataset-name>/cache/`

## Output Schemas

`generated_questions.jsonl`: one row per original seed question, with all generated variants.

Example:

```json
{
  "seed_id": "mmlu_auxiliary_train::auxiliary_train::0",
  "domain": "abstract algebra",
  "question_type": "multiple_choice",
  "question": "Which of these is an element?",
  "answer": "O_{2}",
  "answer_index": 1,
  "choices": ["KBr", "O_{2}", "2KCl", "FeO_{2}"],
  "generated_questions": [
    {
      "variant_id": "mmlu_auxiliary_train::auxiliary_train::0::variant::0",
      "question": "Which of the following represents an element rather than a compound or mixture?",
      "answer": "N_{2}",
      "answer_index": 1,
      "choices": ["NaCl", "N_{2}", "3MgO", "CaCO_{3}"],
      "generator_model": "gpt-5.4",
      "generation_prompt_version": "prompt-xxxx"
    }
  ]
}
```

`distill_data.jsonl`: one row per kept correct answer.

Example:

```json
{
  "source_type": "direct_answer",
  "seed_id": "mmlu_auxiliary_train::auxiliary_train::0",
  "variant_id": "mmlu_auxiliary_train::auxiliary_train::0::variant::0",
  "model_name": "gemini-3-flash-preview",
  "question_type": "multiple_choice",
  "question": "Which of the following represents an element rather than a compound or mixture?",
  "gold_answer": "N_{2}",
  "gold_answer_index": 1,
  "model_answer": "N_{2}",
  "model_answer_index": 1,
  "choices": ["NaCl", "N_{2}", "3MgO", "CaCO_{3}"]
}
```

For recovered rows, `source_type` is `optimized_answer`, and the row also includes:

- `direct_model_answer`
- `direct_model_answer_index`
- `optimized_system_prompt`
- `optimized_prompt_version`

`unresolved_failures.jsonl`: one row per variant that all models still failed after prompt optimization.

## Upload Notes

Before pushing to GitHub:

- do not commit `.env`
- commit `.env.example`
- commit `config/world_pipeline.yaml` because it now only contains generic env-prefix wiring
- large output folders under `data/` are usually better left out unless you intentionally want to publish artifacts
