# Seed-Driven Distillation

This repo runs one pipeline:

1. read seed questions from a local file
2. auto-download and prepare the dataset if the local seed file is missing
3. use `gpt-5.4` to generate related variants
4. let 4 target models answer every generated variant
5. retry failed answers with model-specific optimized prompts
6. keep correct rows as distillation data
7. keep fully unresolved rows in a separate failure file

## Quick Start

### 1. Install dependencies

```bash
bash scripts/bootstrap.sh
```

### 2. Fill API key

Copy `.env.example` to `.env` and fill the key:

```bash
cp .env.example .env
```

Required field in `.env`:

```bash
REELXAI_API_KEY=your_api_key_here
```

If you want to change provider, base URL, or model names, edit:

```bash
config/world_pipeline.yaml
```

Current default config already uses:

- base URL: `https://reelxai.com/v1`
- generator model: `gpt-5.4`
- prompt optimizer model: `gpt-5.4`
- answer models:
  - `gemini-3-flash-preview`
  - `gpt-5-2025-08-07`
  - `claude-sonnet-4-5-20250929`
  - `grok-4.2`

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
- commit `config/world_pipeline.yaml` only if you want to keep the default provider and model names public
- large output folders under `data/` are usually better left out unless you intentionally want to publish artifacts
