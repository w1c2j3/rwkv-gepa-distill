# RWKV GEPA Distill Smoke Scaffold

Minimal runnable scaffold for:

- API teacher model
- GEPA prompt optimization
- RWKV student distillation handoff

The first milestone is a smoke test that:

1. selects an optimized teacher system prompt on a tiny validation seed set,
2. saves the prompt to `artifacts/best_prompt.txt`,
3. generates `data/distill/distill_small.jsonl`.

## Commands

```bash
bash scripts/bootstrap.sh
bash scripts/step1_smoke.sh
```

Optional vendor fetch:

```bash
bash scripts/fetch_vendor.sh
```

## MMLU Pool Prep

Prepare normalized MMLU / MMLU-Pro / MMLU-Redux JSONL pools. If the Hugging Face datasets are not cached locally yet, the script will download them automatically through `datasets.load_dataset()`.

```bash
python scripts/prepare_mmlu_pools.py
python scripts/prepare_mmlu_pools.py --only primary
python scripts/prepare_mmlu_pools.py --include-community
```

## MMLU Batch Run

Run the next end-to-end MMLU batch test:

```bash
bash scripts/step2_mmlu_batch.sh
python scripts/scheduler.py batch-100
```

This step:

1. refreshes the primary MMLU question pools,
2. optimizes an MMLU teacher system prompt with GEPA on small `dev` / `validation` slices,
3. generates 100 teacher responses from `mmlu_auxiliary_train`,
4. writes a random 40-row review sample.

## Full Auxiliary Train Run

Run the first full-pass distillation over all `cais/mmlu` `auxiliary_train` rows:

```bash
bash scripts/step3_mmlu_full.sh
python scripts/scheduler.py full-mmlu
```

This step:

1. checks whether the primary MMLU question pools already exist and only rebuilds them when missing,
2. optimizes an MMLU teacher system prompt with GEPA,
3. generates teacher responses for all `auxiliary_train` rows,
4. filters to the rows whose parsed answer matches the gold label,
5. exports a compact SFT-ready JSONL dataset,
6. writes a 40-row review sample from the filtered full dataset.

The scheduler prints stage boundaries, elapsed time, and per-stage progress. During pool download or cache reads you may still see:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

That is only a rate-limit warning. Set `HF_TOKEN` if you want faster Hugging Face downloads.

## Scheduler Commands

The Python scheduler is the unified task entrypoint:

```bash
python scripts/scheduler.py prepare-pools --only primary
python scripts/scheduler.py optimize-mmlu
python scripts/scheduler.py generate-mmlu
python scripts/scheduler.py batch-100
python scripts/scheduler.py full-mmlu
```

Optional flags for the full run:

```bash
python scripts/scheduler.py full-mmlu --force-pools
python scripts/scheduler.py full-mmlu --max-concurrency 6
```

The full run is resumable. If generation is interrupted, running the same command again continues from the existing partial `data/distill/mmlu_batch_all.jsonl`.

## Teacher API configuration

If `TEACHER_MODEL` and `TEACHER_API_KEY` are set, the scripts use a real OpenAI-compatible teacher endpoint through LiteLLM. `TEACHER_BASE_URL` is supported and optional for the official OpenAI API.

If any of those variables are missing, the project automatically falls back to an offline mock teacher so the smoke test still runs.

Optional GEPA reflection configuration for real API mode:

```bash
export TEACHER_BASE_URL="https://your-openai-compatible-endpoint/v1"
export TEACHER_API_KEY="your-key"
export TEACHER_MODEL="gpt-5.2"

export GEPA_BASE_URL="https://your-openai-compatible-endpoint/v1"
export GEPA_API_KEY="your-gepa-key"
export GEPA_MODEL="Kimi-K2.5"
```

If no GEPA reflection model is available, `optimize_prompt.py` falls back to a deterministic local prompt sweep and still produces the required artifacts.

## Outputs

After the smoke test completes, these files should exist:

- `artifacts/best_prompt.txt`
- `artifacts/optimization_report.json`
- `data/distill/distill_small.jsonl`

## Next step

Use `train/run_rwkv_peft.sh` or `train/run_rwkv_lm.sh` as the handoff point where `data/distill/mmlu_batch_all_sft.jsonl` can be wired into real RWKV training and preprocessing.

## GitHub Upload Checklist

Upload:

- `distill_gepa/`
- `scripts/`
- `train/`
- `configs/`
- `data/seeds/`
- `README.md`
- `pyproject.toml`
- `.python-version`
- `.gitignore`

Do not upload:

- `.venv/`
- `.env`
- `vendor/`
- `artifacts/`
- `data/question_pools/`
- `data/eval_pools/`
- `data/distill/`
