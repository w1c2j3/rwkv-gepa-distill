# RWKV GEPA Distill

World-knowledge synthetic data pipeline for:

- preparing a unified benchmark pool from dataset TOML files in `config/datasets/`
- running four base models through an OpenAI-compatible API with repeated shuffled-choice evaluation
- classifying questions into `direct_distill`, `needs_optimization`, and `suspected_anomaly`
- optimizing grouped user-side prompts with GEPA for `target_model x domain`
- rewriting successful complex questions into simpler distillation prompts
- exporting merged SFT and RWKV datasets

All model calls use an OpenAI-compatible API surface. To switch from a remote endpoint to a local model server,
change the configured `base_url` to `http://127.0.0.1:<port>/v1`.

## Config

Edit [config/world_pipeline.yaml](/home/chase/GitHub/rwkv-gepa-distill/config/world_pipeline.yaml).

Model roles are separated:

- `base_models`: the four benchmark models
- `gepa_reflection_model`: the large model used by GEPA
- `rewrite_model`: the large model used to rewrite complex questions

Each entry uses the same OpenAI-compatible fields:

- `name`
- `model`
- `base_url`
- `api_key`
- `api_protocol`

`config/world_pipeline.example.yaml` is the template copy.

## Pipeline

The default entrypoint is:

```bash
python scripts/scheduler.py
```

With no arguments it runs `full-world`, which does:

1. prepare the dataset declared in `config/datasets/<dataset>.toml`
2. run four-model repeated evaluation
3. classify questions
4. run grouped GEPA user-prompt optimization
5. rewrite complex GEPA questions into simpler prompts and revalidate them under shuffled choices
6. merge all distillation sources
7. export RWKV training jsonl

Useful subcommands:

```bash
python scripts/scheduler.py prepare-world --dataset-name mmlu_auxiliary_train
python scripts/scheduler.py run-benchmark --dataset-name mmlu_auxiliary_train
python scripts/scheduler.py classify-questions --dataset-name mmlu_auxiliary_train
python scripts/scheduler.py optimize-gepa --dataset-name mmlu_auxiliary_train
python scripts/scheduler.py rewrite-questions --dataset-name mmlu_auxiliary_train
python scripts/scheduler.py merge-distill --dataset-name mmlu_auxiliary_train
python scripts/scheduler.py export-rwkv --dataset-name mmlu_auxiliary_train
```

## Default Outputs

For `--dataset-name mmlu_auxiliary_train` the main outputs are:

- questions: `data/mmlu_auxiliary_train/questions.jsonl`
- per-question decisions: `data/mmlu_auxiliary_train/question_decisions.jsonl`
- GEPA grouped results: `data/mmlu_auxiliary_train/gepa_results.jsonl`
- rewrite distill rows: `data/mmlu_auxiliary_train/rewrite_distill.jsonl`
- merged SFT dataset: `data/mmlu_auxiliary_train/distill_sft.jsonl`
- RWKV dataset: `data/mmlu_auxiliary_train/distill_rwkv.jsonl`
- pipeline summary: `data/mmlu_auxiliary_train/pipeline_summary.json`
- cache directory: `data/mmlu_auxiliary_train/cache/`

## Notes

- all multiple-choice model calls use shuffled options, not the source order
- assistant responses must be strict JSON and keep reasoning inside `<think>...</think>`
- `direct_distill`: all four base models are stably correct; export additionally requires `<think>`-tagged reasoning
- `suspected_anomaly`: all four base models are stably wrong
- `needs_optimization`: anything else enters grouped GEPA optimization
- grouped GEPA works at `target_model x domain` granularity
- grouped GEPA keeps the JSON system prompt fixed and searches a user-side prompt prefix
- rewrite revalidates simple prompts with repeated shuffled-choice checks
- root dataset folders keep only JSON/JSONL outputs; large intermediate state lives under `data/<dataset>/cache/`
