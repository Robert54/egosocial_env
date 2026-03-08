# Experiment Log

Last updated: 2026-03-08 UTC

## Scope

Current focus is the `3 vs 4` comparison:

- `3`: `robertzty/EgoNormia-Cosmos-Reason2-2B-v6b-shortcot` as `SFT only`
- `4`: the same checkpoint plus `GRPO` on the local `EgoNormia` 2-turn OpenEnv

## Data Alignment

The environment is now aligned to the same metadata family used by the earlier SFT pipeline.

- Env metadata source of truth: [final_data.json](/root/openenv-hack/egosocial_env/data/egonormia/final_data.json)
- Heldout split: [verified_split.json](/root/openenv-hack/egosocial_env/data/splits/verified_split.json)
- Heldout size: `200`
- Env overlap with heldout: `200/200`

Important note:

- `train-norm-updated.parquet` has only `1743` rows
- `final_data.json` has `1853` scenes
- the earlier SFT pipeline used `final_data.json + verified_split`, not the parquet-only view

## SFT Baseline

### Official/OpenEnv caveat

Do not directly compare the OpenEnv 2-turn metrics below with the earlier official-style SFT metrics such as:

- `78.5 / 88.5 / 70.5 / 0.6450 / 100.0`

Those are a different evaluation protocol. The numbers below are from the OpenEnv 2-turn benchmark-mode evaluator.

### Text-only env eval

Output:

- [/tmp/egosocial_eval/eval_v6b_verified_finaldata.json](/tmp/egosocial_eval/eval_v6b_verified_finaldata.json)

Summary:

- `accuracy = 0.555`
- `avg_reward = 0.4487`
- `avg_sensibility = 0.6900`
- `avg_taxonomy_match = 0.2995`
- `avg_justification_alignment = 0.3867`
- `avg_rubric_average = 0.5504`

### Image-conditioned env eval

Output:

- [/tmp/egosocial_eval/eval_v6b_verified_finaldata_image_200.json](/tmp/egosocial_eval/eval_v6b_verified_finaldata_image_200.json)

Summary:

- model: `robertzty/EgoNormia-Cosmos-Reason2-2B-v6b-shortcot`
- `accuracy = 0.685`
- `avg_reward = 0.5305`
- `avg_action_selection = 0.685`
- `avg_sensibility = 0.765`
- `avg_taxonomy_match = 0.3029`
- `avg_justification_alignment = 0.4800`
- `avg_rubric_average = 0.6205`

This is the current `SFT only` baseline to use for OpenEnv.

## GRPO Status

### What was fixed

The GRPO training path in [train_grpo_reason2.py](/root/openenv-hack/egosocial_env/scripts/train_grpo_reason2.py) was upgraded to:

- use real image-conditioned 2-turn observations instead of text-only prompts
- pack a full episode as one training example
- preserve environment-feedback masking via `env_mask`
- pass multimodal tensors needed by `Qwen3VL/Cosmos-Reason2`
- add `mm_token_type_ids` handling on the training path
- downscale training-time images with `--image-max-edge` to avoid H100 OOM on the very wide `frame_all_prev.jpg` / `frame_all_during.jpg` panoramas

### Smoke test result

A real `1-step` GRPO smoke test completed successfully on the H100.

Run config:

- model: `robertzty/EgoNormia-Cosmos-Reason2-2B-v6b-shortcot`
- env mode: `benchmark`
- `scene_limit = 1`
- `scene_repeats = 2`
- `num_generations = 2`
- `per_device_train_batch_size = 2`
- `gradient_accumulation_steps = 1`
- `max_steps = 1`
- `max_completion_length = 128`
- `image_max_edge = 320`

Observed training summary:

- `train_loss = 0.01375`
- `train_runtime = 138.4s`

Artifacts:

- output dir: [/tmp/egosocial_train/grpo_v6b_multimodal_smoke_gpu](/tmp/egosocial_train/grpo_v6b_multimodal_smoke_gpu)
- checkpoint: [/tmp/egosocial_train/grpo_v6b_multimodal_smoke_gpu/checkpoint-1](/tmp/egosocial_train/grpo_v6b_multimodal_smoke_gpu/checkpoint-1)
- final weights: [/tmp/egosocial_train/grpo_v6b_multimodal_smoke_gpu/model.safetensors](/tmp/egosocial_train/grpo_v6b_multimodal_smoke_gpu/model.safetensors)

This means the `TRL + OpenEnv + Cosmos-Reason2` training path is now operational.

## Cosmos Predict

Cosmos Predict integration remains available as the world-model backend for `train` mode, but it is not the current training truth source.

Current positioning:

- `benchmark mode`: main training/eval truth path
- `train mode + Cosmos Predict`: rollout/demo/scaling path

## Recommended Next Step

Run the real `4` experiment:

- `v6b SFT only` on the fixed 200-scene heldout
- `v6b + GRPO` on non-heldout `EgoNormia`, then evaluate on the same heldout

Suggested starting command:

```bash
cd /root/openenv-hack/egosocial_env
HF_HOME=/tmp/hf TRANSFORMERS_CACHE=/tmp/hf/hub \
.venv/bin/python scripts/train_grpo_reason2.py \
  --model-id /tmp/hf/hub/models--robertzty--EgoNormia-Cosmos-Reason2-2B-v6b-shortcot/snapshots/358d42e154c403a07f3cc2bac9e4f17551146484 \
  --trust-remote-code \
  --env-mode benchmark \
  --scene-limit 256 \
  --scene-repeats 2 \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --max-steps 30 \
  --max-completion-length 128 \
  --image-max-edge 320 \
  --output-dir /tmp/egosocial_train/grpo_v6b_benchmark_run
```

Then evaluate the resulting checkpoint with:

```bash
cd /root/openenv-hack/egosocial_env
HF_HOME=/tmp/hf TRANSFORMERS_CACHE=/tmp/hf/hub \
.venv/bin/python scripts/eval_reason2.py \
  --model-id /tmp/egosocial_train/grpo_v6b_benchmark_run \
  --trust-remote-code \
  --bf16 \
  --env-mode benchmark \
  --output-path /tmp/egosocial_eval/eval_grpo_v6b_verified_image_200.json
```

## Active Run

An actual long GRPO benchmark run has been started.

Run directory:

- [/tmp/egosocial_train/grpo_v6b_benchmark_run](/tmp/egosocial_train/grpo_v6b_benchmark_run)

Command:

```bash
cd /root/openenv-hack/egosocial_env
HF_HOME=/tmp/hf TRANSFORMERS_CACHE=/tmp/hf/hub \
.venv/bin/python scripts/train_grpo_reason2.py \
  --model-id /tmp/hf/hub/models--robertzty--EgoNormia-Cosmos-Reason2-2B-v6b-shortcot/snapshots/358d42e154c403a07f3cc2bac9e4f17551146484 \
  --trust-remote-code \
  --bf16 \
  --env-mode benchmark \
  --scene-limit 0 \
  --scene-repeats 2 \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --max-steps 60 \
  --max-completion-length 128 \
  --image-max-edge 320 \
  --output-dir /tmp/egosocial_train/grpo_v6b_benchmark_run
```

Observed status at launch:

- training loop entered successfully
- progress reached `0/60`
- GPU visible on H100

## Hugging Face Space

A lightweight deploy copy was pushed to Hugging Face Spaces.

Space:

- https://huggingface.co/spaces/robertzty/egosocial-env

Deployment note:

- this Space was pushed from a lightweight directory without the local `61G` `data/egonormia` asset tree
- the online Space therefore falls back to the synthetic fixture instead of bundling the full local video dataset
- local training and evaluation remain unchanged

## Known Remaining Gap

The OpenEnv 2-turn evaluator is now valid for the `3 vs 4` experiment, but it is still a different protocol from the earlier official-style SFT benchmark. If needed for reporting, add a separate official-style eval table rather than mixing the two protocols.
