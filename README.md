---
title: Egosocial Env
emoji: 🎬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - social
  - rl
---

# Egosocial Env

Minimal OpenEnv project for the README's main idea: a two-turn social interaction environment with `prev` and `during` observations.

When local `EgoNormia` assets are present under `data/egonormia/`, the environment loads the real parquet metadata plus the matching `video_prev.mp4` / `video_during.mp4` clips. If those assets are missing, it falls back to the small synthetic fixture file so you can still validate the OpenEnv loop and deployment workflow.

The env now supports two transition modes:

- `benchmark`: replay the fixed `during` clip from EgoNormia for reproducible evaluation.
- `train`: retrieve a norm- and text-matched `during` clip from another local EgoNormia scene as an action-conditioned transition baseline, with a stub fallback if retrieval is unavailable.

Train mode can also prefer a configured world-model adapter before retrieval:

- `EGOSOCIAL_WORLD_MODEL_PROVIDER=mock`: exercise the generated-video path locally by materializing assets under `/tmp/egosocial_env/generated/`.
- `EGOSOCIAL_WORLD_MODEL_PROVIDER=cosmos`: call a local Cosmos Predict 2.5 checkout via `examples/inference.py` and use Video2World for the first-turn transition.
- Any other real provider should be wired behind `world_model.py` while keeping the same transition contract.

## Local Setup

```bash
cd /root/openenv-hack/egosocial_env
UV_CACHE_DIR=/tmp/uv-cache uv sync
source .venv/bin/activate
python scripts/smoke_test.py

# Optional: exercise the adapter-first path
EGOSOCIAL_WORLD_MODEL_PROVIDER=mock python scripts/smoke_test.py

# Optional: run Cosmos Predict 2.5 if the official repo is installed locally
EGOSOCIAL_WORLD_MODEL_PROVIDER=cosmos \
EGOSOCIAL_COSMOS_REPO=/root/cosmos-predict2.5-main \
EGOSOCIAL_COSMOS_MODEL=2B/post-trained \
EGOSOCIAL_COSMOS_RESOLUTION=192,320 \
EGOSOCIAL_COSMOS_NUM_STEPS=4 \
EGOSOCIAL_COSMOS_NUM_OUTPUT_FRAMES=9 \
python scripts/smoke_test.py
```

## Run The Server

```bash
cd /root/openenv-hack/egosocial_env
UV_CACHE_DIR=/tmp/uv-cache uv sync
source .venv/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Train A Reason2 Policy With GRPO

The shortest training path is:

- use a `Cosmos-Reason2` checkpoint that already has EgoNormia SFT or instruction tuning
- keep the env in `benchmark` mode first
- train the decision model with GRPO
- keep `Cosmos Predict` as the optional `train`-mode transition backend, not the first reward truth

Install training dependencies:

```bash
cd /root/openenv-hack/egosocial_env
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra train
source .venv/bin/activate
```

Run a minimal benchmark-mode GRPO job:

```bash
cd /root/openenv-hack/egosocial_env
source .venv/bin/activate
python scripts/train_grpo_reason2.py \
  --model-id your-org/your-egonormia-sft-checkpoint \
  --trust-remote-code \
  --bf16 \
  --env-mode benchmark \
  --scene-limit 128 \
  --max-steps 60 \
  --output-dir outputs/grpo_reason2_benchmark
```

By default, training excludes the fixed heldout split at
[`data/splits/verified_split.json`](/root/openenv-hack/egosocial_env/data/splits/verified_split.json).

Optional: switch to the generated-transition path after the benchmark route is stable:

```bash
cd /root/openenv-hack/egosocial_env
source .venv/bin/activate
python scripts/train_grpo_reason2.py \
  --model-id your-org/your-egonormia-sft-checkpoint \
  --trust-remote-code \
  --bf16 \
  --env-mode train \
  --world-model-provider cosmos \
  --scene-limit 64 \
  --max-steps 30 \
  --output-dir outputs/grpo_reason2_trainmode
```

Current script behavior:

- Runs the local `EgosocialEnvironment` directly instead of going through HTTP.
- Uses the env's text observation contract plus local frame paths in the prompt.
- Treats `benchmark` mode as the default and recommended first training target.
- Supports `train` mode, but that path should be considered an extension once the benchmark path is stable.

Evaluate an SFT or GRPO checkpoint on the fixed 200-scene heldout split:

```bash
cd /root/openenv-hack/egosocial_env
source .venv/bin/activate
python scripts/eval_reason2.py \
  --model-id robertzty/EgoNormia-Cosmos-Reason2-2B-v6b-shortcot \
  --trust-remote-code \
  --bf16 \
  --env-mode benchmark \
  --output-path outputs/eval_v6b_verified.json
```

## Client Example

```python
from egosocial_env import EgosocialAction, EgosocialEnv

with EgosocialEnv(base_url="http://localhost:8000") as env:
    first = env.reset(mode="benchmark")
    print(first.observation.phase)
    print(first.observation.env_mode)
    print(first.observation.frame_descriptions)
    print(first.observation.frame_paths)

    second = env.step(
        EgosocialAction(
            predicted_norms=["safety", "politeness", "cooperation"],
            selected_option="A",
            rationale="Make space first, then offer help.",
            proposed_behavior="Move aside, make room, and check whether help is needed.",
        )
    )
    print(second.observation.phase, second.reward)
    print(second.observation.transition_source)

    final = env.step(
        EgosocialAction(
            predicted_norms=["safety", "communication", "cooperation"],
            selected_option="A",
            rationale="The reaction confirms that the safe cooperative option is best.",
            proposed_behavior="Keep the cooperative action and communicate clearly.",
        )
    )
    print(final.done, final.reward, final.observation.metadata["correct"])
```

## Environment Contract

The episode flow is:

1. `reset()` returns the `prev` observation.
2. `step(action_1)` returns the `during` observation and an intermediate norm-matching reward.
3. `step(action_2)` returns the final reward and ends the episode.

Mode-specific transition behavior:

- In `benchmark` mode, `step(action_1)` reveals the fixed `during` clip from the dataset.
- In `train` mode, `step(action_1)` retrieves a proxy `during` clip from a similar local scene and reports retrieval diagnostics such as `retrieved_scene_id`, norm similarity, and text similarity.
- If a world-model provider is configured, `train` mode tries the adapter first, then falls back to retrieval, then to the text-only stub.

Rewards now expose official-task-aligned components:

- Intermediate reward: taxonomy-match score, with an additional transition-consistency factor in `train` mode.
- Final reward: weighted combination of `action_selection`, `sensibility`, `justification_alignment`, and `taxonomy_match`.
- `train` mode additionally scales the final reward by the transition consistency score from retrieval or the stub fallback.
- Early termination: severe safety violation in turn 1 returns `-1.0`.

Observations now expose real-media pointers when available:

- `frame_paths`: local stitched-frame image paths for the active phase.
- `video_paths`: local `prev` and `during` clip paths keyed by phase.
- `env_mode`: whether the episode is running in `benchmark` or `train`.
- `transition_source`: whether the current observation came from dataset replay, retrieval, or the world-model stub fallback.
- `transition_source`: whether the current observation came from dataset replay, retrieval, the mock world-model path, or Cosmos.
- `consistency_checks`: provider-specific diagnostics such as generated paths, retrieval ids, latency, and filter status.

## Files You Will Likely Edit First

- `server/egosocial_env_environment.py`: tune the real EgoNormia loader, reward heuristics, and media packaging.
- `world_model.py`: connect a real provider such as Cosmos behind the stable adapter interface.
- `models.py`: extend action and observation schemas if you switch from local file paths to hosted URLs or tensors.
- `data/sample_episodes.json`: keep this as a fallback fixture for development without the full dataset.
- `openenv.yaml`: keep this for `openenv validate` and `openenv push`.

## Project Structure

```
egosocial_env/
├── __init__.py
├── README.md
├── client.py
├── data/
│   ├── egonormia/
│   └── sample_episodes.json
├── models.py
├── openenv.yaml
├── pyproject.toml
├── scripts/
│   └── smoke_test.py
└── server/
    ├── app.py
    ├── Dockerfile
    └── egosocial_env_environment.py
```

## Validate And Deploy

```bash
cd /root/openenv-hack/egosocial_env
source .venv/bin/activate
UV_CACHE_DIR=/tmp/uv-cache openenv validate
openenv push
```
