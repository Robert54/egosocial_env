# Future Work

## Scope Freeze

Until the current deadline, keep the project centered on `EgoNormia`.

Use `EgoNormia` as:
- the main benchmark and evaluation anchor
- the source of real `prev` / `during` clips
- the source of social-norm reward components

Do not switch the main dataset before the deadline.

## Current Operating Model

The environment should stay split into two modes:

- `benchmark` mode
  - Uses real `EgoNormia` `prev -> during`
  - This is the stable, reproducible evaluation path
- `train` mode
  - Uses `Cosmos Predict` as the transition generator
  - Falls back to retrieval, then stub, if generation fails

The intended training target is the decision model, not the video model:
- `Cosmos Reason2` or another policy / critic model learns action selection, rationale, and multi-turn behavior
- `Cosmos Predict` acts as the world-model transition backend

## Key Principle For EgoNormia

Do not treat `video_during.mp4` as a passive second clip.

Use `during` as:
- the gold next-state anchor in `benchmark` mode
- the target for `generated_during` alignment checks in `train` mode
- the filter signal for deciding whether a generated transition is good enough to continue a rollout

Recommended hybrid rule:
- Generate `during_generated`
- Compare it against real `during`
- If alignment is too low, either:
  - snap back to real `during`, or
  - terminate the rollout with penalty

## Best Post-Deadline Expansion Order

If more time becomes available, add datasets in this order:

1. `EgoToM`
   - Purpose: belief, goal, and theory-of-mind supervision
   - Why: complements `EgoNormia` norms with hidden-state reasoning

2. `HoloAssist` or `EgoCom`
   - Purpose: real multi-turn interaction trajectories
   - Why: better support for turn-taking, correction, and dialogue flow

3. `Ego4D`
   - Purpose: chain mining and additional raw egocentric video coverage
   - Why: large reservoir for same-scene or same-source extensions

## Minimal Research Direction

The most defensible long-term framing is:

"A hybrid multi-turn social interaction environment anchored by EgoNormia, with real replay for evaluation and world-model rollouts for training."

Avoid stronger claims such as:
- "fully realistic social simulator"
- "native multi-turn dataset"
- "ready to train a video model directly with RL"

## After-Deadline TODOs

1. Add a `during_alignment` score to `reward_breakdown`
2. Add a frozen critic path for generated transitions
3. Add `strict`, `hybrid`, and `full_rollout` train modes
4. Add structured social-state fields such as:
   - `comfort`
   - `confusion`
   - `engagement`
   - `norm_risk`
   - `task_progress`
5. Add `EgoToM` as an auxiliary supervision source
6. Add a real multi-turn dataset layer with `HoloAssist` or `EgoCom`

## What Not To Do Right Now

Do not spend the current deadline window on:
- replacing `EgoNormia` as the core dataset
- retraining `Cosmos Predict`
- adding a large perception stack
- integrating multiple new datasets at once

For now, the correct move is:

`EgoNormia first -> benchmark stable -> Cosmos train mode usable -> submit`
