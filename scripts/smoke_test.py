#!/usr/bin/env python3

"""Small direct test for the two-turn environment."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import EgosocialAction
from server.egosocial_env_environment import EgosocialEnvironment


def run_episode(env: EgosocialEnvironment, scene_id: str, mode: str) -> None:
    first = env.reset(seed=7, scene_id=scene_id, mode=mode)
    gold_option = env._current_episode["gold_option"]
    gold_norms = env._current_episode["gold_norms"][:3]
    proposed_behavior = env._current_episode["options"][gold_option]
    print(f"RESET ({mode})")
    print(json.dumps(first.model_dump(), indent=2))

    second = env.step(
        EgosocialAction(
            predicted_norms=gold_norms,
            selected_option=gold_option,
            rationale="Initial action matches the norms highlighted in the clip.",
            proposed_behavior=proposed_behavior,
        )
    )
    print(f"\nSTEP 1 ({mode})")
    print(json.dumps(second.model_dump(), indent=2))

    final = env.step(
        EgosocialAction(
            predicted_norms=gold_norms,
            selected_option=gold_option,
            rationale="The reaction clip supports keeping the same socially appropriate choice.",
            proposed_behavior=proposed_behavior,
        )
    )
    print(f"\nSTEP 2 ({mode})")
    print(json.dumps(final.model_dump(), indent=2))
    print(f"\nSTATE ({mode})")
    print(json.dumps(env.state.model_dump(), indent=2))


def main() -> None:
    env = EgosocialEnvironment()
    first_scene_id = env._episodes[0]["scene_id"]
    run_episode(env, first_scene_id, "benchmark")
    print("\n" + ("=" * 80) + "\n")
    run_episode(env, first_scene_id, "train")
    print("\n" + ("=" * 80) + "\n")
    provider = os.environ.get("EGOSOCIAL_WORLD_MODEL_PROVIDER", "mock")
    world_model_env = EgosocialEnvironment(world_model_provider=provider)
    run_episode(world_model_env, first_scene_id, "train")


if __name__ == "__main__":
    main()
