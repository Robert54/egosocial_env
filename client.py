# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Egosocial environment client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import EgosocialAction, EgosocialObservation


class EgosocialEnv(
    EnvClient[EgosocialAction, EgosocialObservation, State]
):
    """Client for the two-turn social interaction environment."""

    def _step_payload(self, action: EgosocialAction) -> Dict[str, Any]:
        """Convert an action model into the JSON payload expected by the server."""
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[EgosocialObservation]:
        """Parse a server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = EgosocialObservation(
            env_mode=obs_data.get("env_mode", "benchmark"),
            phase=obs_data.get("phase", "prev"),
            scene_id=obs_data.get("scene_id", ""),
            turn_index=obs_data.get("turn_index", 1),
            social_context=obs_data.get("social_context", ""),
            question=obs_data.get("question", ""),
            prompt=obs_data.get("prompt", ""),
            video_clip_id=obs_data.get("video_clip_id", ""),
            transition_source=obs_data.get("transition_source", "dataset_replay"),
            frame_descriptions=obs_data.get("frame_descriptions", []),
            frame_paths=obs_data.get("frame_paths", []),
            video_paths=obs_data.get("video_paths", {}),
            available_options=obs_data.get("available_options", {}),
            expected_output=obs_data.get("expected_output", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state while preserving custom fields."""
        return State(**payload)
