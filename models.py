# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Egosocial environment."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import Field, field_validator

from openenv.core.env_server.types import Action, Observation


class EgosocialAction(Action):
    """Action for the two-turn social interaction environment."""

    predicted_norms: List[str] = Field(
        default_factory=list,
        description="Predicted social norms that matter in the current scene.",
    )
    selected_option: str = Field(
        ...,
        description="Selected multiple-choice option. Must be one of the available A-E options.",
    )
    rationale: str = Field(
        default="",
        description="Short natural-language explanation for the selected option.",
    )
    proposed_behavior: str = Field(
        default="",
        description="Free-form action text that a world model could condition on.",
    )

    @field_validator("selected_option")
    @classmethod
    def validate_selected_option(cls, value: str) -> str:
        normalized = value.strip().upper()
        if normalized not in {"A", "B", "C", "D", "E"}:
            raise ValueError("selected_option must be one of A, B, C, D or E")
        return normalized

    @field_validator("predicted_norms")
    @classmethod
    def normalize_norms(cls, values: List[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for value in values:
            token = value.strip().lower().replace(" ", "_")
            if not token or token in seen:
                continue
            normalized.append(token)
            seen.add(token)
        return normalized


class EgosocialObservation(Observation):
    """Observation emitted by the two-turn social interaction environment."""

    env_mode: Literal["benchmark", "train"] = Field(
        default="benchmark",
        description="Whether the env is replaying benchmark data or using a train-time transition model.",
    )
    phase: Literal["prev", "during", "final"] = Field(
        default="prev",
        description="Current episode phase.",
    )
    scene_id: str = Field(default="", description="Stable identifier for the scene.")
    turn_index: int = Field(default=1, ge=1, description="Current decision turn.")
    social_context: str = Field(
        default="",
        description="Short text summary of the social setting.",
    )
    question: str = Field(
        default="",
        description="MCQ question that the policy is answering.",
    )
    prompt: str = Field(
        default="",
        description="Turn-specific instruction shown to the policy.",
    )
    video_clip_id: str = Field(
        default="",
        description="Identifier for the current clip chunk.",
    )
    transition_source: str = Field(
        default="dataset_replay",
        description="How the current observation was produced.",
    )
    frame_descriptions: List[str] = Field(
        default_factory=list,
        description="Text descriptions for the current clip when available.",
    )
    frame_paths: List[str] = Field(
        default_factory=list,
        description="Local image paths for the current clip context.",
    )
    video_paths: Dict[str, str] = Field(
        default_factory=dict,
        description="Local video paths keyed by phase, such as prev and during.",
    )
    available_options: Dict[str, str] = Field(
        default_factory=dict,
        description="Multiple-choice options available to the policy.",
    )
    expected_output: str = Field(
        default="",
        description="What the policy should return for this phase.",
    )
