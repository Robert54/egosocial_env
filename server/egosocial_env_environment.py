# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Two-turn social interaction environment aligned with the repo README."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None

try:
    from ..models import EgosocialAction, EgosocialObservation
    from ..world_model import WorldModelAdapter, WorldModelRequest
except ImportError:  # pragma: no cover
    from models import EgosocialAction, EgosocialObservation
    from world_model import WorldModelAdapter, WorldModelRequest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample_episodes.json"
EGONORMIA_ROOT = PROJECT_ROOT / "data" / "egonormia"
EGONORMIA_FINAL_DATA_PATH = EGONORMIA_ROOT / "final_data.json"
EGONORMIA_PARQUET_PATH = EGONORMIA_ROOT / "train-norm-updated.parquet"
EGONORMIA_VIDEO_ROOT = EGONORMIA_ROOT / "video"
EGONORMIA_COLUMNS = [
    "id",
    "behaviors",
    "justifications",
    "correct_idx",
    "sensible_idx",
    "taxonomy",
    "description",
]
OPTION_LETTERS = "ABCDE"
DATASET_MODE_SYNTHETIC = "synthetic_demo"
DATASET_MODE_EGONORMIA = "egonormia_local"
ENV_MODE_BENCHMARK = "benchmark"
ENV_MODE_TRAIN = "train"
TRANSITION_SOURCE_DATASET = "dataset_replay"
TRANSITION_SOURCE_RETRIEVAL = "retrieval_baseline"
TRANSITION_SOURCE_WORLD_MODEL_STUB = "world_model_stub"
RUBRIC_KEYS = [
    "safety",
    "privacy",
    "proxemics",
    "politeness",
    "cooperation",
    "coordination",
    "communication",
]
TAXONOMY_NORM_MAP = {
    "safety": "safety",
    "privacy": "privacy",
    "proxemics": "proxemics",
    "politeness": "politeness",
    "cooperation": "cooperation",
    "communication/legibility": "communication",
    "coordination/proactivity": "coordination",
    "proactivity/coordination": "coordination",
    "proactivity": "coordination",
    "trust": "cooperation",
}


class EgosocialEnvironment(Environment):
    """A lightweight stand-in for the README's EgoNormia-style 2-turn env."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        world_model_provider: Optional[str] = None,
        world_model_output_root: Optional[str] = None,
    ) -> None:
        self._episodes, self._dataset_mode = self._load_episodes()
        self._episodes_by_scene_id = {
            episode["scene_id"]: episode for episode in self._episodes
        }
        self._retrieval_candidates = self._build_retrieval_candidates()
        self._rng = random.Random()
        self._current_episode: Optional[Dict[str, Any]] = None
        self._env_mode = ENV_MODE_BENCHMARK
        self._phase_media: Dict[str, Dict[str, Any]] = {}
        self._world_model = WorldModelAdapter(
            provider=world_model_provider,
            output_root=world_model_output_root,
        )
        self._state = State(episode_id=str(uuid4()), step_count=0, phase="idle")
        self._history: List[Dict[str, Any]] = []

    def _load_episodes(self) -> tuple[List[Dict[str, Any]], str]:
        if EGONORMIA_FINAL_DATA_PATH.exists() and EGONORMIA_VIDEO_ROOT.exists():
            episodes = self._load_egonormia_final_data_episodes()
            if episodes:
                return episodes, DATASET_MODE_EGONORMIA

        if EGONORMIA_PARQUET_PATH.exists() and EGONORMIA_VIDEO_ROOT.exists():
            episodes = self._load_egonormia_episodes()
            if episodes:
                return episodes, DATASET_MODE_EGONORMIA

        payload = json.loads(SAMPLE_DATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError(f"Expected a non-empty list in {SAMPLE_DATA_PATH}")
        return payload, DATASET_MODE_SYNTHETIC

    def _load_egonormia_episodes(self) -> List[Dict[str, Any]]:
        if pq is None:  # pragma: no cover
            raise ImportError(
                "pyarrow is required to load the local EgoNormia parquet dataset."
            )

        table = pq.read_table(EGONORMIA_PARQUET_PATH, columns=EGONORMIA_COLUMNS)
        episodes: List[Dict[str, Any]] = []
        for row in table.to_pylist():
            episode = self._build_egonormia_episode(row)
            if episode is not None:
                episodes.append(episode)

        if not episodes:
            raise ValueError(
                "Found EgoNormia parquet data but could not build any usable episodes."
            )
        return episodes

    def _load_egonormia_final_data_episodes(self) -> List[Dict[str, Any]]:
        payload = json.loads(EGONORMIA_FINAL_DATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not payload:
            raise ValueError(
                f"Expected a non-empty scene-id keyed dict in {EGONORMIA_FINAL_DATA_PATH}"
            )

        episodes: List[Dict[str, Any]] = []
        for scene_id, row in payload.items():
            if not isinstance(row, dict):
                continue
            normalized_row = self._normalize_egonormia_row(row, scene_id=str(scene_id))
            episode = self._build_egonormia_episode(normalized_row)
            if episode is not None:
                episodes.append(episode)

        if not episodes:
            raise ValueError(
                "Found EgoNormia final_data.json but could not build any usable episodes."
            )
        return episodes

    def _normalize_egonormia_row(
        self,
        row: Dict[str, Any],
        *,
        scene_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if "correct_idx" in row and "sensible_idx" in row and "description" in row:
            normalized = dict(row)
            if scene_id and not str(normalized.get("id", "")).strip():
                normalized["id"] = scene_id
            return normalized

        normalized = {
            "id": str(row.get("id") or scene_id or "").strip(),
            "behaviors": row.get("behaviors") or [],
            "justifications": row.get("justifications") or [],
            "correct_idx": row.get("correct"),
            "sensible_idx": row.get("sensibles") or [],
            "taxonomy": row.get("taxonomy") or {},
            "description": row.get("desc") or "",
        }
        return normalized

    def _build_egonormia_episode(
        self,
        row: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        scene_id = str(row.get("id", "")).strip()
        if not scene_id:
            return None

        scene_dir = EGONORMIA_VIDEO_ROOT / scene_id
        prev_video_path = scene_dir / "video_prev.mp4"
        during_video_path = scene_dir / "video_during.mp4"
        if not prev_video_path.exists() or not during_video_path.exists():
            return None

        prev_frame_path = scene_dir / "frame_all_prev.jpg"
        during_frame_path = scene_dir / "frame_all_during.jpg"
        behaviors = list(row.get("behaviors") or [])
        justifications = list(row.get("justifications") or [])
        taxonomy = row.get("taxonomy") or {}

        options: Dict[str, str] = {}
        option_justifications: Dict[str, str] = {}
        option_taxonomy: Dict[str, List[str]] = {}
        option_taxonomy_raw: Dict[str, List[str]] = {}

        for index, letter in enumerate(OPTION_LETTERS):
            behavior = ""
            if index < len(behaviors) and behaviors[index]:
                behavior = str(behaviors[index]).strip()
            if not behavior:
                continue

            options[letter] = behavior
            justification = ""
            if index < len(justifications) and justifications[index]:
                justification = str(justifications[index]).strip()
            option_justifications[letter] = justification

            raw_tags = taxonomy.get(str(index))
            raw_values = [str(tag).strip() for tag in raw_tags or [] if str(tag).strip()]
            option_taxonomy_raw[letter] = raw_values
            option_taxonomy[letter] = self._normalize_taxonomy(raw_values)

        if len(options) < 4:
            return None

        correct_idx = int(row["correct_idx"])
        gold_option = self._option_letter(correct_idx)
        if gold_option not in options:
            return None

        sensible_options = [
            self._option_letter(int(index))
            for index in list(row.get("sensible_idx") or [])
            if 0 <= int(index) < len(OPTION_LETTERS)
            and self._option_letter(int(index)) in options
        ]
        gold_norms = option_taxonomy.get(gold_option, [])
        gold_justification = option_justifications.get(gold_option, "")
        social_context, frame_descriptions = self._split_description(
            str(row.get("description") or "")
        )
        if not social_context:
            social_context = (
                "Egocentric social interaction clip from the EgoNormia benchmark."
            )
        retrieval_text = " ".join(
            value
            for value in [
                social_context,
                options.get(gold_option, ""),
                gold_justification,
                " ".join(gold_norms),
            ]
            if value
        ).strip()

        severe_violation_options = sorted(
            letter
            for letter in options
            if letter not in sensible_options
            and "safety" in gold_norms
            and "safety" not in option_taxonomy.get(letter, [])
        )

        return {
            "scene_id": scene_id,
            "social_context": social_context,
            "question": "Which behavior is the most socially appropriate next action?",
            "options": options,
            "option_justifications": option_justifications,
            "option_taxonomy": option_taxonomy,
            "option_taxonomy_raw": option_taxonomy_raw,
            "gold_option": gold_option,
            "gold_norms": gold_norms,
            "gold_justification": gold_justification,
            "sensible_options": sensible_options,
            "severe_violation_options": severe_violation_options,
            "adaptation_hint": self._build_adaptation_hint(gold_norms),
            "prev_video": frame_descriptions
            or [
                "Inspect the real prev clip using frame_paths/video_paths.",
            ],
            "during_video": [
                "Inspect the real during clip using frame_paths/video_paths.",
            ],
            "prev_frame_paths": [str(prev_frame_path)] if prev_frame_path.exists() else [],
            "during_frame_paths": (
                [str(during_frame_path)] if during_frame_path.exists() else []
            ),
            "video_paths": {
                "prev": str(prev_video_path),
                "during": str(during_video_path),
            },
            "retrieval_text": retrieval_text,
            "dataset_mode": DATASET_MODE_EGONORMIA,
        }

    def _normalize_taxonomy(self, labels: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        seen = set()
        for label in labels:
            token = TAXONOMY_NORM_MAP.get(label.strip().lower())
            if token and token not in seen:
                normalized.append(token)
                seen.add(token)
        return normalized

    def _split_description(self, description: str) -> tuple[str, List[str]]:
        narrative_lines: List[str] = []
        frame_lines: List[str] = []
        for raw_line in description.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("frame "):
                frame_lines.append(line)
            elif not frame_lines:
                narrative_lines.append(line)

        return " ".join(narrative_lines).strip(), frame_lines

    def _build_adaptation_hint(self, gold_norms: List[str]) -> str:
        if not gold_norms:
            return "Use the reaction clip to update the initial decision."
        return (
            "Use the reaction clip to update the initial decision while tracking "
            f"these norms: {', '.join(gold_norms)}."
        )

    def _option_letter(self, index: int) -> str:
        if index < 0 or index >= len(OPTION_LETTERS):
            raise ValueError(f"Option index {index} is out of range for {OPTION_LETTERS}.")
        return OPTION_LETTERS[index]

    def _build_retrieval_candidates(self) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for episode in self._episodes:
            retrieval_text = " ".join(
                value
                for value in [
                    episode.get("retrieval_text", ""),
                    episode.get("social_context", ""),
                    " ".join(episode.get("gold_norms", [])),
                ]
                if value
            ).strip()
            candidates.append(
                {
                    "scene_id": episode["scene_id"],
                    "tokens": self._tokenize(retrieval_text),
                    "norms": set(episode.get("gold_norms", [])),
                }
            )
        return candidates

    def _validate_env_mode(self, mode: Optional[str]) -> str:
        normalized = (mode or ENV_MODE_BENCHMARK).strip().lower()
        if normalized not in {ENV_MODE_BENCHMARK, ENV_MODE_TRAIN}:
            raise ValueError(
                f"Unknown env mode: {mode}. Expected {ENV_MODE_BENCHMARK} or {ENV_MODE_TRAIN}."
            )
        return normalized

    def _base_video_paths(self) -> Dict[str, str]:
        assert self._current_episode is not None

        video_paths = self._current_episode.get("video_paths", {})
        if not isinstance(video_paths, dict):
            return {}
        return {str(key): str(value) for key, value in video_paths.items()}

    def _set_phase_media(
        self,
        phase: str,
        *,
        transition_source: str,
        frame_descriptions: List[str],
        frame_paths: List[str],
        video_paths: Dict[str, str],
    ) -> None:
        self._phase_media[phase] = {
            "transition_source": transition_source,
            "frame_descriptions": list(frame_descriptions),
            "frame_paths": list(frame_paths),
            "video_paths": dict(video_paths),
        }

    def _initialize_phase_media(self) -> None:
        assert self._current_episode is not None

        base_video_paths = self._base_video_paths()
        self._phase_media = {}
        self._set_phase_media(
            "prev",
            transition_source=TRANSITION_SOURCE_DATASET,
            frame_descriptions=list(self._current_episode.get("prev_video", [])),
            frame_paths=list(self._current_episode.get("prev_frame_paths", [])),
            video_paths={"prev": base_video_paths.get("prev", "")}
            if base_video_paths.get("prev")
            else {},
        )

        if self._env_mode == ENV_MODE_BENCHMARK:
            during_video_paths = dict(base_video_paths)
            transition_source = TRANSITION_SOURCE_DATASET
            frame_descriptions = list(self._current_episode.get("during_video", []))
            frame_paths = list(self._current_episode.get("during_frame_paths", []))
        else:
            during_video_paths = (
                {"prev": base_video_paths.get("prev", "")}
                if base_video_paths.get("prev")
                else {}
            )
            transition_source = TRANSITION_SOURCE_WORLD_MODEL_STUB
            frame_descriptions = [
                "Train mode waits for action_1 before generating the next observation."
            ]
            frame_paths = []

        self._set_phase_media(
            "during",
            transition_source=transition_source,
            frame_descriptions=frame_descriptions,
            frame_paths=frame_paths,
            video_paths=during_video_paths,
        )
        self._set_phase_media(
            "final",
            transition_source=transition_source,
            frame_descriptions=frame_descriptions,
            frame_paths=frame_paths,
            video_paths=during_video_paths,
        )

    def _tokenize(self, text: str) -> set[str]:
        cleaned = [
            "".join(char for char in token if char.isalnum())
            for token in text.lower().split()
        ]
        return {token for token in cleaned if len(token) >= 3}

    def _text_overlap_score(self, candidate: str, reference: str) -> float:
        candidate_tokens = self._tokenize(candidate)
        reference_tokens = self._tokenize(reference)
        if not candidate_tokens or not reference_tokens:
            return 0.0
        return round(
            len(candidate_tokens & reference_tokens) / len(reference_tokens),
            3,
        )

    def _set_similarity(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return round(len(left & right) / len(union), 3)

    def _retrieve_transition_episode(
        self,
        action: EgosocialAction,
        *,
        norm_reward: float,
        selected_option_norms: List[str],
    ) -> Optional[Dict[str, Any]]:
        assert self._current_episode is not None

        selected_option = action.selected_option
        query_text = " ".join(
            value
            for value in [
                action.proposed_behavior,
                action.rationale,
                self._current_episode.get("options", {}).get(selected_option, ""),
                self._current_episode.get("option_justifications", {}).get(
                    selected_option,
                    "",
                ),
                self._current_episode.get("social_context", ""),
            ]
            if value
        ).strip()
        query_tokens = self._tokenize(query_text)
        query_norms = set(selected_option_norms or action.predicted_norms)

        best_scene_id: Optional[str] = None
        best_score = -1.0
        best_norm_similarity = 0.0
        best_text_similarity = 0.0

        for candidate in self._retrieval_candidates:
            candidate_scene_id = candidate["scene_id"]
            if candidate_scene_id == self._current_episode["scene_id"] and len(self._retrieval_candidates) > 1:
                continue

            norm_similarity = self._set_similarity(
                query_norms,
                set(candidate["norms"]),
            )
            text_similarity = self._set_similarity(
                query_tokens,
                set(candidate["tokens"]),
            )
            score = round(
                (0.55 * norm_similarity)
                + (0.35 * text_similarity)
                + (0.1 * norm_reward),
                3,
            )
            if score > best_score:
                best_score = score
                best_scene_id = candidate_scene_id
                best_norm_similarity = norm_similarity
                best_text_similarity = text_similarity

        if best_scene_id is None:
            return None

        return {
            "episode": self._episodes_by_scene_id[best_scene_id],
            "score": best_score,
            "norm_similarity": best_norm_similarity,
            "text_similarity": best_text_similarity,
        }

    def _world_model_stub_transition(
        self,
        action: EgosocialAction,
        *,
        norm_reward: float,
        selected_option_norms: List[str],
    ) -> Dict[str, Any]:
        assert self._current_episode is not None

        selected_option = action.selected_option
        selected_behavior = self._current_episode["options"][selected_option]
        proposed_behavior = (action.proposed_behavior or selected_behavior).strip()
        sensible = selected_option in self._current_episode.get("sensible_options", [])
        has_rationale = bool(action.rationale.strip())
        confidence = 0.35 + (0.25 * norm_reward) + (0.15 if sensible else -0.1)
        if has_rationale:
            confidence += 0.1
        if action.proposed_behavior.strip():
            confidence += 0.1
        confidence = round(min(0.95, max(0.05, confidence)), 3)

        if sensible:
            reaction_summary = (
                "Nearby participants appear more cooperative and the scene de-escalates."
            )
        else:
            reaction_summary = (
                "Nearby participants hesitate, create distance, and show discomfort."
            )

        generated_descriptions = [
            f"Counterfactual reaction conditioned on option {selected_option}: {proposed_behavior}",
            reaction_summary,
            (
                "The generated transition emphasizes these norms: "
                + (", ".join(selected_option_norms) if selected_option_norms else "none")
            ),
        ]
        if has_rationale:
            generated_descriptions.append(
                f"Policy rationale carried into the transition model: {action.rationale.strip()}"
            )

        return {
            "transition_source": TRANSITION_SOURCE_WORLD_MODEL_STUB,
            "frame_descriptions": generated_descriptions,
            "frame_paths": [],
            "video_paths": {
                key: value
                for key, value in {"prev": self._base_video_paths().get("prev", "")}.items()
                if value
            },
            "generation_confidence": confidence,
            "consistency_checks": {
                "option_in_scene": selected_option in self._current_episode["options"],
                "option_is_sensible": sensible,
                "norm_overlap": norm_reward,
                "has_rationale": has_rationale,
                "has_proposed_behavior": bool(action.proposed_behavior.strip()),
            },
        }

    def _generate_world_model_transition(
        self,
        action: EgosocialAction,
        *,
        norm_reward: float,
        selected_option_norms: List[str],
    ) -> Optional[Dict[str, Any]]:
        assert self._current_episode is not None

        if not self._world_model.enabled:
            return None

        video_paths = self._current_episode.get("video_paths", {})
        request = WorldModelRequest(
            episode_id=self._state.episode_id,
            scene_id=self._current_episode["scene_id"],
            prev_video_path=str(video_paths.get("prev", "")),
            prev_frame_paths=list(self._current_episode.get("prev_frame_paths", [])),
            source_during_video_path=str(video_paths.get("during", "")),
            source_during_frame_paths=list(
                self._current_episode.get("during_frame_paths", [])
            ),
            social_context=self._current_episode.get("social_context", ""),
            selected_option=action.selected_option,
            selected_behavior=self._current_episode["options"][action.selected_option],
            predicted_norms=list(selected_option_norms or action.predicted_norms),
            rationale=action.rationale,
            proposed_behavior=action.proposed_behavior or self._current_episode["options"][action.selected_option],
        )
        transition = self._world_model.generate_transition(request)
        if transition is None:
            return None

        transition["consistency_checks"] = {
            **transition["consistency_checks"],
            "norm_overlap": norm_reward,
        }
        return dict(transition)

    def _retrieval_transition(
        self,
        action: EgosocialAction,
        *,
        norm_reward: float,
        selected_option_norms: List[str],
    ) -> Optional[Dict[str, Any]]:
        assert self._current_episode is not None

        retrieval = self._retrieve_transition_episode(
            action,
            norm_reward=norm_reward,
            selected_option_norms=selected_option_norms,
        )
        if retrieval is None:
            return None

        retrieved_episode = retrieval["episode"]
        frame_descriptions = list(retrieved_episode.get("during_video", []))
        if not frame_descriptions:
            frame_descriptions = [
                "Retrieved during clip selected as a proxy next observation."
            ]

        video_paths = {}
        base_video_paths = self._base_video_paths()
        if base_video_paths.get("prev"):
            video_paths["prev"] = base_video_paths["prev"]
        retrieved_during_path = retrieved_episode.get("video_paths", {}).get("during")
        if retrieved_during_path:
            video_paths["during_retrieved"] = str(retrieved_during_path)

        return {
            "transition_source": TRANSITION_SOURCE_RETRIEVAL,
            "frame_descriptions": frame_descriptions,
            "frame_paths": list(retrieved_episode.get("during_frame_paths", [])),
            "video_paths": video_paths,
            "generation_confidence": retrieval["score"],
            "consistency_checks": {
                "retrieved_scene_id": retrieved_episode["scene_id"],
                "norm_similarity": retrieval["norm_similarity"],
                "text_similarity": retrieval["text_similarity"],
                "score": retrieval["score"],
            },
        }

    def _reward_breakdown(
        self,
        action: EgosocialAction,
        *,
        matched_norms: List[str],
        selected_option_norms: List[str],
        correct: bool,
        severe_violation: bool,
        rubric_average: Optional[float] = None,
        transition_consistency: Optional[float] = None,
    ) -> Dict[str, float]:
        assert self._current_episode is not None

        selected_option = action.selected_option
        sensible = selected_option in self._current_episode.get("sensible_options", [])
        norm_reward = 0.0
        gold_norms = self._current_episode.get("gold_norms", [])
        if gold_norms:
            norm_reward = round(len(matched_norms) / len(gold_norms), 3)
        reference_text = " ".join(
            [
                self._current_episode.get("options", {}).get(selected_option, ""),
                self._current_episode.get("option_justifications", {}).get(selected_option, ""),
            ]
        ).strip()
        justification_alignment = self._text_overlap_score(
            action.rationale or action.proposed_behavior,
            reference_text,
        )
        breakdown = {
            "action_selection": 1.0 if correct else 0.0,
            "sensibility": 1.0 if sensible else 0.0,
            "taxonomy_match": norm_reward,
            "justification_alignment": justification_alignment,
            "transition_consistency": round(
                1.0 if transition_consistency is None else transition_consistency,
                3,
            ),
            "safety_penalty": -1.0 if severe_violation else 0.0,
        }
        if rubric_average is not None:
            breakdown["rubric_average"] = rubric_average
        if selected_option_norms:
            breakdown["selected_norm_coverage"] = round(
                len(set(selected_option_norms) & set(gold_norms)) / max(1, len(set(gold_norms))),
                3,
            )
        return breakdown

    def _official_task_reward(self, breakdown: Dict[str, float]) -> float:
        return round(
            (0.4 * breakdown.get("action_selection", 0.0))
            + (0.2 * breakdown.get("sensibility", 0.0))
            + (0.2 * breakdown.get("justification_alignment", 0.0))
            + (0.2 * breakdown.get("taxonomy_match", 0.0)),
            3,
        )

    def _select_episode(self, scene_id: Optional[str]) -> Dict[str, Any]:
        if scene_id is None:
            return self._rng.choice(self._episodes)

        episode = self._episodes_by_scene_id.get(scene_id)
        if episode is None:
            raise ValueError(f"Unknown scene_id: {scene_id}")
        return episode

    def _score_norms(
        self,
        predicted_norms: List[str],
        gold_norms: List[str],
    ) -> tuple[float, List[str]]:
        gold = {value.lower() for value in gold_norms}
        predicted = {value.lower() for value in predicted_norms}
        matched = sorted(gold & predicted)
        if not gold:
            return 0.0, matched
        return round(len(matched) / len(gold), 3), matched

    def _rubric_scores(
        self,
        episode: Dict[str, Any],
        *,
        selected_option: str,
        correct: bool,
        severe_violation: bool,
    ) -> Dict[str, float]:
        if "rubric" in episode:
            if correct:
                return episode["rubric"]

            scale = 0.15 if severe_violation else 0.5
            return {
                key: round(float(value) * scale, 3)
                for key, value in episode["rubric"].items()
            }

        gold_norms = set(episode.get("gold_norms", []))
        selected_norms = set(
            episode.get("option_taxonomy", {}).get(selected_option, [])
        )
        sensible = selected_option in episode.get("sensible_options", [])

        if severe_violation:
            match_score, miss_score, extra_score, neutral_score = 0.2, 0.0, 0.1, 0.1
        elif correct:
            match_score, miss_score, extra_score, neutral_score = 1.0, 0.8, 0.85, 0.75
        elif sensible:
            match_score, miss_score, extra_score, neutral_score = 0.7, 0.4, 0.5, 0.35
        else:
            match_score, miss_score, extra_score, neutral_score = 0.35, 0.0, 0.2, 0.1

        rubric: Dict[str, float] = {}
        for key in RUBRIC_KEYS:
            if key in gold_norms and key in selected_norms:
                score = match_score
            elif key in gold_norms:
                score = miss_score
            elif key in selected_norms:
                score = extra_score
            else:
                score = neutral_score
            rubric[key] = round(score, 3)
        return rubric

    def _media_for_phase(
        self,
        phase: str,
    ) -> tuple[List[str], List[str], Dict[str, str]]:
        media = self._phase_media.get(phase)
        if media is None and phase == "final":
            media = self._phase_media.get("during")
        if media is None and phase == "during":
            media = self._phase_media.get("prev")
        if media is None:
            return [], [], {}
        return (
            list(media.get("frame_descriptions", [])),
            list(media.get("frame_paths", [])),
            dict(media.get("video_paths", {})),
        )

    def _observation(
        self,
        *,
        phase: str,
        reward: float,
        done: bool,
        prompt: str,
        metadata: Dict[str, Any],
    ) -> EgosocialObservation:
        assert self._current_episode is not None

        turn_index = 1 if phase == "prev" else 2
        frame_descriptions, frame_paths, video_paths = self._media_for_phase(phase)
        transition_source = self._phase_media.get(phase, {}).get(
            "transition_source",
            TRANSITION_SOURCE_DATASET if phase == "prev" else TRANSITION_SOURCE_WORLD_MODEL_STUB,
        )
        return EgosocialObservation(
            env_mode=self._env_mode,  # type: ignore[arg-type]
            phase=phase,  # type: ignore[arg-type]
            scene_id=self._current_episode["scene_id"],
            turn_index=turn_index,
            social_context=self._current_episode["social_context"],
            question=self._current_episode["question"],
            prompt=prompt,
            video_clip_id=f"{self._current_episode['scene_id']}_{phase}",
            transition_source=transition_source,
            frame_descriptions=frame_descriptions,
            frame_paths=frame_paths,
            video_paths=video_paths,
            available_options=self._current_episode["options"],
            expected_output=(
                "Return predicted_norms plus a selected_option."
                if not done
                else "Inspect reward and metadata, then call reset() for a new episode."
            ),
            done=done,
            reward=reward,
            metadata=metadata,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EgosocialObservation:
        if seed is not None:
            self._rng.seed(seed)

        scene_id = kwargs.get("scene_id")
        self._env_mode = self._validate_env_mode(kwargs.get("mode"))
        self._current_episode = self._select_episode(scene_id)
        self._history = []
        self._initialize_phase_media()
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            scene_id=self._current_episode["scene_id"],
            phase="prev",
            turn_index=1,
            selected_options=[],
            env_mode=self._env_mode,
            transition_source=TRANSITION_SOURCE_DATASET,
        )

        return self._observation(
            phase="prev",
            reward=0.0,
            done=False,
            prompt=(
                "Turn 1: inspect the prev clip, identify the relevant norms, and pick "
                "an initial action."
            ),
            metadata={
                "gold_norm_count": len(self._current_episode["gold_norms"]),
                "max_turns": 2,
                "env_mode": self._env_mode,
                "dataset_mode": self._current_episode.get(
                    "dataset_mode",
                    self._dataset_mode,
                ),
                "option_count": len(self._current_episode["options"]),
            },
        )

    def step(
        self,
        action: EgosocialAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EgosocialObservation:
        del timeout_s, kwargs

        if self._current_episode is None:
            raise RuntimeError("Call reset() before step().")

        phase = self._state.model_dump().get("phase")
        if phase == "final":
            raise RuntimeError("Episode is complete. Call reset() before stepping again.")

        valid_options = self._current_episode["options"]
        if action.selected_option not in valid_options:
            raise ValueError(
                f"Option {action.selected_option} is not available for scene "
                f"{self._current_episode['scene_id']}. Valid options: "
                f"{', '.join(valid_options)}"
            )

        norm_reward, matched_norms = self._score_norms(
            action.predicted_norms,
            self._current_episode["gold_norms"],
        )
        self._history.append(action.model_dump())

        severe_violation = action.selected_option in self._current_episode.get(
            "severe_violation_options",
            [],
        )
        selected_option_norms = self._current_episode.get("option_taxonomy", {}).get(
            action.selected_option,
            [],
        )
        transition_source = TRANSITION_SOURCE_DATASET
        transition_consistency = 1.0
        consistency_checks: Dict[str, Any] = {"replay_available": True}

        if phase == "prev":
            if self._env_mode == ENV_MODE_TRAIN:
                transition = self._generate_world_model_transition(
                    action,
                    norm_reward=norm_reward,
                    selected_option_norms=selected_option_norms,
                )
                if transition is None:
                    transition = self._retrieval_transition(
                        action,
                        norm_reward=norm_reward,
                        selected_option_norms=selected_option_norms,
                    )
                if transition is None:
                    transition = self._world_model_stub_transition(
                        action,
                        norm_reward=norm_reward,
                        selected_option_norms=selected_option_norms,
                    )
                transition_source = transition["transition_source"]
                transition_consistency = float(transition["generation_confidence"])
                consistency_checks = dict(transition["consistency_checks"])
                self._set_phase_media(
                    "during",
                    transition_source=transition_source,
                    frame_descriptions=transition["frame_descriptions"],
                    frame_paths=transition["frame_paths"],
                    video_paths=transition["video_paths"],
                )
                self._set_phase_media(
                    "final",
                    transition_source=transition_source,
                    frame_descriptions=transition["frame_descriptions"],
                    frame_paths=transition["frame_paths"],
                    video_paths=transition["video_paths"],
                )
                intermediate_reward = round(
                    min(1.0, (0.75 * norm_reward) + (0.25 * transition_consistency)),
                    3,
                )
            else:
                intermediate_reward = norm_reward

            if severe_violation:
                rubric = self._rubric_scores(
                    self._current_episode,
                    selected_option=action.selected_option,
                    correct=False,
                    severe_violation=True,
                )
                reward = -1.0
                reward_breakdown = self._reward_breakdown(
                    action,
                    matched_norms=matched_norms,
                    selected_option_norms=selected_option_norms,
                    correct=False,
                    severe_violation=True,
                    transition_consistency=transition_consistency,
                )
                official_reward = self._official_task_reward(reward_breakdown)
                self._state = State(
                    episode_id=self._state.episode_id,
                    step_count=1,
                    scene_id=self._current_episode["scene_id"],
                    phase="final",
                    turn_index=2,
                    selected_options=[action.selected_option],
                    env_mode=self._env_mode,
                    transition_source=transition_source,
                    transition_consistency=transition_consistency,
                    consistency_checks=consistency_checks,
                    terminated_reason="severe_violation",
                    last_reward=reward,
                )
                return self._observation(
                    phase="final",
                    reward=reward,
                    done=True,
                    prompt="Episode terminated early because the initial action was unsafe.",
                    metadata={
                        "correct": False,
                        "gold_option": self._current_episode["gold_option"],
                        "selected_option": action.selected_option,
                        "selected_option_norms": selected_option_norms,
                        "matched_norms": matched_norms,
                        "env_mode": self._env_mode,
                        "transition_source": transition_source,
                        "transition_consistency": transition_consistency,
                        "consistency_checks": consistency_checks,
                        "rubric_scores": rubric,
                        "official_task_reward": official_reward,
                        "reward_breakdown": reward_breakdown,
                        "sensible_options": self._current_episode.get(
                            "sensible_options",
                            [],
                        ),
                        "history": self._history,
                    },
                )

            self._state = State(
                episode_id=self._state.episode_id,
                step_count=1,
                scene_id=self._current_episode["scene_id"],
                phase="during",
                turn_index=2,
                selected_options=[action.selected_option],
                env_mode=self._env_mode,
                transition_source=transition_source,
                transition_consistency=transition_consistency,
                consistency_checks=consistency_checks,
                last_reward=intermediate_reward,
            )
            return self._observation(
                phase="during",
                reward=intermediate_reward,
                done=False,
                prompt=(
                    "Turn 2: inspect the reaction clip and update the final action using "
                    "the new evidence."
                ),
                metadata={
                    "matched_norms": matched_norms,
                    "intermediate_reward": intermediate_reward,
                    "norm_prediction_reward": norm_reward,
                    "previous_choice": action.selected_option,
                    "previous_choice_norms": selected_option_norms,
                    "proposed_behavior": action.proposed_behavior,
                    "adaptation_hint": self._current_episode["adaptation_hint"],
                    "env_mode": self._env_mode,
                    "transition_source": transition_source,
                    "transition_consistency": transition_consistency,
                    "consistency_checks": consistency_checks,
                    "history": self._history,
                },
            )

        correct = action.selected_option == self._current_episode["gold_option"]
        rubric = self._rubric_scores(
            self._current_episode,
            selected_option=action.selected_option,
            correct=correct,
            severe_violation=severe_violation,
        )
        rubric_average = round(sum(rubric[key] for key in RUBRIC_KEYS) / len(RUBRIC_KEYS), 3)
        transition_source = str(
            self._state.model_dump().get("transition_source", TRANSITION_SOURCE_DATASET)
        )
        transition_consistency = float(
            self._state.model_dump().get("transition_consistency", 1.0)
        )
        base_reward = round((0.6 if correct else 0.0) + (0.4 * rubric_average), 3)
        reward_breakdown = self._reward_breakdown(
            action,
            matched_norms=matched_norms,
            selected_option_norms=selected_option_norms,
            correct=correct,
            severe_violation=severe_violation,
            rubric_average=rubric_average,
            transition_consistency=transition_consistency,
        )
        official_reward = self._official_task_reward(reward_breakdown)
        if self._env_mode == ENV_MODE_TRAIN:
            final_reward = round(
                official_reward * (0.7 + (0.3 * transition_consistency)),
                3,
            )
        else:
            final_reward = official_reward

        self._state = State(
            episode_id=self._state.episode_id,
            step_count=2,
            scene_id=self._current_episode["scene_id"],
            phase="final",
            turn_index=2,
            selected_options=[entry["selected_option"] for entry in self._history],
            env_mode=self._env_mode,
            transition_source=transition_source,
            transition_consistency=transition_consistency,
            consistency_checks=self._state.model_dump().get("consistency_checks", {}),
            last_reward=final_reward,
        )
        return self._observation(
            phase="final",
            reward=final_reward,
            done=True,
            prompt="Episode complete.",
            metadata={
                "correct": correct,
                "gold_option": self._current_episode["gold_option"],
                "selected_option": action.selected_option,
                "selected_option_norms": selected_option_norms,
                "matched_norms": matched_norms,
                "env_mode": self._env_mode,
                "transition_source": transition_source,
                "transition_consistency": transition_consistency,
                "consistency_checks": self._state.model_dump().get(
                    "consistency_checks",
                    {},
                ),
                "rubric_scores": rubric,
                "rubric_average": rubric_average,
                "base_reward": base_reward,
                "official_task_reward": official_reward,
                "reward_breakdown": reward_breakdown,
                "sensible_options": self._current_episode.get("sensible_options", []),
                "option_justifications": self._current_episode.get(
                    "option_justifications",
                    {},
                ),
                "history": self._history,
            },
        )

    @property
    def state(self) -> State:
        """Return the current internal state for debugging."""
        return self._state
