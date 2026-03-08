# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adapter layer for action-conditioned world-model transitions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict


WORLD_MODEL_PROVIDER_DISABLED = "disabled"
WORLD_MODEL_PROVIDER_MOCK = "mock"
WORLD_MODEL_PROVIDER_COSMOS = "cosmos"
TRANSITION_SOURCE_WORLD_MODEL_MOCK = "world_model_mock"
TRANSITION_SOURCE_WORLD_MODEL_COSMOS = "world_model_cosmos"
DEFAULT_OUTPUT_ROOT = Path("/tmp/egosocial_env/generated")
DEFAULT_COSMOS_REPO = Path("/root/cosmos-predict2.5-main")


class WorldModelTransition(TypedDict):
    """Normalized transition payload returned to the environment."""

    transition_source: str
    frame_descriptions: List[str]
    frame_paths: List[str]
    video_paths: Dict[str, str]
    generation_confidence: float
    consistency_checks: Dict[str, Any]


@dataclass
class WorldModelRequest:
    """Inputs required to generate the next observation."""

    episode_id: str
    scene_id: str
    prev_video_path: str
    prev_frame_paths: List[str] = field(default_factory=list)
    source_during_video_path: str = ""
    source_during_frame_paths: List[str] = field(default_factory=list)
    social_context: str = ""
    selected_option: str = ""
    selected_behavior: str = ""
    predicted_norms: List[str] = field(default_factory=list)
    rationale: str = ""
    proposed_behavior: str = ""


class WorldModelAdapter:
    """Provider wrapper used by the environment to request transitions."""

    def __init__(
        self,
        provider: Optional[str] = None,
        output_root: Optional[str] = None,
    ) -> None:
        resolved_provider = (
            provider
            or os.environ.get("EGOSOCIAL_WORLD_MODEL_PROVIDER")
            or WORLD_MODEL_PROVIDER_DISABLED
        )
        self.provider = resolved_provider.strip().lower()
        self.output_root = Path(
            output_root
            or os.environ.get("EGOSOCIAL_WORLD_MODEL_OUTPUT_ROOT")
            or DEFAULT_OUTPUT_ROOT
        )

    @property
    def enabled(self) -> bool:
        """Whether a concrete provider is configured."""
        return self.provider != WORLD_MODEL_PROVIDER_DISABLED

    def generate_transition(
        self,
        request: WorldModelRequest,
    ) -> Optional[WorldModelTransition]:
        """Generate an action-conditioned transition or return None when disabled."""

        if self.provider == WORLD_MODEL_PROVIDER_DISABLED:
            return None
        if self.provider == WORLD_MODEL_PROVIDER_MOCK:
            return self._generate_mock_transition(request)
        if self.provider == WORLD_MODEL_PROVIDER_COSMOS:
            return self._generate_cosmos_transition(request)
        raise ValueError(f"Unsupported world model provider: {self.provider}")

    def _generate_mock_transition(
        self,
        request: WorldModelRequest,
    ) -> WorldModelTransition:
        """Materialize a generated-looking artifact bundle for local testing."""

        started = time.time()
        run_dir = self.output_root / request.episode_id
        run_dir.mkdir(parents=True, exist_ok=True)

        generated_video_path = run_dir / "during_generated.mp4"
        generated_frame_path = run_dir / "frame_all_during.jpg"
        metadata_path = run_dir / "metadata.json"

        if request.source_during_video_path:
            shutil.copy2(request.source_during_video_path, generated_video_path)
        elif request.prev_video_path:
            shutil.copy2(request.prev_video_path, generated_video_path)

        source_frame = ""
        if request.source_during_frame_paths:
            source_frame = request.source_during_frame_paths[0]
        elif request.prev_frame_paths:
            source_frame = request.prev_frame_paths[0]
        if source_frame:
            shutil.copy2(source_frame, generated_frame_path)

        metadata = {
            "provider": self.provider,
            "request": asdict(request),
            "created_at_s": round(time.time(), 3),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        frame_descriptions = [
            (
                "Mock world model conditioned on option "
                f"{request.selected_option}: "
                f"{request.proposed_behavior or request.selected_behavior}"
            ),
            (
                "Predicted norms: "
                + (", ".join(request.predicted_norms) if request.predicted_norms else "none")
            ),
        ]
        if request.rationale.strip():
            frame_descriptions.append(f"Rationale: {request.rationale.strip()}")

        latency_ms = int((time.time() - started) * 1000)
        frame_paths = [str(generated_frame_path)] if generated_frame_path.exists() else []
        video_paths = {}
        if request.prev_video_path:
            video_paths["prev"] = request.prev_video_path
        if generated_video_path.exists():
            video_paths["during_generated"] = str(generated_video_path)

        return {
            "transition_source": TRANSITION_SOURCE_WORLD_MODEL_MOCK,
            "frame_descriptions": frame_descriptions,
            "frame_paths": frame_paths,
            "video_paths": video_paths,
            "generation_confidence": 0.92,
            "consistency_checks": {
                "provider": self.provider,
                "generated_path": str(generated_video_path),
                "metadata_path": str(metadata_path),
                "latency_ms": latency_ms,
                "safety_filter_passed": True,
            },
        }

    def _generate_cosmos_transition(
        self,
        request: WorldModelRequest,
    ) -> Optional[WorldModelTransition]:
        """Run Cosmos Predict Video2World against the local prev clip."""

        started = time.time()
        run_dir = self.output_root / request.episode_id / "cosmos"
        run_dir.mkdir(parents=True, exist_ok=True)

        repo_root = Path(
            os.environ.get("EGOSOCIAL_COSMOS_REPO", str(DEFAULT_COSMOS_REPO))
        )
        python_path = Path(
            os.environ.get(
                "EGOSOCIAL_COSMOS_PYTHON",
                str(repo_root / ".venv" / "bin" / "python"),
            )
        )
        if not repo_root.exists() or not python_path.exists():
            self._write_failure_marker(
                run_dir / "failure.json",
                {
                    "provider": self.provider,
                    "reason": "cosmos_runtime_missing",
                    "repo_root": str(repo_root),
                    "python_path": str(python_path),
                },
            )
            return None

        input_json_path = run_dir / "input.json"
        log_path = run_dir / "cosmos.log"
        metadata_path = run_dir / "metadata.json"
        output_dir = run_dir / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        sample_name = "during_generated"
        prompt = self._build_cosmos_prompt(request)
        input_video_path = self._prepare_cosmos_input_video(
            request=request,
            repo_root=repo_root,
            python_path=python_path,
            run_dir=run_dir,
        )
        if input_video_path is None:
            self._write_failure_marker(
                run_dir / "failure.json",
                {
                    "provider": self.provider,
                    "reason": "cosmos_input_preprocess_failed",
                    "repo_root": str(repo_root),
                    "python_path": str(python_path),
                },
            )
            return None

        resolution = os.environ.get("EGOSOCIAL_COSMOS_RESOLUTION", "192,320")
        num_steps = max(1, int(os.environ.get("EGOSOCIAL_COSMOS_NUM_STEPS", "4")))
        num_output_frames = max(
            9, int(os.environ.get("EGOSOCIAL_COSMOS_NUM_OUTPUT_FRAMES", "9"))
        )
        seed = int(os.environ.get("EGOSOCIAL_COSMOS_SEED", "7"))
        disable_guardrails = (
            os.environ.get("EGOSOCIAL_COSMOS_DISABLE_GUARDRAILS", "1").strip().lower()
            not in {"0", "false", "no"}
        )
        model_name = os.environ.get(
            "EGOSOCIAL_COSMOS_MODEL",
            "2B/post-trained",
        )
        timeout_s = max(
            60, int(os.environ.get("EGOSOCIAL_COSMOS_TIMEOUT_S", "3600"))
        )

        input_payload = {
            "inference_type": "video2world",
            "name": sample_name,
            "prompt": prompt,
            "input_path": str(input_video_path),
            "resolution": resolution,
            "num_output_frames": num_output_frames,
            "num_steps": num_steps,
            "seed": seed,
        }
        input_json_path.write_text(
            json.dumps(input_payload, indent=2),
            encoding="utf-8",
        )

        command = [
            str(python_path),
            "examples/inference.py",
            "-i",
            str(input_json_path),
            "-o",
            str(output_dir),
            "--model",
            model_name,
            "--inference-type",
            "video2world",
        ]
        if disable_guardrails:
            command.append("--disable-guardrails")

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["LD_LIBRARY_PATH"] = self._merge_library_paths(
            env.get("LD_LIBRARY_PATH", ""),
            self._cosmos_library_paths(repo_root),
        )

        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except Exception as exc:
            self._write_failure_marker(
                run_dir / "failure.json",
                {
                    "provider": self.provider,
                    "reason": "cosmos_subprocess_failed",
                    "error": repr(exc),
                    "command": command,
                },
            )
            return None

        log_path.write_text(
            "\n".join(
                [
                    "$ " + " ".join(command),
                    "",
                    completed.stdout or "",
                    completed.stderr or "",
                ]
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        generated_video_path = output_dir / f"{sample_name}.mp4"
        if completed.returncode != 0 or not generated_video_path.exists():
            self._write_failure_marker(
                run_dir / "failure.json",
                {
                    "provider": self.provider,
                    "reason": "cosmos_inference_failed",
                    "returncode": completed.returncode,
                    "command": command,
                    "log_path": str(log_path),
                    "generated_video_path": str(generated_video_path),
                },
            )
            return None

        generated_frame_path = self._copy_best_frame(
            request=request,
            destination=run_dir / "frame_all_during.jpg",
        )
        video_paths = {}
        if request.prev_video_path:
            video_paths["prev"] = request.prev_video_path
        video_paths["during_generated"] = str(generated_video_path)

        metadata = {
            "provider": self.provider,
            "command": command,
            "repo_root": str(repo_root),
            "python_path": str(python_path),
            "request": asdict(request),
            "input_payload": input_payload,
            "created_at_s": round(time.time(), 3),
            "returncode": completed.returncode,
            "log_path": str(log_path),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        latency_ms = int((time.time() - started) * 1000)
        frame_paths = [str(generated_frame_path)] if generated_frame_path else []
        return {
            "transition_source": TRANSITION_SOURCE_WORLD_MODEL_COSMOS,
            "frame_descriptions": [
                "Cosmos Predict Video2World rollout conditioned on the current scene and selected behavior.",
                f"Selected option {request.selected_option}: {request.proposed_behavior or request.selected_behavior}",
                (
                    "Predicted norms: "
                    + (
                        ", ".join(request.predicted_norms)
                        if request.predicted_norms
                        else "none"
                    )
                ),
            ],
            "frame_paths": frame_paths,
            "video_paths": video_paths,
            "generation_confidence": 0.9,
            "consistency_checks": {
                "provider": self.provider,
                "model": model_name,
                "seed": seed,
                "num_steps": num_steps,
                "num_output_frames": num_output_frames,
                "resolution": resolution,
                "input_video_path": str(input_video_path),
                "generated_path": str(generated_video_path),
                "metadata_path": str(metadata_path),
                "log_path": str(log_path),
                "latency_ms": latency_ms,
                "safety_filter_passed": not disable_guardrails,
            },
        }

    def _build_cosmos_prompt(self, request: WorldModelRequest) -> str:
        prompt_parts = [
            request.social_context.strip(),
            "The clip shows a first-person social interaction.",
            f"The robot chooses this next behavior: {request.proposed_behavior or request.selected_behavior}.",
        ]
        if request.predicted_norms:
            prompt_parts.append(
                "This behavior should follow these social norms: "
                + ", ".join(request.predicted_norms)
                + "."
            )
        if request.rationale.strip():
            prompt_parts.append(
                "Reasoning behind the choice: " + request.rationale.strip()
            )
        prompt_parts.append(
            "Continue the scene with a realistic short social reaction from the other person, preserving the camera viewpoint and physical layout."
        )
        return " ".join(part for part in prompt_parts if part).strip()

    def _copy_best_frame(
        self,
        *,
        request: WorldModelRequest,
        destination: Path,
    ) -> Optional[Path]:
        source_frame = ""
        if request.source_during_frame_paths:
            source_frame = request.source_during_frame_paths[0]
        elif request.prev_frame_paths:
            source_frame = request.prev_frame_paths[0]
        if not source_frame:
            return None
        try:
            shutil.copy2(source_frame, destination)
        except OSError:
            return None
        return destination if destination.exists() else None

    def _write_failure_marker(self, path: Path, payload: Dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _prepare_cosmos_input_video(
        self,
        *,
        request: WorldModelRequest,
        repo_root: Path,
        python_path: Path,
        run_dir: Path,
    ) -> Optional[Path]:
        if not request.prev_video_path:
            return None

        target_resolution = os.environ.get("EGOSOCIAL_COSMOS_RESOLUTION", "192,320")
        try:
            target_h, target_w = [
                int(value.strip()) for value in target_resolution.split(",", maxsplit=1)
            ]
        except ValueError:
            target_h, target_w = 192, 320
        min_frames = max(
            5, int(os.environ.get("EGOSOCIAL_COSMOS_MIN_INPUT_FRAMES", "5"))
        )
        fps = float(os.environ.get("EGOSOCIAL_COSMOS_INPUT_FPS", "8.0"))
        prepared_video_path = run_dir / "input_conditioning.mp4"
        preprocess_log_path = run_dir / "preprocess.log"

        script = "\n".join(
            [
                "import cv2",
                "import sys",
                "from pathlib import Path",
                "src = Path(sys.argv[1])",
                "dst = Path(sys.argv[2])",
                "target_h = int(sys.argv[3])",
                "target_w = int(sys.argv[4])",
                "min_frames = int(sys.argv[5])",
                "fps = float(sys.argv[6])",
                "cap = cv2.VideoCapture(str(src))",
                "frames = []",
                "while len(frames) < min_frames:",
                "    ok, frame = cap.read()",
                "    if not ok:",
                "        break",
                "    frame = cv2.resize(frame, (target_w, target_h))",
                "    frames.append(frame)",
                "cap.release()",
                "if not frames:",
                "    raise RuntimeError(f'No frames read from {src}')",
                "while len(frames) < min_frames:",
                "    frames.append(frames[-1].copy())",
                "fourcc = cv2.VideoWriter_fourcc(*'mp4v')",
                "writer = cv2.VideoWriter(str(dst), fourcc, fps, (target_w, target_h))",
                "for frame in frames[:min_frames]:",
                "    writer.write(frame)",
                "writer.release()",
                "print(dst)",
            ]
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["LD_LIBRARY_PATH"] = self._merge_library_paths(
            env.get("LD_LIBRARY_PATH", ""),
            self._cosmos_library_paths(repo_root),
        )
        completed = subprocess.run(
            [
                str(python_path),
                "-c",
                script,
                request.prev_video_path,
                str(prepared_video_path),
                str(target_h),
                str(target_w),
                str(min_frames),
                str(fps),
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        preprocess_log_path.write_text(
            "\n".join(
                [
                    completed.stdout or "",
                    completed.stderr or "",
                ]
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode != 0 or not prepared_video_path.exists():
            return None
        return prepared_video_path

    def _cosmos_library_paths(self, repo_root: Path) -> List[str]:
        site_packages = repo_root / ".venv" / "lib" / "python3.10" / "site-packages"
        candidates = [
            site_packages / "nvidia" / "cu13" / "lib",
            site_packages / "nvidia" / "cudnn" / "lib",
            site_packages / "nvidia" / "cusparselt" / "lib",
            site_packages / "nvidia" / "nccl" / "lib",
            site_packages / "nvidia" / "nvshmem" / "lib",
            site_packages / "triton" / "backends" / "nvidia" / "lib",
        ]
        return [str(path) for path in candidates if path.exists()]

    def _merge_library_paths(
        self,
        existing: str,
        additions: List[str],
    ) -> str:
        merged: List[str] = []
        for entry in additions + [part for part in existing.split(":") if part]:
            if entry and entry not in merged:
                merged.append(entry)
        return ":".join(merged)
