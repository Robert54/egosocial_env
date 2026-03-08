"""Run GRPO against the local EgoNormia 2-turn environment.

This script is intentionally benchmark-first:
- default env mode is `benchmark`
- observations are packed into multimodal turns when frame assets are available
- `train` mode is supported, but generated world-model transitions should be
  treated as an extension path rather than the first training target

The training target is the decision model (for example a Cosmos-Reason2
checkpoint), not the video world model.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from egosocial_env.models import EgosocialAction, EgosocialObservation
from egosocial_env.server.egosocial_env_environment import (
    ENV_MODE_BENCHMARK,
    ENV_MODE_TRAIN,
    EgosocialEnvironment,
)


SYSTEM_PROMPT = """You are a social interaction policy for an egocentric robot assistant.
Read the current observation, identify the relevant social norms, and choose the
best next action. Return JSON only.

Required JSON schema:
{
  "predicted_norms": ["safety", "politeness"],
  "selected_option": "A",
  "rationale": "Short explanation grounded in the observation.",
  "proposed_behavior": "Concrete action text for the next step."
}
"""

OPTION_RE = re.compile(r'"selected_option"\s*:\s*"?(?P<option>[A-E])"?', re.IGNORECASE)
DEFAULT_VERIFIED_SPLIT_PATH = (
    REPO_ROOT / "egosocial_env" / "data" / "splits" / "verified_split.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Reason2-style policy with GRPO on the local EgoNormia env.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF model id or local checkpoint path. This can be your EgoNormia SFT checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/grpo_reason2",
        help="Directory for trainer checkpoints and logs.",
    )
    parser.add_argument(
        "--env-mode",
        choices=[ENV_MODE_BENCHMARK, ENV_MODE_TRAIN],
        default=ENV_MODE_BENCHMARK,
        help="Benchmark mode is the recommended starting point.",
    )
    parser.add_argument(
        "--world-model-provider",
        default=None,
        help="Optional world-model provider for train mode, such as 'cosmos'.",
    )
    parser.add_argument(
        "--scene-limit",
        type=int,
        default=256,
        help="Maximum number of EgoNormia scenes to include in the training dataset.",
    )
    parser.add_argument(
        "--scene-repeats",
        type=int,
        default=1,
        help="How many times to repeat each scene id in the prompt dataset.",
    )
    parser.add_argument(
        "--exclude-split-path",
        default=str(DEFAULT_VERIFIED_SPLIT_PATH),
        help="JSON file with {'split': [...]} scene ids to exclude from training.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=4,
        help="GRPO samples this many rollouts per prompt.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=320,
        help="Budget for one action JSON plus rationale.",
    )
    parser.add_argument(
        "--image-max-edge",
        type=int,
        default=448,
        help="Downscale rollout images to this max edge before passing them to the processor.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Set to -1 to disable and rely on epochs instead.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--report-to",
        default="none",
        help="Comma-separated integrations such as 'none' or 'wandb'.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Recommended for Cosmos checkpoints that require remote code.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training and rollout generation.",
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        help="Use TRL's vLLM rollout path if available.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate", "server"],
        default="colocate",
    )
    parser.add_argument(
        "--vllm-server-base-url",
        default=None,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def _load_train_stack() -> Dict[str, Any]:
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoProcessor, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Training dependencies are missing. Install them with "
            "`UV_CACHE_DIR=/tmp/uv-cache uv sync --extra train`."
        ) from exc

    return {
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "Dataset": Dataset,
        "GRPOConfig": GRPOConfig,
        "GRPOTrainer": GRPOTrainer,
        "LoraConfig": LoraConfig,
    }


def _parse_reports(value: str) -> List[str]:
    reports = [token.strip() for token in value.split(",") if token.strip()]
    return [] if reports == ["none"] else reports


def _resolve_split_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    normalized = path_str.strip()
    if not normalized:
        return None
    path = Path(normalized)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _load_scene_id_set(path_str: str | None) -> Set[str]:
    path = _resolve_split_path(path_str)
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        values = payload.get("split", [])
    elif isinstance(payload, list):
        values = payload
    else:
        raise ValueError(f"Unsupported split payload in {path}")
    return {str(value).strip() for value in values if str(value).strip()}


def _load_processing_class(
    model_id: str,
    *,
    trust_remote_code: bool,
    auto_processor_cls: Any,
    auto_tokenizer_cls: Any,
) -> Any:
    try:
        processor = auto_processor_cls.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        processor = auto_tokenizer_cls.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor


def _extract_scene_id(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("scene_id="):
            return line.split("=", 1)[1].strip()
    raise ValueError(f"Could not parse scene_id from prompt: {prompt!r}")


def _build_dataset(
    env: EgosocialEnvironment,
    *,
    scene_limit: int,
    scene_repeats: int,
    dataset_cls: Any,
    seed: int,
    exclude_scene_ids: Set[str],
) -> Any:
    scene_ids = [episode["scene_id"] for episode in env._episodes]  # noqa: SLF001
    scene_ids = [scene_id for scene_id in scene_ids if scene_id not in exclude_scene_ids]
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    if scene_limit > 0:
        scene_ids = scene_ids[:scene_limit]
    prompts = []
    for scene_id in scene_ids:
        for _ in range(max(1, scene_repeats)):
            prompts.append(
                {
                    "prompt": f"scene_id={scene_id}",
                    "scene_id": scene_id,
                }
            )
    return dataset_cls.from_list(prompts)


def _json_schema_hint() -> str:
    return (
        '{"predicted_norms":["safety"],"selected_option":"A",'
        '"rationale":"...","proposed_behavior":"..."}'
    )


def _format_options(options: Dict[str, str]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in sorted(options.items()))


def _format_history(history: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for index, action in enumerate(history, start=1):
        norms = ", ".join(action.get("predicted_norms", [])) or "none"
        lines.append(
            f"Turn {index} policy action: option={action.get('selected_option', '')}, "
            f"norms={norms}, rationale={action.get('rationale', '').strip()}"
        )
    return "\n".join(lines)


def _build_user_prompt(observation: EgosocialObservation) -> str:
    metadata = observation.metadata or {}
    parts = [
        f"Scene ID: {observation.scene_id}",
        f"Env mode: {observation.env_mode}",
        f"Current phase: {observation.phase}",
        f"Turn index: {observation.turn_index}",
        f"Task prompt: {observation.prompt}",
        f"Question: {observation.question}",
        f"Social context: {observation.social_context}",
        "Available options:",
        _format_options(observation.available_options),
    ]

    if observation.frame_descriptions:
        parts.extend(
            [
                "Visual evidence descriptions:",
                "\n".join(f"- {line}" for line in observation.frame_descriptions),
            ]
        )

    history = metadata.get("history", [])
    if history:
        parts.extend(
            [
                "Episode history:",
                _format_history(history),
            ]
        )

    if metadata.get("adaptation_hint"):
        parts.append(f"Adaptation hint: {metadata['adaptation_hint']}")
    if metadata.get("previous_choice"):
        parts.append(f"Previous choice: {metadata['previous_choice']}")
    if metadata.get("previous_choice_norms"):
        previous_norms = ", ".join(metadata["previous_choice_norms"])
        parts.append(f"Previous choice norms: {previous_norms}")

    parts.extend(
        [
            "Return JSON only.",
            f"JSON schema example: {_json_schema_hint()}",
        ]
    )
    return "\n\n".join(parts)


def _chat_to_text(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("Processor or tokenizer does not support apply_chat_template.")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _batch_decode(processor: Any, token_ids: Sequence[int]) -> str:
    if hasattr(processor, "decode"):
        return processor.decode(token_ids, skip_special_tokens=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Processor does not expose a tokenizer for decoding.")
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _tokenizer_from_processor(processor: Any) -> Any:
    return getattr(processor, "tokenizer", processor)


def _extract_json_candidate(text: str) -> Dict[str, Any] | None:
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = []
    if code_block:
        candidates.append(code_block.group(1))
    candidates.extend(re.findall(r"\{.*?\}", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _fallback_action_from_text(
    text: str,
    observation: EgosocialObservation,
) -> Dict[str, Any]:
    option_match = OPTION_RE.search(text)
    selected_option = (
        option_match.group("option").upper()
        if option_match
        else sorted(observation.available_options)[0]
    )
    predicted_norms = re.findall(r"\b(safety|privacy|proxemics|politeness|cooperation|coordination|communication)\b", text.lower())
    option_text = observation.available_options.get(selected_option, "")
    return {
        "predicted_norms": predicted_norms,
        "selected_option": selected_option,
        "rationale": text.strip()[:500] or f"Fallback parse for option {selected_option}.",
        "proposed_behavior": option_text,
    }


def _parse_action(
    completion_text: str,
    observation: EgosocialObservation,
) -> tuple[EgosocialAction, float]:
    payload = _extract_json_candidate(completion_text)
    if payload is None:
        payload = _fallback_action_from_text(completion_text, observation)
        format_reward = 0.0
    else:
        if not isinstance(payload.get("predicted_norms"), list):
            payload["predicted_norms"] = []
        payload.setdefault("rationale", "")
        selected_option = str(payload.get("selected_option", "")).strip().upper()
        if selected_option not in observation.available_options:
            payload["selected_option"] = sorted(observation.available_options)[0]
            format_reward = 0.5
        else:
            payload["selected_option"] = selected_option
            format_reward = 1.0
        payload.setdefault(
            "proposed_behavior",
            observation.available_options.get(payload["selected_option"], ""),
        )

    action = EgosocialAction.model_validate(payload)
    return action, format_reward


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reward_from_key(key: str):
    def reward_func(completions: Sequence[str], **kwargs: Any) -> List[float]:
        values = kwargs.get(key, None)
        if values is None:
            return [0.0] * len(completions)
        return [_as_float(value) for value in values]

    reward_func.__name__ = f"reward_{key}"
    return reward_func


def _collate_rollouts(episodes: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    keys = episodes[0].keys()
    collated: Dict[str, Any] = {}
    for key in keys:
        values = [episode[key] for episode in episodes]
        collated[key] = None if values and all(value is None for value in values) else values
    return collated


def _build_messages(observation: EgosocialObservation) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(observation)},
    ]


def _tokenize_prompt(processor: Any, prompt_text: str, device: Any) -> Dict[str, Any]:
    inputs = processor(
        text=[prompt_text],
        return_tensors="pt",
        padding=True,
    )
    return {key: value.to(device) for key, value in inputs.items()}


def _existing_frame_paths(observation: EgosocialObservation) -> List[str]:
    return [path for path in observation.frame_paths if Path(path).exists()]


def _user_message_content(
    observation: EgosocialObservation,
    image_paths: Sequence[str],
) -> Any:
    prompt_text = _build_user_prompt(observation)
    if not image_paths:
        return prompt_text
    content = [{"type": "image", "image": str(path)} for path in image_paths]
    content.append({"type": "text", "text": prompt_text})
    return content


def _build_turn_messages(
    observation: EgosocialObservation,
    image_paths: Sequence[str],
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_message_content(observation, image_paths)},
    ]


def _conversation_image_paths(messages: Sequence[Dict[str, Any]]) -> List[str]:
    image_paths: List[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image":
                image_paths.append(str(item["image"]))
    return image_paths


def _load_images(
    image_paths: Sequence[str],
    *,
    image_max_edge: int | None,
) -> List[Any]:
    from PIL import Image

    images: List[Any] = []
    for path in image_paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if image_max_edge and image_max_edge > 0:
                rgb.thumbnail((image_max_edge, image_max_edge), Image.Resampling.LANCZOS)
            images.append(rgb.copy())
    return images


def _move_inputs_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _prepare_processor_inputs(
    processor: Any,
    messages: Sequence[Dict[str, Any]],
    *,
    device: Any | None,
    add_generation_prompt: bool,
    image_max_edge: int | None,
) -> Dict[str, Any]:
    prompt_text = _chat_to_text(
        processor,
        list(messages),
        add_generation_prompt=add_generation_prompt,
    )
    image_paths = _conversation_image_paths(messages)
    has_images = bool(image_paths) and hasattr(processor, "image_processor")
    if has_images:
        inputs = processor(
            text=[prompt_text],
            images=_load_images(image_paths, image_max_edge=image_max_edge),
            return_tensors="pt",
            padding=True,
        )
        input_mode = "image"
    else:
        inputs = processor(
            text=[prompt_text],
            return_tensors="pt",
            padding=True,
        )
        input_mode = "text"
    if device is not None:
        inputs = _move_inputs_to_device(inputs, device)
    return {
        "prompt_text": prompt_text,
        "inputs": inputs,
        "image_paths": image_paths,
        "input_mode": input_mode,
    }


def _generate_turn_with_model(
    trainer: Any,
    processor: Any,
    observation: EgosocialObservation,
    max_completion_length: int,
) -> Dict[str, Any]:
    import torch

    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()
    image_paths = _existing_frame_paths(observation)
    messages = _build_turn_messages(observation, image_paths)
    prepared = _prepare_processor_inputs(
        processor,
        messages,
        device=trainer.accelerator.device,
        add_generation_prompt=True,
        image_max_edge=getattr(trainer, "_egosocial_image_max_edge", 448),
    )
    inputs = prepared["inputs"]
    input_ids = inputs["input_ids"]
    input_length = int(input_ids.shape[1])
    generation_config = getattr(trainer, "_egosocial_generation_config", {})
    temperature = float(generation_config.get("temperature", 0.8))
    top_p = float(generation_config.get("top_p", 0.95))
    tokenizer = _tokenizer_from_processor(processor)

    with torch.no_grad():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        full_sequence = sequences[0]
        prompt_ids = full_sequence[:input_length]
        completion_ids = full_sequence[input_length:]

        if completion_ids.numel() == 0:
            completion_ids = full_sequence.new_tensor([tokenizer.eos_token_id])

        forward_inputs = {
            "input_ids": full_sequence.unsqueeze(0),
            "attention_mask": torch.ones_like(full_sequence).unsqueeze(0),
        }
        logits = model(**forward_inputs).logits[:, :-1, :]
        target_ids = full_sequence[1:].unsqueeze(0)
        token_logprobs = torch.log_softmax(logits, dim=-1).gather(
            -1, target_ids.unsqueeze(-1)
        )
        start_index = max(0, input_length - 1)
        completion_logprobs = token_logprobs[0, start_index:, 0]

    return {
        "text": _batch_decode(processor, completion_ids.tolist()),
        "prompt_ids": prompt_ids.tolist(),
        "completion_ids": completion_ids.tolist(),
        "logprobs": completion_logprobs.tolist(),
        "messages": messages,
        "input_mode": prepared["input_mode"],
        "image_paths": list(prepared["image_paths"]),
    }


def _generate_with_vllm(
    trainer: Any,
    prompt_text: str,
) -> Dict[str, Any]:
    from trl.experimental.openenv import generate_rollout_completions

    output = generate_rollout_completions(trainer, [prompt_text])[0]
    return {
        "prompt_ids": output["prompt_ids"],
        "completion_ids": output["completion_ids"],
        "logprobs": output["logprobs"],
        "text": output["text"],
    }


def _run_episode(
    trainer: Any,
    processor: Any,
    env: EgosocialEnvironment,
    *,
    scene_id: str,
    env_mode: str,
) -> Dict[str, Any]:
    import torch

    observation = env.reset(
        scene_id=scene_id,
        mode=env_mode,
    )

    format_score = 1.0
    intermediate_reward = 0.0
    final_reward = 0.0
    trajectory: List[Dict[str, Any]] = []

    while not observation.done:
        if trainer.args.use_vllm:
            raise ValueError(
                "The local EgoSocial rollout path currently requires transformers generation, not vLLM."
            )
        generated = _generate_turn_with_model(
            trainer,
            processor,
            observation,
            trainer.args.max_completion_length,
        )

        action, action_format_score = _parse_action(generated["text"], observation)
        format_score = min(format_score, action_format_score)
        trajectory.append(
            {
                "observation": observation.model_copy(deep=True),
                "completion_text": generated["text"],
                "image_paths": list(generated["image_paths"]),
                "input_mode": generated["input_mode"],
            }
        )
        observation = env.step(action)
        if not observation.done:
            intermediate_reward = _as_float(
                observation.metadata.get("intermediate_reward", observation.reward)
            )
        else:
            final_reward = _as_float(observation.reward)

    metadata = observation.metadata or {}
    reward_breakdown = metadata.get("reward_breakdown", {})
    conversation: List[Dict[str, Any]] = []
    for index, step in enumerate(trajectory):
        if index == 0:
            conversation.append({"role": "system", "content": SYSTEM_PROMPT})
        conversation.append(
            {
                "role": "user",
                "content": _user_message_content(
                    step["observation"],
                    step["image_paths"],
                ),
            }
        )
        conversation.append(
            {
                "role": "assistant",
                "content": step["completion_text"],
            }
        )

    if len(conversation) < 3:
        raise ValueError(f"Episode for {scene_id} did not yield any model turns.")

    prompt_messages = conversation[:2]
    prompt_bundle = _prepare_processor_inputs(
        processor,
        prompt_messages,
        device=None,
        add_generation_prompt=True,
        image_max_edge=getattr(trainer, "_egosocial_image_max_edge", 448),
    )
    full_bundle = _prepare_processor_inputs(
        processor,
        conversation,
        device=None,
        add_generation_prompt=False,
        image_max_edge=getattr(trainer, "_egosocial_image_max_edge", 448),
    )
    prompt_input_ids = prompt_bundle["inputs"]["input_ids"][0].cpu()
    full_input_ids = full_bundle["inputs"]["input_ids"][0].cpu()
    prompt_length = int(prompt_input_ids.shape[0])
    full_length = int(full_input_ids.shape[0])
    completion_input_ids = full_input_ids[prompt_length:]

    env_mask: List[int] = []
    previous_length = prompt_length
    for cutoff in range(3, len(conversation) + 1):
        partial_bundle = _prepare_processor_inputs(
            processor,
            conversation[:cutoff],
            device=None,
            add_generation_prompt=False,
            image_max_edge=getattr(trainer, "_egosocial_image_max_edge", 448),
        )
        partial_length = int(partial_bundle["inputs"]["input_ids"][0].shape[0])
        role = conversation[cutoff - 1]["role"]
        env_mask.extend([1 if role == "assistant" else 0] * (partial_length - previous_length))
        previous_length = partial_length
    if previous_length != full_length:
        raise ValueError(
            f"Packed conversation length mismatch for {scene_id}: "
            f"{previous_length} != {full_length}"
        )

    forward_extras: Dict[str, Any] = {"env_mask": env_mask}
    for key, value in full_bundle["inputs"].items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if key == "mm_token_type_ids":
            prompt_mm = value[0, :prompt_length].cpu()
            completion_mm = value[0, prompt_length:].cpu()
            forward_extras["prompt_mm_token_type_ids"] = prompt_mm
            forward_extras["completion_mm_token_type_ids"] = completion_mm
            continue
        if hasattr(value, "detach"):
            forward_extras[key] = value.detach().cpu()
        else:
            forward_extras[key] = value

    return {
        "prompt_ids": prompt_input_ids.tolist(),
        "completion_ids": completion_input_ids.tolist(),
        "logprobs": None,
        "scene_id": scene_id,
        "env_reward": final_reward,
        "intermediate_reward": intermediate_reward,
        "action_selection_reward": _as_float(
            reward_breakdown.get("action_selection")
        ),
        "sensibility_reward": _as_float(reward_breakdown.get("sensibility")),
        "taxonomy_reward": _as_float(reward_breakdown.get("taxonomy_match")),
        "justification_reward": _as_float(
            reward_breakdown.get("justification_alignment")
        ),
        "transition_consistency_reward": _as_float(
            reward_breakdown.get("transition_consistency", 1.0),
            default=1.0,
        ),
        "format_reward": format_score,
        "correct_reward": 1.0 if bool(metadata.get("correct")) else 0.0,
        **forward_extras,
    }


def _build_rollout_func(args: argparse.Namespace, processor: Any):
    def rollout_func(prompts: Sequence[str], trainer: Any) -> Dict[str, List[Any]]:
        env = getattr(trainer, "_egosocial_local_env", None)
        if env is None:
            env = EgosocialEnvironment(world_model_provider=args.world_model_provider)
            trainer._egosocial_local_env = env

        episodes = []
        for prompt in prompts:
            scene_id = _extract_scene_id(prompt)
            episodes.append(
                _run_episode(
                    trainer,
                    processor,
                    env,
                    scene_id=scene_id,
                    env_mode=args.env_mode,
                )
            )
        return _collate_rollouts(episodes)

    return rollout_func


class EgosocialGRPOTrainer:  # loaded dynamically via local inheritance in main()
    """Local patch layer for TRL GRPO with multimodal OpenEnv rollouts."""

    def _generate_single_turn(self, prompts: list):  # type: ignore[override]
        if not self.use_vllm and self.rollout_func is not None:
            output = self.rollout_func(prompts, self)
            required_keys = {"prompt_ids", "completion_ids", "logprobs"}
            self._egosocial_rollout_extra_fields = {
                key: value for key, value in output.items() if key not in required_keys
            }
            return (
                output["prompt_ids"],
                output["completion_ids"],
                output["logprobs"],
                self._egosocial_rollout_extra_fields,
            )
        return super()._generate_single_turn(prompts)

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        import torch

        output = super()._generate_and_score_completions(inputs)
        extra_fields = getattr(self, "_egosocial_rollout_extra_fields", None)
        if not extra_fields:
            return output

        device = output["prompt_ids"].device
        prompt_width = int(output["prompt_ids"].shape[1])
        completion_width = int(output["completion_ids"].shape[1])

        def _pad_1d_tensors(
            values: Sequence[Any],
            width: int,
            *,
            padding_value: int,
            padding_side: str,
        ) -> Any:
            padded = []
            for value in values:
                tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.long)
                tensor = tensor.to(device=device, dtype=torch.long)
                pad_width = width - int(tensor.shape[0])
                if pad_width < 0:
                    raise ValueError("Encountered overlong multimodal token-type tensor.")
                if pad_width == 0:
                    padded.append(tensor)
                    continue
                pad_tensor = torch.full((pad_width,), padding_value, device=device, dtype=tensor.dtype)
                if padding_side == "left":
                    padded.append(torch.cat([pad_tensor, tensor], dim=0))
                else:
                    padded.append(torch.cat([tensor, pad_tensor], dim=0))
            return torch.stack(padded, dim=0)

        prompt_mm = extra_fields.get("prompt_mm_token_type_ids")
        completion_mm = extra_fields.get("completion_mm_token_type_ids")
        if prompt_mm is not None and completion_mm is not None:
            prompt_mm_padded = _pad_1d_tensors(
                prompt_mm,
                prompt_width,
                padding_value=0,
                padding_side="left",
            )
            completion_mm_padded = _pad_1d_tensors(
                completion_mm,
                completion_width,
                padding_value=0,
                padding_side="right",
            )
            output["mm_token_type_ids"] = torch.cat(
                [prompt_mm_padded, completion_mm_padded],
                dim=1,
            )

        pixel_values = extra_fields.get("pixel_values")
        image_grid_thw = extra_fields.get("image_grid_thw")
        if pixel_values is not None and image_grid_thw is not None:
            output["pixel_values"] = torch.cat(
                [value.to(device=device) for value in pixel_values],
                dim=0,
            )
            output["image_grid_thw"] = torch.cat(
                [value.to(device=device) for value in image_grid_thw],
                dim=0,
            )
            output["num_images"] = [int(value.shape[0]) for value in image_grid_thw]

        self._egosocial_rollout_extra_fields = None
        return output

    def _compute_loss(self, model, inputs):  # type: ignore[override]
        import torch

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            mm_token_type_ids=inputs.get("mm_token_type_ids"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = (
            per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        )

        if self.off_policy_mask_threshold is not None:
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)
            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)
            per_token_loss1 = coef_1 * advantages
            per_token_loss2 = coef_2 * advantages
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(value):
            if value.shape[1] == 1:
                return value.mean()
            return (value * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped
            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())
            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(torch.nan_to_num(gathered_low_clip, nan=float("inf")).min().item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(torch.nan_to_num(gathered_high_clip, nan=float("-inf")).max().item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())
        return loss


def main() -> None:
    args = parse_args()
    stack = _load_train_stack()

    AutoProcessor = stack["AutoProcessor"]
    AutoTokenizer = stack["AutoTokenizer"]
    Dataset = stack["Dataset"]
    GRPOConfig = stack["GRPOConfig"]
    BaseGRPOTrainer = stack["GRPOTrainer"]
    LoraConfig = stack["LoraConfig"]

    class GRPOTrainer(EgosocialGRPOTrainer, BaseGRPOTrainer):
        pass

    processor = _load_processing_class(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        auto_processor_cls=AutoProcessor,
        auto_tokenizer_cls=AutoTokenizer,
    )

    env = EgosocialEnvironment(world_model_provider=args.world_model_provider)
    exclude_scene_ids = _load_scene_id_set(args.exclude_split_path)
    available_scene_ids = {episode["scene_id"] for episode in env._episodes}  # noqa: SLF001
    missing_exclude_scene_ids = sorted(exclude_scene_ids - available_scene_ids)
    if missing_exclude_scene_ids:
        print(
            "Warning: "
            f"{len(missing_exclude_scene_ids)} exclude-scene ids are not present in the local env."
        )
        print(
            "Missing exclude scene ids: "
            + ", ".join(missing_exclude_scene_ids[:10])
        )
    train_dataset = _build_dataset(
        env,
        scene_limit=args.scene_limit,
        scene_repeats=args.scene_repeats,
        dataset_cls=Dataset,
        seed=args.seed,
        exclude_scene_ids=exclude_scene_ids,
    )

    reward_funcs = [
        _reward_from_key("env_reward"),
        _reward_from_key("intermediate_reward"),
        _reward_from_key("format_reward"),
    ]
    reward_weights = [1.0, 0.2, 0.05]
    if args.env_mode == ENV_MODE_TRAIN:
        reward_funcs.append(_reward_from_key("transition_consistency_reward"))
        reward_weights.append(0.1)

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to=_parse_reports(args.report_to),
        bf16=args.bf16,
        use_cache=True,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        reward_weights=reward_weights,
        seed=args.seed,
    )

    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )

    trainer = GRPOTrainer(
        model=args.model_id,
        args=training_args,
        processing_class=processor,
        train_dataset=train_dataset,
        reward_funcs=reward_funcs,
        rollout_func=_build_rollout_func(args, processor),
        peft_config=peft_config,
    )
    trainer._egosocial_generation_config = {  # noqa: SLF001
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    trainer._egosocial_image_max_edge = args.image_max_edge  # noqa: SLF001
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
