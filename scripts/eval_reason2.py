"""Evaluate a Reason2-style checkpoint on the fixed EgoNormia heldout split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from train_grpo_reason2 import (
    DEFAULT_VERIFIED_SPLIT_PATH,
    SYSTEM_PROMPT,
    _as_float,
    _batch_decode,
    _build_user_prompt,
    _load_scene_id_set,
    _parse_action,
    _tokenize_prompt,
    _tokenizer_from_processor,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from egosocial_env.server.egosocial_env_environment import (
    ENV_MODE_BENCHMARK,
    ENV_MODE_TRAIN,
    EgosocialEnvironment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Reason2-style checkpoint on the fixed EgoNormia heldout split.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF model id or local checkpoint path.",
    )
    parser.add_argument(
        "--split-path",
        default=str(DEFAULT_VERIFIED_SPLIT_PATH),
        help="JSON file with {'split': [...]} scene ids to evaluate.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/eval_reason2_verified.json",
        help="Where to write the evaluation summary and per-scene results.",
    )
    parser.add_argument(
        "--env-mode",
        choices=[ENV_MODE_BENCHMARK, ENV_MODE_TRAIN],
        default=ENV_MODE_BENCHMARK,
    )
    parser.add_argument(
        "--world-model-provider",
        default=None,
        help="Optional world-model provider for train mode, such as 'cosmos'.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Evaluate only the first N scene ids from the split.",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Use 0.0 for deterministic evaluation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    return parser.parse_args()


def _load_eval_stack() -> Dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Evaluation dependencies are missing. Install them with "
            "`UV_CACHE_DIR=/tmp/uv-cache uv sync --extra train`."
        ) from exc

    extra_model_classes: List[Any] = []
    try:
        from transformers import AutoModelForImageTextToText

        extra_model_classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        extra_model_classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass

    return {
        "torch": torch,
        "Image": Image,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "model_classes": [AutoModelForCausalLM, *extra_model_classes],
    }


def _load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    stack = _load_eval_stack()
    torch = stack["torch"]
    AutoProcessor = stack["AutoProcessor"]
    AutoTokenizer = stack["AutoTokenizer"]
    model_classes = stack["model_classes"]

    try:
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )

    dtype = torch.bfloat16 if args.bf16 else None
    load_errors = []
    model = None
    for model_cls in model_classes:
        try:
            model = model_cls.from_pretrained(
                args.model_id,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=dtype,
                device_map="auto",
            )
            break
        except Exception as exc:  # pragma: no cover
            load_errors.append(f"{model_cls.__name__}: {exc}")
    if model is None:  # pragma: no cover
        raise RuntimeError(
            "Failed to load model. Tried: " + " | ".join(load_errors)
        )
    model.eval()
    return model, processor


def _move_inputs_to_device(inputs: Dict[str, Any], device: Any) -> Dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _existing_frame_paths(observation: Any) -> List[str]:
    return [path for path in observation.frame_paths if Path(path).exists()]


def _build_multimodal_messages(
    observation: Any,
    image_paths: Sequence[str],
) -> List[Dict[str, Any]]:
    user_prompt = _build_user_prompt(observation)
    if not image_paths:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    user_content: List[Dict[str, str]] = [
        {"type": "image", "image": str(path)}
        for path in image_paths
    ]
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _apply_chat_template(processor: Any, messages: List[Dict[str, Any]]) -> str:
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    raise ValueError("Processor/tokenizer does not support apply_chat_template.")


def _load_images(image_paths: Sequence[str]) -> List[Any]:
    stack = _load_eval_stack()
    Image = stack["Image"]
    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def _generate_completion(
    model: Any,
    processor: Any,
    observation: Any,
    *,
    max_completion_length: int,
    temperature: float,
    top_p: float,
) -> tuple[str, str]:
    stack = _load_eval_stack()
    torch = stack["torch"]

    device = next(model.parameters()).device
    image_paths = _existing_frame_paths(observation)
    input_mode = "text"
    if image_paths and hasattr(processor, "image_processor"):
        messages = _build_multimodal_messages(observation, image_paths)
        prompt_text = _apply_chat_template(processor, messages)
        inputs = processor(
            text=[prompt_text],
            images=_load_images(image_paths),
            return_tensors="pt",
            padding=True,
        )
        input_mode = "image"
        inputs = _move_inputs_to_device(inputs, device)
    else:
        messages = _build_multimodal_messages(observation, [])
        prompt_text = _apply_chat_template(processor, messages)
        inputs = _tokenize_prompt(processor, prompt_text, device)
    input_ids = inputs["input_ids"]
    input_length = int(input_ids.shape[1])
    do_sample = temperature > 0.0
    tokenizer = _tokenizer_from_processor(processor)

    with torch.no_grad():
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = sequences[0][input_length:]
    return _batch_decode(processor, completion_ids.tolist()), input_mode


def _evaluate_scene(
    model: Any,
    processor: Any,
    env: EgosocialEnvironment,
    *,
    scene_id: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    observation = env.reset(scene_id=scene_id, mode=args.env_mode)
    turns: List[Dict[str, Any]] = []

    while not observation.done:
        completion_text, input_mode = _generate_completion(
            model,
            processor,
            observation,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        action, format_score = _parse_action(completion_text, observation)
        turns.append(
            {
                "phase": observation.phase,
                "raw_completion": completion_text,
                "action": action.model_dump(),
                "format_reward": format_score,
                "input_mode": input_mode,
                "frame_paths": list(observation.frame_paths),
            }
        )
        observation = env.step(action)

    metadata = observation.metadata or {}
    reward_breakdown = metadata.get("reward_breakdown", {})
    return {
        "scene_id": scene_id,
        "reward": _as_float(observation.reward),
        "correct": bool(metadata.get("correct")),
        "transition_source": metadata.get(
            "transition_source",
            observation.transition_source,
        ),
        "reward_breakdown": reward_breakdown,
        "rubric_average": _as_float(metadata.get("rubric_average")),
        "turns": turns,
    }


def _mean(values: Iterable[float]) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return round(sum(numbers) / len(numbers), 4)


def _safe_output_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _resolved_path_str(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _take_scene_ids(args: argparse.Namespace) -> List[str]:
    scene_ids = sorted(_load_scene_id_set(args.split_path))
    if args.max_samples > 0:
        scene_ids = scene_ids[: args.max_samples]
    return scene_ids


def _summary(results: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model_id": args.model_id,
        "env_mode": args.env_mode,
        "world_model_provider": args.world_model_provider,
        "split_path": _resolved_path_str(args.split_path),
        "num_samples": len(results),
        "avg_reward": _mean(result["reward"] for result in results),
        "accuracy": _mean(1.0 if result["correct"] else 0.0 for result in results),
        "avg_action_selection": _mean(
            _as_float(result["reward_breakdown"].get("action_selection"))
            for result in results
        ),
        "avg_sensibility": _mean(
            _as_float(result["reward_breakdown"].get("sensibility"))
            for result in results
        ),
        "avg_taxonomy_match": _mean(
            _as_float(result["reward_breakdown"].get("taxonomy_match"))
            for result in results
        ),
        "avg_justification_alignment": _mean(
            _as_float(result["reward_breakdown"].get("justification_alignment"))
            for result in results
        ),
        "avg_rubric_average": _mean(result["rubric_average"] for result in results),
    }


def main() -> None:
    args = parse_args()
    model, processor = _load_model_and_processor(args)
    env = EgosocialEnvironment(world_model_provider=args.world_model_provider)
    requested_scene_ids = _take_scene_ids(args)
    if not requested_scene_ids:
        raise ValueError("No scene ids found in the evaluation split.")

    available_scene_ids = {episode["scene_id"] for episode in env._episodes}  # noqa: SLF001
    missing_scene_ids = [
        scene_id for scene_id in requested_scene_ids if scene_id not in available_scene_ids
    ]
    scene_ids = [
        scene_id for scene_id in requested_scene_ids if scene_id in available_scene_ids
    ]
    if not scene_ids:
        raise ValueError("No evaluation scenes overlap with the local EgoNormia env.")
    if missing_scene_ids:
        print(
            "Warning: skipping "
            f"{len(missing_scene_ids)} scene ids that are not present in the local env."
        )
        preview = ", ".join(missing_scene_ids[:10])
        print(f"Missing scene ids: {preview}")

    results = []
    for index, scene_id in enumerate(scene_ids, start=1):
        result = _evaluate_scene(
            model,
            processor,
            env,
            scene_id=scene_id,
            args=args,
        )
        results.append(result)
        print(
            f"[{index}/{len(scene_ids)}] {scene_id} "
            f"reward={result['reward']:.3f} correct={int(result['correct'])}"
        )

    payload = {
        "summary": {
            **_summary(results, args),
            "requested_num_samples": len(requested_scene_ids),
            "available_num_samples": len(scene_ids),
            "missing_num_samples": len(missing_scene_ids),
        },
        "missing_scene_ids": missing_scene_ids,
        "results": results,
    }
    output_path = _safe_output_path(args.output_path)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
