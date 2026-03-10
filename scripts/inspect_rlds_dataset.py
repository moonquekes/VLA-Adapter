#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from PIL import Image
import tensorflow_datasets as tfds


EXPECTED_STEP_KEYS = {
    "action",
    "discount",
    "is_first",
    "is_last",
    "is_terminal",
    "language_instruction",
    "observation",
    "reward",
}

EXPECTED_OBSERVATION_KEYS = {
    "image",
    "joint_state",
    "state",
    "wrist_image",
}


@dataclass
class EpisodeSummary:
    episode_index: int
    num_steps: int
    language: str
    file_path: str
    action_shape: Sequence[int]
    state_shape: Sequence[int]
    joint_state_shape: Sequence[int]
    image_shape: Sequence[int]
    wrist_image_shape: Sequence[int]
    action_min: List[float]
    action_max: List[float]
    reward_sum: float
    last_reward: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect TFDS/RLDS-format datasets used by VLA-Adapter and sample a few episodes."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Path to dataset root or version directory, for example data/libero/libero_spatial_no_noops or .../1.0.0",
    )
    parser.add_argument("--split", default="train", help="TFDS split to inspect. Default: train")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Number of episodes to sample from the split. Default: 3",
    )
    parser.add_argument(
        "--preview-steps",
        type=int,
        default=3,
        help="How many leading steps to print per sampled episode. Default: 3",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional directory to export first/middle/last RGB frames for sampled episodes.",
    )
    parser.add_argument(
        "--quiet-tf",
        action="store_true",
        help="Reduce TensorFlow logging noise by setting TF_CPP_MIN_LOG_LEVEL=2 before import in the shell.",
    )
    return parser.parse_args()


def resolve_version_dir(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.expanduser().resolve()
    if (dataset_dir / "dataset_info.json").exists() and (dataset_dir / "features.json").exists():
        return dataset_dir

    version_dirs = sorted(
        child for child in dataset_dir.iterdir() if child.is_dir() and (child / "dataset_info.json").exists()
    )
    if len(version_dirs) == 1:
        return version_dirs[0]
    if not version_dirs:
        raise FileNotFoundError(f"No TFDS version directory found under {dataset_dir}")
    raise ValueError(
        f"Multiple version directories found under {dataset_dir}: {[child.name for child in version_dirs]}. "
        "Please pass the exact version directory."
    )


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def episode_steps(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(episode["steps"])


def compute_episode_summary(episode: Dict[str, Any], episode_index: int) -> EpisodeSummary:
    steps = episode_steps(episode)
    if not steps:
        raise ValueError(f"Episode {episode_index} contains no steps")

    actions = np.stack([step["action"] for step in steps], axis=0)
    states = np.stack([step["observation"]["state"] for step in steps], axis=0)
    joint_states = np.stack([step["observation"]["joint_state"] for step in steps], axis=0)
    rewards = np.asarray([step["reward"] for step in steps], dtype=np.float32)

    first_step = steps[0]
    last_step = steps[-1]
    return EpisodeSummary(
        episode_index=episode_index,
        num_steps=len(steps),
        language=decode_text(first_step["language_instruction"]),
        file_path=decode_text(episode["episode_metadata"]["file_path"]),
        action_shape=list(first_step["action"].shape),
        state_shape=list(first_step["observation"]["state"].shape),
        joint_state_shape=list(first_step["observation"]["joint_state"].shape),
        image_shape=list(first_step["observation"]["image"].shape),
        wrist_image_shape=list(first_step["observation"]["wrist_image"].shape),
        action_min=np.round(actions.min(axis=0), 6).tolist(),
        action_max=np.round(actions.max(axis=0), 6).tolist(),
        reward_sum=float(rewards.sum()),
        last_reward=float(last_step["reward"]),
    )


def validate_episode(episode: Dict[str, Any], episode_index: int) -> List[str]:
    issues: List[str] = []
    steps = episode_steps(episode)
    if not steps:
        return [f"episode {episode_index}: empty steps"]

    step_keys = set(steps[0].keys())
    missing_step_keys = sorted(EXPECTED_STEP_KEYS - step_keys)
    if missing_step_keys:
        issues.append(f"episode {episode_index}: missing step keys {missing_step_keys}")

    obs_keys = set(steps[0]["observation"].keys())
    missing_obs_keys = sorted(EXPECTED_OBSERVATION_KEYS - obs_keys)
    if missing_obs_keys:
        issues.append(f"episode {episode_index}: missing observation keys {missing_obs_keys}")

    if not bool(steps[0]["is_first"]):
        issues.append(f"episode {episode_index}: first step does not set is_first=True")
    if not bool(steps[-1]["is_last"]):
        issues.append(f"episode {episode_index}: last step does not set is_last=True")
    if float(steps[-1]["reward"]) <= 0:
        issues.append(f"episode {episode_index}: last reward is non-positive ({steps[-1]['reward']})")

    for step_idx, step in enumerate(steps):
        action = np.asarray(step["action"])
        if action.shape != (7,):
            issues.append(f"episode {episode_index}: step {step_idx} action shape is {tuple(action.shape)}, expected (7,)")
            break

    return issues


def format_vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{value:.4f}" for value in values) + "]"


def export_episode_frames(steps: List[Dict[str, Any]], export_dir: Path, episode_index: int) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = sorted({0, len(steps) // 2, len(steps) - 1})
    episode_dir = export_dir / f"episode_{episode_index:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    for step_idx in frame_indices:
        step = steps[step_idx]
        Image.fromarray(step["observation"]["image"]).save(episode_dir / f"step_{step_idx:04d}_image.png")
        Image.fromarray(step["observation"]["wrist_image"]).save(episode_dir / f"step_{step_idx:04d}_wrist.png")


def print_dataset_metadata(dataset_dir: Path) -> None:
    info = load_json(dataset_dir / "dataset_info.json")
    features = load_json(dataset_dir / "features.json")

    print("=== Dataset Metadata ===")
    print(f"dataset_dir: {dataset_dir}")
    print(f"name: {info.get('name')}")
    print(f"version: {info.get('version')}")
    print(f"file_format: {info.get('fileFormat')}")

    splits = info.get("splits", [])
    for split in splits:
        shard_lengths = [int(value) for value in split.get("shardLengths", [])]
        print(
            f"split={split.get('name')} shards={len(shard_lengths)} episodes={sum(shard_lengths)} bytes={split.get('numBytes')}"
        )

    step_features = features["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]["features"]
    observation_features = step_features["observation"]["featuresDict"]["features"]
    print(f"step_keys: {sorted(step_features.keys())}")
    print(f"observation_keys: {sorted(observation_features.keys())}")
    print()


def iter_sampled_episodes(builder: tfds.core.DatasetBuilder, split: str, limit: int) -> Iterable[Dict[str, Any]]:
    dataset = builder.as_dataset(split=f"{split}[:{limit}]")
    for episode in tfds.as_numpy(dataset):
        yield episode


def main() -> None:
    args = parse_args()
    dataset_dir = resolve_version_dir(args.dataset_dir)
    print_dataset_metadata(dataset_dir)

    builder = tfds.builder_from_directory(str(dataset_dir))
    print("=== Runtime Check ===")
    print(f"builder_name: {builder.info.name}")
    print(f"available_splits: {list(builder.info.splits.keys())}")
    print()

    sampled_summaries: List[EpisodeSummary] = []
    validation_issues: List[str] = []

    for episode_index, episode in enumerate(iter_sampled_episodes(builder, args.split, args.num_episodes)):
        steps = episode_steps(episode)
        summary = compute_episode_summary(episode, episode_index)
        sampled_summaries.append(summary)
        validation_issues.extend(validate_episode(episode, episode_index))

        print(f"=== Episode {episode_index} ===")
        print(f"language: {summary.language}")
        print(f"file_path: {summary.file_path}")
        print(f"num_steps: {summary.num_steps}")
        print(f"action_shape: {summary.action_shape}")
        print(f"state_shape: {summary.state_shape}")
        print(f"joint_state_shape: {summary.joint_state_shape}")
        print(f"image_shape: {summary.image_shape}")
        print(f"wrist_image_shape: {summary.wrist_image_shape}")
        print(f"action_min: {format_vector(summary.action_min)}")
        print(f"action_max: {format_vector(summary.action_max)}")
        print(f"reward_sum: {summary.reward_sum:.4f}")
        print(f"last_reward: {summary.last_reward:.4f}")
        print("preview_steps:")
        for step_idx, step in enumerate(steps[: args.preview_steps]):
            print(
                f"  step={step_idx} is_first={bool(step['is_first'])} is_last={bool(step['is_last'])} "
                f"reward={float(step['reward']):.4f} action={format_vector(step['action'])}"
            )
        print()

        if args.export_dir is not None:
            export_episode_frames(steps, args.export_dir, episode_index)

    if sampled_summaries:
        action_mins = np.stack([summary.action_min for summary in sampled_summaries], axis=0)
        action_maxs = np.stack([summary.action_max for summary in sampled_summaries], axis=0)
        num_steps = np.asarray([summary.num_steps for summary in sampled_summaries], dtype=np.int32)

        print("=== Aggregate Summary ===")
        print(f"sampled_episodes: {len(sampled_summaries)}")
        print(f"min_steps: {int(num_steps.min())}")
        print(f"max_steps: {int(num_steps.max())}")
        print(f"mean_steps: {float(num_steps.mean()):.2f}")
        print(f"global_action_min: {format_vector(action_mins.min(axis=0))}")
        print(f"global_action_max: {format_vector(action_maxs.max(axis=0))}")
        print()

    print("=== Validation ===")
    if validation_issues:
        for issue in validation_issues:
            print(f"FAIL: {issue}")
    else:
        print("PASS: sampled episodes match the expected VLA-Adapter LIBERO-style RLDS schema.")

    print()
    print("=== Dataset Creation Checklist ===")
    print("1. Store the dataset as TFDS/RLDS with dataset_info.json, features.json, and sharded TFRecord files.")
    print("2. Keep each top-level example as an episode with a nested steps sequence.")
    print("3. For LIBERO-style fine-tuning, keep step keys action/reward/discount/is_first/is_last/is_terminal/language_instruction/observation.")
    print("4. Provide observation.image, observation.wrist_image, observation.state, and observation.joint_state.")
    print("5. Keep action shape at 7, where the last dimension is the gripper command.")
    print("6. Ensure the first step has is_first=True and the last step has is_last=True with a positive terminal reward for demos.")


if __name__ == "__main__":
    main()