#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import h5py
import numpy as np
import tensorflow_datasets as tfds


DEFAULT_DATASET_NAME = "libero_suction_no_noops"
DEFAULT_VERSION = "1.0.0"


@dataclass(frozen=True)
class EpisodeRef:
    source_path: Path
    demo_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LIBERO-style converted_hdf5 files into a TFDS/RLDS dataset that VLA-Adapter can train on."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing converted_hdf5 files generated from LIBERO replay.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/libero"),
        help="Root directory where the TFDS dataset directory will be created.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Dataset directory name. Use libero_suction_no_noops to match the built-in config added in this repo.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=DEFAULT_VERSION,
        help="Dataset version directory name.",
    )
    parser.add_argument(
        "--max_episodes_per_shard",
        type=int,
        default=32,
        help="Maximum number of episodes to store in each TFRecord shard.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.0,
        help="Optional validation split ratio in [0, 1). If 0, only a train split is created.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for train/val splitting.",
    )
    parser.add_argument(
        "--rotate_180",
        action="store_true",
        help="Rotate both camera images by 180 degrees before writing RLDS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output dataset directory before writing.",
    )
    parser.add_argument(
        "--include_failed",
        action="store_true",
        help="Keep episodes whose demo attr success != 1. By default only successful demos are exported.",
    )
    return parser.parse_args()


def build_features() -> tfds.features.FeaturesDict:
    return tfds.features.FeaturesDict(
        {
            "steps": tfds.features.Dataset(
                tfds.features.FeaturesDict(
                    {
                        "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                        "discount": tfds.features.Scalar(dtype=np.float32),
                        "is_first": tfds.features.Scalar(dtype=np.bool_),
                        "is_last": tfds.features.Scalar(dtype=np.bool_),
                        "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                        "language_instruction": tfds.features.Text(),
                        "observation": tfds.features.FeaturesDict(
                            {
                                "image": tfds.features.Image(shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"),
                                "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                "wrist_image": tfds.features.Image(
                                    shape=(256, 256, 3), dtype=np.uint8, encoding_format="jpeg"
                                ),
                            }
                        ),
                        "reward": tfds.features.Scalar(dtype=np.float32),
                    }
                )
            ),
            "episode_metadata": tfds.features.FeaturesDict(
                {
                    "file_path": tfds.features.Text(),
                }
            ),
        }
    )


def collect_episode_refs(input_dir: Path, include_failed: bool) -> List[EpisodeRef]:
    refs: List[EpisodeRef] = []
    for file_path in sorted(input_dir.glob("*.hdf5")):
        with h5py.File(file_path, "r") as h5_file:
            if "data" not in h5_file:
                continue
            data_group = h5_file["data"]
            for demo_key in data_group.keys():
                demo_group = data_group[demo_key]
                success = int(demo_group.attrs.get("success", 1))
                if include_failed or success == 1:
                    refs.append(EpisodeRef(file_path, demo_key))
    return refs


def split_episode_refs(refs: List[EpisodeRef], val_ratio: float, seed: int) -> dict[str, List[EpisodeRef]]:
    shuffled = list(refs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    if val_ratio <= 0:
        return {"train": shuffled}

    val_count = int(round(len(shuffled) * val_ratio))
    val_count = min(max(val_count, 1), len(shuffled) - 1)
    return {
        "train": shuffled[val_count:],
        "val": shuffled[:val_count],
    }


def maybe_rotate(images: np.ndarray, rotate_180: bool) -> np.ndarray:
    if not rotate_180:
        return images
    return np.rot90(images, k=2, axes=(1, 2))


def to_gripper_pair(gripper_states: np.ndarray) -> np.ndarray:
    gripper_states = np.asarray(gripper_states, dtype=np.float32)
    if gripper_states.ndim == 1:
        return np.repeat(gripper_states[:, None], 2, axis=1)
    if gripper_states.ndim == 2 and gripper_states.shape[1] == 1:
        return np.repeat(gripper_states, 2, axis=1)
    if gripper_states.ndim == 2 and gripper_states.shape[1] == 2:
        return gripper_states
    raise ValueError(f"Unsupported gripper state shape: {gripper_states.shape}")


def load_language_instruction(data_group: h5py.Group) -> str:
    problem_info_raw = data_group.attrs.get("problem_info")
    if problem_info_raw is None:
        return ""
    problem_info = json.loads(problem_info_raw)
    return str(problem_info.get("language_instruction", ""))


def validate_episode_lengths(lengths: Iterable[int], episode_ref: EpisodeRef) -> int:
    unique_lengths = sorted(set(int(length) for length in lengths))
    if len(unique_lengths) != 1:
        raise ValueError(f"Mismatched step counts in {episode_ref.source_path}::{episode_ref.demo_key}: {unique_lengths}")
    return unique_lengths[0]


def build_episode_example(episode_ref: EpisodeRef, rotate_180: bool) -> dict:
    with h5py.File(episode_ref.source_path, "r") as h5_file:
        data_group = h5_file["data"]
        demo_group = data_group[episode_ref.demo_key]
        obs_group = demo_group["obs"]

        actions = np.asarray(demo_group["actions"], dtype=np.float32)
        rewards = np.asarray(demo_group["rewards"], dtype=np.float32)
        dones = np.asarray(demo_group["dones"], dtype=np.bool_)
        ee_states = np.asarray(obs_group["ee_states"], dtype=np.float32)
        joint_states = np.asarray(obs_group["joint_states"], dtype=np.float32)
        gripper_pair = to_gripper_pair(np.asarray(obs_group["gripper_states"]))
        image = maybe_rotate(np.asarray(obs_group["agentview_rgb"], dtype=np.uint8), rotate_180)
        wrist_image = maybe_rotate(np.asarray(obs_group["eye_in_hand_rgb"], dtype=np.uint8), rotate_180)
        language_instruction = load_language_instruction(data_group)

        num_steps = validate_episode_lengths(
            [
                len(actions),
                len(rewards),
                len(dones),
                len(ee_states),
                len(joint_states),
                len(gripper_pair),
                len(image),
                len(wrist_image),
            ],
            episode_ref,
        )

        state = np.concatenate([ee_states, gripper_pair], axis=1).astype(np.float32)
        discounts = np.ones((num_steps,), dtype=np.float32)

        steps = []
        for step_idx in range(num_steps):
            is_last = step_idx == num_steps - 1
            steps.append(
                {
                    "action": actions[step_idx],
                    "discount": discounts[step_idx],
                    "is_first": step_idx == 0,
                    "is_last": is_last,
                    "is_terminal": bool(dones[step_idx]) if is_last else False,
                    "language_instruction": language_instruction,
                    "observation": {
                        "image": image[step_idx],
                        "joint_state": joint_states[step_idx],
                        "state": state[step_idx],
                        "wrist_image": wrist_image[step_idx],
                    },
                    "reward": rewards[step_idx],
                }
            )

        return {
            "steps": steps,
            "episode_metadata": {
                "file_path": f"{episode_ref.source_path}::{episode_ref.demo_key}",
            },
        }


def build_dataset_info(dataset_name: str, version: str, version_dir: Path, features: tfds.features.FeaturesDict) -> tfds.core.DatasetInfo:
    identity = tfds.core.DatasetIdentity(
        name=dataset_name,
        version=tfds.core.Version(version),
        data_dir=str(version_dir),
        module_name="local",
    )
    return tfds.core.DatasetInfo(
        builder=identity,
        description="Custom LIBERO suction dataset converted from converted_hdf5 into RLDS/TFDS format.",
        features=features,
        homepage="https://github.com/Lifelong-Robot-Learning/LIBERO",
    )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    dataset_dir = output_root / args.dataset_name
    version_dir = dataset_dir / args.version

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val_ratio must be in [0, 1).")
    if args.max_episodes_per_shard <= 0:
        raise ValueError("--max_episodes_per_shard must be > 0.")

    episode_refs = collect_episode_refs(input_dir, include_failed=args.include_failed)
    if not episode_refs:
        raise ValueError(f"No eligible HDF5 episodes found under {input_dir}")

    splits = split_episode_refs(episode_refs, val_ratio=args.val_ratio, seed=args.seed)

    if version_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output dataset already exists: {version_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(version_dir)
    version_dir.mkdir(parents=True, exist_ok=True)

    features = build_features()
    ds_info = build_dataset_info(args.dataset_name, args.version, version_dir, features)
    writer = tfds.core.SequentialWriter(
        ds_info=ds_info,
        max_examples_per_shard=args.max_episodes_per_shard,
        overwrite=True,
        file_format="tfrecord",
    )
    writer.initialize_splits(list(splits.keys()))

    written_counts: dict[str, int] = {split_name: 0 for split_name in splits}
    for split_name, refs in splits.items():
        for episode_ref in refs:
            example = build_episode_example(episode_ref, rotate_180=args.rotate_180)
            writer.add_examples({split_name: [example]})
            written_counts[split_name] += 1

    writer.close_all()

    print("RLDS conversion complete")
    print(f"input_dir={input_dir}")
    print(f"dataset_dir={dataset_dir}")
    print(f"version_dir={version_dir}")
    for split_name, count in written_counts.items():
        print(f"split={split_name} episodes={count}")
    print(f"dataset_name={args.dataset_name}")
    print("Train with --data_root_dir data/libero and --dataset_name libero_suction_no_noops")


if __name__ == "__main__":
    main()