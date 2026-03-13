#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds
import torch
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import AutoTokenizer

from experiments.robot.openvla_utils import (
    DEVICE,
    normalize_proprio,
    prepare_images_for_vla,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models import load as load_prismatic_vlm
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType


DEFAULT_DATASET_DIR = Path("/home/x/vla/VLA-Adapter-main/data/libero/libero_suction_no_noops/1.0.0")
DEFAULT_HDF5_DIR = Path("/home/x/vla/libero/data/suction_dataset/converted_hdf5")
DEFAULT_CHECKPOINT = Path(
    "/home/x/vla/VLA-Adapter-main/outputs/"
    "configs+libero_suction_no_noops+b8+lr-0.0002+lora-r64+dropout-0.0"
    "--image_aug--suction-gpu13-resume3200-20260312_101352--20000_chkpt"
)
DEFAULT_BASE_VLM_DIR = Path(
    "/home/x/vla/VLA-Adapter-main/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"
)
DEFAULT_BASE_CONFIG_DIR = Path("/home/x/vla/VLA-Adapter-main/pretrained_models/configs")
DEFAULT_FOCUS_STEPS = [0, 5, 10, 15, 70, 85, 100, 110]
AXIS_NAMES = ["x", "y", "z"]
ACTION_DIM_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "delta_roll",
    "delta_pitch",
    "delta_yaw",
    "gripper",
]
STATE_DIM_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_roll",
    "eef_pitch",
    "eef_yaw",
    "gripper_left",
    "gripper_right",
]
PHASE_TRACKED_LABELS = ["+x", "-y", "-z"]
MODEL_DTYPE = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32


@dataclass(frozen=True)
class EpisodeRef:
    source_path: Path
    demo_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose HDF5 -> RLDS -> normalized training chunk -> checkpoint prediction alignment for suction."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR, help="Path to TFDS version directory.")
    parser.add_argument("--hdf5-dir", type=Path, default=DEFAULT_HDF5_DIR, help="Path to converted_hdf5 directory.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint directory to inspect.")
    parser.add_argument(
        "--base-vlm-dir",
        type=Path,
        default=DEFAULT_BASE_VLM_DIR,
        help="Base Prismatic VLM directory used to reconstruct LoRA-only checkpoints.",
    )
    parser.add_argument(
        "--base-config-dir",
        type=Path,
        default=DEFAULT_BASE_CONFIG_DIR,
        help="Directory containing local HF config/tokenizer files for OpenVLA reconstruction.",
    )
    parser.add_argument("--split", type=str, default="train", help="TFDS split. Default: train")
    parser.add_argument("--episode-index", type=int, default=0, help="Episode index inside the split.")
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help="Run the diagnosis over every episode in the split and write an aggregate report.",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=0,
        help="Optional cap on the number of episodes to analyze in --all-episodes mode. <=0 means no cap.",
    )
    parser.add_argument("--start-step", type=int, default=0, help="First analyzed step index, inclusive.")
    parser.add_argument("--end-step", type=int, default=120, help="Last analyzed step index, inclusive.")
    parser.add_argument(
        "--focus-steps",
        type=str,
        default=",".join(str(step) for step in DEFAULT_FOCUS_STEPS),
        help="Comma-separated focus steps for snapshots and summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: tmp/diagnose_suction_alignment_<timestamp>",
    )
    parser.add_argument(
        "--snapshot-episodes",
        type=str,
        default="",
        help="Comma-separated episode indices that should export focus-step image snapshots. Default: current episode only.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output directory before writing.")
    parser.add_argument("--center-crop", dest="center_crop", action="store_true", default=True)
    parser.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    parser.add_argument("--use-minivlm", dest="use_minivlm", action="store_true", default=True)
    parser.add_argument("--no-use-minivlm", dest="use_minivlm", action="store_false")
    parser.add_argument("--num-open-loop-steps", type=int, default=8, help="Predicted action chunk length.")
    parser.add_argument(
        "--unnorm-key",
        type=str,
        default="libero_suction_no_noops",
        help="Normalization key inside checkpoint dataset_statistics.json",
    )
    return parser.parse_args()


def parse_int_list(csv_value: str) -> List[int]:
    values = []
    for chunk in csv_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return sorted(set(values))


def decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def ensure_output_dir(path: Path, overwrite: bool) -> Path:
    path = path.resolve()
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        for child in sorted(path.iterdir(), reverse=True):
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_gripper_pair(gripper_states: np.ndarray) -> np.ndarray:
    gripper_states = np.asarray(gripper_states, dtype=np.float32)
    if gripper_states.ndim == 1:
        return np.repeat(gripper_states[:, None], 2, axis=1)
    if gripper_states.ndim == 2 and gripper_states.shape[1] == 1:
        return np.repeat(gripper_states, 2, axis=1)
    if gripper_states.ndim == 2 and gripper_states.shape[1] == 2:
        return gripper_states
    raise ValueError(f"Unsupported gripper_states shape: {gripper_states.shape}")


def load_rlds_episode(dataset_dir: Path, split: str, episode_index: int) -> Dict[str, Any]:
    builder = tfds.builder_from_directory(str(dataset_dir))
    dataset = builder.as_dataset(split=split).skip(episode_index).take(1)
    try:
        return next(iter(tfds.as_numpy(dataset)))
    except StopIteration as exc:
        raise IndexError(f"Episode index {episode_index} is out of range for split {split!r}") from exc


def get_split_num_episodes(dataset_dir: Path, split: str) -> int:
    builder = tfds.builder_from_directory(str(dataset_dir))
    split_info = builder.info.splits.get(split)
    if split_info is None:
        raise KeyError(f"Split {split!r} not found in dataset {dataset_dir}")
    return int(split_info.num_examples)


def parse_episode_ref(file_path_value: str) -> EpisodeRef:
    source_path_raw, demo_key = file_path_value.rsplit("::", 1)
    return EpisodeRef(source_path=Path(source_path_raw), demo_key=demo_key)


def load_hdf5_episode(episode_ref: EpisodeRef) -> Dict[str, Any]:
    with h5py.File(episode_ref.source_path, "r") as h5_file:
        data_group = h5_file["data"]
        demo_group = data_group[episode_ref.demo_key]
        obs_group = demo_group["obs"]

        problem_info_raw = data_group.attrs.get("problem_info")
        instruction = ""
        if problem_info_raw is not None:
            instruction = str(json.loads(problem_info_raw).get("language_instruction", ""))

        ee_states = np.asarray(obs_group["ee_states"], dtype=np.float32)
        gripper_pair = to_gripper_pair(np.asarray(obs_group["gripper_states"]))

        return {
            "instruction": instruction,
            "actions": np.asarray(demo_group["actions"], dtype=np.float32),
            "rewards": np.asarray(demo_group["rewards"], dtype=np.float32),
            "dones": np.asarray(demo_group["dones"], dtype=np.bool_),
            "state": np.concatenate([ee_states, gripper_pair], axis=1).astype(np.float32),
            "joint_state": np.asarray(obs_group["joint_states"], dtype=np.float32),
            "image": np.asarray(obs_group["agentview_rgb"], dtype=np.uint8),
            "wrist_image": np.asarray(obs_group["eye_in_hand_rgb"], dtype=np.uint8),
        }


def load_checkpoint_dataset_stats(checkpoint_dir: Path, unnorm_key: str) -> Dict[str, Any]:
    stats_path = checkpoint_dir / "dataset_statistics.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if unnorm_key not in all_stats:
        if len(all_stats) == 1:
            unnorm_key = next(iter(all_stats))
        else:
            raise KeyError(f"Normalization key {unnorm_key!r} not found in {stats_path}")
    return {"key": unnorm_key, "all_stats": all_stats, "stats": all_stats[unnorm_key]}


def normalize_array(values: np.ndarray, stats: Dict[str, Any], clip: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    normalized = np.where(mask, 2 * (values - low) / (high - low + 1e-8) - 1, values)
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)

    if "min" in stats and "max" in stats:
        zeros_mask = np.asarray(stats["min"], dtype=np.float32) == np.asarray(stats["max"], dtype=np.float32)
        normalized = np.where(zeros_mask, 0.0, normalized)

    return normalized.astype(np.float32)


def unnormalize_array(values: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        low = np.asarray(stats["q01"], dtype=np.float32)
        high = np.asarray(stats["q99"], dtype=np.float32)
    else:
        raise ValueError(f"Unsupported normalization type: {ACTION_PROPRIO_NORMALIZATION_TYPE}")

    mask = np.asarray(stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    return np.where(mask, 0.5 * (values + 1) * (high - low + 1e-8) + low, values).astype(np.float32)


def vector_stats(values: np.ndarray, dim_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected 2D array for stats, got shape {values.shape}")

    stats: Dict[str, Dict[str, float]] = {}
    for index, dim_name in enumerate(dim_names):
        column = values[:, index]
        stats[dim_name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "q01": float(np.quantile(column, 0.01)),
            "q99": float(np.quantile(column, 0.99)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
            "saturation_rate": float(np.mean(np.abs(column) >= 0.999)),
        }
    return stats


def motion_phase(vector: Optional[np.ndarray], dominance_threshold: float = 0.8, eps: float = 1e-6) -> Dict[str, Any]:
    if vector is None:
        return {
            "argmax_axis": None,
            "dominant_axis_score": None,
            "phase_label": None,
            "signed_phase_label": None,
            "dominant_sign": None,
        }

    vector = np.asarray(vector, dtype=np.float32)
    translation = vector[:3]
    abs_translation = np.abs(translation)
    total = float(np.sum(abs_translation))

    if total < eps or float(np.max(abs_translation)) < eps:
        return {
            "argmax_axis": None,
            "dominant_axis_score": 0.0,
            "phase_label": "noop",
            "signed_phase_label": "noop",
            "dominant_sign": None,
        }

    axis_index = int(np.argmax(abs_translation))
    axis_name = AXIS_NAMES[axis_index]
    score = float(abs_translation[axis_index] / (total + 1e-8))
    sign = "+" if float(translation[axis_index]) >= 0 else "-"

    if score < dominance_threshold:
        return {
            "argmax_axis": axis_name,
            "dominant_axis_score": score,
            "phase_label": "mixed",
            "signed_phase_label": "mixed",
            "dominant_sign": sign,
        }

    return {
        "argmax_axis": axis_name,
        "dominant_axis_score": score,
        "phase_label": axis_name,
        "signed_phase_label": f"{sign}{axis_name}",
        "dominant_sign": sign,
    }


def same_axis_phase(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[bool]:
    if a["phase_label"] is None or b["phase_label"] is None:
        return None
    if a["phase_label"] in {"noop", "mixed"} or b["phase_label"] in {"noop", "mixed"}:
        return None
    return bool(a["signed_phase_label"] == b["signed_phase_label"])


def phase_length_stats(signed_phase_labels: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    labels = list(signed_phase_labels)
    output: Dict[str, Dict[str, Any]] = {}
    for tracked in PHASE_TRACKED_LABELS:
        run_lengths: List[int] = []
        run_length = 0
        sample_count = 0
        for label in labels:
            if label == tracked:
                sample_count += 1
                run_length += 1
            elif run_length:
                run_lengths.append(run_length)
                run_length = 0
        if run_length:
            run_lengths.append(run_length)

        output[tracked] = {
            "sample_count": sample_count,
            "num_runs": len(run_lengths),
            "run_lengths": run_lengths,
            "max_run_length": max(run_lengths) if run_lengths else 0,
        }
    return output


def serialize_array(values: Optional[np.ndarray], decimals: int = 6) -> Optional[List[Optional[float]]]:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"serialize_array expects 1D arrays, got shape {array.shape}")
    rounded = np.round(array, decimals)
    result: List[Optional[float]] = []
    for value in rounded.tolist():
        if value is None:
            result.append(None)
        elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            result.append(None)
        else:
            result.append(float(value))
    return result


def serialize_matrix(values: np.ndarray, decimals: int = 6) -> List[List[Optional[float]]]:
    array = np.asarray(values, dtype=np.float32)
    return [serialize_array(row, decimals=decimals) for row in array]


def deserialize_array(values: Optional[Sequence[Optional[float]]]) -> Optional[np.ndarray]:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def deserialize_matrix(values: Sequence[Sequence[Optional[float]]]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def image_mae(image_a: np.ndarray, image_b: np.ndarray) -> float:
    diff = np.asarray(image_a, dtype=np.float32) - np.asarray(image_b, dtype=np.float32)
    return float(np.mean(np.abs(diff)))


def max_abs_diff(values_a: np.ndarray, values_b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values_a, dtype=np.float32) - np.asarray(values_b, dtype=np.float32))))


def save_image(array: np.ndarray, output_path: Path) -> None:
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(output_path)


def build_contact_sheet(
    output_path: Path,
    title: str,
    primary_hdf5: np.ndarray,
    primary_rlds: np.ndarray,
    primary_policy: Image.Image,
    wrist_hdf5: np.ndarray,
    wrist_rlds: np.ndarray,
    wrist_policy: Image.Image,
) -> None:
    tiles = [
        ("primary_hdf5", Image.fromarray(primary_hdf5)),
        ("primary_rlds", Image.fromarray(primary_rlds)),
        ("primary_policy", primary_policy.convert("RGB")),
        ("wrist_hdf5", Image.fromarray(wrist_hdf5)),
        ("wrist_rlds", Image.fromarray(wrist_rlds)),
        ("wrist_policy", wrist_policy.convert("RGB")),
    ]
    tile_width, tile_height = tiles[0][1].size
    header_height = 28
    label_height = 22
    margin = 8
    canvas = Image.new("RGB", (3 * tile_width + 4 * margin, 2 * (tile_height + label_height) + header_height + 3 * margin), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 6), title, fill="black")

    for index, (label, image) in enumerate(tiles):
        row, col = divmod(index, 3)
        x = margin + col * (tile_width + margin)
        y = header_height + margin + row * (tile_height + label_height + margin)
        canvas.paste(image.resize((tile_width, tile_height)), (x, y))
        draw.text((x, y + tile_height + 2), label, fill="black")

    canvas.save(output_path)


def build_cfg(args: argparse.Namespace, unnorm_key: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=str(args.checkpoint),
        base_vlm_dir=args.base_vlm_dir,
        base_config_dir=args.base_config_dir,
        use_l1_regression=True,
        use_minivlm=args.use_minivlm,
        num_diffusion_steps=50,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=args.center_crop,
        num_open_loop_steps=args.num_open_loop_steps,
        load_in_8bit=False,
        load_in_4bit=False,
        unnorm_key=unnorm_key,
        task_suite_name="libero_suction",
        use_pro_version=True,
        save_version="suction",
        phase="Inference",
    )


def per_dim_max_abs_diff(values_a: np.ndarray, values_b: np.ndarray, dim_names: Sequence[str]) -> Dict[str, float]:
    delta = np.max(np.abs(np.asarray(values_a, dtype=np.float32) - np.asarray(values_b, dtype=np.float32)), axis=0)
    return {dim_name: float(delta[idx]) for idx, dim_name in enumerate(dim_names)}


def safe_mean_bool(values: Sequence[Optional[bool]]) -> float:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return 0.0
    return float(np.mean(filtered))


def nearest_centroid_metrics(class_a: np.ndarray, class_b: np.ndarray) -> Dict[str, Any]:
    class_a = np.asarray(class_a, dtype=np.float32)
    class_b = np.asarray(class_b, dtype=np.float32)
    centroid_a = class_a.mean(axis=0)
    centroid_b = class_b.mean(axis=0)

    pred_a = np.linalg.norm(class_a - centroid_a, axis=1) <= np.linalg.norm(class_a - centroid_b, axis=1)
    pred_b = np.linalg.norm(class_b - centroid_b, axis=1) <= np.linalg.norm(class_b - centroid_a, axis=1)
    accuracy = float((pred_a.sum() + pred_b.sum()) / (len(class_a) + len(class_b)))

    return {
        "num_samples": {"-y": int(len(class_a)), "-z": int(len(class_b))},
        "centroid_-y": serialize_array(centroid_a),
        "centroid_-z": serialize_array(centroid_b),
        "centroid_distance_l2": float(np.linalg.norm(centroid_a - centroid_b)),
        "nearest_centroid_accuracy": accuracy,
        "confusion": {
            "-y_as_-y": int(pred_a.sum()),
            "-y_as_-z": int((~pred_a).sum()),
            "-z_as_-z": int(pred_b.sum()),
            "-z_as_-y": int((~pred_b).sum()),
        },
    }


def per_dim_boundary_stats(normalized_values: np.ndarray, unclipped_values: np.ndarray, dim_names: Sequence[str]) -> Dict[str, Dict[str, float]]:
    normalized_values = np.asarray(normalized_values, dtype=np.float32)
    unclipped_values = np.asarray(unclipped_values, dtype=np.float32)
    stats: Dict[str, Dict[str, float]] = {}
    for idx, dim_name in enumerate(dim_names):
        clipped_column = normalized_values[:, idx]
        unclipped_column = unclipped_values[:, idx]
        stats[dim_name] = {
            "saturation_rate": float(np.mean(np.abs(clipped_column) >= 0.999)),
            "upper_clip_rate": float(np.mean(unclipped_column > 1.0)),
            "lower_clip_rate": float(np.mean(unclipped_column < -1.0)),
            "clip_rate": float(np.mean(np.logical_or(unclipped_column > 1.0, unclipped_column < -1.0))),
        }
    return stats


def compute_state_semantics_analysis(
    episode_reports: Sequence[Dict[str, Any]],
    state_dim_names: Sequence[str],
) -> Dict[str, Any]:
    hdf5_states = []
    rlds_states = []
    mismatch_episodes = []
    mismatch_steps = 0
    for report in episode_reports:
        episode_has_mismatch = False
        for record in report["step_records"]:
            hdf5_state = deserialize_array(record["hdf5_state"])
            rlds_state = deserialize_array(record["rlds_state"])
            hdf5_states.append(hdf5_state)
            rlds_states.append(rlds_state)
            if not bool(record["hdf5_vs_rlds_state_allclose"]):
                mismatch_steps += 1
                episode_has_mismatch = True
        if episode_has_mismatch:
            mismatch_episodes.append(int(report["episode_summary"]["episode_index"]))

    hdf5_states_np = np.stack(hdf5_states, axis=0)
    rlds_states_np = np.stack(rlds_states, axis=0)
    max_diff_per_dim = per_dim_max_abs_diff(hdf5_states_np, rlds_states_np, state_dim_names)
    max_diff_value = float(np.max(np.abs(hdf5_states_np - rlds_states_np)))
    conclusion = "consistent_no_mismatch_detected" if mismatch_steps == 0 and max_diff_value <= 1e-6 else "mismatch_detected"
    conclusion_reason = (
        "HDF5 ee_states+gripper_pair and RLDS state match exactly across analyzed episodes; no state-semantic mismatch is evidenced."
        if conclusion == "consistent_no_mismatch_detected"
        else "At least one analyzed step differs between HDF5 state and RLDS state; inspect the reported dimensions."
    )

    return {
        "conclusion": conclusion,
        "reason": conclusion_reason,
        "num_episodes_analyzed": int(len(episode_reports)),
        "num_steps_analyzed": int(hdf5_states_np.shape[0]),
        "mismatch_step_count": int(mismatch_steps),
        "mismatch_episode_indices": mismatch_episodes,
        "max_abs_diff_overall": max_diff_value,
        "max_abs_diff_per_dim": max_diff_per_dim,
        "eval_state_contract": {
            "eef_position": "robot0_eef_pos",
            "eef_orientation": "quat2axisangle(robot0_eef_quat)",
            "gripper_state": "repeat(robot0_gripper_qpos, 2)[:2]",
        },
    }


def compute_normalization_analysis(
    episode_reports: Sequence[Dict[str, Any]],
    action_stats: Dict[str, Any],
    action_dim_names: Sequence[str],
) -> Dict[str, Any]:
    gt_raw_by_phase: Dict[str, List[np.ndarray]] = defaultdict(list)
    gt_norm_by_phase: Dict[str, List[np.ndarray]] = defaultdict(list)
    gt_norm_unclipped_by_phase: Dict[str, List[np.ndarray]] = defaultdict(list)

    for report in episode_reports:
        for record in report["step_records"]:
            phase = record["phase_rlds_action"]["signed_phase_label"]
            if phase not in {"-y", "-z"}:
                continue
            raw = deserialize_array(record["rlds_action"])
            norm = deserialize_array(record["rlds_action_normalized"])
            norm_unclipped = normalize_array(raw, action_stats, clip=False)
            gt_raw_by_phase[phase].append(raw)
            gt_norm_by_phase[phase].append(norm)
            gt_norm_unclipped_by_phase[phase].append(norm_unclipped)

    if not gt_raw_by_phase["-y"] or not gt_raw_by_phase["-z"]:
        return {
            "conclusion": "insufficient_phase_samples",
            "reason": "The analyzed window does not contain enough -y and -z samples to measure normalization separability.",
            "raw_phase_separability": None,
            "normalized_phase_separability": None,
            "raw_boundary_stats": None,
            "normalized_boundary_stats": None,
            "phase_sample_counts": {
                "-y": int(len(gt_raw_by_phase["-y"])),
                "-z": int(len(gt_raw_by_phase["-z"])),
            },
        }

    raw_y = np.stack(gt_raw_by_phase["-y"], axis=0)
    raw_z = np.stack(gt_raw_by_phase["-z"], axis=0)
    norm_y = np.stack(gt_norm_by_phase["-y"], axis=0)
    norm_z = np.stack(gt_norm_by_phase["-z"], axis=0)
    norm_unclipped_y = np.stack(gt_norm_unclipped_by_phase["-y"], axis=0)
    norm_unclipped_z = np.stack(gt_norm_unclipped_by_phase["-z"], axis=0)

    raw_metrics = nearest_centroid_metrics(raw_y, raw_z)
    normalized_metrics = nearest_centroid_metrics(norm_y, norm_z)
    raw_boundary_stats = per_dim_boundary_stats(
        np.concatenate([raw_y, raw_z], axis=0),
        np.concatenate([raw_y, raw_z], axis=0),
        action_dim_names,
    )
    normalized_boundary_stats = per_dim_boundary_stats(
        np.concatenate([norm_y, norm_z], axis=0),
        np.concatenate([norm_unclipped_y, norm_unclipped_z], axis=0),
        action_dim_names,
    )

    if normalized_metrics["nearest_centroid_accuracy"] + 1e-6 < raw_metrics["nearest_centroid_accuracy"]:
        conclusion = "normalization_reduces_phase_separability"
        reason = "The -y / -z nearest-centroid separation accuracy drops after bounds_q99 normalization."
    else:
        conclusion = "normalization_not_primary_cause"
        reason = "The -y / -z phase remains equally separable after bounds_q99 normalization; normalization is not the primary collapse source."

    return {
        "conclusion": conclusion,
        "reason": reason,
        "raw_phase_separability": raw_metrics,
        "normalized_phase_separability": normalized_metrics,
        "raw_boundary_stats": raw_boundary_stats,
        "normalized_boundary_stats": normalized_boundary_stats,
        "phase_sample_counts": {
            "-y": int(len(raw_y)),
            "-z": int(len(raw_z)),
        },
    }


def compute_phase_confusion_summary(episode_reports: Sequence[Dict[str, Any]], focus_steps: Sequence[int]) -> Dict[str, Any]:
    overall_counter: Counter[Tuple[str, str]] = Counter()
    focus_step_summary: Dict[str, Dict[str, Any]] = {}
    per_episode_diagnosis: List[Dict[str, Any]] = []

    for report in episode_reports:
        per_episode_diagnosis.append(
            {
                "episode_index": int(report["episode_summary"]["episode_index"]),
                "diagnosis": report["alignment_checks"]["root_cause_priority_decision"],
                "reason": report["alignment_checks"]["decision_reason"],
                "prediction_vs_action_phase_match_rate": report["alignment_checks"]["prediction_vs_action_phase_match_rate"],
            }
        )
        for record in report["step_records"]:
            gt_phase = record["phase_rlds_action"]["signed_phase_label"]
            pred_phase = record["phase_checkpoint_pred"]["signed_phase_label"]
            if gt_phase not in {"+x", "-y", "-z"} or pred_phase is None:
                continue
            overall_counter[(gt_phase, pred_phase)] += 1

    for focus_step in focus_steps:
        step_counter: Counter[Tuple[str, str]] = Counter()
        raw_examples = []
        for report in episode_reports:
            match = next((item for item in report["focus_steps"] if item["step"] == focus_step), None)
            if match is None:
                continue
            gt_phase = match["phase_rlds_action"]["signed_phase_label"]
            pred_phase = match["phase_checkpoint_pred"]["signed_phase_label"]
            step_counter[(gt_phase, pred_phase)] += 1
            raw_examples.append(
                {
                    "episode_index": int(report["episode_summary"]["episode_index"]),
                    "gt_phase": gt_phase,
                    "pred_phase": pred_phase,
                    "gt_action": match["rlds_action"],
                    "pred_action": match["checkpoint_pred_action"],
                }
            )
        focus_step_summary[str(focus_step)] = {
            "confusion_counts": {
                f"{gt}->{pred}": int(count) for (gt, pred), count in sorted(step_counter.items())
            },
            "examples": raw_examples,
        }

    return {
        "overall_confusion_counts": {
            f"{gt}->{pred}": int(count) for (gt, pred), count in sorted(overall_counter.items())
        },
        "focus_step_confusion": focus_step_summary,
        "per_episode_diagnosis": per_episode_diagnosis,
    }


def find_checkpoint_component(checkpoint_dir: Path, component_prefix: str) -> Path:
    matches = sorted(path for path in checkpoint_dir.iterdir() if component_prefix in path.name and "checkpoint" in path.name)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one component matching {component_prefix!r} in {checkpoint_dir}, found {matches}")
    return matches[0]


def load_component_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[7:]] = value
        else:
            cleaned_state_dict[key] = value
    return cleaned_state_dict


def patch_action_head_predict_dtype(action_head: L1RegressionActionHead, chunk_len: int) -> None:
    def predict_action_dtype_safe(
        actions_hidden_states: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        proprio_projector: Optional[torch.nn.Module] = None,
        phase: str = "Inference",
    ) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        device = actions_hidden_states.device
        dtype = actions_hidden_states.dtype

        proprio_tensor = torch.as_tensor(proprio, device=device, dtype=dtype).reshape(batch_size, -1)
        proprio_features = proprio_projector(proprio_tensor).unsqueeze(dim=1)

        task_hidden_states = actions_hidden_states[:, :, : action_head.num_task_tokens, :]
        action_hidden_states = actions_hidden_states[:, :, action_head.num_task_tokens :, :]

        cond_actions_hidden_states = torch.zeros(
            (batch_size, action_head.action_dim * chunk_len, action_head.hidden_dim),
            device=device,
            dtype=action_hidden_states.dtype,
        ).detach()

        rearranged_actions_hidden_states = cond_actions_hidden_states.reshape(batch_size, chunk_len, -1)
        return action_head.model(
            rearranged_actions_hidden_states,
            h_a=action_hidden_states,
            p=proprio_features,
            h_t=task_hidden_states,
        )

    action_head.predict_action = predict_action_dtype_safe


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    ]

    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replace_map:
            if old in new_key:
                new_key = new_key.replace(old, new)
        new_state_dict[new_key] = value
    return new_state_dict


def resolve_config_dir(checkpoint_dir: Path, base_config_dir: Path) -> Path:
    if (checkpoint_dir / "config.json").exists():
        return checkpoint_dir

    adapter_config_path = checkpoint_dir / "lora_adapter" / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text())
        raw_path = adapter_config.get("base_model_name_or_path", "")
        if raw_path:
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = (base_config_dir.parent / candidate).resolve()
            if candidate.is_file():
                return candidate.parent
            if candidate.is_dir():
                return candidate

    return base_config_dir


def load_base_vla_with_lora(
    checkpoint_dir: Path,
    cfg: SimpleNamespace,
) -> OpenVLAForActionPrediction:
    config_dir = resolve_config_dir(checkpoint_dir, cfg.base_config_dir)
    config = OpenVLAConfig.from_pretrained(str(config_dir), trust_remote_code=False)
    vla = OpenVLAForActionPrediction(config).to(device=DEVICE, dtype=MODEL_DTYPE)

    vlm = load_prismatic_vlm(str(cfg.base_vlm_dir), hf_token="", load_for_training=True)
    raw_state_dict = rename_state_dict_keys(vlm.state_dict())
    vla.load_state_dict(raw_state_dict, strict=False)
    del vlm
    del raw_state_dict

    adapter_dir = checkpoint_dir / "lora_adapter"
    peft_model = PeftModel.from_pretrained(vla, str(adapter_dir), is_trainable=False)
    merged_vla = peft_model.merge_and_unload()
    merged_vla.eval()
    return merged_vla.to(device=DEVICE, dtype=MODEL_DTYPE)


def load_proprio_projector(checkpoint_dir: Path, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim).to(device=DEVICE, dtype=MODEL_DTYPE)
    projector.load_state_dict(load_component_state_dict(find_checkpoint_component(checkpoint_dir, "proprio_projector")))
    projector.eval()
    return projector


def load_action_head(checkpoint_dir: Path, llm_dim: int, use_pro_version: bool, chunk_len: int) -> L1RegressionActionHead:
    action_head = L1RegressionActionHead(
        input_dim=llm_dim,
        hidden_dim=llm_dim,
        action_dim=len(ACTION_DIM_NAMES),
        use_pro_version=use_pro_version,
    ).to(device=DEVICE, dtype=MODEL_DTYPE)
    action_head.load_state_dict(load_component_state_dict(find_checkpoint_component(checkpoint_dir, "action_head")))
    action_head.eval()
    patch_action_head_predict_dtype(action_head, chunk_len=chunk_len)
    return action_head


def load_readonly_vla(checkpoint_dir: Path, all_stats: Dict[str, Any], cfg: SimpleNamespace) -> OpenVLAForActionPrediction:
    if (checkpoint_dir / "config.json").exists():
        config = OpenVLAConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=False)
        vla = OpenVLAForActionPrediction.from_pretrained(
            str(checkpoint_dir),
            config=config,
            trust_remote_code=False,
            torch_dtype=MODEL_DTYPE,
            low_cpu_mem_usage=False,
        )
    else:
        vla = load_base_vla_with_lora(checkpoint_dir, cfg)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    vla = vla.to(device=DEVICE, dtype=MODEL_DTYPE)
    vla.norm_stats = all_stats
    return vla


def predict_action_chunk(
    cfg: SimpleNamespace,
    vla: OpenVLAForActionPrediction,
    processor: PrismaticProcessor,
    action_head: torch.nn.Module,
    proprio_projector: torch.nn.Module,
    instruction: str,
    primary_image: np.ndarray,
    wrist_image: np.ndarray,
    state: np.ndarray,
    action_stats: Dict[str, Any],
) -> Dict[str, Any]:
    with torch.inference_mode():
        processed_images = prepare_images_for_vla([primary_image, wrist_image], cfg)
        if not cfg.use_minivlm:
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        else:
            prompt = (
                "<|im_start|>system\n"
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n"
                f"What action should the robot take to {instruction.lower()}?<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        inputs = processor(prompt, processed_images[0]).to(DEVICE, dtype=MODEL_DTYPE)
        wrist_inputs = [processor(prompt, image).to(DEVICE, dtype=MODEL_DTYPE) for image in processed_images[1:]]
        if wrist_inputs:
            wrist_pixels = [item["pixel_values"] for item in wrist_inputs]
            inputs["pixel_values"] = torch.cat([inputs["pixel_values"]] + wrist_pixels, dim=1)

        proprio_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        normalized_state = normalize_proprio(np.asarray(state, dtype=np.float32), proprio_stats)
        predicted_actions, _ = vla.predict_action(
            **inputs,
            unnorm_key=cfg.unnorm_key,
            do_sample=False,
            proprio=normalized_state,
            proprio_projector=proprio_projector,
            action_head=action_head,
            use_film=cfg.use_film,
        )

    predicted_actions = np.asarray(predicted_actions, dtype=np.float32)
    predicted_actions_normalized = normalize_array(predicted_actions, action_stats)
    return {
        "predicted_actions": predicted_actions,
        "predicted_actions_normalized": predicted_actions_normalized,
        "prepared_images": processed_images,
        "normalized_state": normalized_state,
    }


def select_window(actions: np.ndarray, step_index: int, horizon: int) -> np.ndarray:
    return np.asarray(actions[step_index : step_index + horizon], dtype=np.float32)


def detect_prediction_phase_collapse(focus_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gt_to_pred: Dict[str, set[str]] = {}
    for record in focus_records:
        gt = record["phase_rlds_action"]["signed_phase_label"]
        pred = record["phase_checkpoint_pred"]["signed_phase_label"]
        if gt in {"noop", "mixed", None}:
            continue
        if pred is None:
            continue
        gt_to_pred.setdefault(gt, set()).add(pred)

    collapse_pairs = []
    for gt_a, gt_b in combinations(sorted(gt_to_pred.keys()), 2):
        preds_a = gt_to_pred[gt_a]
        preds_b = gt_to_pred[gt_b]
        if len(preds_a) == 1 and preds_a == preds_b:
            collapse_pairs.append(
                {
                    "gt_phases": [gt_a, gt_b],
                    "collapsed_pred_phase": next(iter(preds_a)),
                }
            )

    return {
        "detected": bool(collapse_pairs),
        "pairs": collapse_pairs,
        "gt_to_pred_map": {key: sorted(values) for key, values in sorted(gt_to_pred.items())},
    }


def diagnosis_from_metrics(
    hdf5_mismatch_count: int,
    next_match_rate: float,
    prev_match_rate: float,
    prediction_collapse: Dict[str, Any],
) -> Tuple[str, str]:
    if hdf5_mismatch_count > 0:
        return (
            "conversion_issue",
            "RLDS action does not exactly match the source converted_hdf5 action at one or more analyzed steps.",
        )
    if next_match_rate < 0.5 and prev_match_rate > next_match_rate + 0.2:
        return (
            "temporal_or_label_misalignment",
            "action_t aligns better with state_t - state_{t-1} than with state_{t+1} - state_t, which suggests an off-by-one issue.",
        )
    if prediction_collapse["detected"]:
        return (
            "model_learning_or_config_issue",
            "HDF5/RLDS alignment is intact, but checkpoint predictions collapse distinct ground-truth phases into the same predicted phase.",
        )
    if next_match_rate < 0.8:
        return (
            "temporal_or_label_misalignment",
            "RLDS action and delta_state do not maintain a stable phase match across the analyzed window.",
        )
    return (
        "aligned_within_first_pass",
        "No conversion or obvious temporal shift issue was detected in the first-pass diagnostic window.",
    )


def write_episode_summary_md(report: Dict[str, Any], output_path: Path) -> None:
    checks = report["alignment_checks"]
    lines = [
        "# Suction Alignment Diagnosis",
        "",
        f"- diagnosis: `{checks['root_cause_priority_decision']}`",
        f"- reason: {checks['decision_reason']}",
        f"- episode: `{report['episode_summary']['episode_index']}`",
        f"- analyzed_steps: `{report['episode_summary']['start_step']}..{report['episode_summary']['end_step']}`",
        f"- instruction: `{report['episode_summary']['instruction']}`",
        "",
        "## Key Metrics",
        "",
        f"- hdf5_vs_rlds_action_mismatch_count: `{checks['hdf5_vs_rlds_action_mismatch_count']}`",
        f"- action_vs_delta_next_phase_match_rate: `{checks['action_vs_delta_next_phase_match_rate']:.4f}`",
        f"- action_vs_delta_prev_phase_match_rate: `{checks['action_vs_delta_prev_phase_match_rate']:.4f}`",
        f"- prediction_vs_action_phase_match_rate: `{checks['prediction_vs_action_phase_match_rate']:.4f}`",
        f"- prediction_phase_collapse_detected: `{checks['prediction_phase_collapse']['detected']}`",
        "",
        "## Focus Steps",
        "",
        "| step | gt_phase | pred_phase | delta_next_phase | pred_action_xyz | gt_action_xyz |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record in report["focus_steps"]:
        pred_xyz = record["checkpoint_pred_action"][:3]
        gt_xyz = record["rlds_action"][:3]
        lines.append(
            "| {step} | {gt} | {pred} | {delta} | [{px:.4f}, {py:.4f}, {pz:.4f}] | [{gx:.4f}, {gy:.4f}, {gz:.4f}] |".format(
                step=record["step"],
                gt=record["phase_rlds_action"]["signed_phase_label"],
                pred=record["phase_checkpoint_pred"]["signed_phase_label"],
                delta=record["phase_delta_state_next"]["signed_phase_label"],
                px=pred_xyz[0],
                py=pred_xyz[1],
                pz=pred_xyz[2],
                gx=gt_xyz[0],
                gy=gt_xyz[1],
                gz=gt_xyz[2],
            )
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_episode_diagnosis(
    args: argparse.Namespace,
    cfg: SimpleNamespace,
    stats_bundle: Dict[str, Any],
    processor: PrismaticProcessor,
    vla: OpenVLAForActionPrediction,
    action_head: torch.nn.Module,
    proprio_projector: torch.nn.Module,
    episode_index: int,
    focus_steps: Sequence[int],
    episode_output_dir: Path,
    export_snapshots: bool,
) -> Dict[str, Any]:
    episode_output_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = episode_output_dir / "snapshots"
    if export_snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading RLDS episode {episode_index} from {args.dataset_dir}...")
    rlds_episode = load_rlds_episode(args.dataset_dir, args.split, episode_index)
    rlds_steps = list(rlds_episode["steps"])
    instruction = decode_text(rlds_steps[0]["language_instruction"])
    episode_ref = parse_episode_ref(decode_text(rlds_episode["episode_metadata"]["file_path"]))

    print(f"Loading HDF5 episode from {episode_ref.source_path}::{episode_ref.demo_key}...")
    hdf5_episode = load_hdf5_episode(episode_ref)

    num_steps = len(rlds_steps)
    if args.start_step < 0 or args.end_step < args.start_step or args.end_step >= num_steps:
        raise ValueError(f"Invalid step range [{args.start_step}, {args.end_step}] for episode length {num_steps}")

    if instruction != hdf5_episode["instruction"] and hdf5_episode["instruction"]:
        print("WARNING: RLDS instruction differs from HDF5 instruction.")

    action_stats = stats_bundle["stats"]["action"]
    proprio_stats = stats_bundle["stats"]["proprio"]

    effective_num_steps = num_steps - (cfg.num_open_loop_steps - 1)
    if args.end_step >= effective_num_steps:
        raise ValueError(
            f"end_step={args.end_step} exceeds the valid training-chunk range {effective_num_steps - 1} "
            f"for num_open_loop_steps={cfg.num_open_loop_steps}"
        )

    step_records: List[Dict[str, Any]] = []
    focus_records: List[Dict[str, Any]] = []
    analyzed_rlds_actions: List[np.ndarray] = []
    analyzed_rlds_actions_normalized: List[np.ndarray] = []
    analyzed_pred_actions: List[np.ndarray] = []
    analyzed_pred_actions_normalized: List[np.ndarray] = []

    all_episode_actions = np.stack([np.asarray(step["action"], dtype=np.float32) for step in rlds_steps], axis=0)
    print(f"Running diagnosis over episode={episode_index} steps {args.start_step}..{args.end_step}...")
    for step_index in range(args.start_step, args.end_step + 1):
        rlds_step = rlds_steps[step_index]
        rlds_action = np.asarray(rlds_step["action"], dtype=np.float32)
        rlds_state = np.asarray(rlds_step["observation"]["state"], dtype=np.float32)
        hdf5_action = np.asarray(hdf5_episode["actions"][step_index], dtype=np.float32)
        hdf5_state = np.asarray(hdf5_episode["state"][step_index], dtype=np.float32)
        next_state = (
            np.asarray(rlds_steps[step_index + 1]["observation"]["state"], dtype=np.float32)
            if step_index + 1 < num_steps
            else None
        )
        prev_state = (
            np.asarray(rlds_steps[step_index - 1]["observation"]["state"], dtype=np.float32)
            if step_index - 1 >= 0
            else None
        )
        delta_state_next = (next_state - rlds_state).astype(np.float32) if next_state is not None else None
        delta_state_prev = (rlds_state - prev_state).astype(np.float32) if prev_state is not None else None

        hdf5_vs_rlds_match = bool(np.allclose(hdf5_action, rlds_action, atol=1e-6))
        hdf5_vs_rlds_state_match = bool(np.allclose(hdf5_state, rlds_state, atol=1e-6))
        rlds_action_normalized_unclipped = normalize_array(rlds_action, action_stats, clip=False)
        rlds_action_normalized = normalize_array(rlds_action, action_stats, clip=True)
        rlds_action_reconstructed = unnormalize_array(rlds_action_normalized, action_stats)
        state_normalized = normalize_array(rlds_state, proprio_stats, clip=True)
        state_reconstructed = unnormalize_array(state_normalized, proprio_stats)
        action_chunk_raw = select_window(all_episode_actions, step_index, cfg.num_open_loop_steps)
        action_chunk_normalized = normalize_array(action_chunk_raw, action_stats, clip=True)

        prediction = predict_action_chunk(
            cfg=cfg,
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            instruction=instruction,
            primary_image=np.asarray(rlds_step["observation"]["image"], dtype=np.uint8),
            wrist_image=np.asarray(rlds_step["observation"]["wrist_image"], dtype=np.uint8),
            state=rlds_state,
            action_stats=action_stats,
        )

        predicted_chunk = prediction["predicted_actions"]
        predicted_chunk_normalized = prediction["predicted_actions_normalized"]
        checkpoint_pred_action = predicted_chunk[0]
        checkpoint_pred_action_normalized = predicted_chunk_normalized[0]

        phase_rlds_action = motion_phase(rlds_action)
        phase_delta_next = motion_phase(delta_state_next)
        phase_delta_prev = motion_phase(delta_state_prev)
        phase_checkpoint_pred = motion_phase(checkpoint_pred_action)

        record = {
            "episode_index": episode_index,
            "step": step_index,
            "instruction": instruction,
            "hdf5_raw_action": serialize_array(hdf5_action),
            "hdf5_state": serialize_array(hdf5_state),
            "rlds_action": serialize_array(rlds_action),
            "rlds_action_normalized": serialize_array(rlds_action_normalized),
            "rlds_action_normalized_unclipped": serialize_array(rlds_action_normalized_unclipped),
            "rlds_action_reconstructed": serialize_array(rlds_action_reconstructed),
            "rlds_state": serialize_array(rlds_state),
            "rlds_state_normalized": serialize_array(state_normalized),
            "rlds_state_reconstructed": serialize_array(state_reconstructed),
            "state_semantics": {
                "eef_position": serialize_array(rlds_state[:3]),
                "eef_orientation": serialize_array(rlds_state[3:6]),
                "gripper_state": serialize_array(rlds_state[6:8]),
            },
            "delta_state_next": serialize_array(delta_state_next),
            "delta_state_prev": serialize_array(delta_state_prev),
            "train_action_chunk_raw": serialize_matrix(action_chunk_raw),
            "train_action_chunk_normalized": serialize_matrix(action_chunk_normalized),
            "checkpoint_pred_action": serialize_array(checkpoint_pred_action),
            "checkpoint_pred_action_normalized": serialize_array(checkpoint_pred_action_normalized),
            "checkpoint_pred_action_chunk": serialize_matrix(predicted_chunk),
            "checkpoint_pred_action_chunk_normalized": serialize_matrix(predicted_chunk_normalized),
            "phase_rlds_action": phase_rlds_action,
            "phase_delta_state_next": phase_delta_next,
            "phase_delta_state_prev": phase_delta_prev,
            "phase_checkpoint_pred": phase_checkpoint_pred,
            "hdf5_vs_rlds_action_allclose": hdf5_vs_rlds_match,
            "hdf5_vs_rlds_state_allclose": hdf5_vs_rlds_state_match,
            "hdf5_vs_rlds_action_max_abs_diff": max_abs_diff(hdf5_action, rlds_action),
            "hdf5_vs_rlds_state_max_abs_diff": max_abs_diff(hdf5_state, rlds_state),
            "phase_match_rlds_vs_delta_next": same_axis_phase(phase_rlds_action, phase_delta_next),
            "phase_match_rlds_vs_delta_prev": same_axis_phase(phase_rlds_action, phase_delta_prev),
            "phase_match_pred_vs_rlds": same_axis_phase(phase_checkpoint_pred, phase_rlds_action),
        }

        step_records.append(record)
        analyzed_rlds_actions.append(rlds_action)
        analyzed_rlds_actions_normalized.append(rlds_action_normalized)
        analyzed_pred_actions.append(checkpoint_pred_action)
        analyzed_pred_actions_normalized.append(checkpoint_pred_action_normalized)

        if step_index in focus_steps:
            focus_record = dict(record)
            if export_snapshots:
                primary_hdf5 = np.asarray(hdf5_episode["image"][step_index], dtype=np.uint8)
                primary_rlds = np.asarray(rlds_step["observation"]["image"], dtype=np.uint8)
                wrist_hdf5 = np.asarray(hdf5_episode["wrist_image"][step_index], dtype=np.uint8)
                wrist_rlds = np.asarray(rlds_step["observation"]["wrist_image"], dtype=np.uint8)
                primary_policy, wrist_policy = prediction["prepared_images"]

                primary_hdf5_path = snapshots_dir / f"step_{step_index:04d}_primary_hdf5.png"
                primary_rlds_path = snapshots_dir / f"step_{step_index:04d}_primary_rlds.png"
                primary_policy_path = snapshots_dir / f"step_{step_index:04d}_primary_policy.png"
                wrist_hdf5_path = snapshots_dir / f"step_{step_index:04d}_wrist_hdf5.png"
                wrist_rlds_path = snapshots_dir / f"step_{step_index:04d}_wrist_rlds.png"
                wrist_policy_path = snapshots_dir / f"step_{step_index:04d}_wrist_policy.png"
                contact_sheet_path = snapshots_dir / f"step_{step_index:04d}_contact_sheet.png"

                save_image(primary_hdf5, primary_hdf5_path)
                save_image(primary_rlds, primary_rlds_path)
                primary_policy.save(primary_policy_path)
                save_image(wrist_hdf5, wrist_hdf5_path)
                save_image(wrist_rlds, wrist_rlds_path)
                wrist_policy.save(wrist_policy_path)
                build_contact_sheet(
                    output_path=contact_sheet_path,
                    title=f"episode={episode_index} step={step_index} instruction={instruction}",
                    primary_hdf5=primary_hdf5,
                    primary_rlds=primary_rlds,
                    primary_policy=primary_policy,
                    wrist_hdf5=wrist_hdf5,
                    wrist_rlds=wrist_rlds,
                    wrist_policy=wrist_policy,
                )

                focus_record["snapshot_paths"] = {
                    "primary_hdf5": str(primary_hdf5_path),
                    "primary_rlds": str(primary_rlds_path),
                    "primary_policy": str(primary_policy_path),
                    "wrist_hdf5": str(wrist_hdf5_path),
                    "wrist_rlds": str(wrist_rlds_path),
                    "wrist_policy": str(wrist_policy_path),
                    "contact_sheet": str(contact_sheet_path),
                }
                focus_record["image_diff_metrics"] = {
                    "primary_hdf5_vs_rlds_mae": image_mae(primary_hdf5, primary_rlds),
                    "wrist_hdf5_vs_rlds_mae": image_mae(wrist_hdf5, wrist_rlds),
                }
            focus_records.append(focus_record)

    analyzed_rlds_actions_np = np.stack(analyzed_rlds_actions, axis=0)
    analyzed_rlds_actions_normalized_np = np.stack(analyzed_rlds_actions_normalized, axis=0)
    analyzed_pred_actions_np = np.stack(analyzed_pred_actions, axis=0)
    analyzed_pred_actions_normalized_np = np.stack(analyzed_pred_actions_normalized, axis=0)

    hdf5_mismatch_count = sum(0 if record["hdf5_vs_rlds_action_allclose"] else 1 for record in step_records)
    next_match_rate = safe_mean_bool([record["phase_match_rlds_vs_delta_next"] for record in step_records])
    prev_match_rate = safe_mean_bool([record["phase_match_rlds_vs_delta_prev"] for record in step_records])
    pred_match_rate = safe_mean_bool([record["phase_match_pred_vs_rlds"] for record in step_records])
    prediction_collapse = detect_prediction_phase_collapse(focus_records)
    diagnosis, decision_reason = diagnosis_from_metrics(
        hdf5_mismatch_count=hdf5_mismatch_count,
        next_match_rate=next_match_rate,
        prev_match_rate=prev_match_rate,
        prediction_collapse=prediction_collapse,
    )

    report = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(DEVICE),
            "normalization_type": str(ACTION_PROPRIO_NORMALIZATION_TYPE),
        },
        "paths": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "hdf5_dir": str(args.hdf5_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "output_dir": str(episode_output_dir),
            "episode_source": str(episode_ref.source_path),
            "episode_demo_key": episode_ref.demo_key,
        },
        "semantics": {
            "action_dim_names": ACTION_DIM_NAMES,
            "state_dim_names": STATE_DIM_NAMES,
            "state_layout": {
                "eef_position": [0, 1, 2],
                "eef_orientation": [3, 4, 5],
                "gripper_state": [6, 7],
            },
            "training_chunk_contract": {
                "window_size": 1,
                "future_action_window_size": cfg.num_open_loop_steps - 1,
                "current_action_is_action_t": True,
                "future_actions_cover": f"action_t ... action_t+{cfg.num_open_loop_steps - 1}",
            },
        },
        "episode_summary": {
            "split": args.split,
            "episode_index": episode_index,
            "instruction": instruction,
            "num_steps": num_steps,
            "effective_train_steps": effective_num_steps,
            "start_step": args.start_step,
            "end_step": args.end_step,
            "focus_steps": list(focus_steps),
            "snapshots_exported": bool(export_snapshots),
        },
        "dataset_reference_stats": {
            "unnorm_key": stats_bundle["key"],
            "action": stats_bundle["stats"]["action"],
            "proprio": stats_bundle["stats"]["proprio"],
        },
        "action_stats": {
            "rlds_action_raw_window": vector_stats(analyzed_rlds_actions_np, ACTION_DIM_NAMES),
            "rlds_action_normalized_window": vector_stats(analyzed_rlds_actions_normalized_np, ACTION_DIM_NAMES),
            "checkpoint_pred_action_raw_window": vector_stats(analyzed_pred_actions_np, ACTION_DIM_NAMES),
            "checkpoint_pred_action_normalized_window": vector_stats(analyzed_pred_actions_normalized_np, ACTION_DIM_NAMES),
        },
        "phase_length_stats": phase_length_stats(record["phase_rlds_action"]["signed_phase_label"] for record in step_records),
        "alignment_checks": {
            "hdf5_vs_rlds_action_mismatch_count": hdf5_mismatch_count,
            "action_vs_delta_next_phase_match_rate": next_match_rate,
            "action_vs_delta_prev_phase_match_rate": prev_match_rate,
            "prediction_vs_action_phase_match_rate": pred_match_rate,
            "prediction_phase_collapse": prediction_collapse,
            "root_cause_priority_decision": diagnosis,
            "decision_reason": decision_reason,
        },
        "focus_steps": focus_records,
        "step_records": step_records,
    }

    report_path = episode_output_dir / "report.json"
    summary_path = episode_output_dir / "summary.md"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_episode_summary_md(report, summary_path)
    print(f"Wrote report: {report_path}")
    print(f"Wrote summary: {summary_path}")
    return report


def write_aggregate_summary_md(report: Dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Suction Aggregate Diagnosis",
        "",
        f"- checkpoint: `{Path(report['paths']['checkpoint']).name}`",
        f"- split: `{report['aggregate_summary']['split']}`",
        f"- num_episodes_analyzed: `{report['aggregate_summary']['num_episodes_analyzed']}`",
        f"- analyzed_step_range: `{report['aggregate_summary']['start_step']}..{report['aggregate_summary']['end_step']}`",
        "",
        "## Conclusions",
        "",
        f"- state_semantics: `{report['state_semantics_analysis']['conclusion']}`",
        f"- state_reason: {report['state_semantics_analysis']['reason']}",
        f"- normalization: `{report['normalization_analysis']['conclusion']}`",
        f"- normalization_reason: {report['normalization_analysis']['reason']}",
        "",
        "## Episode Diagnoses",
        "",
    ]
    diagnosis_counts = Counter(item["diagnosis"] for item in report["phase_confusion_analysis"]["per_episode_diagnosis"])
    for diagnosis, count in sorted(diagnosis_counts.items()):
        lines.append(f"- {diagnosis}: `{count}`")
    lines.extend(
        [
            "",
            "## Phase Confusion",
            "",
        ]
    )
    for key, count in sorted(report["phase_confusion_analysis"]["overall_confusion_counts"].items()):
        lines.append(f"- {key}: `{count}`")
    lines.extend(
        [
            "",
            "## Key Steps",
            "",
            "| step | confusion |",
            "| --- | --- |",
        ]
    )
    for step, details in sorted(report["phase_confusion_analysis"]["focus_step_confusion"].items(), key=lambda item: int(item[0])):
        confusion = ", ".join(f"{key}={value}" for key, value in sorted(details["confusion_counts"].items()))
        lines.append(f"| {step} | {confusion or 'n/a'} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    focus_steps = parse_int_list(args.focus_steps)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("/home/x/vla/VLA-Adapter-main/tmp") / f"diagnose_suction_alignment_{timestamp}"
    output_dir = ensure_output_dir(output_dir, overwrite=args.overwrite)

    snapshot_episodes = parse_int_list(args.snapshot_episodes) if args.snapshot_episodes else [args.episode_index]
    stats_bundle = load_checkpoint_dataset_stats(args.checkpoint, args.unnorm_key)
    cfg = build_cfg(args, stats_bundle["key"])

    num_episodes = get_split_num_episodes(args.dataset_dir, args.split)
    if args.all_episodes:
        episode_indices = list(range(num_episodes))
        if args.episode_limit > 0:
            episode_indices = episode_indices[: args.episode_limit]
    else:
        episode_indices = [args.episode_index]

    print(f"Loading checkpoint components from {args.checkpoint} on {DEVICE}...")
    image_processor = PrismaticImageProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), trust_remote_code=False)
    processor = PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)
    vla = load_readonly_vla(args.checkpoint, stats_bundle["all_stats"], cfg)
    action_head = load_action_head(
        args.checkpoint,
        vla.llm_dim,
        use_pro_version=cfg.use_pro_version,
        chunk_len=cfg.num_open_loop_steps,
    )
    proprio_projector = load_proprio_projector(args.checkpoint, vla.llm_dim, proprio_dim=8)

    episode_reports: List[Dict[str, Any]] = []
    episodes_root = output_dir / "episodes"
    if len(episode_indices) > 1:
        episodes_root.mkdir(parents=True, exist_ok=True)

    for episode_index in episode_indices:
        episode_output_dir = output_dir if len(episode_indices) == 1 else episodes_root / f"episode_{episode_index:03d}"
        report = run_episode_diagnosis(
            args=args,
            cfg=cfg,
            stats_bundle=stats_bundle,
            processor=processor,
            vla=vla,
            action_head=action_head,
            proprio_projector=proprio_projector,
            episode_index=episode_index,
            focus_steps=focus_steps,
            episode_output_dir=episode_output_dir,
            export_snapshots=episode_index in snapshot_episodes,
        )
        episode_reports.append(report)

    if len(episode_reports) == 1:
        return

    aggregate_report = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(DEVICE),
            "normalization_type": str(ACTION_PROPRIO_NORMALIZATION_TYPE),
        },
        "paths": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "hdf5_dir": str(args.hdf5_dir.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "output_dir": str(output_dir.resolve()),
            "episodes_root": str(episodes_root.resolve()),
        },
        "aggregate_summary": {
            "split": args.split,
            "num_episodes_analyzed": len(episode_reports),
            "episode_indices": episode_indices,
            "start_step": args.start_step,
            "end_step": args.end_step,
            "focus_steps": focus_steps,
            "snapshot_episodes": snapshot_episodes,
        },
        "state_semantics_analysis": compute_state_semantics_analysis(episode_reports, STATE_DIM_NAMES),
        "normalization_analysis": compute_normalization_analysis(episode_reports, stats_bundle["stats"]["action"], ACTION_DIM_NAMES),
        "phase_confusion_analysis": compute_phase_confusion_summary(episode_reports, focus_steps),
    }

    aggregate_report_path = output_dir / "aggregate_report.json"
    aggregate_summary_path = output_dir / "aggregate_summary.md"
    aggregate_report_path.write_text(json.dumps(aggregate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_aggregate_summary_md(aggregate_report, aggregate_summary_path)
    print(f"Wrote aggregate report: {aggregate_report_path}")
    print(f"Wrote aggregate summary: {aggregate_summary_path}")


if __name__ == "__main__":
    main()
