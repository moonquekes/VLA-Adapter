"""Utilities for transition-focused action loss weighting and offline metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransitionLossConfig:
    """Resolved transition-weighting configuration for action-chunk training."""

    loss_type: str
    future_horizon_weights: Tuple[float, ...]
    transition_chunk_weight: float
    transition_step_radius: int
    transition_dim_weights: Tuple[float, ...]
    transition_gripper_threshold: float
    transition_weight_warmup_steps: int


def parse_float_list(raw_value: str | Iterable[float], expected_len: int, field_name: str) -> Tuple[float, ...]:
    """Parse a comma-separated float list and validate its length."""
    if isinstance(raw_value, str):
        values = tuple(float(item.strip()) for item in raw_value.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in raw_value)

    if len(values) != expected_len:
        raise ValueError(f"{field_name} must contain exactly {expected_len} values, got {len(values)}: {values}")

    return values


def resolve_transition_loss_config(cfg, action_dim: int, num_actions_chunk: int) -> TransitionLossConfig:
    """Resolve transition-loss config values from CLI / config strings."""
    loss_type = str(getattr(cfg, "loss_type", "l1")).strip().lower()
    if loss_type not in {"l1", "smooth_l1"}:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    future_horizon_weights = parse_float_list(
        getattr(cfg, "future_horizon_weights", "1.0,1.2,1.5,2.0,2.5,3.0,3.0"),
        num_actions_chunk - 1,
        "future_horizon_weights",
    )
    transition_dim_weights = parse_float_list(
        getattr(cfg, "transition_dim_weights", "1.25,1.25,2.5,1.0,1.0,1.0,4.0"),
        action_dim,
        "transition_dim_weights",
    )

    return TransitionLossConfig(
        loss_type=loss_type,
        future_horizon_weights=future_horizon_weights,
        transition_chunk_weight=float(getattr(cfg, "transition_chunk_weight", 4.0)),
        transition_step_radius=int(getattr(cfg, "transition_step_radius", 1)),
        transition_dim_weights=transition_dim_weights,
        transition_gripper_threshold=float(getattr(cfg, "transition_gripper_threshold", 0.5)),
        transition_weight_warmup_steps=int(getattr(cfg, "transition_weight_warmup_steps", 2000)),
    )


def get_warmup_progress(global_step: int, max_steps: int, config: TransitionLossConfig) -> float:
    """Return linear warmup progress for transition-specific weights."""
    if config.transition_weight_warmup_steps <= 0:
        return 1.0

    capped_warmup_steps = min(config.transition_weight_warmup_steps, max(1, int(0.1 * max_steps)))
    if capped_warmup_steps <= 0:
        return 1.0

    return min(max(float(global_step) / float(capped_warmup_steps), 0.0), 1.0)


def _interpolate_weight(target_weight: float | torch.Tensor, warmup_progress: float) -> float | torch.Tensor:
    """Interpolate an extra weight from 1.0 to its target value."""
    return 1.0 + warmup_progress * (target_weight - 1.0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the masked mean or return NaN when the mask is empty."""
    if mask.dtype != torch.bool:
        mask = mask.bool()

    valid_count = mask.sum()
    if valid_count.item() == 0:
        return values.new_tensor(float("nan"))

    return values.masked_select(mask).mean()


def compute_transition_masks(actions: torch.Tensor, gripper_threshold: float, step_radius: int) -> dict[str, torch.Tensor]:
    """
    Compute chunk- and step-level transition masks from gripper actions.

    A transition step is centered on the first timestep where the binarized gripper
    state changes, then expanded by `step_radius` on both sides.
    """
    if actions.ndim != 3:
        raise ValueError(f"Expected actions with shape (batch, chunk, dim), got {tuple(actions.shape)}")

    batch_size, chunk_len, _ = actions.shape
    gripper_binary = actions[..., -1] > gripper_threshold
    flip_pair_mask = gripper_binary[:, 1:] != gripper_binary[:, :-1]
    chunk_has_flip = flip_pair_mask.any(dim=1)

    transition_step_mask = torch.zeros((batch_size, chunk_len), dtype=torch.bool, device=actions.device)
    flip_locations = torch.nonzero(flip_pair_mask, as_tuple=False)
    for batch_idx, pair_idx in flip_locations.tolist():
        center_step = pair_idx + 1
        start_step = max(0, center_step - step_radius)
        end_step = min(chunk_len, center_step + step_radius + 1)
        transition_step_mask[batch_idx, start_step:end_step] = True

    return {
        "gripper_binary": gripper_binary,
        "flip_pair_mask": flip_pair_mask,
        "chunk_has_flip": chunk_has_flip,
        "transition_step_mask": transition_step_mask,
    }


def _get_base_action_loss(loss_type: str, predicted_actions: torch.Tensor, ground_truth_actions: torch.Tensor) -> torch.Tensor:
    """Return unreduced elementwise action loss."""
    if loss_type == "smooth_l1":
        # `smooth_l1_cuda` is not implemented for bf16 on this stack, so compute the
        # elementwise loss in fp32 and let autograd cast gradients back as needed.
        return F.smooth_l1_loss(predicted_actions.float(), ground_truth_actions.float(), reduction="none")
    if loss_type == "l1":
        return torch.abs(predicted_actions - ground_truth_actions)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def compute_transition_weighted_loss(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    config: TransitionLossConfig,
    global_step: int,
    max_steps: int,
    curr_action_loss_weight: float = 1.0,
    future_action_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute current-step loss plus transition-weighted future loss."""
    if predicted_actions.shape != ground_truth_actions.shape:
        raise ValueError(
            f"Predicted and ground-truth actions must match shape, got {tuple(predicted_actions.shape)} vs "
            f"{tuple(ground_truth_actions.shape)}"
        )

    base_loss = _get_base_action_loss(config.loss_type, predicted_actions, ground_truth_actions)
    transition_masks = compute_transition_masks(
        ground_truth_actions,
        gripper_threshold=config.transition_gripper_threshold,
        step_radius=config.transition_step_radius,
    )
    warmup_progress = get_warmup_progress(global_step, max_steps, config)

    current_loss = base_loss[:, 0, :].mean()

    if predicted_actions.shape[1] == 1:
        future_loss = current_loss.new_zeros(())
        per_chunk_future_loss = current_loss.new_empty((predicted_actions.shape[0],)).fill_(float("nan"))
    else:
        future_base_loss = base_loss[:, 1:, :]
        horizon_weights = torch.tensor(
            config.future_horizon_weights,
            dtype=future_base_loss.dtype,
            device=future_base_loss.device,
        ).view(1, -1, 1)

        future_transition_mask = transition_masks["transition_step_mask"][:, 1:]
        target_dim_weights = torch.tensor(
            config.transition_dim_weights,
            dtype=future_base_loss.dtype,
            device=future_base_loss.device,
        )
        warmed_dim_weights = _interpolate_weight(target_dim_weights, warmup_progress)
        dim_weights = torch.ones_like(future_base_loss)
        dim_weights = torch.where(
            future_transition_mask.unsqueeze(-1),
            warmed_dim_weights.view(1, 1, -1).expand_as(future_base_loss),
            dim_weights,
        )

        future_element_weights = horizon_weights * dim_weights
        per_chunk_future_loss = (future_base_loss * future_element_weights).sum(dim=(1, 2))
        per_chunk_future_loss = per_chunk_future_loss / future_element_weights.sum(dim=(1, 2)).clamp_min(1e-8)

        sample_chunk_weight = future_base_loss.new_ones((future_base_loss.shape[0],))
        warmed_chunk_weight = float(_interpolate_weight(config.transition_chunk_weight, warmup_progress))
        sample_chunk_weight = torch.where(
            transition_masks["chunk_has_flip"],
            sample_chunk_weight.new_full(sample_chunk_weight.shape, warmed_chunk_weight),
            sample_chunk_weight,
        )
        future_loss = (per_chunk_future_loss * sample_chunk_weight).sum()
        future_loss = future_loss / sample_chunk_weight.sum().clamp_min(1e-8)

    total_loss = curr_action_loss_weight * current_loss + future_action_loss_weight * future_loss

    return total_loss, {
        "base_loss": base_loss,
        "current_loss": current_loss,
        "future_loss": future_loss,
        "per_chunk_future_loss": per_chunk_future_loss,
        "warmup_progress": predicted_actions.new_tensor(warmup_progress),
        **transition_masks,
    }


def compute_transition_metrics(
    predicted_actions: torch.Tensor,
    ground_truth_actions: torch.Tensor,
    loss_details: dict[str, torch.Tensor],
    config: TransitionLossConfig,
) -> dict[str, float]:
    """Compute offline diagnostics for chunk quality around gripper transitions."""
    abs_error = torch.abs(predicted_actions - ground_truth_actions)
    batch_size, chunk_len, _ = abs_error.shape

    metrics: dict[str, float] = {
        "curr_action_loss": float(loss_details["current_loss"].item()),
        "future_action_loss": float(loss_details["future_loss"].item()),
        "transition_weight_warmup_progress": float(loss_details["warmup_progress"].item()),
    }

    for step_idx in range(chunk_len):
        metrics[f"horizon_mae_step{step_idx}"] = float(abs_error[:, step_idx, :].mean().item())

    metrics["curr_action_mae"] = float(abs_error[:, 0, :].mean().item())
    if chunk_len > 1:
        metrics["future_action_mae"] = float(abs_error[:, 1:, :].mean().item())
    else:
        metrics["future_action_mae"] = float("nan")

    future_transition_step_mask = loss_details["transition_step_mask"][:, 1:] if chunk_len > 1 else None
    metrics["future_z_mae"] = float(abs_error[:, 1:, 2].mean().item()) if chunk_len > 1 else float("nan")
    metrics["transition_future_z_mae"] = (
        float(_masked_mean(abs_error[:, 1:, 2], future_transition_step_mask).item())
        if chunk_len > 1
        else float("nan")
    )
    metrics["future_gripper_mae"] = float(abs_error[:, 1:, -1].mean().item()) if chunk_len > 1 else float("nan")
    metrics["transition_future_gripper_mae"] = (
        float(_masked_mean(abs_error[:, 1:, -1], future_transition_step_mask).item())
        if chunk_len > 1
        else float("nan")
    )

    predicted_gripper_binary = predicted_actions[..., -1] > config.transition_gripper_threshold
    predicted_flip_pair_mask = predicted_gripper_binary[:, 1:] != predicted_gripper_binary[:, :-1]
    gt_flip_pair_mask = loss_details["flip_pair_mask"]
    pairwise_flip_accuracy = (predicted_flip_pair_mask == gt_flip_pair_mask).float()
    metrics["gripper_flip_accuracy"] = float(pairwise_flip_accuracy.mean().item())
    metrics["transition_gripper_flip_accuracy"] = float(
        _masked_mean(pairwise_flip_accuracy, gt_flip_pair_mask).item()
    )

    per_chunk_future_loss = loss_details["per_chunk_future_loss"]
    chunk_has_flip = loss_details["chunk_has_flip"]
    metrics["transition_chunk_loss"] = float(_masked_mean(per_chunk_future_loss, chunk_has_flip).item())
    metrics["non_transition_chunk_loss"] = float(_masked_mean(per_chunk_future_loss, ~chunk_has_flip).item())
    metrics["transition_chunk_fraction"] = float(chunk_has_flip.float().mean().item())

    return metrics
