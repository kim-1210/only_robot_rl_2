from __future__ import annotations

import torch


def absolute_joint_targets_to_policy_action(
    joint_targets: torch.Tensor,
    default_joint_pos: torch.Tensor,
    arm_joint_ids: torch.Tensor,
    left_finger_id: int,
    right_finger_id: int,
    arm_action_scale: float,
    close_threshold: float,
) -> torch.Tensor:
    arm_actions = (joint_targets[:, arm_joint_ids] - default_joint_pos[:, arm_joint_ids]) / max(arm_action_scale, 1e-6)
    finger_mean = 0.5 * (joint_targets[:, left_finger_id] + joint_targets[:, right_finger_id])
    gripper_action = torch.where(
        finger_mean <= close_threshold,
        -torch.ones_like(finger_mean),
        torch.ones_like(finger_mean),
    ).unsqueeze(-1)
    return torch.cat((arm_actions, gripper_action), dim=-1)
