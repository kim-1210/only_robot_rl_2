from __future__ import annotations

import torch
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz


def sample_pose_command(
    num_envs: int,
    device: torch.device | str,
    *,
    pos_x: tuple[float, float],
    pos_y: tuple[float, float],
    pos_z: tuple[float, float],
    roll: tuple[float, float],
    pitch: tuple[float, float],
    yaw: tuple[float, float],
) -> torch.Tensor:
    pos = torch.empty((num_envs, 3), device=device)
    pos[:, 0].uniform_(*pos_x)
    pos[:, 1].uniform_(*pos_y)
    pos[:, 2].uniform_(*pos_z)

    roll_t = torch.empty(num_envs, device=device).uniform_(*roll)
    pitch_t = torch.empty(num_envs, device=device).uniform_(*pitch)
    yaw_t = torch.empty(num_envs, device=device).uniform_(*yaw)
    quat = quat_from_euler_xyz(roll_t, pitch_t, yaw_t)
    return torch.cat((pos, quat), dim=-1)


def target_pos_world(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    pose_command: torch.Tensor,
) -> torch.Tensor:
    target_pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, pose_command[:, :3])
    return target_pos_w
