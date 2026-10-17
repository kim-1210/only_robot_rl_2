from __future__ import annotations

import torch
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul


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


def sample_object_poses(
    num_envs: int,
    device: torch.device | str,
    *,
    pos_x: tuple[float, float],
    pos_y: tuple[float, float],
    pos_z: float,
    yaw: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.empty((num_envs, 3), device=device)
    pos[:, 0].uniform_(*pos_x)
    pos[:, 1].uniform_(*pos_y)
    pos[:, 2] = pos_z

    yaw_t = torch.empty(num_envs, device=device).uniform_(*yaw)
    quat = quat_from_euler_xyz(
        torch.zeros(num_envs, device=device),
        torch.zeros(num_envs, device=device),
        yaw_t,
    )
    return pos, quat


__all__ = ["quat_mul", "sample_pose_command", "sample_object_poses"]
