from __future__ import annotations

import torch
from isaaclab.assets import Articulation


def build_policy_observation(
    robot: Articulation,
    pose_command: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            robot.data.joint_pos,
            robot.data.joint_vel,
            pose_command,
        ),
        dim=-1,
    )
