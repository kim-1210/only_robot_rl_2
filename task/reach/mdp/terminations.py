from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def success_cube(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_pose",
    goal_threshold: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["ee_link"]),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]

    asset: Articulation = env.scene[asset_cfg.name]

    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        des_pos_b,
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    return torch.norm(curr_pos_w - des_pos_w, dim=1) < goal_threshold
