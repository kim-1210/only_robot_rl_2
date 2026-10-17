from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def position_command_error(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, 
        des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  
    return torch.norm(curr_pos_w - des_pos_w, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRLEnv, 
    std: float, 
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, 
        des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]] 
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)

def orientation_command_error(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    command_name: str,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(
        asset.data.root_quat_w, 
        des_quat_b
    )
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    return quat_error_magnitude(curr_quat_w, des_quat_w)