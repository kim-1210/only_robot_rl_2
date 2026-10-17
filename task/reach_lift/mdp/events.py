from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_mul, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def change_phase(
    env,
    env_ids,
    robot_cfg:SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg:SceneEntityCfg = SceneEntityCfg("ee_frame"),
    obj_cfg:SceneEntityCfg = SceneEntityCfg("object"),
    command_name = "obj_pose",
    threshold_1: float = 0.02,
    threshold_2: float = 0.02,
):
    robot = env.scene[robot_cfg.name]
    obj = env.scene[obj_cfg.name]
    ee = env.scene[ee_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        command[:, :3]
    )
    dist = torch.norm(obj.data.root_pos_w - des_pos_w, dim=1)
    mask = (env.phase == 1) & (dist <= threshold_2)
    env.phase[mask] = 2

    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w, dim=1)
    mask = (env.phase == 0) & (dist <= threshold_1)
    env.phase[mask] = 1

def zero_phase(
    env,
    env_ids,
    threshold_1: float = 0.02,
    ee_cfg:SceneEntityCfg = SceneEntityCfg("ee_frame"),
    obj_cfg:SceneEntityCfg = SceneEntityCfg("object"),
):
    obj = env.scene[obj_cfg.name]
    ee = env.scene[ee_cfg.name]

    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w, dim=1)
    mask = (dist > threshold_1)
    env.phase[mask] = 0
    
def reset_phase(
    env,
    env_ids,
):
    env.phase[env_ids] = 0