from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_mul, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_ee_distance(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    return torch.norm(cube_pos_w - ee_w, dim=1)

# 물체와 ee의 거리
def object_ee_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)

# 물체를 들었는가에 대한 여부 
def object_is_lifted(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    object = env.scene[object_cfg.name]
    condition = (object.data.root_pos_w[:, 2] > minimal_height) & (env.phase >= 1)
    
    return torch.where(
        condition, 
        torch.ones_like(condition, dtype=torch.float32), 
        torch.zeros_like(condition, dtype=torch.float32)
    )

def orientation_command_error(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    command_name: str = "obj_pose",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    asset = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(robot.data.root_quat_w, des_quat_b)
    
    obj_quat_w = object.data.root_quat_w
    curr_quat_w = asset.data.target_quat_w[:, 0, :]
    phase = env.phase

    error_to_obj = quat_error_magnitude(curr_quat_w, obj_quat_w)
    error_to_des = quat_error_magnitude(curr_quat_w, des_quat_w)

    return torch.where(phase == 0, error_to_obj, error_to_des)

# 물체와 lift할 목표의 거리
def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)

    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], 
        des_pos_b
    )
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    condition = (object.data.root_pos_w[:, 2] > minimal_height) & (env.phase >= 1)
    
    return condition.float() * (1 - torch.tanh(distance / std))

# 물체와 ee의 거리와 들었는가 여부
def object_ee_distance_and_lifted(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    reach_reward = object_ee_distance(
        env, std, object_cfg, ee_frame_cfg
    )
    lift_reward = object_is_lifted(
        env, minimal_height, object_cfg
    )
    
    return reach_reward * lift_reward * (env.phase >= 1).float()