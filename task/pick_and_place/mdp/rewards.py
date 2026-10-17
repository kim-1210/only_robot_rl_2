from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _gripper_open_close_progress(
    env,
    robot_cfg: SceneEntityCfg,
    progress_scale: float,
):
    """Returns averaged open/close progress for the two gripper fingers."""
    robot = env.scene[robot_cfg.name]

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
    open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32, device=env.device)

    finger1 = robot.data.joint_pos[:, gripper_joint_ids[0]]
    finger2 = robot.data.joint_pos[:, gripper_joint_ids[1]]

    open_progress_1 = torch.clamp(1.0 - torch.abs(open_val - finger1) / progress_scale, min=0.0, max=1.0)
    open_progress_2 = torch.clamp(1.0 - torch.abs(open_val - finger2) / progress_scale, min=0.0, max=1.0)
    open_progress = 0.5 * (open_progress_1 + open_progress_2)
    close_progress = 1.0 - open_progress

    return open_progress, close_progress


def _lift_progress(
    object_height: torch.Tensor,
    minimal_height: float,
    base_height: float,
) -> torch.Tensor:
    """Normalized lift progress from the tabletop height to the lift threshold."""
    height_span = max(minimal_height - base_height, 1e-6)
    return torch.clamp((object_height - base_height) / height_span, min=0.0, max=1.0)

def obj_ee_dist_reward(
    env,
    std,
    ee_cfg = SceneEntityCfg("ee_frame"),
    obj1_cfg = SceneEntityCfg("obj1")
):
    ee = env.scene[ee_cfg.name]
    obj1 = env.scene[obj1_cfg.name]

    dist = torch.norm(ee.data.target_pos_w[..., 0, :] - obj1.data.root_pos_w, dim=1)
    return 1 - torch.tanh(dist / std)

def obj_grasp_reward(
    env,
    obj1_cfg = SceneEntityCfg("obj1"),
    ee_cfg = SceneEntityCfg("ee_frame"),
    robot_cfg = SceneEntityCfg("robot"),
    diff_threshold = 0.04,
):
    ee = env.scene[ee_cfg.name]
    obj1 = env.scene[obj1_cfg.name]
    robot = env.scene[robot_cfg.name]

    obj1_pos = obj1.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 0, :]
    pos_diff = torch.norm(obj1_pos - ee_pos, dim=1)

    if hasattr(env.cfg, "gripper_joint_names"):
        gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
        open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)

        grasped = torch.logical_and(
            pos_diff < diff_threshold,
            torch.abs(robot.data.joint_pos[:, gripper_joint_ids[0]] - open_val) > env.cfg.gripper_threshold,
        )
        grasped = torch.logical_and(
            grasped,
            torch.abs(robot.data.joint_pos[:, gripper_joint_ids[1]] - open_val) > env.cfg.gripper_threshold,
        )
        return grasped.float()

    return torch.zeros(env.num_envs, device=env.device)


def obj_grasp_dense_reward(
    env,
    diff_threshold=0.04,
    gripper_close_scale=0.02,
    obj1_cfg=SceneEntityCfg("obj1"),
    ee_cfg=SceneEntityCfg("ee_frame"),
    robot_cfg=SceneEntityCfg("robot"),
):
    """Dense grasp reward from EE-object proximity and gripper closing progress."""
    ee = env.scene[ee_cfg.name]
    obj1 = env.scene[obj1_cfg.name]

    obj1_pos = obj1.data.root_pos_w
    ee_pos = ee.data.target_pos_w[..., 0, :]
    pos_diff = torch.norm(obj1_pos - ee_pos, dim=1)
    proximity_reward = 1.0 - torch.tanh(pos_diff / diff_threshold)

    if hasattr(env.cfg, "gripper_joint_names"):
        _, close_reward = _gripper_open_close_progress(env, robot_cfg, gripper_close_scale)
        return proximity_reward * (0.35 + 0.65 * close_reward)

    return proximity_reward
    
def lifting_obj_reward(
    env,
    minimal_height,
    obj1_cfg = SceneEntityCfg("obj1")
):
    obj1 = env.scene[obj1_cfg.name]
    return torch.where(obj1.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def lifting_obj_progress_reward(
    env,
    minimal_height,
    base_height=1.066,
    obj1_cfg=SceneEntityCfg("obj1"),
):
    """Dense lifting reward from current object height progress."""
    obj1 = env.scene[obj1_cfg.name]
    object_height = obj1.data.root_pos_w[:, 2]
    return _lift_progress(object_height, minimal_height, base_height)
    
def obj_stacking_reward(
    env,
    std,
    minimal_height,
    base_height,
    add_target_height,
    height_std=0.05,
    obj1_cfg = SceneEntityCfg("obj1"),
    obj2_cfg = SceneEntityCfg("obj2"),
):
    """Dense stage-3 reward for moving the lifted object above the target object."""
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    pos_diff = obj1.data.root_pos_w - obj2.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    target_height = obj2.data.root_pos_w[:, 2] + add_target_height
    height_error = torch.abs(obj1.data.root_pos_w[:, 2] - target_height)

    lift_progress = _lift_progress(obj1.data.root_pos_w[:, 2], minimal_height, base_height)
    xy_reward = 1.0 - torch.tanh(xy_dist / std)
    height_reward = 1.0 - torch.tanh(height_error / height_std)
    above_object = torch.clamp(pos_diff[:, 2] / add_target_height, min=0.0, max=1.0)

    return xy_reward * height_reward * above_object * (0.25 + 0.75 * lift_progress)
    
def obj_height_alignment_reward(
    env,
    std,
    target_height,
    xy_std=0.08,
    minimal_height=1.13,
    base_height=1.066,
    obj1_cfg = SceneEntityCfg("obj1"),
    obj2_cfg = SceneEntityCfg("obj2"),
):
    """Dense stage-4 reward for matching the final stack height after XY alignment."""
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    pos_diff = obj1.data.root_pos_w - obj2.data.root_pos_w
    height_diff = pos_diff[:, 2]

    height_error = torch.abs(height_diff - target_height)
    height_reward = 1 - torch.tanh(height_error / std)

    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    xy_reward = 1 - torch.tanh(xy_dist / xy_std)

    lift_progress = _lift_progress(obj1.data.root_pos_w[:, 2], minimal_height, base_height)
    above_object = torch.clamp(height_diff / target_height, min=0.0, max=1.0)
    xy_gate = torch.clamp(1.0 - xy_dist / (xy_std * 1.5), min=0.0, max=1.0)

    return height_reward * xy_reward * xy_gate * above_object * (0.25 + 0.75 * lift_progress)
    
def stacking_success_reward(
    env,
    obj1_cfg: SceneEntityCfg = SceneEntityCfg("obj1"),
    obj2_cfg: SceneEntityCfg = SceneEntityCfg("obj2"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.04,
    xy_std: float = 0.04,
    height_std: float = 0.01,
    release_std: float = 0.01,
):
    """Dense near-success reward for stacked geometry plus gripper release."""
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    pos_diff = obj1.data.root_pos_w - obj2.data.root_pos_w
    xy_dist = torch.norm(pos_diff[:, :2], dim=1)
    height_error = torch.abs(pos_diff[:, 2] - height_diff)
    xy_reward = 1.0 - torch.tanh(xy_dist / xy_std)
    height_reward = 1.0 - torch.tanh(height_error / height_std)
    above_object = torch.clamp(pos_diff[:, 2] / height_diff, min=0.0, max=1.0)

    near_stack_xy = torch.clamp(1.0 - xy_dist / xy_threshold, min=0.0, max=1.0)
    near_stack_height = torch.clamp(1.0 - height_error / height_threshold, min=0.0, max=1.0)
    placement_reward = (
        xy_reward
        * height_reward
        * above_object
        * (0.25 + 0.75 * near_stack_xy)
        * (0.25 + 0.75 * near_stack_height)
    )

    if hasattr(env.cfg, "gripper_joint_names"):
        open_reward, _ = _gripper_open_close_progress(env, robot_cfg, release_std)
    else:
        open_reward = torch.ones(env.num_envs, device=env.device)

    return placement_reward * (0.35 + 0.65 * open_reward)
