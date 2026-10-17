from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import Articulation

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_robot_with_init_pose(
    env,
    env_ids,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[robot_cfg.name]

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    joint_names = robot.data.joint_names

    base = joint_names.index("base")
    shoulder = joint_names.index("shoulder")
    elbow = joint_names.index("elbow")
    wrist1 = joint_names.index("wrist1")
    wrist2 = joint_names.index("wrist2")
    wrist3 = joint_names.index("wrist3")

    joint_pos[:, base] = -1.57
    joint_pos[:, shoulder] = 0.0
    joint_pos[:, elbow] = 0.7
    joint_pos[:, wrist1] = 0.87
    joint_pos[:, wrist2] = 1.57
    joint_pos[:, wrist3] = 1.57

    robot.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids
    )

def reset_env_phase(
    env,
    env_ids,
):
    env.phase[env_ids] = 0

def _get_gripper_finger_mean(robot):
    names = robot.data.joint_names
    li = names.index("robotiq_hande_left_finger_joint")
    ri = names.index("robotiq_hande_right_finger_joint")
    return 0.5 * (robot.data.joint_pos[:, li] + robot.data.joint_pos[:, ri])

def change_phase(
    env,
    env_ids,
    robot_cfg=SceneEntityCfg("robot"),
    ee_cfg=SceneEntityCfg("ee_frame"),
    obj_cfg=SceneEntityCfg("cube"),
):
    robot = env.scene[robot_cfg.name]
    ee = env.scene[ee_cfg.name]
    obj = env.scene[obj_cfg.name]

    ee_pos = ee.data.target_pos_w[:, 0, :].squeeze(1)
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(ee_pos - obj_pos, dim=1)

    finger_mean = _get_gripper_finger_mean(robot)
    closed = finger_mean <= 0.018

    mask01 = (env.phase[env_ids] == 0) & (dist[env_ids] <= 0.15)
    env.phase[env_ids[mask01]] = 1

    mask12 = (env.phase[env_ids] == 1) & (dist[env_ids] <= 0.03) & (closed[env_ids])
    env.phase[env_ids[mask12]] = 2