from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def objs_pose(
    env,
    obj1_cfg = SceneEntityCfg("obj1"),
    obj2_cfg = SceneEntityCfg("obj2"),
):
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    obj1_pos = obj1.data.root_pos_w
    obj2_pos = obj2.data.root_pos_w

    obj1_quat = obj1.data.root_quat_w
    obj2_quat = obj2.data.root_quat_w

    return torch.cat((obj1_pos, obj1_quat, 
                      obj2_pos, obj2_quat), dim=1)

def ee_pose(
    env,
    ee_cfg = SceneEntityCfg("ee_frame")
):
    ee = env.scene[ee_cfg.name]
    return torch.cat((ee.data.target_pos_w[..., 0, :], ee.data.target_quat_w[..., 0, :]), dim=1)

def ee_pos_to_each_env(
    env,
    ee_cfg = SceneEntityCfg("ee_frame")
):
    ee = env.scene[ee_cfg.name]
    return ee.data.target_pos_w[..., 0, :] - env.scene.env_origins[:, 0:3]

def gripper_pos(
    env,
    robot_cfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    finger_ids, _ = robot.find_bodies(".*_finger")    
    l_finger = robot.data.body_pos_w[:, finger_ids[0], :]
    r_finger = robot.data.body_pos_w[:, finger_ids[1], :]

    return torch.cat((l_finger, r_finger), dim=-1)

def object_grasp(
    env,
    robot_cfg = SceneEntityCfg("robot"),
    obj1_cfg = SceneEntityCfg("obj1"),
    ee_cfg = SceneEntityCfg("ee_frame"),
    diff_threshold = 0.011
):
    robot = env.scene[robot_cfg.name]
    obj1 = env.scene[obj1_cfg.name]
    ee = env.scene[ee_cfg.name]

    obj_pos = obj1.data.root_pos_w
    ee_pos = ee.data.target_pos_w[:, 0, :]
    pose_diff = torch.linalg.vector_norm(obj_pos - ee_pos, dim=1)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

    grasped = torch.logical_and(
        pose_diff < diff_threshold,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[0]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        )
        > env.cfg.gripper_threshold,
    )
    grasped = torch.logical_and(
        grasped,
        torch.abs(
            robot.data.joint_pos[:, gripper_joint_ids[1]]
            - torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
        )
        > env.cfg.gripper_threshold,
    )
    return grasped
    
def object_stacked(
    env,
    robot_cfg,
    upper_object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    xy_threshold: float = 0.05,
    height_threshold: float = 0.005,
    height_diff: float = 0.04,
):
    robot: Articulation = env.scene[robot_cfg.name]
    upper_object: RigidObject = env.scene[upper_object_cfg.name]
    lower_object: RigidObject = env.scene[lower_object_cfg.name]

    pos_diff = upper_object.data.root_pos_w - lower_object.data.root_pos_w
    height_error = torch.abs(pos_diff[:, 2] - height_diff)
    xy_dist = torch.linalg.vector_norm(pos_diff[:, :2], dim=1)

    stacked = torch.logical_and(xy_dist < xy_threshold, height_error < height_threshold)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=1e-4,
            rtol=1e-4,
        ),
        stacked,
    )
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=1e-4,
            rtol=1e-4,
        ),
        stacked,
    )
    return stacked
    
def objs_poses_to_robot(
    env,
    robot_cfg = SceneEntityCfg("robot"),
    obj1_cfg = SceneEntityCfg("obj1"),
    obj2_cfg = SceneEntityCfg("obj2")
):
    robot = env.scene[robot_cfg.name]
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    obj1_to_robot_pos, obj1_to_robot_quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        obj1.data.root_pos_w, obj1.data.root_qaut_w
    )

    obj2_to_robot_pos, obj2_to_robot_quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        obj2.data.root_pos_w, obj2.data.root_qaut_w
    )

    return torch.cat((obj1_to_robot_pos, obj1_to_robot_quat,
                    obj2_to_robot_pos, obj2_to_robot_quat), dim=1)

def ee_poses_to_robot(
    env,
    robot_cfg = SceneEntityCfg("robot"),
    ee_cfg = SceneEntityCfg("ee_frame"),
):
    robot = env.scene[robot_cfg.name]
    ee = env.scene[ee_cfg.name]

    ee_to_robot_pos, ee_to_robot_quat = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        ee.data.root_pos_w, ee.data.root_quat_w
    )

    return torch.cat((ee_to_robot_pos, ee_to_robot_quat), dim=1)

def ee_obj_dist(
    env,
    ee_cfg = SceneEntityCfg("ee_frame"),
    obj1_cfg = SceneEntityCfg("obj1"),
    obj2_cfg = SceneEntityCfg("obj2")
):
    ee = env.scene[ee_cfg.name]
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    ee_pos = ee.data.target_pos_w[..., 0, :]
    ee_obj1_dist = torch.norm(ee_pos - obj1.data.root_pos_w, dim=1)
    ee_obj2_dist = torch.norm(ee_pos - obj2.data.root_pos_w, dim=1)

    return torch.stack((ee_obj1_dist, ee_obj2_dist), dim=1)
