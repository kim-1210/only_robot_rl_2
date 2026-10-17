from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import subtract_frame_transforms


def object_pos_in_robot_root_frame(robot: Articulation, obj: RigidObject) -> torch.Tensor:
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
    )
    return obj_pos_b


def build_policy_observation(
    robot: Articulation,
    obj: RigidObject,
    pose_command: torch.Tensor,
) -> torch.Tensor:
    joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel
    obj_pos_b = object_pos_in_robot_root_frame(robot, obj)
    return torch.cat((joint_pos_rel, joint_vel_rel, obj_pos_b, pose_command), dim=-1)
