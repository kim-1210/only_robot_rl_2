from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def obj_pos_in_robot_root_frame(
    env,
    robot_cfg:SceneEntityCfg = SceneEntityCfg("robot"),
    obj_cfg:SceneEntityCfg = SceneEntityCfg("obj"),
):
    robot = env.scene[robot_cfg.name]
    obj = env.scene[obj_cfg.name]

    obj_pos_w = obj.data.root_pos_w[:, :3]
    obj_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        obj_pos_w
    )

    return obj_pos_b