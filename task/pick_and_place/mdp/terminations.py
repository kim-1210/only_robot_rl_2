from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def cubes_stacked(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("obj1"),
    cube_2_cfg: SceneEntityCfg = SceneEntityCfg("obj2"),
    xy_threshold: float = 0.04,
    height_threshold: float = 0.005,
    height_diff: float = 0.04,
    atol=0.006,
    rtol=0.0,
):
    robot: Articulation = env.scene[robot_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    cube_2: RigidObject = env.scene[cube_2_cfg.name]

    pos_diff_c12 = cube_1.data.root_pos_w - cube_2.data.root_pos_w

    # Compute cube position difference in x-y plane
    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)

    # Compute cube height difference
    height_error_c12 = torch.abs(pos_diff_c12[:, 2] - height_diff)

    # Check cube positions
    stacked = xy_dist_c12 < xy_threshold
    stacked = torch.logical_and(height_error_c12 < height_threshold, stacked)
    stacked = torch.logical_and(pos_diff_c12[:, 2] > 0.0, stacked)

    gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)

    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[0]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        stacked,
    )
    stacked = torch.logical_and(
        torch.isclose(
            robot.data.joint_pos[:, gripper_joint_ids[1]],
            torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device),
            atol=atol,
            rtol=rtol,
        ),
        stacked,
    )

    return stacked
