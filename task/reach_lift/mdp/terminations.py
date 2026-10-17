from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def cube_stacked(
    env,
    robot_cfg = SceneEntityCfg("robot"),
    obj1_cfg = SceneEntityCfg("obj1"),
    obj2_cfg = SceneEntityCfg("obj2"),
    xy_threshold = 0.04,
    height_threshold = 0.005,
    height_diff = 0.0468,
    atol = 1e-4,
    rtol = 1e-4,
):
    robot = env.scene[robot_cfg.name]
    obj1 = env.scene[obj1_cfg.name]
    obj2 = env.scene[obj2_cfg.name]

    pos_diff_c12 = obj1.data.root_pos_w - obj2.data.root_pos_w

    xy_dist_c12 = torch.norm(pos_diff_c12[:, :2], dim=1)
    h_dist_c12 = torch.norm(pos_diff_c12[:, 2], dim=1)
    
    stacked = xy_dist_c12 < xy_threshold
    stacked = torch.logical_and(
        h_dist_c12 - height_diff < height_threshold,
        stacked
    )
    stacked = torch.logical_and(
        pos_diff_c12[:, :2] < 0.0,
        stacked
    )

    # hasattr -> env.scene에 surface_grippers속성이 있는지 확인
    # suction cup이 열려있을때만 stacked상태를 True로 인정
    if hasattr(env.scene, "surface_grippers") and len(env.scene.surface_grippers) >0:
        surface_gripper = env.scene.surface_grippers["surface_gripper"]
        suction_cup_status = surface_gripper.state.view(-1, 1)  # 1: closed, 0: closing, -1: open
        suction_cup_is_open = (suction_cup_status == -1).to(torch.float32)
        stacked = torch.logical_and(suction_cup_is_open, stacked)

    else:
        if hasattr(env.cfg, "gripper_joint_names"):
            gripper_joint_ids, _ = robot.find_joints(env.cfg.gripper_joint_names)
            if len(gripper_joint_ids) == 0:
                raise ValueError("No gripper joints found from gripper_joint_names")

            open_val = torch.tensor(env.cfg.gripper_open_val, dtype=torch.float32).to(env.device)
            for joint_id in gripper_joint_ids:
                stacked = torch.logical_and(
                    # mimic gripper can expose a single commanded joint, while
                    # parallel grippers may expose both finger joints.
                    torch.isclose(
                        robot.data.joint_pos[:, joint_id],
                        open_val,
                        atol=atol,
                        rtol=rtol,
                    ),
                    stacked,
                )

        else:
            raise ValueError("No gripper_joint_names found in environment config")

    return stacked


def object_goal_reached(
    env: ManagerBasedRLEnv,
    threshold: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("obj"),
) -> torch.Tensor:
    """Terminate when the lifted object is sufficiently close to the commanded goal."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    return is_lifted & (distance < threshold)
