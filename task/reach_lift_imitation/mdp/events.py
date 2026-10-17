from __future__ import annotations

import torch

from .actions import absolute_joint_targets_to_policy_action
from .commands import quat_mul, sample_object_poses, sample_pose_command


def maybe_resample_object_pose(env) -> None:
    resample_env_ids = torch.nonzero(
        env.object_resample_counter >= env.object_resample_interval_steps,
        as_tuple=False,
    ).squeeze(-1)
    if len(resample_env_ids) > 0:
        randomize_object_pose(env, resample_env_ids)
        env.object_resample_counter[resample_env_ids] = 0
    env.object_resample_counter += 1


def randomize_object_pose(env, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
        return

    obj_root_state = env.obj.data.default_root_state[env_ids].clone()
    obj_pos_offset, obj_quat_delta = sample_object_poses(
        len(env_ids),
        env.device,
        pos_x=env.cfg.randomization.object_reset_pos_x,
        pos_y=env.cfg.randomization.object_reset_pos_y,
        pos_z=0.0,
        yaw=env.cfg.randomization.object_reset_yaw,
    )

    obj_pos = obj_root_state[:, 0:3] + env.scene.env_origins[env_ids] + obj_pos_offset
    obj_quat = quat_mul(obj_root_state[:, 3:7], obj_quat_delta)
    obj_pose = torch.cat((obj_pos, obj_quat), dim=-1)
    obj_vel = torch.zeros_like(obj_root_state[:, 7:13])
    env.obj.write_root_pose_to_sim(obj_pose, env_ids)
    env.obj.write_root_velocity_to_sim(obj_vel, env_ids)


def reset_robot_state(env, env_ids: torch.Tensor) -> torch.Tensor:
    default_root_state = env.robot.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] += env.scene.env_origins[env_ids]
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    env.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
    env.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env.robot.set_joint_position_target(
        joint_pos[:, env.actuated_joint_ids], env_ids=env_ids, joint_ids=env.actuated_joint_ids
    )
    return joint_pos


def reset_pose_command(env, env_ids: torch.Tensor) -> None:
    env.pose_command[env_ids] = sample_pose_command(
        len(env_ids),
        env.device,
        pos_x=env.cfg.commands.pos_x,
        pos_y=env.cfg.commands.pos_y,
        pos_z=env.cfg.commands.pos_z,
        roll=env.cfg.commands.roll,
        pitch=env.cfg.commands.pitch,
        yaw=env.cfg.commands.yaw,
    )


def reset_last_policy_action(env, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> None:
    env.last_policy_action[env_ids] = absolute_joint_targets_to_policy_action(
        joint_pos,
        env.robot.data.default_joint_pos[env_ids],
        env.arm_joint_ids,
        env.left_finger_joint_id,
        env.right_finger_joint_id,
        env.cfg.actions.arm_action_scale,
        env.cfg.actions.gripper_close_threshold,
    )
