from __future__ import annotations

import torch

from .commands import sample_pose_command


def sample_initial_pose_command(env, env_ids: torch.Tensor) -> None:
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


def resample_moving_pose_command(env) -> torch.Tensor:
    resample_mask = (env.episode_length_buf > 0) & (
        env.episode_length_buf % env.command_resample_interval_steps == 0
    )
    if not torch.any(resample_mask):
        return torch.empty(0, device=env.device, dtype=torch.long)

    env_ids = torch.nonzero(resample_mask, as_tuple=False).squeeze(-1)
    pos = env.pose_command[env_ids, :3].clone()
    pos[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(
        -env.cfg.commands.move_range_x, env.cfg.commands.move_range_x
    )
    pos[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(
        -env.cfg.commands.move_range_y, env.cfg.commands.move_range_y
    )
    pos[:, 2] += torch.empty(len(env_ids), device=env.device).uniform_(
        -env.cfg.commands.move_range_z, env.cfg.commands.move_range_z
    )

    pos[:, 0].clamp_(*env.cfg.commands.pos_x)
    pos[:, 1].clamp_(*env.cfg.commands.pos_y)
    pos[:, 2].clamp_(*env.cfg.commands.pos_z)
    env.pose_command[env_ids, :3] = pos
    return env_ids


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
