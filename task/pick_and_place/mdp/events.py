from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math
import random

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def set_default_joint_pose(
    env,
    env_ids,
    default_pose,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    asset.data_default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)
    
def randomize_joint_by_gaussian_offset(
    env,
    env_ids,
    mean,
    std,
    asset_cfg
):
    asset = env.scene[asset_cfg.name]

    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    joint_pos += math_utils.sample_gaussian(mean, std, joint_pos.shape, joint_pos.device)

    joint_pos[:, -2:] = asset.data.default_joint_pos[env_ids, -2:]

    asset.set_joint_position_target(joint_pos, env_ids=env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break
            
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list

def randomize_object_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    pose_ranges: list[dict[str, tuple[float, float]]] | None = None,
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    if pose_ranges is None:
        pose_ranges = [pose_range] * len(asset_cfgs)

    for cur_env in env_ids.tolist():
        pose_list = []
        for asset_index in range(len(asset_cfgs)):
            sampled_pose = sample_object_poses(
                num_objects=1,
                min_separation=0.0,
                pose_range=pose_ranges[asset_index],
                max_sample_tries=max_sample_tries,
            )
            pose_list.append(sampled_pose[0])

        for sample_try in range(max_sample_tries):
            separation_check = [
                math.dist(pose_list[i][:3], pose_list[j][:3]) > min_separation
                for i in range(len(pose_list))
                for j in range(i + 1, len(pose_list))
            ]
            if all(separation_check):
                break
            if sample_try == max_sample_tries - 1:
                break
            pose_list = []
            for asset_index in range(len(asset_cfgs)):
                sampled_pose = sample_object_poses(
                    num_objects=1,
                    min_separation=0.0,
                    pose_range=pose_ranges[asset_index],
                    max_sample_tries=max_sample_tries,
                )
                pose_list.append(sampled_pose[0])

        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env, 0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5]
            )
            asset.write_root_pose_to_sim(
                torch.cat([positions, orientations], dim=-1), 
                env_ids = torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), 
                env_ids=torch.tensor([cur_env], device=env.device)
            )
    
def sample_random_color(
    base=(0.75, 0.75, 0.75), 
    variation=0.1
):
    offsets = [random.uniform(-variation, variation) for _ in range(3)]
    avg_offset = sum(offsets) / 3
    balanced_offsets = [offset - avg_offset for offset in offsets]
    new_color = tuple(max(0, min(1, base_component + offset)) 
                for base_component, offset in zip(base, balanced_offsets))

    return new_color
