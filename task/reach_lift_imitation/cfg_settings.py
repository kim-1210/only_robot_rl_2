from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass

from robot.finger_material import apply_rubber_finger_material
from task.imitation_scene_cfg import ReachLiftStyleSceneCfg

from . import mdp
from .cfg import ReachLiftImitationEnvCfg


@configclass
class ReachLiftImitationCFG(ReachLiftImitationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        scene_cfg = ReachLiftStyleSceneCfg(
            num_envs=self.scene.num_envs,
            env_spacing=self.scene.env_spacing,
            replicate_physics=getattr(self.scene, "replicate_physics", True),
        )
        self.scene.ground = scene_cfg.ground
        self.scene.light = scene_cfg.light
        self.scene.table = scene_cfg.table
        self.scene.robot = scene_cfg.robot
        self.scene.ee_frame = scene_cfg.ee_frame
        self.scene.obj = scene_cfg.obj
        self.scene.obj_frame = scene_cfg.obj_frame


class ReachLiftImitationEnv(DirectRLEnv):
    cfg: ReachLiftImitationCFG

    def __init__(self, cfg: ReachLiftImitationCFG, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self.obj: RigidObject = self.scene["obj"]

        self.arm_joint_ids, _ = self.robot.find_joints(list(self.cfg.actions.arm_joint_names))
        self.actuated_joint_ids, _ = self.robot.find_joints(list(self.cfg.actions.actuated_joint_names))
        self.left_finger_joint_id = self.robot.find_joints(self.cfg.actions.left_finger_joint_name)[0][0]
        self.right_finger_joint_id = self.robot.find_joints(self.cfg.actions.right_finger_joint_name)[0][0]

        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[..., 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[..., 1]

        self.pose_command = torch.zeros((self.num_envs, 7), device=self.device)
        self.last_policy_action = torch.zeros((self.num_envs, 7), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos.clone()
        self.object_resample_interval_steps = max(
            1, int(round(self.cfg.randomization.object_resample_interval_s / self.step_dt))
        )
        self.object_resample_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        apply_rubber_finger_material()

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        if actions.shape[-1] != self.cfg.observations.action_space:
            raise ValueError(f"Expected action dim {self.cfg.observations.action_space}, got {actions.shape[-1]}")

        mdp.maybe_resample_object_pose(self)

        self.joint_targets[:] = actions
        self.joint_targets[:] = torch.clamp(self.joint_targets, self.joint_lower_limits, self.joint_upper_limits)

        default_joint_pos = self.robot.data.default_joint_pos
        self.last_policy_action[:] = mdp.absolute_joint_targets_to_policy_action(
            self.joint_targets,
            default_joint_pos,
            self.arm_joint_ids,
            self.left_finger_joint_id,
            self.right_finger_joint_id,
            self.cfg.actions.arm_action_scale,
            self.cfg.actions.gripper_close_threshold,
        )

    def _apply_action(self):
        self.robot.set_joint_position_target(self.joint_targets[:, self.actuated_joint_ids], joint_ids=self.actuated_joint_ids)

    def _get_observations(self) -> dict:
        obs = mdp.build_policy_observation(self.robot, self.obj, self.pose_command)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        object_height = self.obj.data.root_pos_w[:, 2].clone()
        failed = mdp.object_below_minimum_height(
            self.obj.data.root_pos_w,
            self.cfg.terminations.minimum_object_height,
        )
        success = mdp.object_at_or_above_height(
            self.obj.data.root_pos_w,
            self.cfg.terminations.success_object_height,
        )
        self.extras["object_height_before_reset"] = object_height
        self.extras["is_success_step"] = success.clone()
        self.extras["is_failure_step"] = failed.clone()
        terminated = failed | success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        self.extras["terminated_before_reset"] = terminated.clone()
        self.extras["truncated_before_reset"] = truncated.clone()
        return terminated, truncated

    def _randomize_object_pose(self, env_ids: torch.Tensor):
        mdp.randomize_object_pose(self, env_ids)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        joint_pos = mdp.reset_robot_state(self, env_ids)
        self._randomize_object_pose(env_ids)

        mdp.reset_pose_command(self, env_ids)
        self.joint_targets[env_ids] = joint_pos
        mdp.reset_last_policy_action(self, env_ids, joint_pos)
        self.object_resample_counter[env_ids] = 0
