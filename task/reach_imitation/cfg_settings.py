from __future__ import annotations

from collections.abc import Sequence

import omni.usd
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils import configclass
from pxr import Gf, UsdGeom

from robot.finger_material import apply_rubber_finger_material
from task.imitation_scene_cfg import ReachLiftStyleRobotSceneCfg

from . import mdp
from .cfg import ReachImitationEnvCfg


class TargetPositionVisualizer:
    def __init__(self, num_envs: int, root_path: str = "/World/ReachImitationTargets", radius: float = 0.015):
        self.num_envs = num_envs
        self.root_path = root_path
        self.radius = radius
        self.stage = omni.usd.get_context().get_stage()
        self.marker_paths = [f"{self.root_path}/env_{env_idx:03d}" for env_idx in range(num_envs)]
        self._define_prims()

    def _define_prims(self):
        UsdGeom.Xform.Define(self.stage, self.root_path)
        for marker_path in self.marker_paths:
            sphere = UsdGeom.Sphere.Define(self.stage, marker_path)
            sphere.GetRadiusAttr().Set(self.radius)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.2, 0.2)])
            xformable = UsdGeom.Xformable(sphere.GetPrim())
            if not xformable.GetOrderedXformOps():
                xformable.AddTranslateOp()

    def update(self, positions: torch.Tensor, env_ids: torch.Tensor | None = None):
        positions_np = positions.detach().cpu().numpy()
        if env_ids is None:
            env_indices = range(len(positions_np))
        else:
            env_indices = env_ids.detach().cpu().tolist()
        for marker_env_idx, position in zip(env_indices, positions_np):
            prim = self.stage.GetPrimAtPath(self.marker_paths[int(marker_env_idx)])
            xformable = UsdGeom.Xformable(prim)
            xformable.GetOrderedXformOps()[0].Set(
                Gf.Vec3d(float(position[0]), float(position[1]), float(position[2]))
            )


@configclass
class ReachImitationCFG(ReachImitationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        scene_cfg = ReachLiftStyleRobotSceneCfg(
            num_envs=self.scene.num_envs,
            env_spacing=self.scene.env_spacing,
            replicate_physics=getattr(self.scene, "replicate_physics", True),
        )
        self.scene.ground = scene_cfg.ground
        self.scene.light = scene_cfg.light
        self.scene.table = scene_cfg.table
        self.scene.robot = scene_cfg.robot
        self.scene.ee_frame = scene_cfg.ee_frame


class ReachImitationEnv(DirectRLEnv):
    cfg: ReachImitationCFG

    def __init__(self, cfg: ReachImitationCFG, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot: Articulation = self.scene["robot"]
        self.arm_joint_ids, _ = self.robot.find_joints(list(self.cfg.actions.arm_joint_names))
        self.actuated_joint_ids, _ = self.robot.find_joints(list(self.cfg.actions.actuated_joint_names))
        self.left_finger_joint_id = self.robot.find_joints(self.cfg.actions.left_finger_joint_name)[0][0]
        self.right_finger_joint_id = self.robot.find_joints(self.cfg.actions.right_finger_joint_name)[0][0]
        self.ee_body_idx = self.robot.find_bodies("ee_link")[0][0]

        self.joint_lower_limits = self.robot.data.soft_joint_pos_limits[..., 0]
        self.joint_upper_limits = self.robot.data.soft_joint_pos_limits[..., 1]

        self.pose_command = torch.zeros((self.num_envs, 7), device=self.device)
        self.last_policy_action = torch.zeros((self.num_envs, 7), device=self.device)
        self.joint_targets = self.robot.data.default_joint_pos.clone()
        self.command_resample_interval_steps = max(
            1, int(round(self.cfg.commands.resample_interval_s / self.step_dt))
        )
        self.target_visualizer = TargetPositionVisualizer(
            num_envs=self.num_envs,
            radius=self.cfg.observations.target_marker_radius,
        )

    def _setup_scene(self):
        apply_rubber_finger_material()

    def _update_target_visualizer(self, env_ids: torch.Tensor):
        target_pos_w = mdp.target_pos_world(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.pose_command[env_ids],
        )
        self.target_visualizer.update(target_pos_w, env_ids)

    def _sample_initial_pose_command(self, env_ids: torch.Tensor):
        mdp.sample_initial_pose_command(self, env_ids)
        self._update_target_visualizer(env_ids)

    def _resample_moving_pose_command(self):
        env_ids = mdp.resample_moving_pose_command(self)
        if env_ids.numel() > 0:
            self._update_target_visualizer(env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = actions.to(self.device)
        if actions.shape[-1] != self.cfg.observations.action_space:
            raise ValueError(f"Expected action dim {self.cfg.observations.action_space}, got {actions.shape[-1]}")

        self._resample_moving_pose_command()

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
        obs = mdp.build_policy_observation(self.robot, self.pose_command)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = mdp.no_termination(self.num_envs, self.device)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        joint_pos = mdp.reset_robot_state(self, env_ids)

        self._sample_initial_pose_command(env_ids)
        self.last_policy_action[env_ids] = 0.0
        self.joint_targets[env_ids] = joint_pos
