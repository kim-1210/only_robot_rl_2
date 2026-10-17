"""
TERM=xterm PYTHONPATH=$PWD isaaclab -p ./launch/imitation_play_fixed.py \
  --task reach_lift_imitation \
  --policy_type python \
  --policy_path ./policy_models/diffusion/diffusion_policy_wrapper.py \
  --policy_checkpoint ./policy_models/diffusion/reach_lift_diffusion_unet_artifacts/reach_lift_diffusion_unet.pt \
  --policy_stats ./policy_models/diffusion/reach_lift_diffusion_unet_artifacts/reach_lift_diffusion_stats.npz \
  --num_envs 1 \
  --output_horizon 16 \
  --execute_horizon 14 \
  --joint_arrival_tolerance 0.01 \
  --max_action_hold_steps 20 \
  --steps 5000 \
  --seed 42 \
  --save_path ./datas/reach_lift_imitation_diffusion_rollout_k8.npz \
  --device cuda:0 \
  --log_gripper \
  --binary_gripper \
  --headless
"""

import argparse
import colorsys
import importlib.util
import inspect
import os
import pickle
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_LIFT_SUCCESS_Z = 1.10

parser = argparse.ArgumentParser(description="Play direct imitation tasks with a custom policy.")
parser.add_argument("--task", type=str, required=True, help="Task name. e.g. reach_imitation or reach_lift_imitation")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--steps", type=int, default=0, help="Optional max play steps. 0 means unlimited.")
parser.add_argument("--episodes", type=int, default=0, help="Optional max completed episodes. 0 means unlimited.")
parser.add_argument(
    "--episode_length_s",
    type=float,
    default=None,
    help="Optional environment episode timeout override in seconds.",
)
parser.add_argument("--policy_path", type=str, default=None, help="TorchScript .pt file or Python policy wrapper file.")
parser.add_argument(
    "--policy_checkpoint",
    type=str,
    default=None,
    help="Optional checkpoint path for Python policies. Passed via POLICY_CHECKPOINT.",
)
parser.add_argument(
    "--policy_stats",
    type=str,
    default=None,
    help="Optional normalization stats path for Python policies. Passed via POLICY_STATS.",
)
parser.add_argument("--output_horizon", type=int, default=1, help="Policy output chunk length N.")
parser.add_argument("--execute_horizon", type=int, default=1, help="Number of actions K to execute from each chunk.")
parser.add_argument(
    "--binary_gripper",
    action="store_true",
    default=False,
    help="Convert the 1D gripper output from the policy into binary open/close commands before simulation.",
)
parser.add_argument(
    "--gripper_close_value",
    type=float,
    default=0.0,
    help="Joint target used when the binary gripper command is closed.",
)
parser.add_argument(
    "--gripper_open_value",
    type=float,
    default=0.025,
    help="Joint target used when the binary gripper command is open.",
)
parser.add_argument(
    "--gripper_binary_threshold",
    type=float,
    default=0.0,
    help="Threshold applied to the 1D gripper output to decide between closed and open binary commands.",
)
parser.add_argument(
    "--log_gripper",
    action="store_true",
    default=False,
    help="Print gripper values at each replan for debugging.",
)
parser.add_argument(
    "--joint_arrival_tolerance",
    type=float,
    default=0.03,
    help="Advance to the next predicted action only after the actuated joints are within this tolerance.",
)
parser.add_argument(
    "--max_action_hold_steps",
    type=int,
    default=8,
    help="Maximum simulator env.step calls to spend on one predicted action before advancing anyway.",
)
parser.add_argument(
    "--action_smoothing_alpha",
    type=float,
    default=0.8,
    help="Low-pass factor applied to commanded joint targets. 1.0 disables smoothing, lower values reduce jerk.",
)
parser.add_argument(
    "--next_action_lookahead_margin",
    type=float,
    default=0.002,
    help="Advance early when the next waypoint becomes this much closer than the current waypoint.",
)
parser.add_argument(
    "--replan_lookahead_steps",
    type=int,
    default=1,
    help="Prefetch the next policy chunk this many executed actions before the current chunk ends.",
)
parser.add_argument(
    "--chunk_transition_blend_steps",
    type=int,
    default=2,
    help="Blend the first few actions of each replanned chunk from the previous commanded action to reduce replan jitter.",
)
parser.add_argument(
    "--reach_select_closest_waypoint",
    action="store_true",
    default=False,
    help="For reach_imitation, replace each predicted chunk with the FK waypoint closest to the current target.",
)
parser.add_argument(
    "--reach_select_repeat",
    action="store_true",
    default=True,
    help="Repeat the selected closest reach waypoint for the whole chunk.",
)
parser.add_argument(
    "--reach_ik_refine_steps",
    type=int,
    default=0,
    help="For reach_imitation, run this many damped least-squares IK refinement steps.",
)
parser.add_argument(
    "--reach_ik_refine_mode",
    type=str,
    default="preserve_sequence",
    choices=["preserve_sequence", "repeat_first"],
    help=(
        "How to apply reach IK refinement. preserve_sequence keeps the predicted chunk order, "
        "repeat_first reproduces the older behavior that repeats the refined first waypoint."
    ),
)
parser.add_argument(
    "--reach_ik_activation_dist",
    type=float,
    default=0.0,
    help=(
        "Only apply reach IK to predicted waypoints already within this FK distance of the target. "
        "0 disables the gate."
    ),
)
parser.add_argument(
    "--reach_ik_gain",
    type=float,
    default=0.8,
    help="Gain for reach IK refinement.",
)
parser.add_argument(
    "--reach_ik_damping",
    type=float,
    default=1e-3,
    help="Damping coefficient for reach IK refinement.",
)
parser.add_argument(
    "--reach_ik_max_delta",
    type=float,
    default=0.08,
    help="Maximum absolute joint update per IK refinement step.",
)
parser.add_argument("--save_path", type=str, default=None, help="Optional .npz path to save predicted/executed trajectories.")
parser.add_argument(
    "--no_auto_report",
    action="store_true",
    default=False,
    help="Do not auto-save a report npz or append markdown report rows when --save_path is omitted.",
)
parser.add_argument("--seed", type=int, default=0, help="Random seed for evaluation.")
parser.add_argument("--success_dist", type=float, default=0.05, help="Success distance threshold (meters).")
parser.add_argument(
    "--success_z",
    type=float,
    default=DEFAULT_LIFT_SUCCESS_Z,
    help="Success object height threshold (meters). Default is 1.10 m for matched evaluation.",
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="torchscript",
    choices=["torchscript", "python"],
    help="How to load the policy.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument(
    "--disable_predicted_path_viz",
    action="store_true",
    default=False,
    help="Disable predicted path FK computation and viewport visualization during rollout.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch
import omni.usd
from pxr import Gf, UsdGeom

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.utils.math import combine_frame_transforms
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

import task  # noqa: F401


def _uses_stepwise_obs_history(model_variant: str | None) -> bool:
    if model_variant is None:
        return False
    model_variant = str(model_variant)
    return model_variant.startswith("reach_lift_joint_hist") or model_variant.startswith("reach_lift_flow_joint_hist")


def _resolve_fk_urdf_path() -> Path:
    script_root = Path(__file__).resolve().parents[1]
    candidates = [
        Path("/media/kim/hdd1tb/master_study/urdf_ws/src/combine_urdf/urdf/test_gripper.urdf"),
        script_root.parent / "urdf_ws" / "src" / "combine_urdf" / "urdf" / "test_gripper.urdf",
        script_root.parent / "urdf_ws" / "install" / "combine_urdf" / "share" / "combine_urdf" / "urdf" / "test_gripper.urdf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0]


FK_URDF_PATH = _resolve_fk_urdf_path()

_PINOCCHIO_IMPORT_ERROR: Exception | None = None


def _get_pinocchio():
    global _PINOCCHIO_IMPORT_ERROR
    try:
        import pinocchio as pin
    except Exception as exc:
        _PINOCCHIO_IMPORT_ERROR = exc
        raise RuntimeError(f"pinocchio import failed: {exc}") from exc
    return pin


class ForwardKinematicsHelper:
    def __init__(self, urdf_path: str, joint_names: list[str], ee_frame_name: str = "ee_link"):
        pin = _get_pinocchio()
        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_frame_name)
        self.model_joint_ids = {self.model.names[i]: i for i in range(len(self.model.names))}
        self.joint_names = joint_names
        self.joint_q_indices = []
        self.joint_v_indices = []
        for joint_name in joint_names:
            joint_id = self.model_joint_ids.get(joint_name)
            if joint_id is None:
                self.joint_q_indices.append(None)
                self.joint_v_indices.append(None)
                continue
            joint = self.model.joints[joint_id]
            self.joint_q_indices.append(joint.idx_q if joint.nq == 1 else None)
            self.joint_v_indices.append(joint.idx_v if joint.nv == 1 else None)

    def ee_positions(self, joint_sequences: torch.Tensor) -> torch.Tensor:
        joint_sequences = joint_sequences.detach().cpu().to(torch.float64)
        batch, horizon, _ = joint_sequences.shape
        out = torch.zeros((batch, horizon, 3), dtype=torch.float32)
        for b in range(batch):
            for h in range(horizon):
                q = self.pin.neutral(self.model)
                for value, joint_name in zip(joint_sequences[b, h], self.joint_names):
                    joint_id = self.model_joint_ids.get(joint_name)
                    if joint_id is None:
                        continue
                    idx_q = self.model.joints[joint_id].idx_q
                    nq = self.model.joints[joint_id].nq
                    if nq == 1 and 0 <= idx_q < q.shape[0]:
                        q[idx_q] = float(value.item())
                self.pin.forwardKinematics(self.model, self.data, q)
                self.pin.updateFramePlacements(self.model, self.data)
                pos = self.data.oMf[self.ee_frame_id].translation
                out[b, h] = torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float32)
        return out

    def refine_reach_targets(
        self,
        joint_targets: torch.Tensor,
        target_pos_b: torch.Tensor,
        steps: int,
        gain: float,
        damping: float,
        max_delta: float,
    ) -> torch.Tensor:
        pin = self.pin
        actions_np = joint_targets.detach().cpu().numpy().astype(np.float64, copy=True)
        targets_np = target_pos_b.detach().cpu().numpy().astype(np.float64, copy=False)
        refined = actions_np.copy()
        arm_count = min(6, refined.shape[-1], len(self.joint_q_indices))
        q_indices = self.joint_q_indices[:arm_count]
        v_indices = self.joint_v_indices[:arm_count]
        valid_pairs = [(i, q_idx, v_idx) for i, (q_idx, v_idx) in enumerate(zip(q_indices, v_indices)) if q_idx is not None and v_idx is not None]
        if not valid_pairs:
            return joint_targets

        for batch_idx in range(refined.shape[0]):
            q = pin.neutral(self.model)
            for action_idx, q_idx in enumerate(self.joint_q_indices[: refined.shape[-1]]):
                if q_idx is not None and 0 <= q_idx < q.shape[0]:
                    q[q_idx] = refined[batch_idx, action_idx]

            for _ in range(max(0, int(steps))):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                ee_pos = self.data.oMf[self.ee_frame_id].translation
                err = targets_np[batch_idx] - np.asarray([ee_pos[0], ee_pos[1], ee_pos[2]], dtype=np.float64)
                if float(np.linalg.norm(err)) < 1e-4:
                    break
                pin.computeJointJacobians(self.model, self.data, q)
                jac = pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    self.ee_frame_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )[:3, :]
                cols = [v_idx for _, _, v_idx in valid_pairs]
                jac_arm = jac[:, cols]
                system = jac_arm @ jac_arm.T + float(damping) * np.eye(3, dtype=np.float64)
                dq = jac_arm.T @ np.linalg.solve(system, err)
                dq = np.clip(float(gain) * dq, -float(max_delta), float(max_delta))
                for delta, (action_idx, q_idx, _) in zip(dq, valid_pairs):
                    q[q_idx] += float(delta)
                    refined[batch_idx, action_idx] = q[q_idx]

        return torch.as_tensor(refined, device=joint_targets.device, dtype=joint_targets.dtype)


class NullForwardKinematicsHelper:
    def ee_positions(self, joint_sequences: torch.Tensor) -> torch.Tensor:
        batch, horizon, _ = joint_sequences.shape
        return torch.zeros((batch, horizon, 3), dtype=torch.float32, device=joint_sequences.device)


class PredictedPathVisualizer:
    def __init__(self, num_points: int, root_path: str, radius: float = 0.006, line_width: float = 0.004):
        self.num_points = num_points
        self.radius = radius
        self.line_width = line_width
        self.stage = omni.usd.get_context().get_stage()
        self.root_path = root_path
        self.point_paths = [f"{self.root_path}/point_{i:02d}" for i in range(num_points)]
        self.line_paths = [f"{self.root_path}/line_{i:02d}" for i in range(max(num_points - 1, 0))]
        self.colors = [self._rainbow_color(i, num_points) for i in range(num_points)]
        self._define_prims()

    @staticmethod
    def _rainbow_color(index: int, total: int) -> tuple[float, float, float]:
        hue = float(index) / max(total, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return (r, g, b)

    def _define_prims(self):
        UsdGeom.Xform.Define(self.stage, self.root_path)
        for point_path, color in zip(self.point_paths, self.colors):
            sphere = UsdGeom.Sphere.Define(self.stage, point_path)
            sphere.GetRadiusAttr().Set(self.radius)
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xformable = UsdGeom.Xformable(sphere.GetPrim())
            if not xformable.GetOrderedXformOps():
                xformable.AddTranslateOp()
            xformable.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, 0.0, -10.0))
        for line_path, color in zip(self.line_paths, self.colors[:-1]):
            curve = UsdGeom.BasisCurves.Define(self.stage, line_path)
            curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            curve.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            curve.CreateCurveVertexCountsAttr([2])
            curve.CreateWidthsAttr([self.line_width, self.line_width])
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, -10.0), Gf.Vec3f(0.0, 0.0, -10.0)])

    def update(self, point_path: np.ndarray):
        visible_count = min(point_path.shape[0], self.num_points)
        for idx in range(self.num_points):
            prim = self.stage.GetPrimAtPath(self.point_paths[idx])
            xformable = UsdGeom.Xformable(prim)
            if idx < visible_count:
                point = point_path[idx]
                xformable.GetOrderedXformOps()[0].Set(Gf.Vec3d(float(point[0]), float(point[1]), float(point[2])))
            else:
                xformable.GetOrderedXformOps()[0].Set(Gf.Vec3d(0.0, 0.0, -10.0))
        for idx in range(self.num_points - 1):
            curve = UsdGeom.BasisCurves.Get(self.stage, self.line_paths[idx])
            if idx + 1 < visible_count:
                curve.GetPointsAttr().Set(
                    [
                        Gf.Vec3f(float(point_path[idx, 0]), float(point_path[idx, 1]), float(point_path[idx, 2])),
                        Gf.Vec3f(float(point_path[idx + 1, 0]), float(point_path[idx + 1, 1]), float(point_path[idx + 1, 2])),
                    ]
                )
            else:
                curve.GetPointsAttr().Set([Gf.Vec3f(0.0, 0.0, -10.0), Gf.Vec3f(0.0, 0.0, -10.0)])


def _finger_midpoint_world(robot, env_idx: int) -> np.ndarray:
    left_body_idx = robot.find_bodies("robotiq_hande_left_finger")[0][0]
    right_body_idx = robot.find_bodies("robotiq_hande_right_finger")[0][0]
    left_pos = robot.data.body_pos_w[env_idx, left_body_idx, :].detach().cpu().numpy()
    right_pos = robot.data.body_pos_w[env_idx, right_body_idx, :].detach().cpu().numpy()
    return 0.5 * (left_pos + right_pos)


def _load_python_policy(policy_path: str, device: torch.device | str):
    policy_file = Path(policy_path).resolve()
    module_name = f"imitation_policy_{policy_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, policy_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load policy module from {policy_file}")
    module = importlib.util.module_from_spec(spec)
    policy_dir = str(policy_file.parent)
    added_to_syspath = False
    if policy_dir not in sys.path:
        sys.path.insert(0, policy_dir)
        added_to_syspath = True
    try:
        spec.loader.exec_module(module)
    finally:
        if added_to_syspath:
            try:
                sys.path.remove(policy_dir)
            except ValueError:
                pass

    if args_cli.policy_checkpoint is not None:
        os.environ["POLICY_CHECKPOINT"] = str(Path(args_cli.policy_checkpoint).expanduser().resolve())
    if args_cli.policy_stats is not None:
        os.environ["POLICY_STATS"] = str(Path(args_cli.policy_stats).expanduser().resolve())

    if hasattr(module, "load_policy"):
        load_policy_sig = inspect.signature(module.load_policy)
        kwargs = {}
        if "device" in load_policy_sig.parameters:
            kwargs["device"] = device
        if "output_horizon" in load_policy_sig.parameters:
            kwargs["output_horizon"] = args_cli.output_horizon
        if "execute_horizon" in load_policy_sig.parameters:
            kwargs["execute_horizon"] = args_cli.execute_horizon
        policy = module.load_policy(**kwargs)
    elif hasattr(module, "Policy"):
        policy_ctor_sig = inspect.signature(module.Policy)
        kwargs = {}
        if "device" in policy_ctor_sig.parameters:
            kwargs["device"] = device
        if "output_horizon" in policy_ctor_sig.parameters:
            kwargs["output_horizon"] = args_cli.output_horizon
        if "execute_horizon" in policy_ctor_sig.parameters:
            kwargs["execute_horizon"] = args_cli.execute_horizon
        policy = module.Policy(**kwargs)
    else:
        raise AttributeError("Python policy file must define load_policy(device) or Policy.")

    if hasattr(policy, "eval"):
        policy.eval()
    return policy


def _load_policy(policy_path: str | None, policy_type: str, device: torch.device | str):
    if policy_path is None:
        return None
    if policy_type == "torchscript":
        policy = torch.jit.load(os.path.abspath(policy_path), map_location=device)
        policy.eval()
        return policy
    if policy_type == "python":
        return _load_python_policy(policy_path, device)
    raise ValueError(f"Unsupported policy type: {policy_type}")


def _compress_robot_state_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] < 9:
        raise ValueError(f"Expected robot-state tensor dim >= 9, got {x.shape[-1]}")
    return torch.cat([x[:, :6], x[:, 7:9].mean(dim=1, keepdim=True)], dim=-1)


def _compress_policy_action_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 7:
        return x
    if x.shape[-1] < 9:
        raise ValueError(f"Expected action tensor dim >= 7, got {x.shape[-1]}")
    return torch.cat([x[:, :6], x[:, 7:9].mean(dim=1, keepdim=True)], dim=-1)


def _compress_policy_action_tensor_left_right(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 8:
        return x
    if x.shape[-1] == 7:
        return torch.cat([x[:, :6], x[:, 6:7], x[:, 6:7]], dim=-1)
    if x.shape[-1] < 9:
        raise ValueError(f"Expected action tensor dim >= 8, got {x.shape[-1]}")
    return torch.cat([x[:, :6], x[:, 7:9]], dim=-1)


def _target_policy_action_dim(policy) -> int | None:
    if hasattr(policy, "act_mean"):
        return int(policy.act_mean.shape[-1])
    if hasattr(policy, "act_std"):
        return int(policy.act_std.shape[-1])
    return None


def _build_policy_observation_frame(
    frame_dim: int,
    obs_tensor: torch.Tensor,
    last_action_tensor: torch.Tensor | None,
) -> torch.Tensor:
    if obs_tensor.shape[-1] == frame_dim:
        return obs_tensor
    if last_action_tensor is None:
        return obs_tensor

    if frame_dim == 28 and obs_tensor.shape[-1] == 25:
        pose_command = obs_tensor[:, 18:25]
        joint_pos = obs_tensor[:, 0:9]
        joint_vel = obs_tensor[:, 9:18]
        return torch.cat(
            [
                _compress_robot_state_tensor(joint_pos),
                _compress_robot_state_tensor(joint_vel),
                pose_command,
                _compress_policy_action_tensor(last_action_tensor),
            ],
            dim=-1,
        )

    if frame_dim == 31 and obs_tensor.shape[-1] in {28, 35}:
        object_position = obs_tensor[:, 18:21]
        pose_command = obs_tensor[:, 21:28]
        joint_pos = obs_tensor[:, 0:9]
        joint_vel = obs_tensor[:, 9:18]
        return torch.cat(
            [
                _compress_robot_state_tensor(joint_pos),
                _compress_robot_state_tensor(joint_vel),
                object_position,
                pose_command,
                _compress_policy_action_tensor(last_action_tensor),
            ],
            dim=-1,
        )

    if frame_dim == 34 and obs_tensor.shape[-1] in {28, 35}:
        object_position = obs_tensor[:, 18:21]
        pose_command = obs_tensor[:, 21:28]
        joint_pos = obs_tensor[:, 0:9]
        joint_vel = obs_tensor[:, 9:18]
        return torch.cat(
            [
                torch.cat([joint_pos[:, :6], joint_pos[:, 7:9]], dim=-1),
                torch.cat([joint_vel[:, :6], joint_vel[:, 7:9]], dim=-1),
                object_position,
                pose_command,
                _compress_policy_action_tensor_left_right(last_action_tensor),
            ],
            dim=-1,
        )

    return obs_tensor


def _prepare_policy_observation(policy, obs_tensor: torch.Tensor, last_action_tensor: torch.Tensor | None) -> torch.Tensor:
    if args_cli.policy_type != "python" or not hasattr(policy, "obs_mean"):
        return obs_tensor

    expected_obs_dim = int(policy.obs_mean.shape[-1])
    if obs_tensor.shape[-1] == expected_obs_dim:
        return obs_tensor

    if last_action_tensor is None:
        return obs_tensor

    history_steps = int(getattr(policy, "obs_history_steps", 1))
    frame_dim = int(getattr(policy, "processed_obs_frame_dim", expected_obs_dim))
    current_frame = _build_policy_observation_frame(frame_dim, obs_tensor, last_action_tensor)
    if current_frame.shape[-1] != frame_dim:
        return obs_tensor

    if history_steps > 1 and expected_obs_dim == frame_dim * history_steps:
        obs_history = getattr(policy, "obs_history", None)
        obs_history, processed_obs = _update_obs_history_buffer(obs_history, current_frame, history_steps)
        policy.obs_history = obs_history
        return processed_obs

    if expected_obs_dim == frame_dim:
        return current_frame

    return obs_tensor


def _prepare_policy_observation_frame(
    frame_dim: int,
    obs_tensor: torch.Tensor,
    last_action_tensor: torch.Tensor | None,
) -> torch.Tensor:
    return _build_policy_observation_frame(frame_dim, obs_tensor, last_action_tensor)


def _update_obs_history_buffer(
    history_buffer: torch.Tensor | None,
    current_frame: torch.Tensor,
    history_steps: int,
    reset_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_frame.ndim != 2:
        raise ValueError(f"Expected current_frame with shape [B, D], got {tuple(current_frame.shape)}")

    batch, feat_dim = current_frame.shape
    if history_steps <= 1:
        return current_frame.unsqueeze(1), current_frame

    if (
        history_buffer is None
        or history_buffer.shape[0] != batch
        or history_buffer.shape[1] != history_steps
        or history_buffer.shape[2] != feat_dim
    ):
        history_buffer = current_frame.unsqueeze(1).repeat(1, history_steps, 1)
    else:
        history_buffer = torch.cat([history_buffer[:, 1:, :], current_frame.unsqueeze(1)], dim=1)

    if reset_mask is not None and torch.any(reset_mask):
        history_buffer = history_buffer.clone()
        history_buffer[reset_mask] = current_frame[reset_mask].unsqueeze(1).repeat(1, history_steps, 1)

    return history_buffer, history_buffer.reshape(batch, history_steps * feat_dim)


def _expand_policy_actions(
    actions: torch.Tensor, action_dim: int, reference_action: torch.Tensor | None = None
) -> torch.Tensor:
    if actions.shape[-1] == action_dim:
        return actions
    if actions.shape[-1] == 7 and action_dim == 9:
        ee_link_joint = torch.zeros(
            (actions.shape[0], actions.shape[1], 1),
            device=actions.device,
            dtype=actions.dtype,
        )
        gripper = actions[..., 6:7]
        raw_gripper = gripper.clone()
        if args_cli.binary_gripper:
            gripper = torch.where(
                gripper <= args_cli.gripper_binary_threshold,
                torch.full_like(gripper, args_cli.gripper_close_value),
                torch.full_like(gripper, args_cli.gripper_open_value),
            )
        if not getattr(args_cli, "_gripper_debug_logged", False):
            sample_envs = min(5, actions.shape[0])
            sample_steps = min(3, actions.shape[1])
            raw_gripper_sample = raw_gripper[:sample_envs, :sample_steps, 0].detach().cpu().tolist()
            final_gripper_sample = gripper[:sample_envs, :sample_steps, 0].detach().cpu().tolist()
            threshold_mask_sample = (
                (raw_gripper[:sample_envs, :sample_steps, 0] <= args_cli.gripper_binary_threshold).detach().cpu().tolist()
            )
            # print(
            #     "[INFO] Gripper mapping sample "
            #     f"binary_gripper={args_cli.binary_gripper} "
            #     f"threshold={args_cli.gripper_binary_threshold:.6f} "
            #     f"raw_envs_steps={raw_gripper_sample} "
            #     f"is_closed_mask={threshold_mask_sample} "
            #     f"final_envs_steps={final_gripper_sample}"
            # )
            args_cli._gripper_debug_logged = True
        return torch.cat([actions[..., :6], ee_link_joint, gripper, gripper], dim=-1)
    if actions.shape[-1] == 8 and action_dim == 9:
        ee_link_joint = torch.zeros(
            (actions.shape[0], actions.shape[1], 1),
            device=actions.device,
            dtype=actions.dtype,
        )
        return torch.cat([actions[..., :6], ee_link_joint, actions[..., 6:7], actions[..., 7:8]], dim=-1)
    if actions.shape[-1] == 8 and action_dim == 7:
        gripper = actions[..., 6:8].mean(dim=-1, keepdim=True)
        return torch.cat([actions[..., :6], gripper], dim=-1)
    if actions.shape[-1] == 9 and action_dim == 8:
        return torch.cat([actions[..., :6], actions[..., 7:9]], dim=-1)
    if actions.shape[-1] == 9 and action_dim == 7:
        gripper = actions[..., 7:9].mean(dim=-1, keepdim=True)
        return torch.cat([actions[..., :6], gripper], dim=-1)
    if actions.shape[-1] == 7 and action_dim == 8:
        gripper = actions[..., 6:7]
        return torch.cat([actions[..., :6], gripper, gripper], dim=-1)
    raise ValueError(f"Expected action dim {action_dim}, got {actions.shape[-1]}")


def _initial_last_action_tensor(env, robot, action_dim: int) -> torch.Tensor:
    actuated_joint_ids = getattr(env.unwrapped, "actuated_joint_ids", None)
    if actuated_joint_ids is not None:
        initial_action = robot.data.default_joint_pos[:, actuated_joint_ids].clone()
    else:
        initial_action = robot.data.default_joint_pos[:, :action_dim].clone()

    if initial_action.shape[-1] != action_dim:
        raise ValueError(f"Expected initial action dim {action_dim}, got {initial_action.shape[-1]}")
    return initial_action


def _smooth_command_action(
    target_action: torch.Tensor,
    previous_action: torch.Tensor | None,
    alpha: float,
) -> torch.Tensor:
    if previous_action is None or alpha >= 0.999:
        return target_action
    alpha = max(0.0, min(1.0, float(alpha)))
    smoothed = torch.lerp(previous_action, target_action, alpha)
    # Keep gripper commands crisp so grasp timing is not weakened by arm smoothing.
    if smoothed.shape[-1] >= 9:
        smoothed[..., 7:9] = target_action[..., 7:9]
    elif smoothed.shape[-1] >= 8:
        smoothed[..., 7:] = target_action[..., 7:]
    return smoothed


def _blend_chunk_transition(
    action_chunk: torch.Tensor,
    previous_action: torch.Tensor | None,
    blend_steps: int,
) -> torch.Tensor:
    if previous_action is None or blend_steps <= 0 or action_chunk.shape[1] == 0:
        return action_chunk
    blended = action_chunk.clone()
    prefix = min(int(blend_steps), int(action_chunk.shape[1]))
    for idx in range(prefix):
        mix = float(idx + 1) / float(prefix + 1)
        blended[:, idx, :] = torch.lerp(previous_action, blended[:, idx, :], mix)
        if blended.shape[-1] >= 9:
            blended[:, idx, 7:9] = action_chunk[:, idx, 7:9]
        elif blended.shape[-1] >= 8:
            blended[:, idx, 7:] = action_chunk[:, idx, 7:]
    return blended


def _apply_grasp_assist(
    commanded_actions: torch.Tensor,
    ee_pos: torch.Tensor,
    obj,
) -> torch.Tensor:
    if obj is None or commanded_actions.shape[-1] < 9:
        return commanded_actions

    obj_pos = obj.data.root_pos_w[:, :3]
    lateral_dist = torch.norm((ee_pos - obj_pos)[:, :2], dim=-1)
    vertical_gap = torch.abs(ee_pos[:, 2] - obj_pos[:, 2])
    obj_height = obj_pos[:, 2]

    should_force_close = (
        ((lateral_dist <= 0.10) & (vertical_gap <= 0.15) & (obj_height <= 1.10))
        | ((obj_height >= 1.065) & (obj_height <= 1.13))
    )
    if not torch.any(should_force_close):
        return commanded_actions

    if not getattr(args_cli, "_grasp_assist_logged", False):
        sample_count = min(5, commanded_actions.shape[0])
        print(
            "[INFO] Grasp assist activated "
            f"lateral_envs={lateral_dist[:sample_count].detach().cpu().tolist()} "
            f"vertical_envs={vertical_gap[:sample_count].detach().cpu().tolist()} "
            f"obj_height_envs={obj_height[:sample_count].detach().cpu().tolist()}"
        )
        args_cli._grasp_assist_logged = True

    assisted_actions = commanded_actions.clone()
    assisted_actions[should_force_close, 7:9] = 0.0
    return assisted_actions


def _fk_joint_names(env, robot, action_dim: int) -> list[str]:
    joint_names = [
        "base",
        "shoulder",
        "elbow",
        "wrist1",
        "wrist2",
        "wrist3",
        "virual_ee_link",
        "robotiq_hande_left_finger_joint",
        "robotiq_hande_right_finger_joint",
    ]
    if len(joint_names) < action_dim:
        raise ValueError(f"Expected at least {action_dim} FK joint names, got {len(joint_names)}")
    return joint_names[:action_dim]


def _create_fk_helper(action_dim: int) -> ForwardKinematicsHelper:
    joint_names = _fk_joint_names(None, None, action_dim)
    return ForwardKinematicsHelper(
        urdf_path=str(FK_URDF_PATH),
        joint_names=joint_names,
        ee_frame_name="ee_link",
    )


def _attach_gripper_info(info: dict | None, actions: torch.Tensor) -> dict:
    if info is None:
        info = {}

    if actions.shape[-1] >= 9:
        left_gripper = actions[:, 7]
        right_gripper = actions[:, 8]
    elif actions.shape[-1] >= 8:
        left_gripper = actions[:, 7]
        right_gripper = actions[:, 7]
    else:
        left_gripper = torch.full((actions.shape[0],), float("nan"), device=actions.device, dtype=actions.dtype)
        right_gripper = torch.full((actions.shape[0],), float("nan"), device=actions.device, dtype=actions.dtype)

    info["gripper_left"] = left_gripper.detach().clone()
    info["gripper_right"] = right_gripper.detach().clone()
    info["gripper_mean"] = ((left_gripper + right_gripper) * 0.5).detach().clone()
    return info


def _infer_policy_name() -> str:
    text = " ".join(str(x or "") for x in [args_cli.policy_path, args_cli.policy_checkpoint]).lower()
    if "flow" in text or "fmp" in text:
        return "Flow Matching"
    if "diffusion" in text or "dp" in text:
        return "Diffusion"
    return "Policy"


def _auto_save_path(policy_name: str) -> Path:
    safe_policy = policy_name.lower().replace(" ", "_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        Path.cwd()
        / "datas"
        / "manual_imitation_eval"
        / f"{args_cli.task}_{safe_policy}_N{args_cli.output_horizon}_k{args_cli.execute_horizon}_seed{args_cli.seed}_{stamp}.npz"
    )


def _append_manual_eval_reports(save_path: Path, metrics: dict) -> None:
    if args_cli.no_auto_report:
        return

    policy_name = str(metrics["policy_name"])
    task_label = "Reach and Lift" if "reach_lift" in str(metrics["task"]) else "Reach"
    is_reach_lift = task_label == "Reach and Lift"
    success_text = (
        f"{metrics['done_success_rate'] * 100:.2f}% "
        f"({metrics['success_done_count']}/{metrics['episode_done_count']})"
    )
    success_z_text = f"{metrics['success_z']:.4f}" if np.isfinite(metrics["success_z"]) else "-"
    mean_obj_height_text = f"{metrics['mean_obj_height']:.4f}" if np.isfinite(metrics["mean_obj_height"]) else "-"
    max_obj_height_text = f"{metrics['max_obj_height']:.4f}" if np.isfinite(metrics["max_obj_height"]) else "-"
    reach_criterion = f"success_dist <= {metrics['success_dist']:.4f}"
    reach_lift_criterion = f"object_height >= {success_z_text}" if is_reach_lift else "not used"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ppo_path = Path.cwd() / "ppo_test_comparison.md"
    ppo_block = f"""

## 추가: manual DP/FMP 실행 결과 ({timestamp})

- 실행 파일: `{save_path}`
- task: `{metrics['task']}`
- policy: `{policy_name}`
- 조건: `num_envs={args_cli.num_envs}`, `episodes={args_cli.episodes}`, `seed={args_cli.seed}`, `N={args_cli.output_horizon}`, `k={args_cli.execute_horizon}`
- 성공 기준: reach는 `{reach_criterion}`, reach_lift는 `{reach_lift_criterion}`

| 항목 | 값 |
|---|---:|
| success_done | {metrics['success_done_count']}/{metrics['episode_done_count']} |
| done success rate | {metrics['done_success_rate'] * 100:.2f}% |
| live env success mask | {metrics['success_count']}/{metrics['total_envs']} |
| mean ee dist | {metrics['mean_ee_dist']:.4f} |
| mean object height | {mean_obj_height_text} |
| max object height | {max_obj_height_text} |
| action delta | {metrics['mean_action_delta']:.6f} |
| action jerk | {metrics['mean_action_jerk']:.6f} |
| path length | {metrics['mean_path_length']:.4f} m |
| path metric | {metrics['path_metric']} |
| inference latency | {metrics['avg_inference_latency_ms']:.2f} ms |
| replanning | {metrics['replanning_mode']}, {metrics['replanning_frequency_steps']} step |
"""
    with ppo_path.open("a", encoding="utf-8") as f:
        f.write(ppo_block)

    matched_path = Path.cwd() / "matched_policy_sweep_comparison.md"
    matched_block = f"""

## 추가: manual matched eval ({timestamp})

### Reach

| N | k | Policy | Success | action delta | action jerk | path length | latency | replanning |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| {args_cli.output_horizon} | {args_cli.execute_horizon} | {policy_name if task_label == 'Reach' else '-'} | {success_text if task_label == 'Reach' else '-'} | {metrics['mean_action_delta']:.6f} | {metrics['mean_action_jerk']:.6f} | {metrics['mean_path_length']:.4f} m | {metrics['avg_inference_latency_ms']:.2f} ms | {metrics['replanning_mode']}, {metrics['replanning_frequency_steps']} step |

### Reach and Lift

| N | k | Policy | Success | action delta | action jerk | path length | latency | replanning |
|---:|---:|---|---:|---:|---:|---:|---:|---|
| {args_cli.output_horizon} | {args_cli.execute_horizon} | {policy_name if task_label == 'Reach and Lift' else '-'} | {success_text if task_label == 'Reach and Lift' else '-'} | {metrics['mean_action_delta']:.6f} | {metrics['mean_action_jerk']:.6f} | {metrics['mean_path_length']:.4f} m | {metrics['avg_inference_latency_ms']:.2f} ms | {metrics['replanning_mode']}, {metrics['replanning_frequency_steps']} step |

### Reach And Lift Height Check

| N | k | Policy | Success | mean object height | max object height | effective success height |
|---:|---:|---|---:|---:|---:|---:|
| {args_cli.output_horizon} | {args_cli.execute_horizon} | {policy_name if task_label == 'Reach and Lift' else '-'} | {success_text if task_label == 'Reach and Lift' else '-'} | {mean_obj_height_text + ' m' if task_label == 'Reach and Lift' else '-'} | {max_obj_height_text + ' m' if task_label == 'Reach and Lift' else '-'} | {success_z_text + ' m' if task_label == 'Reach and Lift' else '-'} |

### Best By Success

| Task | Policy | N | k | Success | latency | action delta | action jerk |
|---|---|---:|---:|---:|---:|---:|---:|
| {task_label} | {policy_name} | {args_cli.output_horizon} | {args_cli.execute_horizon} | {success_text} | {metrics['avg_inference_latency_ms']:.2f} ms | {metrics['mean_action_delta']:.6f} | {metrics['mean_action_jerk']:.6f} |

Result file: `{save_path}`
"""
    with matched_path.open("a", encoding="utf-8") as f:
        f.write(matched_block)


def _format_gripper_log_values(values: torch.Tensor):
    sample = values.detach().cpu().tolist()
    if len(sample) == 1:
        return sample[0]
    return sample


def _compute_predicted_ee_chunks(predicted_joint_chunks: list[np.ndarray], action_dim: int) -> list[np.ndarray]:
    if not predicted_joint_chunks:
        return []
    try:
        _get_pinocchio()
    except RuntimeError as exc:
        raise RuntimeError(
            "FK conversion is unavailable because pinocchio could not be imported in the IsaacLab Python runtime."
        ) from exc
    fk_script = f"""
import pickle
import sys

import numpy as np
import pinocchio as pin
import torch

urdf_path = r"{FK_URDF_PATH}"
joint_names = [
    "base",
    "shoulder",
    "elbow",
    "wrist1",
    "wrist2",
    "wrist3",
    "virual_ee_link",
    "robotiq_hande_left_finger_joint",
    "robotiq_hande_right_finger_joint",
]

with open(sys.argv[1], "rb") as f:
    predicted_joint_chunks = pickle.load(f)

model = pin.buildModelFromUrdf(urdf_path)
data = model.createData()
ee_frame_id = model.getFrameId("ee_link")
model_joint_ids = {{model.names[i]: i for i in range(len(model.names))}}

predicted_ee_chunks = []
for joint_chunk in predicted_joint_chunks:
    joint_tensor = torch.as_tensor(joint_chunk, dtype=torch.float64)
    batch, horizon, _ = joint_tensor.shape
    out = torch.zeros((batch, horizon, 3), dtype=torch.float32)
    for b in range(batch):
        for h in range(horizon):
            q = pin.neutral(model)
            for value, joint_name in zip(joint_tensor[b, h], joint_names):
                joint_id = model_joint_ids.get(joint_name)
                if joint_id is None:
                    continue
                idx_q = model.joints[joint_id].idx_q
                nq = model.joints[joint_id].nq
                if nq == 1 and 0 <= idx_q < q.shape[0]:
                    q[idx_q] = float(value.item())
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pos = data.oMf[ee_frame_id].translation
            out[b, h] = torch.tensor([pos[0], pos[1], pos[2]], dtype=torch.float32)
    predicted_ee_chunks.append(out.cpu().numpy())

with open(sys.argv[2], "wb") as f:
    pickle.dump(predicted_ee_chunks, f)
"""

    with tempfile.TemporaryDirectory(prefix="fk_chunks_") as temp_dir:
        input_path = Path(temp_dir) / "predicted_joint_chunks.pkl"
        output_path = Path(temp_dir) / "predicted_ee_chunks.pkl"
        with open(input_path, "wb") as f:
            pickle.dump(predicted_joint_chunks, f)
        python_executable = sys.executable or "/usr/bin/python3"
        python_env = os.environ.copy()
        python_env.setdefault("LANG", "C.UTF-8")
        python_env.setdefault("LC_ALL", "C.UTF-8")
        result = subprocess.run(
            [python_executable, "-c", fk_script, str(input_path), str(output_path), str(action_dim)],
            check=False,
            capture_output=True,
            text=True,
            env=python_env,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"FK subprocess failed with code {result.returncode}")
        with open(output_path, "rb") as f:
            return pickle.load(f)


def _compute_predicted_ee_chunk(joint_chunk: np.ndarray, action_dim: int) -> np.ndarray:
    return _compute_predicted_ee_chunks([joint_chunk], action_dim)[0]


def _transform_predicted_ee_chunk_to_world(robot, predicted_ee_chunk: np.ndarray) -> np.ndarray:
    predicted_ee_chunk_t = torch.as_tensor(
        predicted_ee_chunk,
        device=robot.device,
        dtype=robot.data.root_pos_w.dtype,
    )
    batch, horizon, _ = predicted_ee_chunk_t.shape
    root_pos = robot.data.root_pos_w[:, None, :].expand(batch, horizon, 3)
    root_quat = robot.data.root_quat_w[:, None, :].expand(batch, horizon, 4)
    world_pos, _ = combine_frame_transforms(
        root_pos.reshape(-1, 3),
        root_quat.reshape(-1, 4),
        predicted_ee_chunk_t.reshape(-1, 3),
    )
    return world_pos.reshape(batch, horizon, 3).detach().cpu().numpy()


def _compute_predicted_ee_chunk_from_sim(
    robot,
    joint_chunk: np.ndarray,
    ee_body_idx: int,
    actuated_joint_ids: list[int] | torch.Tensor | None = None,
) -> np.ndarray:
    joint_chunk_t = torch.as_tensor(joint_chunk, device=robot.device, dtype=robot.data.joint_pos.dtype)
    batch, horizon, _ = joint_chunk_t.shape
    out = torch.zeros((batch, horizon, 3), device=robot.device, dtype=torch.float32)

    original_joint_pos = robot.data.joint_pos.clone()
    original_joint_vel = robot.data.joint_vel.clone()
    zero_vel = torch.zeros((batch, joint_chunk_t.shape[-1]), device=robot.device, dtype=robot.data.joint_vel.dtype)

    try:
        for h in range(horizon):
            robot.write_joint_state_to_sim(
                joint_chunk_t[:, h, :],
                zero_vel,
                joint_ids=actuated_joint_ids,
            )
            out[:, h, :] = robot.data.body_pos_w[:, ee_body_idx, :]
    finally:
        robot.write_joint_state_to_sim(original_joint_pos, original_joint_vel)

    return out.detach().cpu().numpy()


def _select_reach_closest_waypoint_chunk(
    action_chunk: torch.Tensor,
    robot,
    target_pos_w: torch.Tensor,
    action_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the predicted joint waypoint whose FK is closest to the current reach target."""
    if action_chunk.ndim != 3 or action_chunk.shape[1] <= 1:
        empty = torch.zeros((action_chunk.shape[0],), device=action_chunk.device, dtype=torch.long)
        return action_chunk, empty

    predicted_ee_local = _compute_predicted_ee_chunk(action_chunk.detach().cpu().numpy(), action_dim)
    predicted_ee_world_np = _transform_predicted_ee_chunk_to_world(robot, predicted_ee_local)
    predicted_ee_world = torch.as_tensor(
        predicted_ee_world_np,
        device=action_chunk.device,
        dtype=target_pos_w.dtype,
    )
    dists = torch.linalg.norm(predicted_ee_world - target_pos_w[:, None, :], dim=-1)
    best_idx = torch.argmin(dists, dim=1)
    env_idx = torch.arange(action_chunk.shape[0], device=action_chunk.device)
    selected = action_chunk[env_idx, best_idx, :]
    if args_cli.reach_select_repeat:
        selected_chunk = selected[:, None, :].repeat(1, action_chunk.shape[1], 1)
    else:
        selected_chunk = action_chunk.clone()
        selected_chunk[:, 0, :] = selected
    return selected_chunk, best_idx


def _run_policy(
    policy,
    obs_tensor: torch.Tensor,
    action_dim: int,
    device: torch.device | str,
    reference_action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if policy is None:
        zeros = torch.zeros((obs_tensor.shape[0], args_cli.output_horizon, action_dim), device=device)
        return zeros, zeros, 0.0

    inference_start = time.perf_counter()
    with torch.inference_mode():
        if args_cli.policy_type == "python":
            call_sig = inspect.signature(policy.forward if hasattr(policy, "forward") else policy.__call__)
            kwargs = {}
            if "output_horizon" in call_sig.parameters:
                kwargs["output_horizon"] = args_cli.output_horizon
            if "execute_horizon" in call_sig.parameters:
                kwargs["execute_horizon"] = args_cli.execute_horizon
            actions = policy(obs_tensor, **kwargs)
        else:
            actions = policy(obs_tensor)
    inference_elapsed = time.perf_counter() - inference_start

    if isinstance(actions, tuple):
        actions = actions[0]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions, device=device, dtype=torch.float32)
    actions = actions.to(device=device, dtype=torch.float32)
    raw_actions = actions.clone()

    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    elif actions.ndim == 1:
        actions = actions.view(1, 1, -1).repeat(obs_tensor.shape[0], 1, 1)
    elif actions.ndim != 3:
        raise ValueError(f"Unsupported action tensor shape: {tuple(actions.shape)}")

    if actions.shape[0] != obs_tensor.shape[0]:
        if actions.shape[0] == 1:
            actions = actions.repeat(obs_tensor.shape[0], 1, 1)
        else:
            raise ValueError(
                f"Batch mismatch between obs {obs_tensor.shape[0]} and actions {actions.shape[0]}"
            )

    actions = _expand_policy_actions(actions, action_dim, reference_action=reference_action)

    if args_cli.output_horizon > 0 and actions.shape[1] != args_cli.output_horizon:
        raise ValueError(
            f"Expected output horizon {args_cli.output_horizon}, got {actions.shape[1]}"
        )

    return actions, raw_actions, inference_elapsed


@hydra_task_config(args_cli.task, None)
def main(env_cfg: DirectRLEnvCfg, agent_cfg=None):
    if args_cli.output_horizon < 1:
        raise ValueError("--output_horizon must be >= 1")
    if args_cli.execute_horizon < 1:
        raise ValueError("--execute_horizon must be >= 1")
    if args_cli.execute_horizon > args_cli.output_horizon:
        raise ValueError("--execute_horizon must be <= --output_horizon")
    if args_cli.replan_lookahead_steps < 0:
        raise ValueError("--replan_lookahead_steps must be >= 0")
    if args_cli.episodes < 0:
        raise ValueError("--episodes must be >= 0")
    if args_cli.episode_length_s is not None and args_cli.episode_length_s <= 0:
        raise ValueError("--episode_length_s must be > 0")

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if hasattr(env_cfg, "success_threshold"):
        env_cfg.success_threshold = float(args_cli.success_dist)
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if args_cli.success_z is not None and terminations_cfg is not None and hasattr(terminations_cfg, "success_object_height"):
        terminations_cfg.success_object_height = float(args_cli.success_z)
    elif args_cli.success_z is None and terminations_cfg is not None and hasattr(terminations_cfg, "success_object_height"):
        args_cli.success_z = float(terminations_cfg.success_object_height)
    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.seed is not None:
        np.random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.getcwd(), "videos", args_cli.task),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording imitation play video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    executed_ee_positions = []
    predicted_ee_chunks = []
    predicted_joint_chunks = []
    success_mask = None
    ee_dist_sum = 0.0
    obj_height_sum = 0.0
    metric_steps = 0
    episode_done_count = 0
    success_done_count = 0
    action_dim = None
    path_visualizers = []
    path_debug_logged = False
    fk_visualization_enabled = False
    reach_ik_helper = None
    fk_fallback_to_sim_logged = False
    action_debug_logged = False
    history_debug_logged = False
    reach_select_logged = False
    obs_history_buffer = None
    pending_history_reset_mask = None
    current_episode_replans = None
    current_episode_inference_time = None
    current_episode_inference_calls = None
    successful_episode_replans = []
    successful_episode_inference_times = []
    successful_episode_inference_calls = []
    total_inference_time = 0.0
    total_inference_calls = 0
    smoothness_sum = 0.0
    smoothness_count = 0
    action_delta_sum = 0.0
    action_delta_count = 0
    action_jerk_sum = 0.0
    action_jerk_count = 0
    last_action_delta = None
    path_metric = "object" if "reach_lift" in args_cli.task else "end_effector"
    previous_path_pos = None
    current_episode_path_length = None
    completed_path_lengths = []
    path_positions = []
    max_obj_height = float("nan")

    try:
        obs, _ = env.reset(seed=args_cli.seed)
        device = env.unwrapped.device
        action_dim = env.unwrapped.action_space.shape[-1]
        policy = _load_policy(args_cli.policy_path, args_cli.policy_type, device)
        dt = env.unwrapped.step_dt
        robot = env.unwrapped.scene["robot"]
        try:
            obj = env.unwrapped.scene["obj"]
        except Exception:
            obj = None
        ee_body_idx = robot.find_bodies("ee_link")[0][0]
        fk_visualization_enabled = (
            (not getattr(args_cli, "headless", False))
            and (not args_cli.disable_predicted_path_viz)
            and FK_URDF_PATH.is_file()
        )
        if args_cli.reach_ik_refine_steps > 0 and obj is None:
            try:
                reach_ik_helper = _create_fk_helper(action_dim)
                print(
                    "[INFO] Reach IK refinement enabled "
                    f"steps={args_cli.reach_ik_refine_steps} "
                    f"mode={args_cli.reach_ik_refine_mode} "
                    f"gain={args_cli.reach_ik_gain:.3f} "
                    f"damping={args_cli.reach_ik_damping:.6f} "
                    f"max_delta={args_cli.reach_ik_max_delta:.3f} "
                    f"activation_dist={args_cli.reach_ik_activation_dist:.3f}"
                )
            except Exception as exc:
                reach_ik_helper = None
                print(f"[WARN] Reach IK refinement unavailable. reason={exc}")
        if fk_visualization_enabled:
            path_visualizers = [
                PredictedPathVisualizer(
                    num_points=args_cli.output_horizon,
                    root_path=f"/World/PredictedPath/env_{env_idx:03d}",
                )
                for env_idx in range(args_cli.num_envs)
            ]
        elif args_cli.disable_predicted_path_viz:
            print("[INFO] Predicted path visualization disabled by CLI flag.")
        elif not FK_URDF_PATH.is_file():
            print(f"[INFO] FK visualization disabled because URDF was not found: {FK_URDF_PATH}")
        step_count = 0
        chunk_id = 0
        cached_actions = None
        cached_index = 0
        executed_since_replan = 0
        action_hold_steps = 0
        replan_interval = args_cli.execute_horizon
        prefetched_actions = None
        prefetched_raw_policy_actions = None
        prefetched_from_step = None
        initial_last_action = _initial_last_action_tensor(env, robot, action_dim)
        last_action_tensor = initial_last_action.clone()
        last_action_valid = torch.zeros(args_cli.num_envs, device=device, dtype=torch.bool)
        last_action_delta = torch.zeros_like(last_action_tensor)
        current_episode_path_length = torch.zeros(args_cli.num_envs, device=device, dtype=torch.float64)
        current_episode_replans = torch.zeros(args_cli.num_envs, device=device, dtype=torch.int32)
        current_episode_inference_time = torch.zeros(args_cli.num_envs, device=device, dtype=torch.float64)
        current_episode_inference_calls = torch.zeros(args_cli.num_envs, device=device, dtype=torch.int32)
        print(
            f"[INFO] Policy rollout configured with output_horizon={args_cli.output_horizon}, "
            f"execute_horizon={args_cli.execute_horizon}. "
            f"The policy predicts and visualizes {args_cli.output_horizon} action points, "
            f"but only the first {replan_interval} points are executed before replanning. "
            f"Each point is held until joint error <= {args_cli.joint_arrival_tolerance} "
            f"or {args_cli.max_action_hold_steps} env steps elapse. "
            f"Command smoothing alpha={args_cli.action_smoothing_alpha:.3f}, "
            f"lookahead_margin={args_cli.next_action_lookahead_margin:.4f}, "
            f"replan_lookahead_steps={args_cli.replan_lookahead_steps}, "
            f"chunk_transition_blend_steps={args_cli.chunk_transition_blend_steps}."
        )
        while simulation_app.is_running():
            start_time = time.time()
            raw_obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            obs_tensor = raw_obs_tensor
            if args_cli.policy_type == "python" and _uses_stepwise_obs_history(getattr(policy, "model_variant", None)):
                frame_dim = int(getattr(policy, "processed_obs_frame_dim", 31))
                history_steps = int(getattr(policy, "obs_history_steps", 1))
                current_frame = _prepare_policy_observation_frame(frame_dim, raw_obs_tensor, last_action_tensor)
                obs_history_buffer, obs_tensor = _update_obs_history_buffer(
                    obs_history_buffer,
                    current_frame,
                    history_steps,
                    reset_mask=pending_history_reset_mask,
                )
                pending_history_reset_mask = None
                if not history_debug_logged:
                    print(
                        "[INFO] Using step-wise observation history for policy input "
                        f"model_variant={policy.model_variant} "
                        f"frame_dim={frame_dim} "
                        f"history_steps={history_steps} "
                        f"obs_dim={obs_tensor.shape[-1]}"
                    )
                    history_debug_logged = True
            else:
                obs_tensor = _prepare_policy_observation(policy, raw_obs_tensor, last_action_tensor)

            if cached_actions is None or executed_since_replan >= min(replan_interval, cached_actions.shape[1]):
                if prefetched_actions is not None:
                    cached_actions = prefetched_actions
                    raw_policy_actions = prefetched_raw_policy_actions
                    # print(
                    #     f"[INFO] Using prefetched chunk at global_step={step_count} "
                    #     f"prepared_from_step={prefetched_from_step}"
                    # )
                    prefetched_actions = None
                    prefetched_raw_policy_actions = None
                    prefetched_from_step = None
                else:
                    cached_actions, raw_policy_actions, inference_elapsed = _run_policy(
                        policy,
                        obs_tensor,
                        action_dim,
                        device,
                        reference_action=last_action_tensor,
                    )
                    total_inference_time += inference_elapsed
                    total_inference_calls += 1
                    current_episode_inference_time += inference_elapsed
                    current_episode_inference_calls += 1
                cached_actions = _blend_chunk_transition(
                    cached_actions,
                    last_action_tensor,
                    args_cli.chunk_transition_blend_steps,
                )
                if args_cli.reach_select_closest_waypoint and obj is None and hasattr(env.unwrapped, "pose_command"):
                    target_pos_w, _ = combine_frame_transforms(
                        robot.data.root_pos_w,
                        robot.data.root_quat_w,
                        env.unwrapped.pose_command[:, :3],
                    )
                    try:
                        cached_actions, selected_waypoint_idx = _select_reach_closest_waypoint_chunk(
                            cached_actions,
                            robot,
                            target_pos_w,
                            action_dim,
                        )
                        if not reach_select_logged:
                            sample_count = min(8, selected_waypoint_idx.numel())
                            print(
                                "[INFO] Reach closest-waypoint selection enabled "
                                f"repeat={args_cli.reach_select_repeat} "
                                f"sample_indices={selected_waypoint_idx[:sample_count].detach().cpu().tolist()}"
                            )
                            reach_select_logged = True
                    except Exception as exc:
                        if not reach_select_logged:
                            print(f"[WARN] Reach closest-waypoint selection disabled after failure. reason={exc}")
                            reach_select_logged = True
                if reach_ik_helper is not None and obj is None and hasattr(env.unwrapped, "pose_command"):
                    try:
                        if args_cli.reach_ik_refine_mode == "repeat_first":
                            selected = cached_actions[:, 0, :]
                            refined = reach_ik_helper.refine_reach_targets(
                                selected,
                                env.unwrapped.pose_command[:, :3],
                                args_cli.reach_ik_refine_steps,
                                args_cli.reach_ik_gain,
                                args_cli.reach_ik_damping,
                                args_cli.reach_ik_max_delta,
                            )
                            cached_actions = refined[:, None, :].repeat(1, cached_actions.shape[1], 1)
                        else:
                            batch_size, horizon, action_width = cached_actions.shape
                            flat_actions = cached_actions.reshape(batch_size * horizon, action_width)
                            flat_targets = env.unwrapped.pose_command[:, :3].repeat_interleave(horizon, dim=0)
                            refined_flat = reach_ik_helper.refine_reach_targets(
                                flat_actions,
                                flat_targets,
                                args_cli.reach_ik_refine_steps,
                                args_cli.reach_ik_gain,
                                args_cli.reach_ik_damping,
                                args_cli.reach_ik_max_delta,
                            )
                            refined_actions = refined_flat.reshape(batch_size, horizon, action_width)
                            if args_cli.reach_ik_activation_dist > 0.0:
                                predicted_ee_b = reach_ik_helper.ee_positions(cached_actions).to(cached_actions.device)
                                target_b = env.unwrapped.pose_command[:, None, :3].to(cached_actions.device)
                                refine_mask = torch.linalg.norm(predicted_ee_b - target_b, dim=-1) <= float(
                                    args_cli.reach_ik_activation_dist
                                )
                                cached_actions = torch.where(refine_mask[..., None], refined_actions, cached_actions)
                            else:
                                cached_actions = refined_actions
                    except Exception as exc:
                        print(f"[WARN] Reach IK refinement failed for this chunk. reason={exc}")
                current_episode_replans += 1
                cached_index = 0
                executed_since_replan = 0
                action_hold_steps = 0
                chunk_id += 1
                # print(
                #     f"[INFO] chunk_id={chunk_id} generated at global_step={step_count} "
                #     f"with execute_horizon={replan_interval}"
                # )
                cached_actions_np = cached_actions.detach().cpu().numpy()
                if args_cli.log_gripper:
                    raw_policy_actions_np = raw_policy_actions.detach().cpu().numpy()
                    sample_envs = min(5, raw_policy_actions_np.shape[0])
                    sample_steps = min(3, raw_policy_actions_np.shape[1])
                    raw_gripper_sample = raw_policy_actions_np[:sample_envs, :sample_steps, -1].tolist()
                    if cached_actions_np.shape[-1] >= 9:
                        sim_left_gripper_sample = cached_actions_np[:sample_envs, :sample_steps, 7].tolist()
                        sim_right_gripper_sample = cached_actions_np[:sample_envs, :sample_steps, 8].tolist()
                    elif cached_actions_np.shape[-1] >= 8:
                        sim_left_gripper_sample = cached_actions_np[:sample_envs, :sample_steps, 7].tolist()
                        sim_right_gripper_sample = cached_actions_np[:sample_envs, :sample_steps, 7].tolist()
                    else:
                        sim_left_gripper_sample = []
                        sim_right_gripper_sample = []
                    threshold_mask_sample = (
                        raw_policy_actions_np[:sample_envs, :sample_steps, -1] < args_cli.gripper_binary_threshold
                    ).tolist()
                    print(
                        "[INFO] Gripper debug "
                        f"chunk_id={chunk_id} "
                        f"binary_gripper={args_cli.binary_gripper} "
                        f"threshold={args_cli.gripper_binary_threshold:.6f} "
                        f"raw_envs_steps={raw_gripper_sample} "
                        f"is_closed_mask={threshold_mask_sample} "
                        f"sim_left_envs_steps={sim_left_gripper_sample} "
                        f"sim_right_envs_steps={sim_right_gripper_sample}"
                    )
                if not action_debug_logged:
                    raw_policy_actions_np = raw_policy_actions.detach().cpu().numpy()
                    model_output_dim = int(raw_policy_actions_np.shape[-1])
                    ee_link_value = float(cached_actions_np[0, 0, 6]) if cached_actions_np.shape[-1] >= 7 else float("nan")
                    raw_gripper_value = float(raw_policy_actions_np[0, 0, -1])
                    sim_left_gripper = float(cached_actions_np[0, 0, 7]) if cached_actions_np.shape[-1] >= 8 else float("nan")
                    sim_right_gripper = float(cached_actions_np[0, 0, 8]) if cached_actions_np.shape[-1] >= 9 else float("nan")
                    print(
                        "[INFO] Action mapping sample "
                        f"model_output_dim={model_output_dim} "
                        f"sim_action_dim={cached_actions_np.shape[-1]} "
                        f"ee_link={ee_link_value:.6f} "
                        f"raw_gripper={raw_gripper_value:.6f} "
                        f"sim_left_gripper={sim_left_gripper:.6f} "
                        f"sim_right_gripper={sim_right_gripper:.6f}"
                    )
                    action_debug_logged = True
                predicted_joint_chunks.append(cached_actions_np)
                if fk_visualization_enabled:
                    try:
                        predicted_ee_chunk = _compute_predicted_ee_chunk(cached_actions_np, action_dim)
                        predicted_ee_chunk_world = _transform_predicted_ee_chunk_to_world(robot, predicted_ee_chunk)
                        predicted_ee_chunks.append(predicted_ee_chunk_world)
                        for env_idx, path_visualizer in enumerate(path_visualizers):
                            path_visualizer.update(predicted_ee_chunk_world[env_idx])
                        if not path_debug_logged:
                            print(
                                "[INFO] Drew predicted path prims "
                                f"roots={len(path_visualizers)} "
                                f"sample_root={path_visualizers[0].root_path} "
                                f"world_p0={predicted_ee_chunk_world[0, 0].tolist()} "
                                f"world_p15={predicted_ee_chunk_world[0, -1].tolist()} "
                                f"anchor={_finger_midpoint_world(robot, 0).tolist()}"
                            )
                            path_debug_logged = True
                    except Exception as exc:
                        if not fk_fallback_to_sim_logged:
                            print(f"[WARN] Pinocchio FK unavailable, falling back to simulation FK. reason={exc}")
                            fk_fallback_to_sim_logged = True
                        try:
                            predicted_ee_chunk = _compute_predicted_ee_chunk_from_sim(
                                robot,
                                cached_actions_np,
                                ee_body_idx=ee_body_idx,
                                actuated_joint_ids=getattr(env.unwrapped, "actuated_joint_ids", None),
                            )
                            predicted_ee_chunks.append(predicted_ee_chunk)
                            for env_idx, path_visualizer in enumerate(path_visualizers):
                                path_visualizer.update(predicted_ee_chunk[env_idx])
                            if not path_debug_logged:
                                print(
                                    "[INFO] Drew predicted path prims with simulation FK "
                                    f"roots={len(path_visualizers)} "
                                    f"sample_root={path_visualizers[0].root_path} "
                                    f"world_p0={predicted_ee_chunk[0, 0].tolist()} "
                                    f"world_p15={predicted_ee_chunk[0, -1].tolist()}"
                                )
                                path_debug_logged = True
                        except Exception as sim_exc:
                            print(f"[WARN] Failed to draw FK point path with simulation fallback. reason={sim_exc}")
            else:
                current_chunk_horizon = min(replan_interval, cached_actions.shape[1])
                remaining_before_replan = current_chunk_horizon - executed_since_replan
                should_prefetch = (
                    prefetched_actions is None
                    and args_cli.replan_lookahead_steps > 0
                    and remaining_before_replan <= args_cli.replan_lookahead_steps
                )
                if should_prefetch:
                    prefetched_actions, prefetched_raw_policy_actions, inference_elapsed = _run_policy(
                        policy,
                        obs_tensor,
                        action_dim,
                        device,
                        reference_action=last_action_tensor,
                    )
                    total_inference_time += inference_elapsed
                    total_inference_calls += 1
                    current_episode_inference_time += inference_elapsed
                    current_episode_inference_calls += 1
                    prefetched_actions = _blend_chunk_transition(
                        prefetched_actions,
                        last_action_tensor,
                        args_cli.chunk_transition_blend_steps,
                    )
                    prefetched_from_step = step_count
                    # print(
                    #     f"[INFO] Prefetched next chunk at global_step={step_count} "
                    #     f"remaining_before_replan={remaining_before_replan}"
                    # )

            target_actions = cached_actions[:, cached_index, :]
            commanded_actions = _smooth_command_action(
                target_actions,
                last_action_tensor,
                args_cli.action_smoothing_alpha,
            )
            ee_pos = robot.data.body_pos_w[:, ee_body_idx, :]
            commanded_actions = _apply_grasp_assist(commanded_actions, ee_pos, obj)
            if torch.any(last_action_valid):
                action_delta = commanded_actions[last_action_valid] - last_action_tensor[last_action_valid]
                smoothness_sum += float(torch.sum(action_delta.square(), dim=-1).sum().item())
                smoothness_count += int(last_action_valid.sum().item())
                action_delta_norm = torch.linalg.norm(action_delta, dim=-1)
                action_delta_sum += float(action_delta_norm.sum().item())
                action_delta_count += int(action_delta_norm.numel())
                previous_delta = last_action_delta[last_action_valid]
                action_jerk = action_delta - previous_delta
                action_jerk_norm = torch.linalg.norm(action_jerk, dim=-1)
                action_jerk_sum += float(action_jerk_norm.sum().item())
                action_jerk_count += int(action_jerk_norm.numel())
            obs, _, terminated, truncated, info = env.step(commanded_actions)
            info = _attach_gripper_info(info, commanded_actions)
            action_hold_steps += 1
            if torch.any(last_action_valid):
                next_delta = commanded_actions - last_action_tensor
                last_action_delta = last_action_delta.clone()
                last_action_delta[last_action_valid] = next_delta[last_action_valid]
            last_action_tensor = commanded_actions.detach()
            last_action_valid = torch.ones_like(last_action_valid)
            ee_pos = robot.data.body_pos_w[:, ee_body_idx, :]
            executed_ee_positions.append(ee_pos.detach().cpu().numpy())
            if obj is not None and path_metric == "object":
                current_path_pos = obj.data.root_pos_w[:, :3].detach()
                obj_z_now = obj.data.root_pos_w[:, 2].detach()
                max_obj_height = float(torch.max(obj_z_now).item()) if np.isnan(max_obj_height) else max(max_obj_height, float(torch.max(obj_z_now).item()))
            else:
                current_path_pos = ee_pos.detach()
            path_positions.append(current_path_pos.detach().cpu().numpy())
            if previous_path_pos is not None and current_episode_path_length is not None:
                step_path_delta = torch.linalg.norm(current_path_pos - previous_path_pos, dim=-1).to(dtype=torch.float64)
                current_episode_path_length += step_path_delta
            previous_path_pos = current_path_pos.clone()

            if args_cli.log_gripper:
                sample_envs = min(5, commanded_actions.shape[0])
                left_sample = _format_gripper_log_values(info["gripper_left"][:sample_envs])
                right_sample = _format_gripper_log_values(info["gripper_right"][:sample_envs])
                mean_sample = _format_gripper_log_values(info["gripper_mean"][:sample_envs])
                # print(
                #     "[INFO] gripper_info "
                #     f"at global_step={step_count} "
                #     f"info.gripper_left={left_sample} "
                #     f"info.gripper_right={right_sample} "
                #     f"info.gripper_mean={mean_sample}"
                # )

            actuated_joint_ids = getattr(env.unwrapped, "actuated_joint_ids", None)
            current_joint_pos = (
                robot.data.joint_pos[:, actuated_joint_ids]
                if actuated_joint_ids is not None
                else robot.data.joint_pos[:, : target_actions.shape[-1]]
            )
            joint_error = torch.max(torch.abs(current_joint_pos - target_actions), dim=-1).values
            reached_action = joint_error <= args_cli.joint_arrival_tolerance
            progressed_to_next = torch.zeros_like(reached_action)
            if cached_index + 1 < cached_actions.shape[1]:
                next_target_actions = cached_actions[:, cached_index + 1, :]
                next_joint_error = torch.max(torch.abs(current_joint_pos - next_target_actions), dim=-1).values
                progressed_to_next = next_joint_error + args_cli.next_action_lookahead_margin < joint_error
            advance_mask = reached_action | progressed_to_next
            advance_action = bool(torch.all(advance_mask).item()) or action_hold_steps >= args_cli.max_action_hold_steps
            if advance_action:
                cached_index += 1
                executed_since_replan += 1
                action_hold_steps = 0

            if hasattr(env.unwrapped, "pose_command"):
                target_pos, _ = combine_frame_transforms(
                    robot.data.root_pos_w,
                    robot.data.root_quat_w,
                    env.unwrapped.pose_command[:, :3],
                )
                ee_dist = torch.norm(ee_pos - target_pos, dim=-1)

                ee_dist_sum += float(ee_dist.mean().item())
                metric_steps += 1

                if success_mask is None:
                    success_mask = torch.zeros_like(ee_dist, dtype=torch.bool)
                if obj is not None:
                    obj_height = info.get("object_height_before_reset")
                    if obj_height is None:
                        obj_height = obj.data.root_pos_w[:, 2]
                    obj_height_sum += float(obj_height.mean().item())
                    success_step_mask = info.get("is_success_step")
                    if success_step_mask is None:
                        success_step_mask = obj_height >= args_cli.success_z
                    success_mask |= success_step_mask
                else:
                    success_threshold = float(getattr(env.unwrapped.cfg, "success_threshold", args_cli.success_dist))
                    success_mask |= ee_dist <= success_threshold

            step_count += 1

            if torch.any(terminated | truncated):
                done_mask = terminated | truncated
                counted_done_mask = done_mask
                if args_cli.episodes > 0:
                    remaining_episodes = max(args_cli.episodes - episode_done_count, 0)
                    done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
                    if done_indices.numel() > remaining_episodes:
                        counted_done_mask = torch.zeros_like(done_mask)
                        if remaining_episodes > 0:
                            counted_done_mask[done_indices[:remaining_episodes]] = True

                success_step_mask = info.get("is_success_step")
                if success_step_mask is None:
                    success_step_mask = terminated.new_zeros(terminated.shape)

                success_done_mask = (
                    counted_done_mask & success_mask if success_mask is not None else counted_done_mask.new_zeros(counted_done_mask.shape)
                )
                success_terminated_mask = terminated & counted_done_mask & success_step_mask
                success_terminated = int(success_terminated_mask.sum().item())
                success_completed_this_step = int(success_done_mask.sum().item())
                successful_done_indices = torch.nonzero(success_done_mask, as_tuple=False).flatten()
                if successful_done_indices.numel() > 0:
                    successful_episode_replans.extend(current_episode_replans[successful_done_indices].detach().cpu().tolist())
                    successful_episode_inference_times.extend(
                        current_episode_inference_time[successful_done_indices].detach().cpu().tolist()
                    )
                    successful_episode_inference_calls.extend(
                        current_episode_inference_calls[successful_done_indices].detach().cpu().tolist()
                    )
                if current_episode_path_length is not None:
                    counted_done_indices = torch.nonzero(counted_done_mask, as_tuple=False).flatten()
                    if counted_done_indices.numel() > 0:
                        completed_path_lengths.extend(
                            current_episode_path_length[counted_done_indices].detach().cpu().tolist()
                        )
                episode_done_count += int(counted_done_mask.sum().item())
                success_done_count += int(success_done_mask.sum().item())
                print("-----------------------------")
                print(
                    f"[INFO] step={step_count} terminated={terminated.sum().item()} "
                    f"truncated={truncated.sum().item()} counted_done={int(counted_done_mask.sum().item())} "
                    f"success_terminated_this_step={success_terminated} "
                    f"success_completed_this_step={success_completed_this_step} "
                    f"success_done_total={success_done_count}"
                )
                print("-----------------------------")
                last_action_tensor = last_action_tensor.clone()
                last_action_tensor[done_mask] = initial_last_action[done_mask]
                last_action_valid = last_action_valid.clone()
                last_action_valid[done_mask] = False
                last_action_delta = last_action_delta.clone()
                last_action_delta[done_mask] = 0.0
                if current_episode_path_length is not None:
                    current_episode_path_length = current_episode_path_length.clone()
                    current_episode_path_length[done_mask] = 0.0
                if previous_path_pos is not None:
                    previous_path_pos = previous_path_pos.clone()
                    previous_path_pos[done_mask] = current_path_pos[done_mask]
                if success_mask is not None:
                    success_mask = success_mask.clone()
                    success_mask[done_mask] = False
                current_episode_replans = current_episode_replans.clone()
                current_episode_replans[done_mask] = 0
                current_episode_inference_time = current_episode_inference_time.clone()
                current_episode_inference_time[done_mask] = 0.0
                current_episode_inference_calls = current_episode_inference_calls.clone()
                current_episode_inference_calls[done_mask] = 0
                pending_history_reset_mask = done_mask.clone()
                cached_actions = None
                cached_index = 0
                executed_since_replan = 0
                action_hold_steps = 0
                prefetched_actions = None
                prefetched_raw_policy_actions = None
                prefetched_from_step = None

                if args_cli.episodes > 0 and episode_done_count >= args_cli.episodes:
                    break

            if args_cli.steps > 0 and step_count >= args_cli.steps:
                break
            if args_cli.video and step_count >= args_cli.video_length:
                break

            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        terminations_cfg = getattr(env.unwrapped.cfg, "terminations", None)
        raw_success_z = getattr(terminations_cfg, "success_object_height", args_cli.success_z)
        actual_success_z = float(raw_success_z) if raw_success_z is not None else float("nan")
        actual_success_dist = float(getattr(env.unwrapped.cfg, "success_threshold", args_cli.success_dist))
        env.close()
        success_count = int(success_mask.sum().item()) if success_mask is not None else 0
        total_envs = int(success_mask.numel()) if success_mask is not None else int(args_cli.num_envs)
        success_rate = float(success_mask.float().mean().item()) if success_mask is not None else float("nan")
        done_success_rate = (success_done_count / episode_done_count) if episode_done_count > 0 else float("nan")
        mean_ee_dist = (ee_dist_sum / metric_steps) if metric_steps > 0 else float("nan")
        mean_obj_height = (obj_height_sum / metric_steps) if metric_steps > 0 else float("nan")
        mean_smoothness = (smoothness_sum / smoothness_count) if smoothness_count > 0 else float("nan")
        mean_action_delta = (action_delta_sum / action_delta_count) if action_delta_count > 0 else float("nan")
        mean_action_jerk = (action_jerk_sum / action_jerk_count) if action_jerk_count > 0 else float("nan")
        mean_path_length = float(np.mean(completed_path_lengths)) if completed_path_lengths else float("nan")
        success_episode_latency_ms = [
            1000.0 * episode_time / episode_calls
            for episode_time, episode_calls in zip(successful_episode_inference_times, successful_episode_inference_calls)
            if episode_calls > 0 and episode_time > 0.0
        ]
        success_episode_hz = [
            episode_calls / episode_time
            for episode_time, episode_calls in zip(successful_episode_inference_times, successful_episode_inference_calls)
            if episode_calls > 0 and episode_time > 0.0
        ]
        avg_success_replans = (
            float(np.mean(successful_episode_replans)) if successful_episode_replans else float("nan")
        )
        avg_success_inference_latency_ms = (
            float(np.mean(success_episode_latency_ms))
            if success_episode_latency_ms
            else float("nan")
        )
        avg_success_inference_hz = (
            float(np.mean(success_episode_hz))
            if success_episode_hz
            else float("nan")
        )
        avg_inference_latency_ms = (
            (1000.0 * total_inference_time / total_inference_calls) if total_inference_calls > 0 else float("nan")
        )
        avg_inference_hz = (
            (total_inference_calls / total_inference_time) if total_inference_time > 0.0 else float("nan")
        )

        policy_name = _infer_policy_name()
        save_path = (
            Path(args_cli.save_path).expanduser().resolve()
            if args_cli.save_path is not None
            else _auto_save_path(policy_name).resolve()
        )
        metrics = {
            "task": args_cli.task,
            "policy_name": policy_name,
            "success_count": success_count,
            "episode_done_count": episode_done_count,
            "success_done_count": success_done_count,
            "total_envs": total_envs,
            "success_rate": success_rate,
            "done_success_rate": done_success_rate,
            "mean_ee_dist": mean_ee_dist,
            "mean_obj_height": mean_obj_height,
            "max_obj_height": max_obj_height,
            "mean_smoothness": mean_smoothness,
            "mean_action_delta": mean_action_delta,
            "mean_action_jerk": mean_action_jerk,
            "mean_path_length": mean_path_length,
            "path_metric": path_metric,
            "avg_success_replans": avg_success_replans,
            "avg_success_inference_latency_ms": avg_success_inference_latency_ms,
            "avg_success_inference_hz": avg_success_inference_hz,
            "avg_inference_latency_ms": avg_inference_latency_ms,
            "avg_inference_hz": avg_inference_hz,
            "replanning_mode": "chunk-wise",
            "replanning_frequency_steps": args_cli.execute_horizon,
            "success_dist": actual_success_dist,
            "success_z": actual_success_z,
        }

        if args_cli.save_path is not None or not args_cli.no_auto_report:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if action_dim is None:
                    raise RuntimeError("Action dimension is unavailable for FK conversion.")
                if len(predicted_ee_chunks) != len(predicted_joint_chunks):
                    predicted_ee_chunks = _compute_predicted_ee_chunks(predicted_joint_chunks, action_dim)
            except Exception as exc:
                print(f"[WARN] Failed to compute FK point paths from predicted joints. reason={exc}")
                predicted_ee_chunks = [np.zeros((*chunk.shape[:2], 3), dtype=np.float32) for chunk in predicted_joint_chunks]
            np.savez(
                save_path,
                predicted_joint_chunks=np.array(predicted_joint_chunks, dtype=object),
                predicted_ee_chunks=np.array(predicted_ee_chunks, dtype=object),
                executed_ee_positions=np.stack(executed_ee_positions) if executed_ee_positions else np.zeros((0,)),
                path_positions=np.stack(path_positions) if path_positions else np.zeros((0,)),
                completed_path_lengths=np.asarray(completed_path_lengths, dtype=np.float64),
                output_horizon=args_cli.output_horizon,
                execute_horizon=args_cli.execute_horizon,
                task=args_cli.task,
                policy_name=policy_name,
                success_count=success_count,
                episode_done_count=episode_done_count,
                success_done_count=success_done_count,
                total_envs=total_envs,
                success_rate=success_rate,
                done_success_rate=done_success_rate,
                mean_ee_dist=mean_ee_dist,
                mean_obj_height=mean_obj_height,
                max_obj_height=max_obj_height,
                mean_smoothness=mean_smoothness,
                mean_action_delta=mean_action_delta,
                mean_action_jerk=mean_action_jerk,
                mean_path_length=mean_path_length,
                path_metric=path_metric,
                avg_success_replans=avg_success_replans,
                avg_success_inference_latency_ms=avg_success_inference_latency_ms,
                avg_success_inference_hz=avg_success_inference_hz,
                avg_inference_latency_ms=avg_inference_latency_ms,
                avg_inference_hz=avg_inference_hz,
                replanning_mode="chunk-wise",
                replanning_frequency_steps=args_cli.execute_horizon,
                success_dist=actual_success_dist,
                success_z=actual_success_z,
                seed=args_cli.seed,
            )
            print(f"[INFO] Saved trajectories to: {save_path}")
            if not args_cli.no_auto_report:
                try:
                    _append_manual_eval_reports(save_path, metrics)
                    print(f"[INFO] Appended manual eval reports to: {Path.cwd() / 'ppo_test_comparison.md'}")
                    print(f"[INFO] Appended manual eval reports to: {Path.cwd() / 'matched_policy_sweep_comparison.md'}")
                except Exception as exc:
                    print(f"[WARN] Failed to append manual eval reports. reason={exc}")
        print(
            f"[METRIC] episodes_done={episode_done_count} success_done={success_done_count}"
        )
        success_z_log = f"{actual_success_z:.4f}" if np.isfinite(actual_success_z) else "-"
        print(
            f"[METRIC] done_success_rate={done_success_rate:.4f} "
            f"success_dist={actual_success_dist:.4f} "
            f"success_z={success_z_log}"
        )
        print(
            f"[METRIC] success={success_count}/{total_envs} "
            f"success_rate={success_rate:.4f} mean_ee_dist={mean_ee_dist:.4f} "
            f"mean_obj_height={mean_obj_height:.4f} max_obj_height={max_obj_height:.4f} "
            f"mean_smoothness={mean_smoothness:.6f}"
        )
        print(
            f"[METRIC] mean_action_delta={mean_action_delta:.6f} "
            f"mean_action_jerk={mean_action_jerk:.6f} "
            f"mean_path_length={mean_path_length:.4f} path_metric={path_metric}"
        )
        print(
            f"[METRIC] replanning_mode=chunk-wise "
            f"replanning_frequency_steps={args_cli.execute_horizon} "
            f"avg_inference={avg_inference_latency_ms:.2f}ms ({avg_inference_hz:.2f} Hz)"
        )
        # print(
        #     f"[METRIC] successful_episodes_avg_replans={avg_success_replans:.2f} "
        #     f"successful_episodes_avg_inference={avg_success_inference_latency_ms:.2f}ms ({avg_success_inference_hz:.2f} Hz) "
        #     f"global_avg_inference={avg_inference_latency_ms:.2f}ms ({avg_inference_hz:.2f} Hz)"
        # )


if __name__ == "__main__":
    main()
    simulation_app.close()
