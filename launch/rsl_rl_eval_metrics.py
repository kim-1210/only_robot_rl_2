# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate an RSL-RL PPO checkpoint and write simple episode metrics."""

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate an RSL-RL PPO checkpoint with metrics.")
parser.add_argument("--num_envs", type=int, default=100, help="Number of parallel environments.")
parser.add_argument("--episodes", type=int, default=200, help="Number of completed episodes to count.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent cfg entry point.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--episode_length_s", type=float, default=None, help="Override episode length.")
parser.add_argument("--success_height", type=float, default=None, help="Override reach_lift success height.")
parser.add_argument("--success_dist", type=float, default=None, help="Override reach success distance.")
parser.add_argument("--save_path", type=str, default=None, help="Where to save the metric npz.")
parser.add_argument(
    "--metric_chunk_steps",
    type=int,
    default=1,
    help="Compute PPO action delta/jerk every N env steps for metric-only chunk-wise reporting. PPO execution stays step-wise.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real time.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--skip_finger_material", action="store_true", default=False, help="Skip rubber finger material binding.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import inspect
import os
import time
from importlib.metadata import PackageNotFoundError, version

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import combine_frame_transforms
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from rsl_rl.algorithms import PPO
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import task  # noqa: F401


def _installed_rsl_rl_version() -> str:
    try:
        return version("rsl-rl-lib")
    except PackageNotFoundError:
        return "0.0.0"


def _nearest_checkpoint_in_dir(directory: Path, requested_name: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob("model_*.pt"))
    if not matches:
        return None
    try:
        requested_step = int(requested_name.removeprefix("model_").removesuffix(".pt"))
    except ValueError:
        return matches[-1]
    valid = [path for path in matches if path.stem.removeprefix("model_").isdigit()]
    if not valid:
        return matches[-1]
    return min(valid, key=lambda path: (abs(int(path.stem.removeprefix("model_")) - requested_step), -int(path.stem.removeprefix("model_"))))


def _resolve_checkpoint_path(checkpoint_arg: str | None, log_root_path: str, load_run: str | None, load_checkpoint: str | None) -> str:
    if not checkpoint_arg:
        return get_checkpoint_path(log_root_path, load_run, load_checkpoint)
    requested = Path(checkpoint_arg).expanduser()
    if requested.exists():
        return str(retrieve_file_path(str(requested)))
    candidate_dirs = [requested.parent]
    parts = requested.parts
    if "logs" in parts:
        logs_idx = parts.index("logs")
        candidate_dirs.append(Path.cwd().joinpath(*parts[logs_idx:]).parent)
    for candidate_dir in candidate_dirs:
        nearest = _nearest_checkpoint_in_dir(candidate_dir, requested.name)
        if nearest is not None:
            print(f"[WARN] Requested checkpoint not found: {requested}")
            print(f"[WARN] Falling back to nearest checkpoint: {nearest}")
            return str(nearest.resolve())
    raise FileNotFoundError(f"Checkpoint not found: {requested}")


def _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg: RslRlBaseRunnerCfg) -> dict:
    runner_cfg = agent_cfg.to_dict()
    algorithm_cfg = runner_cfg.get("algorithm", {})
    if runner_cfg.get("class_name") != "OnPolicyRunner" or algorithm_cfg.get("class_name") != "PPO":
        return runner_cfg
    ppo_signature = inspect.signature(PPO.__init__)
    allowed_keys = set(ppo_signature.parameters) - {"self", "policy", "device", "multi_gpu_cfg"}
    runner_cfg["algorithm"] = {
        key: value
        for key, value in algorithm_cfg.items()
        if key == "class_name" or key in allowed_keys
    }
    return runner_cfg


def _get_scene_item(base_env, *names):
    for name in names:
        try:
            return base_env.scene[name]
        except Exception:
            pass
    return None


def _as_bool_tensor(value, reference: torch.Tensor) -> torch.Tensor:
    if value is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=reference.device)
    return value.to(device=reference.device, dtype=torch.bool)


def _termination_term(base_env, name: str, reference: torch.Tensor) -> torch.Tensor:
    manager = getattr(base_env, "termination_manager", None)
    if manager is None:
        return torch.zeros_like(reference, dtype=torch.bool)
    try:
        return manager.get_term(name).to(device=reference.device, dtype=torch.bool)
    except Exception:
        return torch.zeros_like(reference, dtype=torch.bool)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    rsl_rl_version = _installed_rsl_rl_version()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, rsl_rl_version)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    if args_cli.episode_length_s is not None:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if args_cli.skip_finger_material:
        try:
            import robot.finger_material as finger_material
            import task.reach_lift.cfg_settings as reach_lift_settings

            finger_material.apply_rubber_finger_material = lambda: None
            reach_lift_settings.apply_rubber_finger_material = lambda: None
            print("[INFO] Skipping rubber finger material binding for evaluation.")
        except Exception as exc:
            print(f"[WARN] Could not disable rubber finger material binding: {exc}")

    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is not None and hasattr(terminations_cfg, "success"):
        success_term = getattr(terminations_cfg, "success")
        params = getattr(success_term, "params", None)
        if isinstance(params, dict):
            if args_cli.success_height is not None and "minimal_height" in params:
                params["minimal_height"] = float(args_cli.success_height)
            if args_cli.success_dist is not None and "goal_threshold" in params:
                params["goal_threshold"] = float(args_cli.success_dist)

    task_name = args_cli.task.split(":")[-1]
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = _resolve_checkpoint_path(args_cli.checkpoint, log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, rsl_rl_version)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cfg = _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", None))

    obs = env.get_observations()
    base_env = env.unwrapped
    device = base_env.device
    dt = base_env.step_dt
    num_envs = int(args_cli.num_envs)
    robot = _get_scene_item(base_env, "robot")
    obj = _get_scene_item(base_env, "obj", "object", "cube")
    ee_body_idx = None
    if robot is not None:
        try:
            ee_body_idx = robot.find_bodies("ee_link")[0][0]
        except Exception:
            ee_body_idx = None

    is_reach_lift = "reach_lift" in task_name
    command_name = "obj_pose" if is_reach_lift else "ee_pose"
    path_metric = "object" if is_reach_lift and obj is not None else "end_effector"
    success_height = float(args_cli.success_height if args_cli.success_height is not None else 1.1)
    success_dist = float(args_cli.success_dist if args_cli.success_dist is not None else (0.005 if is_reach_lift else 0.02))
    metric_chunk_steps = max(1, int(args_cli.metric_chunk_steps))
    metric_replanning = (
        f"metric-only chunk-wise, {metric_chunk_steps} step"
        if metric_chunk_steps > 1
        else "step-wise, 1 step"
    )

    episode_done_count = 0
    success_done_count = 0
    success_mask = torch.zeros(num_envs, device=device, dtype=torch.bool)
    previous_path_pos = None
    current_path_length = torch.zeros(num_envs, device=device, dtype=torch.float64)
    completed_path_lengths = []
    last_step_action = None
    last_step_delta = None
    last_step_action_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
    last_step_delta_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
    last_metric_action = None
    last_metric_delta = None
    last_metric_action_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
    last_metric_delta_valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
    metric_steps_since_sample = torch.zeros(num_envs, device=device, dtype=torch.long)
    step_action_delta_sum = 0.0
    step_action_delta_count = 0
    step_action_jerk_sum = 0.0
    step_action_jerk_count = 0
    action_delta_sum = 0.0
    action_delta_count = 0
    action_jerk_sum = 0.0
    action_jerk_count = 0
    inference_time = 0.0
    inference_calls = 0
    ee_dist_sum = 0.0
    obj_height_sum = 0.0
    metric_steps = 0
    max_obj_height = float("nan")
    global_step = 0

    try:
        while simulation_app.is_running() and episode_done_count < args_cli.episodes:
            start = time.time()
            with torch.no_grad():
                infer_start = time.perf_counter()
                actions = policy(obs)
                if torch.cuda.is_available() and "cuda" in str(device):
                    torch.cuda.synchronize(device=device)
                inference_time += time.perf_counter() - infer_start
                inference_calls += 1

                if last_step_action is None:
                    last_step_action = actions.detach().clone()
                    last_step_delta = torch.zeros_like(actions)
                    last_metric_action = actions.detach().clone()
                    last_metric_delta = torch.zeros_like(actions)

                step_delta_mask = last_step_action_valid
                if torch.any(step_delta_mask):
                    step_delta = actions[step_delta_mask] - last_step_action[step_delta_mask]
                    step_delta_norm = torch.linalg.norm(step_delta, dim=-1)
                    step_action_delta_sum += float(step_delta_norm.sum().item())
                    step_action_delta_count += int(step_delta_norm.numel())
                    step_jerk_mask = step_delta_mask & last_step_delta_valid
                    if torch.any(step_jerk_mask):
                        step_jerk = (
                            actions[step_jerk_mask]
                            - last_step_action[step_jerk_mask]
                            - last_step_delta[step_jerk_mask]
                        )
                        step_jerk_norm = torch.linalg.norm(step_jerk, dim=-1)
                        step_action_jerk_sum += float(step_jerk_norm.sum().item())
                        step_action_jerk_count += int(step_jerk_norm.numel())
                    last_step_delta = last_step_delta.clone()
                    last_step_delta[step_delta_mask] = actions[step_delta_mask] - last_step_action[step_delta_mask]
                    last_step_delta_valid |= step_delta_mask
                last_step_action = actions.detach().clone()
                last_step_action_valid[:] = True

                metric_sample_mask = (~last_metric_action_valid) | (metric_steps_since_sample >= metric_chunk_steps)
                metric_delta_mask = metric_sample_mask & last_metric_action_valid
                if torch.any(metric_delta_mask):
                    metric_delta = actions[metric_delta_mask] - last_metric_action[metric_delta_mask]
                    metric_delta_norm = torch.linalg.norm(metric_delta, dim=-1)
                    action_delta_sum += float(metric_delta_norm.sum().item())
                    action_delta_count += int(metric_delta_norm.numel())
                    metric_jerk_mask = metric_delta_mask & last_metric_delta_valid
                    if torch.any(metric_jerk_mask):
                        metric_jerk = (
                            actions[metric_jerk_mask]
                            - last_metric_action[metric_jerk_mask]
                            - last_metric_delta[metric_jerk_mask]
                        )
                        metric_jerk_norm = torch.linalg.norm(metric_jerk, dim=-1)
                        action_jerk_sum += float(metric_jerk_norm.sum().item())
                        action_jerk_count += int(metric_jerk_norm.numel())
                    last_metric_delta = last_metric_delta.clone()
                    last_metric_delta[metric_delta_mask] = actions[metric_delta_mask] - last_metric_action[metric_delta_mask]
                    last_metric_delta_valid |= metric_delta_mask
                if torch.any(metric_sample_mask):
                    last_metric_action = last_metric_action.clone()
                    last_metric_action[metric_sample_mask] = actions[metric_sample_mask].detach()
                    last_metric_action_valid |= metric_sample_mask
                    metric_steps_since_sample = metric_steps_since_sample.clone()
                    metric_steps_since_sample[metric_sample_mask] = 0

                obs, _, dones, infos = env.step(actions)

            env_success_step = _termination_term(base_env, "success", success_mask)
            success_mask |= env_success_step

            if robot is not None and ee_body_idx is not None:
                ee_pos = robot.data.body_pos_w[:, ee_body_idx, :]
            else:
                ee_pos = None
            obj_pos = obj.data.root_pos_w[:, :3] if obj is not None else None

            current_path_pos = obj_pos if path_metric == "object" and obj_pos is not None else ee_pos
            if current_path_pos is not None:
                if previous_path_pos is not None:
                    current_path_length += torch.linalg.norm(current_path_pos - previous_path_pos, dim=-1).to(torch.float64)
                previous_path_pos = current_path_pos.detach().clone()

            if robot is not None:
                command = base_env.command_manager.get_command(command_name)
                des_pos_b = command[:, :3]
                des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
                if is_reach_lift and obj_pos is not None:
                    metric_dist = torch.norm(des_pos_w - obj_pos, dim=-1)
                    obj_height = obj.data.root_pos_w[:, 2]
                    success_step = (obj_height > success_height) & (metric_dist < success_dist)
                    obj_height_sum += float(obj_height.mean().item())
                    max_obj_height = float(torch.max(obj_height).item()) if np.isnan(max_obj_height) else max(max_obj_height, float(torch.max(obj_height).item()))
                elif ee_pos is not None:
                    metric_dist = torch.norm(des_pos_w - ee_pos, dim=-1)
                    success_step = metric_dist < success_dist
                else:
                    metric_dist = None
                    success_step = torch.zeros(num_envs, device=device, dtype=torch.bool)
                if metric_dist is not None:
                    ee_dist_sum += float(metric_dist.mean().item())
                    metric_steps += 1
                success_mask |= success_step

            done_mask = _as_bool_tensor(dones, success_mask)
            if torch.any(done_mask):
                remaining = max(int(args_cli.episodes) - episode_done_count, 0)
                done_indices = torch.nonzero(done_mask, as_tuple=False).flatten()
                counted_indices = done_indices[:remaining]
                counted_done_mask = torch.zeros_like(done_mask)
                if counted_indices.numel() > 0:
                    counted_done_mask[counted_indices] = True
                    success_done_count += int(success_mask[counted_indices].sum().item())
                    completed_path_lengths.extend(current_path_length[counted_indices].detach().cpu().tolist())
                episode_done_count += int(counted_done_mask.sum().item())
                print(
                    f"[INFO] step={global_step} done={int(done_mask.sum().item())} "
                    f"counted_done={int(counted_done_mask.sum().item())} "
                    f"success_done_total={success_done_count}/{episode_done_count}"
                )
                success_mask[done_mask] = False
                current_path_length[done_mask] = 0.0
                last_step_action_valid[done_mask] = False
                last_step_delta_valid[done_mask] = False
                last_metric_action_valid[done_mask] = False
                last_metric_delta_valid[done_mask] = False
                metric_steps_since_sample[done_mask] = 0
                if last_step_delta is not None:
                    last_step_delta = last_step_delta.clone()
                    last_step_delta[done_mask] = 0.0
                if last_metric_delta is not None:
                    last_metric_delta = last_metric_delta.clone()
                    last_metric_delta[done_mask] = 0.0
                if previous_path_pos is not None:
                    previous_path_pos = previous_path_pos.clone()
                    previous_path_pos[done_mask] = current_path_pos[done_mask]
                if policy_nn is not None and hasattr(policy_nn, "reset"):
                    policy_nn.reset(done_mask)

            metric_steps_since_sample[last_metric_action_valid] += 1
            global_step += 1
            sleep_time = dt - (time.time() - start)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()

    success_rate = success_done_count / episode_done_count if episode_done_count else float("nan")
    mean_action_delta = action_delta_sum / action_delta_count if action_delta_count else float("nan")
    mean_action_jerk = action_jerk_sum / action_jerk_count if action_jerk_count else float("nan")
    mean_step_action_delta = step_action_delta_sum / step_action_delta_count if step_action_delta_count else float("nan")
    mean_step_action_jerk = step_action_jerk_sum / step_action_jerk_count if step_action_jerk_count else float("nan")
    mean_path_length = float(np.mean(completed_path_lengths)) if completed_path_lengths else float("nan")
    mean_metric_dist = ee_dist_sum / metric_steps if metric_steps else float("nan")
    mean_obj_height = obj_height_sum / metric_steps if metric_steps else float("nan")
    avg_latency_ms = 1000.0 * inference_time / inference_calls if inference_calls else float("nan")
    avg_hz = inference_calls / inference_time if inference_time > 0.0 else float("nan")

    metrics = {
        "task": task_name,
        "checkpoint": resume_path,
        "num_envs": num_envs,
        "episodes_requested": int(args_cli.episodes),
        "episode_done_count": episode_done_count,
        "success_done_count": success_done_count,
        "success_rate": success_rate,
        "success_dist": success_dist,
        "success_height": success_height if is_reach_lift else float("nan"),
        "mean_metric_dist": mean_metric_dist,
        "mean_obj_height": mean_obj_height,
        "max_obj_height": max_obj_height,
        "mean_action_delta": mean_action_delta,
        "mean_action_jerk": mean_action_jerk,
        "mean_step_action_delta": mean_step_action_delta,
        "mean_step_action_jerk": mean_step_action_jerk,
        "metric_chunk_steps": metric_chunk_steps,
        "mean_path_length": mean_path_length,
        "path_metric": path_metric,
        "avg_inference_latency_ms": avg_latency_ms,
        "avg_inference_hz": avg_hz,
        "replanning": metric_replanning,
        "replanning_mode": "metric-only chunk-wise" if metric_chunk_steps > 1 else "step-wise",
        "replanning_frequency_steps": metric_chunk_steps,
        "seed": -1 if args_cli.seed is None else int(args_cli.seed),
        "episode_length_s": float(getattr(base_env.cfg, "episode_length_s", float("nan"))),
    }
    save_path = Path(args_cli.save_path) if args_cli.save_path else Path("datas/ppo_eval") / f"{task_name}_ppo_env{num_envs}_ep{args_cli.episodes}_seed{metrics['seed']}.npz"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, **metrics)
    print(f"[INFO] Saved metrics to: {save_path}")
    print(f"[METRIC] episodes_done={episode_done_count} success_done={success_done_count}")
    print(f"[METRIC] success_rate={success_rate:.4f} mean_metric_dist={mean_metric_dist:.4f} mean_obj_height={mean_obj_height:.4f} max_obj_height={max_obj_height:.4f}")
    print(
        f"[METRIC] mean_action_delta={mean_action_delta:.6f} "
        f"mean_action_jerk={mean_action_jerk:.6f} "
        f"mean_path_length={mean_path_length:.4f} path_metric={path_metric}"
    )
    print(
        f"[METRIC] mean_step_action_delta={mean_step_action_delta:.6f} "
        f"mean_step_action_jerk={mean_step_action_jerk:.6f}"
    )
    print(f"[METRIC] avg_inference={avg_latency_ms:.2f}ms ({avg_hz:.2f} Hz) replanning={metric_replanning}")


if __name__ == "__main__":
    main()
    simulation_app.close()
