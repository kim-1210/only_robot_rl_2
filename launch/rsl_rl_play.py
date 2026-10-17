# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# PYTHONPATH=$PWD isaaclab -p launch/play.py --task reach --checkpoint /home/kim/Desktop/robot_arm_rl/logs/rsl_rl/franka_reack_task/2026-02-08_10-48-23/model_9999.pt --num_envs 1

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import time
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import logging
logger = logging.getLogger(__name__)

import task
# import FrankaPickPlace.tasks  # noqa: F401


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

    def step_of(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem.removeprefix("model_")), path.name
        except ValueError:
            return 10**18, path.name

    valid = [path for path in matches if path.stem.removeprefix("model_").isdigit()]
    if not valid:
        return matches[-1]
    return min(valid, key=lambda path: (abs(step_of(path)[0] - requested_step), -step_of(path)[0]))


def _resolve_checkpoint_path(checkpoint_arg: str | None, log_root_path: str, load_run: str | None, load_checkpoint: str | None) -> str:
    if not checkpoint_arg:
        return get_checkpoint_path(log_root_path, load_run, load_checkpoint)

    requested = Path(checkpoint_arg).expanduser()
    if requested.exists():
        return str(retrieve_file_path(str(requested)))

    candidate_dirs: list[Path] = []
    if requested.parent != Path(""):
        candidate_dirs.append(requested.parent)

    # Mirror old absolute paths into the current workspace when the run exists locally.
    parts = requested.parts
    if "logs" in parts:
        logs_idx = parts.index("logs")
        mirrored = Path.cwd().joinpath(*parts[logs_idx:])
        candidate_dirs.append(mirrored.parent)

    for candidate_dir in candidate_dirs:
        nearest = _nearest_checkpoint_in_dir(candidate_dir, requested.name)
        if nearest is not None:
            print(
                f"[WARN] Requested checkpoint was not found: {requested}\n"
                f"[WARN] Falling back to nearest available checkpoint: {nearest}"
            )
            return str(nearest.resolve())

    raise FileNotFoundError(f"Checkpoint not found: {requested}")


def _map_legacy_submodule_state_dict(
    legacy_state_dict: dict[str, torch.Tensor],
    *,
    normalizer_prefix: str,
    module_prefix: str,
    std_key: str | None = None,
) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in legacy_state_dict.items():
        if key.startswith("obs_normalizer."):
            mapped[f"{normalizer_prefix}{key.removeprefix('obs_normalizer.')}"] = value
        elif key.startswith("mlp."):
            mapped[f"{module_prefix}{key.removeprefix('mlp.')}"] = value
        elif key == "distribution.std_param":
            if std_key is None:
                raise KeyError("Legacy actor checkpoint contains distribution.std_param but current policy has no std key.")
            mapped[std_key] = value
        else:
            raise KeyError(f"Unsupported legacy checkpoint key: {key}")
    return mapped


def _load_legacy_checkpoint(runner: OnPolicyRunner | DistillationRunner, resume_path: str, device: str) -> dict | None:
    checkpoint = torch.load(resume_path, weights_only=False, map_location=device)

    if "model_state_dict" in checkpoint:
        runner.load(resume_path)
        return checkpoint.get("infos")

    if "actor_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint must contain either 'model_state_dict' or legacy 'actor_state_dict'/'critic_state_dict'."
        )

    policy = runner.alg.policy
    current_state_dict = policy.state_dict()

    if "std" in current_state_dict:
        std_key = "std"
    elif "log_std" in current_state_dict:
        std_key = "log_std"
    else:
        std_key = None

    remapped_state_dict: dict[str, torch.Tensor] = {}
    remapped_state_dict.update(
        _map_legacy_submodule_state_dict(
            checkpoint["actor_state_dict"],
            normalizer_prefix="actor_obs_normalizer.",
            module_prefix="actor.",
            std_key=std_key,
        )
    )

    if "critic_state_dict" in checkpoint:
        remapped_state_dict.update(
            _map_legacy_submodule_state_dict(
                checkpoint["critic_state_dict"],
                normalizer_prefix="critic_obs_normalizer.",
                module_prefix="critic.",
            )
        )

    merged_state_dict = dict(current_state_dict)
    merged_state_dict.update(remapped_state_dict)
    policy.load_state_dict(merged_state_dict, strict=True)

    if "iter" in checkpoint:
        runner.current_learning_iteration = checkpoint["iter"]

    return checkpoint.get("infos")


def _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg: RslRlBaseRunnerCfg) -> dict:
    """Drop unsupported PPO kwargs for the installed rsl-rl version."""
    runner_cfg = agent_cfg.to_dict()
    algorithm_cfg = runner_cfg.get("algorithm", {})
    if runner_cfg.get("class_name") != "OnPolicyRunner" or algorithm_cfg.get("class_name") != "PPO":
        return runner_cfg

    ppo_signature = inspect.signature(PPO.__init__)
    allowed_keys = set(ppo_signature.parameters) - {"self", "policy", "device", "multi_gpu_cfg"}
    filtered_algorithm_cfg = {
        key: value
        for key, value in algorithm_cfg.items()
        if key == "class_name" or key in allowed_keys
    }
    dropped_keys = sorted(set(algorithm_cfg) - set(filtered_algorithm_cfg))
    if dropped_keys:
        print(f"[INFO] Dropped unsupported PPO args for installed rsl-rl: {dropped_keys}")

    runner_cfg["algorithm"] = filtered_algorithm_cfg
    return runner_cfg


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        print("[ERROR] get_published_pretrained_checkpoint 모듈을 찾을 수 없습니다.")
        return
    else:
        resume_path = _resolve_checkpoint_path(
            args_cli.checkpoint,
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cfg = _sanitize_runner_cfg_for_installed_rsl_rl(agent_cfg)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    _load_legacy_checkpoint(runner, resume_path, agent_cfg.device)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
