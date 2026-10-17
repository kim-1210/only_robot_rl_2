"""
isaaclab -p ./launch/get_data_task.py --task reach_lift \
--checkpoint ./logs/rsl_rl/rb5_850_reach_lift_task/2026-03-28_15-38-57/model_26600.pt \
--save_name reach_lift_data1 --num_envs 8000  \
--max_episode 150000
"""

import argparse
import sys
import os
import numpy as np

from isaaclab.app import AppLauncher

import cli_args 

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
parser.add_argument("--max_episode", type=int, default=1000, help="save data nums")
parser.add_argument("--save_path", type=str, default="./datas", help="save data path")
parser.add_argument("--save_name", type=str, default="data1", help="save data name")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import time
import torch

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

import task.reach as reach
# import FrankaPickPlace.tasks  # noqa: F401


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

    dataset_dir = os.path.join(args_cli.save_path, args_cli.save_name)
    os.makedirs(args_cli.save_path, exist_ok=True)
    if os.path.exists(dataset_dir):
        logging.error("이미 데이터 이름이 존재합니다.")
        return
    os.makedirs(dataset_dir, exist_ok=False)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        print("[ERROR] get_published_pretrained_checkpoint 모듈을 찾을 수 없습니다.")
        return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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
    # obs["policy"] = (num_envs, obs_num)
    obs = env.get_observations()
    num_envs = env.unwrapped.num_envs
    
    # save obs shpae: [num, step, obs]
    # save action shpae: [num, step, actions]
    current_obs = [[] for _ in range(num_envs)]
    current_actions = [[] for _ in range(num_envs)]

    timestep = 0
    episode_count = 0
    episode_id = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            obs_t = obs["policy"]
            # agent stepping
            actions = policy(obs)
            real_actions = env.unwrapped.scene["robot"].data.joint_pos[..., :]
            # env stepping
            next_obs, _, dones, _ = env.step(actions)

            for i in range(num_envs):
                current_obs[i].append(obs_t[i].cpu().numpy())
                current_actions[i].append(real_actions[i].cpu().numpy())

                if dones[i].item():
                    episode_id += 1
                    np.save(
                        os.path.join(dataset_dir, f"obs_ep_{episode_id:06d}.npy"),
                        np.stack(current_obs[i]),
                    )
                    np.save(
                        os.path.join(dataset_dir, f"act_ep_{episode_id:06d}.npy"),
                        np.stack(current_actions[i]),
                    )
                    current_obs[i] = []
                    current_actions[i] = []

            episode_count += dones.sum().item()
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
            obs = next_obs

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if episode_count >= args_cli.max_episode:
            break

    np.savez(
        os.path.join(dataset_dir, "meta.npz"),
        num_episodes=episode_id,
        num_envs=num_envs,
        task=args_cli.task,
        checkpoint=resume_path,
    )


    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
