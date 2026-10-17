import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check whether the gripper mimic joint is working.")
parser.add_argument("--task", type=str, default="reach_lift", help="Task name to launch.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--settle_steps", type=int, default=10, help="Initial zero-action settle steps.")
parser.add_argument("--close_steps", type=int, default=25, help="Steps to command gripper close.")
parser.add_argument("--open_steps", type=int, default=25, help="Steps to command gripper open.")
parser.add_argument("--print_every", type=int, default=5, help="Print interval in steps.")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import task  # noqa: F401


@hydra_task_config(args_cli.task, None)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg=None):
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.reset()
        robot = env.unwrapped.scene["robot"]
        names = robot.data.joint_names
        left_idx = names.index("robotiq_hande_left_finger_joint")
        right_idx = names.index("robotiq_hande_right_finger_joint")

        action_dim = env.unwrapped.action_space.shape[-1]
        print(f"[MIMIC] task={args_cli.task} action_dim={action_dim}")
        print(f"[MIMIC] gripper action index assumption={action_dim - 1}")
        print(
            "[MIMIC] joint indices "
            f"left={left_idx}({names[left_idx]}), right={right_idx}({names[right_idx]})"
        )

        def log_state(tag: str, step: int):
            left = robot.data.joint_pos[0, left_idx].item()
            right = robot.data.joint_pos[0, right_idx].item()
            print(f"[MIMIC] {tag} step={step:03d} left={left:.6f} right={right:.6f} diff={abs(left-right):.6f}")

        zero_action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        close_action = zero_action.clone()
        open_action = zero_action.clone()

        # Binary gripper action is appended after the arm action in this task.
        close_action[:, -1] = -1.0
        open_action[:, -1] = 1.0

        step = 0
        log_state("reset", step)

        for _ in range(args_cli.settle_steps):
            env.step(zero_action)
            step += 1
            if step % args_cli.print_every == 0:
                log_state("settle", step)

        for _ in range(args_cli.close_steps):
            env.step(close_action)
            step += 1
            if step % args_cli.print_every == 0:
                log_state("close", step)

        for _ in range(args_cli.open_steps):
            env.step(open_action)
            step += 1
            if step % args_cli.print_every == 0:
                log_state("open", step)

        log_state("final", step)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
