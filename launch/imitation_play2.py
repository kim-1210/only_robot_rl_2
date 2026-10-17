"""
N=16 (4,6,10,14)

TERM=xterm PYTHONPATH=$PWD isaaclab -p ./launch/imitation_play2.py \
  --task reach_lift_imitation \
  --policy_type python \
  --policy_path ./policy_models/diffusion/diffusion_policy_wrapper.py \
  --policy_checkpoint ./policy_models/diffusion/reach/reach_lift_diffusion_unet_artifacts2/reach_lift_diffusion_unet.pt \
  --policy_stats ./policy_models/diffusion/reach/reach_lift_diffusion_unet_artifacts2/reach_lift_diffusion_stats.npz \
  --num_envs 20 \
  --output_horizon 16 \
  --execute_horizon 14 \
  --steps 500 \
  --seed 42 \
  --save_path ./datas/reach_lift_imitation_rollout_k8_gpu_first.npz

N=8 (2,4,6)
"""

import imitation_play_fixed as base


@base.hydra_task_config(base.args_cli.task, None)
def main(env_cfg: base.DirectRLEnvCfg, agent_cfg=None):
    if base.args_cli.output_horizon < 1:
        raise ValueError("--output_horizon must be >= 1")
    if base.args_cli.execute_horizon < 1:
        raise ValueError("--execute_horizon must be >= 1")
    if base.args_cli.execute_horizon > base.args_cli.output_horizon:
        raise ValueError("--execute_horizon must be <= --output_horizon")

    env_cfg.scene.num_envs = base.args_cli.num_envs
    env = base.gym.make(
        base.args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if base.args_cli.video else None,
    )
    if base.args_cli.seed is not None:
        base.np.random.seed(base.args_cli.seed)
        base.torch.manual_seed(base.args_cli.seed)

    if base.args_cli.video:
        video_kwargs = {
            "video_folder": base.os.path.join(base.os.getcwd(), "videos", base.args_cli.task),
            "step_trigger": lambda step: step == 0,
            "video_length": base.args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording imitation play video.")
        base.print_dict(video_kwargs, nesting=4)
        env = base.gym.wrappers.RecordVideo(env, **video_kwargs)

    predicted_joint_chunks_gpu = []
    executed_ee_positions_gpu = []
    success_mask = None
    ee_dist_sum = 0.0
    obj_height_sum = 0.0
    metric_steps = 0

    try:
        obs, _ = env.reset(seed=base.args_cli.seed)
        device = env.unwrapped.device
        action_dim = env.unwrapped.action_space.shape[-1]
        policy = base._load_policy(base.args_cli.policy_path, base.args_cli.policy_type, device)
        dt = env.unwrapped.step_dt
        robot = env.unwrapped.scene["robot"]
        try:
            obj = env.unwrapped.scene["obj"]
        except Exception:
            obj = None
        ee_body_idx = robot.find_bodies("ee_link")[0][0]

        step_count = 0
        chunk_id = 0
        cached_actions = None
        cached_index = 0
        executed_since_replan = 0
        action_hold_steps = 0
        replan_interval = base.args_cli.execute_horizon
        initial_last_action = base._initial_last_action_tensor(env, robot, action_dim)
        last_action_tensor = initial_last_action.clone()

        print(
            f"[INFO] GPU-first policy rollout configured with output_horizon={base.args_cli.output_horizon}, "
            f"execute_horizon={base.args_cli.execute_horizon}. FK visualization is disabled and tensors stay "
            f"on device until save time. Each point is held until joint error <= "
            f"{base.args_cli.joint_arrival_tolerance} or {base.args_cli.max_action_hold_steps} env steps elapse."
        )

        while base.simulation_app.is_running():
            start_time = base.time.time()
            obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
            obs_tensor = base._prepare_policy_observation(policy, obs_tensor, last_action_tensor)

            if cached_actions is None or executed_since_replan >= min(replan_interval, cached_actions.shape[1]):
                cached_actions, _raw_policy_actions = base._run_policy(
                    policy,
                    obs_tensor,
                    action_dim,
                    device,
                    reference_action=last_action_tensor,
                )
                cached_index = 0
                executed_since_replan = 0
                action_hold_steps = 0
                chunk_id += 1
                print(
                    f"[INFO] chunk_id={chunk_id} generated at global_step={step_count} "
                    f"with execute_horizon={replan_interval}"
                )
                if base.args_cli.save_path is not None:
                    predicted_joint_chunks_gpu.append(cached_actions.detach().clone())

            actions = cached_actions[:, cached_index, :]
            obs, _, terminated, truncated, _ = env.step(actions)
            action_hold_steps += 1
            last_action_tensor = actions.detach()

            actuated_joint_ids = getattr(env.unwrapped, "actuated_joint_ids", None)
            current_joint_pos = (
                robot.data.joint_pos[:, actuated_joint_ids]
                if actuated_joint_ids is not None
                else robot.data.joint_pos[:, : actions.shape[-1]]
            )
            joint_error = base.torch.max(base.torch.abs(current_joint_pos - actions), dim=-1).values
            reached_action = joint_error <= base.args_cli.joint_arrival_tolerance
            advance_action = bool(base.torch.all(reached_action).item()) or action_hold_steps >= base.args_cli.max_action_hold_steps
            if advance_action:
                cached_index += 1
                executed_since_replan += 1
                action_hold_steps = 0

            if base.args_cli.save_path is not None:
                ee_pos = robot.data.body_pos_w[:, ee_body_idx, :].detach().clone()
                executed_ee_positions_gpu.append(ee_pos)
            else:
                ee_pos = robot.data.body_pos_w[:, ee_body_idx, :]

            if obj is not None and hasattr(env.unwrapped, "pose_command"):
                target_pos = env.unwrapped.pose_command[:, :3] + env.unwrapped.scene.env_origins
                ee_dist = base.torch.norm(ee_pos - target_pos, dim=-1)
                obj_height = obj.data.root_pos_w[:, 2]

                ee_dist_sum += float(ee_dist.mean().item())
                obj_height_sum += float(obj_height.mean().item())
                metric_steps += 1

                if success_mask is None:
                    success_mask = base.torch.zeros_like(ee_dist, dtype=base.torch.bool)
                success_mask |= (ee_dist <= base.args_cli.success_dist) & (obj_height >= base.args_cli.success_z)

            step_count += 1
            if base.args_cli.steps > 0 and step_count >= base.args_cli.steps:
                break
            if base.args_cli.video and step_count >= base.args_cli.video_length:
                break

            if base.torch.any(terminated | truncated):
                print(
                    f"[INFO] step={step_count} terminated={terminated.sum().item()} truncated={truncated.sum().item()}"
                )
                done_mask = terminated | truncated
                last_action_tensor = last_action_tensor.clone()
                last_action_tensor[done_mask] = initial_last_action[done_mask]
                cached_actions = None
                cached_index = 0
                executed_since_replan = 0
                action_hold_steps = 0

            sleep_time = dt - (base.time.time() - start_time)
            if base.args_cli.real_time and sleep_time > 0:
                base.time.sleep(sleep_time)
    finally:
        env.close()
        if base.args_cli.save_path is not None:
            save_path = base.Path(base.args_cli.save_path).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)

            predicted_joint_chunks_np = base.np.array(
                [chunk.float().cpu().numpy() for chunk in predicted_joint_chunks_gpu],
                dtype=object,
            )
            executed_ee_positions_np = (
                base.torch.stack(executed_ee_positions_gpu).float().cpu().numpy()
                if executed_ee_positions_gpu
                else base.np.zeros((0,), dtype=base.np.float32)
            )

            success_rate = float(success_mask.float().mean().item()) if success_mask is not None else float("nan")
            mean_ee_dist = (ee_dist_sum / metric_steps) if metric_steps > 0 else float("nan")
            mean_obj_height = (obj_height_sum / metric_steps) if metric_steps > 0 else float("nan")

            base.np.savez(
                save_path,
                predicted_joint_chunks=predicted_joint_chunks_np,
                predicted_ee_chunks=base.np.array([], dtype=object),
                predicted_ee_chunks_available=False,
                executed_ee_positions=executed_ee_positions_np,
                output_horizon=base.args_cli.output_horizon,
                execute_horizon=base.args_cli.execute_horizon,
                task=base.args_cli.task,
                success_rate=success_rate,
                mean_ee_dist=mean_ee_dist,
                mean_obj_height=mean_obj_height,
                success_dist=base.args_cli.success_dist,
                success_z=base.args_cli.success_z,
                seed=base.args_cli.seed,
            )
            print(f"[INFO] Saved trajectories to: {save_path}")
            print(
                f"[METRIC] success_rate={success_rate:.4f} mean_ee_dist={mean_ee_dist:.4f} "
                f"mean_obj_height={mean_obj_height:.4f}"
            )


if __name__ == "__main__":
    main()
    base.simulation_app.close()
