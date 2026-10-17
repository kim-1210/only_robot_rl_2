"""
N = 16

TERM=xterm PYTHONPATH=$PWD isaaclab_4 -p ./launch/imitation_play.py   \
--task reach_imitation  \
--policy_type python   \
--policy_path ./policy_models/diffusion/reach/policy_wrapper.py   \
--policy_checkpoint ./policy_models/diffusion/reach/reach_diffusion_unet_artifacts/reach_diffusion_unet.pt   \
--policy_stats ./policy_models/diffusion/reach/reach_diffusion_unet_artifacts/reach_diffusion_stats.npz   \
--num_envs 30   \
--output_horizon 16   \
--execute_horizon 16   \
--episode_length_s 40   \
--joint_arrival_tolerance 0.01   \
--max_action_hold_steps 20   \
--action_smoothing_alpha 0.7   \
--next_action_lookahead_margin 0.004   \
--episodes 100   \
--seed 42   \
--save_path ./datas/reach_imitation_diffusion_rollout_k8.npz   \
--device cuda:0 \
--headless

TERM=xterm PYTHONPATH=$PWD isaaclab -p ./launch/imitation_play.py \
--task reach_imitation \
--policy_type python \
--policy_path ./policy_models/flow_match/reach/policy_wrapper.py \
--policy_checkpoint ./policy_models/flow_match/reach/reach_flow_match_artifacts/reach_flow_match.pt \
--policy_stats ./policy_models/flow_match/reach/reach_flow_match_artifacts/reach_flow_match_stats.npz \
--num_envs 20 \
--output_horizon 16 \
--execute_horizon 14 \
--episode_length_s 20   \
--joint_arrival_tolerance 0.01   \
--max_action_hold_steps 20   \
--action_smoothing_alpha 0.7   \
--next_action_lookahead_margin 0.004   \
--episodes 500   \
--seed 42   \
--save_path ./datas/reach_imitation_flow_match_rollout_k8.npz   \
--device cuda:0 \
--headless

  
N = 8
(3, 5, 7)

"""

import imitation_play_fixed as base
from imitation_play_fixed import *  # noqa: F401,F403


def _apply_reach_execute_horizon_defaults() -> None:
    # Align the default waypoint execution behavior with imitation_play_fixed_left_right.
    if base.args_cli.joint_arrival_tolerance == base.parser.get_default("joint_arrival_tolerance"):
        base.args_cli.joint_arrival_tolerance = 0.01
    if base.args_cli.max_action_hold_steps == base.parser.get_default("max_action_hold_steps"):
        base.args_cli.max_action_hold_steps = 20
    if base.args_cli.action_smoothing_alpha == base.parser.get_default("action_smoothing_alpha"):
        base.args_cli.action_smoothing_alpha = 0.7
    if base.args_cli.next_action_lookahead_margin == base.parser.get_default("next_action_lookahead_margin"):
        base.args_cli.next_action_lookahead_margin = 0.004


if __name__ == "__main__":
    _apply_reach_execute_horizon_defaults()
    base.main()
    base.simulation_app.close()
