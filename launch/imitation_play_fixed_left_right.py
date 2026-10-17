"""
N=16 (4,6,10,14)
성공 높이 1.14

TERM=xterm PYTHONPATH=$PWD isaaclab -p ./launch/imitation_play_fixed_left_right.py \
  --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error" \
  --task reach_lift_imitation \
  --policy_type python \
  --policy_path ./policy_models/diffusion/reach_lift/policy_wrapper.py \
  --policy_checkpoint ./policy_models/diffusion/reach_lift/reach_lift_diffusion_unet_artifacts_balanced_v3_1/reach_lift_diffusion_unet_balanced_v3.pt \
  --policy_stats ./policy_models/diffusion/reach_lift/reach_lift_diffusion_unet_artifacts_balanced_v3_1/reach_lift_diffusion_balanced_v3_stats.npz \
  --num_envs 100 \
  --output_horizon 16 \
  --execute_horizon 14 \
  --episode_length_s 25 \
  --joint_arrival_tolerance 0.01 \
  --max_action_hold_steps 20 \
  --action_smoothing_alpha 0.7 \
  --next_action_lookahead_margin 0.004 \
  --replan_lookahead_steps 0 \
  --episodes 500 \
  --seed 50 \
  --device cuda:0 \
  --binary_gripper \
  --log_gripper \
  --disable_predicted_path_viz \
  --headless

TERM=xterm PYTHONPATH=$PWD isaaclab -p ./launch/imitation_play_fixed_left_right.py \
    --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error" \
  --task reach_lift_imitation \
  --policy_type python \
  --policy_path ./policy_models/flow_match/reach_lift/policy_wrapper.py \
  --policy_checkpoint ./policy_models/flow_match/reach_lift/reach_lift_flow_match_artifacts_balanced_v5_1/reach_lift_flow_match_balanced_v5.pt \
  --policy_stats ./policy_models/flow_match/reach_lift/reach_lift_flow_match_artifacts_balanced_v5_1/reach_lift_flow_match_balanced_v5_stats.npz \
  --num_envs 100 \
  --output_horizon 16 \
  --execute_horizon 14 \
  --episode_length_s 25   \
  --joint_arrival_tolerance 0.01 \
  --max_action_hold_steps 20 \
  --action_smoothing_alpha 0.7 \
  --next_action_lookahead_margin 0.004 \
  --replan_lookahead_steps 1 \
  --episodes 500 \
  --seed 50 \
  --device cuda:0 \
  --binary_gripper \
  --log_gripper \
  --disable_predicted_path_viz \
  --headless

N=8 (2,4,6)
"""

import imitation_play_fixed as base


if __name__ == "__main__":
    base.main()
    base.simulation_app.close()
