#!/usr/bin/env bash
set -euo pipefail

cd /media/kim/hdd1tb2/master_study/only_robot_rl_2
export PYTHONPATH="${PYTHONPATH:-$PWD}"

ISAACLAB_BIN="${ISAACLAB_BIN:-isaaclab_4}"
OUT_DIR="${OUT_DIR:-./datas/reach_lift_flow_match_diffusion_arch_sequence}"

NUM_EPOCHS="${NUM_EPOCHS:-150}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_TRAIN_WINDOWS_PER_EPOCH="${MAX_TRAIN_WINDOWS_PER_EPOCH:-0}"
MAX_VAL_WINDOWS="${MAX_VAL_WINDOWS:-0}"
LEARNING_RATE="${LEARNING_RATE:-0.00015}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-100}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-50}"
SUCCESS_Z="${SUCCESS_Z:-1.106}"

mkdir -p "${OUT_DIR}"

save_dir_for_n() {
  local n="$1"
  if [[ "${n}" == "8" ]]; then
    echo "./policy_models/flow_match/reach_lift/reach_lift_flow_match_artifacts_balanced_v5_1_8chunk"
  else
    echo "./policy_models/flow_match/reach_lift/reach_lift_flow_match_artifacts_balanced_v5_1"
  fi
}

default_k_for_n() {
  local n="$1"
  if [[ "${n}" == "8" ]]; then
    echo "${EXECUTE_HORIZON_N8:-6}"
  else
    echo "${EXECUTE_HORIZON_N16:-14}"
  fi
}

train_model() {
  local n="$1"
  local save_dir="$2"

  if [[ "${FORCE_RETRAIN:-0}" != "1" ]] && [[ -f "${save_dir}/train_log.txt" ]] && rg -q '^final_val_loss=' "${save_dir}/train_log.txt"; then
    echo "[SKIP] FlowMatching diffusion-arch N=${n}: already trained at ${save_dir}"
    return
  fi

  mkdir -p "${save_dir}"
  echo "================================================================"
  echo "[TRAIN] FlowMatching diffusion-arch N=${n}"
  echo "[ARCH] action_mode=mean processed_obs_dim=124 processed_action_dim=7 down_dims=(192,384,768)"
  echo "[MATCH] diffusion train config except batch: lr=0.00015 no_lr_scheduler no_sampler_oversampling no_episode_cache grasp/lift weights matched"
  echo "[SAVE] ${save_dir}"
  echo "================================================================"

  MODEL_ARCH=stanford_dp \
  ACTION_MODE=mean \
  CHUNK_SIZE="${n}" \
  NUM_EPOCHS="${NUM_EPOCHS}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  WEIGHT_DECAY=0.000001 \
  USE_LR_SCHEDULER=0 \
  NUM_WORKERS="${NUM_WORKERS}" \
  MAX_TRAIN_WINDOWS_PER_EPOCH="${MAX_TRAIN_WINDOWS_PER_EPOCH}" \
  MAX_VAL_WINDOWS="${MAX_VAL_WINDOWS}" \
  CACHE_EPISODES=0 \
  RECOMPUTE_STATS="${RECOMPUTE_STATS:-1}" \
  REUSE_STATS_PATH="${REUSE_STATS_PATH:-./policy_models/diffusion/reach_lift/reach_lift_diffusion_unet_artifacts_balanced_v3_1/reach_lift_diffusion_balanced_v3_stats.npz}" \
  OVERSAMPLE_CRITICAL_WINDOWS=0 \
  PREGRASP_LOOKBACK=8 \
  GRASP_WINDOW_WEIGHT=4.0 \
  LIFT_WINDOW_WEIGHT=5.0 \
  PREGRASP_WINDOW_WEIGHT=10.0 \
  GRASP_STEP_WEIGHT=5.0 \
  LIFT_STEP_WEIGHT=6.0 \
  PREGRASP_STEP_WEIGHT=12.0 \
  GRIPPER_LOSS_WEIGHT=4.0 \
  TRAIN_DEVICE="${TRAIN_DEVICE}" \
  SAVE_DIR_OVERRIDE="${save_dir}" \
  CHECKPOINT_FILENAME="reach_lift_flow_match_balanced_v5.pt" \
  STATS_FILENAME="reach_lift_flow_match_balanced_v5_stats.npz" \
  TQDM_DISABLE="${TQDM_DISABLE:-1}" \
  python ./policy_models/flow_match/reach_lift/train.py
}

test_model() {
  local n="$1"
  local k="$2"
  local save_dir="$3"
  local save_path="${OUT_DIR}/reach_lift_flow_match_diffusion_arch_N${n}_k${k}.npz"
  local log_path="${OUT_DIR}/reach_lift_flow_match_diffusion_arch_N${n}_k${k}.log"

  if [[ "${FORCE_RERUN:-0}" != "1" ]] && [[ -f "${save_path}" ]]; then
    echo "[SKIP] test FlowMatching diffusion-arch N=${n} k=${k}: ${save_path}"
    return
  fi

  echo "================================================================"
  echo "[TEST] FlowMatching diffusion-arch N=${n} k=${k}"
  echo "================================================================"

  TERM=xterm PYTHONPATH="$PWD" "${ISAACLAB_BIN}" -p ./launch/imitation_play_fixed_left_right.py \
    --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error" \
    --task reach_lift_imitation \
    --policy_type python \
    --policy_path ./policy_models/flow_match/reach_lift/policy_wrapper.py \
    --policy_checkpoint "${save_dir}/reach_lift_flow_match_balanced_v5.pt" \
    --policy_stats "${save_dir}/reach_lift_flow_match_balanced_v5_stats.npz" \
    --num_envs "${NUM_ENVS}" \
    --output_horizon "${n}" \
    --execute_horizon "${k}" \
    --episode_length_s "${EPISODE_LENGTH_S:-30}" \
    --joint_arrival_tolerance "${JOINT_ARRIVAL_TOLERANCE:-0.015}" \
    --max_action_hold_steps "${MAX_ACTION_HOLD_STEPS:-15}" \
    --action_smoothing_alpha "${ACTION_SMOOTHING_ALPHA:-0.8}" \
    --next_action_lookahead_margin "${NEXT_ACTION_LOOKAHEAD_MARGIN:-0.004}" \
    --chunk_transition_blend_steps 0 \
    --replan_lookahead_steps 0 \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --success_z "${SUCCESS_Z}" \
    --save_path "${save_path}" \
    --device "${DEVICE}" \
    --disable_predicted_path_viz \
    --headless 2>&1 | tee "${log_path}"
}

append_summary() {
  python - <<'PY'
from pathlib import Path
import numpy as np

out_dir = Path("datas/reach_lift_flow_match_diffusion_arch_sequence")
rows = []
for path in sorted(out_dir.glob("*.npz")):
    data = np.load(path, allow_pickle=True)
    episodes = int(np.asarray(data["episode_done_count"]).item())
    success = int(np.asarray(data["success_done_count"]).item())
    rate = success / episodes if episodes else 0.0
    n = int(np.asarray(data["output_horizon"]).item())
    k = int(np.asarray(data["execute_horizon"]).item())
    latency = float(np.asarray(data.get("avg_inference_latency_ms", np.nan)).item())
    delta = float(np.asarray(data.get("mean_action_delta", np.nan)).item())
    jerk = float(np.asarray(data.get("mean_action_jerk", np.nan)).item())
    rows.append((n, k, success, episodes, rate, latency, delta, jerk, path.name))

summary = out_dir / "summary.md"
with summary.open("w", encoding="utf-8") as f:
    f.write("# FlowMatching Diffusion-Architecture Sequence (_2)\n\n")
    f.write("| N | k | Success | Rate | latency | action delta | action jerk | file |\n")
    f.write("|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for n, k, success, episodes, rate, latency, delta, jerk, name in rows:
        f.write(
            f"| {n} | {k} | {success}/{episodes} | {rate:.1%} | "
            f"{latency:.2f} ms | {delta:.6f} | {jerk:.6f} | `{name}` |\n"
        )
print(summary)
PY
}

for n in 16 8; do
  save_dir="$(save_dir_for_n "${n}")"
  k="$(default_k_for_n "${n}")"
  train_model "${n}" "${save_dir}"
  test_model "${n}" "${k}" "${save_dir}"
done

append_summary
