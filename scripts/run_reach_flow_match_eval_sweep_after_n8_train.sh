#!/usr/bin/env bash
set -euo pipefail

cd /media/kim/hdd1tb2/master_study/only_robot_rl_2
export PYTHONPATH="${PYTHONPATH:-$PWD}"

ISAACLAB_BIN="${ISAACLAB_BIN:-isaaclab_4}"
TRAIN_LOG="${TRAIN_LOG:-./policy_models/flow_match/reach/reach_flow_match_artifacts_8chunk/train_log.txt}"
OUT_DIR="${OUT_DIR:-./datas/reach_flow_match_eval_sweep}"
NUM_ENVS="${NUM_ENVS:-100}"
EPISODES="${EPISODES:-100}"
SEED="${SEED:-50}"
DEVICE="${DEVICE:-cuda:0}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-50}"

mkdir -p "${OUT_DIR}"

wait_for_n8_train() {
  echo "================================================================"
  echo "[WAIT] Flow Matching Reach N=8 training"
  echo "[LOG] ${TRAIN_LOG}"
  echo "================================================================"

  while ! grep -q '^final_val_loss=' "${TRAIN_LOG}" 2>/dev/null; do
    if ! pgrep -f 'policy_models/flow_match/reach/train.py' >/dev/null; then
      echo "[ERROR] N=8 training process is not running and final_val_loss is not in ${TRAIN_LOG}"
      tail -n 30 "${TRAIN_LOG}" 2>/dev/null || true
      exit 1
    fi
    echo "[WAIT] $(date '+%Y-%m-%d %H:%M:%S') $(tail -n 1 "${TRAIN_LOG}" 2>/dev/null || echo 'log not ready')"
    sleep 60
  done

  echo "[DONE] N=8 training finished"
  tail -n 8 "${TRAIN_LOG}"
}

run_eval() {
  local n="$1"
  local k="$2"
  local save_dir="$3"
  local save_path="${OUT_DIR}/reach_flow_match_N${n}_k${k}_seed${SEED}.npz"
  local log_path="${OUT_DIR}/reach_flow_match_N${n}_k${k}_seed${SEED}.log"

  if [[ "${FORCE_RERUN:-0}" != "1" && -f "${save_path}" ]]; then
    echo "[SKIP] Flow Matching Reach N=${n} k=${k}: ${save_path}"
    return
  fi

  echo "================================================================"
  echo "[TEST] Flow Matching Reach N=${n} k=${k}"
  echo "[SAVE] ${save_path}"
  echo "================================================================"

  TERM=xterm PYTHONPATH="$PWD" "${ISAACLAB_BIN}" -p ./launch/imitation_play_fixed_left_right.py \
    --kit_args="--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error" \
    --task reach_imitation \
    --policy_type python \
    --policy_path ./policy_models/flow_match/reach/policy_wrapper.py \
    --policy_checkpoint "${save_dir}/reach_flow_match.pt" \
    --policy_stats "${save_dir}/reach_flow_match_stats.npz" \
    --num_envs "${NUM_ENVS}" \
    --output_horizon "${n}" \
    --execute_horizon "${k}" \
    --episode_length_s "${EPISODE_LENGTH_S}" \
    --joint_arrival_tolerance 0.015 \
    --max_action_hold_steps 15 \
    --action_smoothing_alpha 0.8 \
    --next_action_lookahead_margin 0.004 \
    --replan_lookahead_steps 0 \
    --episodes "${EPISODES}" \
    --seed "${SEED}" \
    --save_path "${save_path}" \
    --device "${DEVICE}" \
    --disable_predicted_path_viz \
    --headless 2>&1 | tee "${log_path}"
}

wait_for_n8_train

for k in 2 4 6 8; do
  run_eval 8 "${k}" ./policy_models/flow_match/reach/reach_flow_match_artifacts_8chunk
done

for k in 6 8 10 14 16; do
  run_eval 16 "${k}" ./policy_models/flow_match/reach/reach_flow_match_artifacts
done

echo "================================================================"
echo "[DONE] Flow Matching Reach eval sweep completed"
echo "[OUT] ${OUT_DIR}"
echo "================================================================"
