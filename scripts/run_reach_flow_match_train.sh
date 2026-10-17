#!/usr/bin/env bash
set -euo pipefail

cd /media/kim/hdd1tb2/master_study/only_robot_rl_2
export PYTHONPATH="${PYTHONPATH:-$PWD}"

CHUNK_SIZE="${CHUNK_SIZE:-16}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_EPOCHS="${NUM_EPOCHS:-200}"
NUM_WORKERS="${NUM_WORKERS:-16}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.000001}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cuda:0}"
TQDM_DISABLE="${TQDM_DISABLE:-1}"
MODEL_ARCH="${MODEL_ARCH:-stanford_dp}"

if [[ "${CHUNK_SIZE}" == "8" ]]; then
  SAVE_DIR_OVERRIDE="${SAVE_DIR_OVERRIDE:-./policy_models/flow_match/reach/reach_flow_match_artifacts_8chunk}"
else
  SAVE_DIR_OVERRIDE="${SAVE_DIR_OVERRIDE:-./policy_models/flow_match/reach/reach_flow_match_artifacts}"
fi

echo "================================================================"
echo "[TRAIN] Flow Matching Reach N=${CHUNK_SIZE}"
echo "[ARCH] ${MODEL_ARCH}"
echo "[SAVE] ${SAVE_DIR_OVERRIDE}"
echo "[DEVICE] ${TRAIN_DEVICE}"
echo "================================================================"

PYTHON_RUNNER=()
if command -v uv >/dev/null 2>&1; then
  PYTHON_RUNNER=(uv run python)
else
  PYTHON_RUNNER=(python)
fi

PYTHONPATH="$PWD" \
MODEL_ARCH="${MODEL_ARCH}" \
CHUNK_SIZE="${CHUNK_SIZE}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_EPOCHS="${NUM_EPOCHS}" \
NUM_WORKERS="${NUM_WORKERS}" \
LEARNING_RATE="${LEARNING_RATE}" \
WEIGHT_DECAY="${WEIGHT_DECAY}" \
TRAIN_DEVICE="${TRAIN_DEVICE}" \
SAVE_DIR_OVERRIDE="${SAVE_DIR_OVERRIDE}" \
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-reach_flow_match.pt}" \
STATS_FILENAME="${STATS_FILENAME:-reach_flow_match_stats.npz}" \
TQDM_DISABLE="${TQDM_DISABLE}" \
"${PYTHON_RUNNER[@]}" ./policy_models/flow_match/reach/train.py
