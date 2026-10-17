import copy
import os
import math
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

try:
    from .flow_matching_unet_model import ConditionalUnet1D
except ImportError:
    local_dir = Path(__file__).resolve().parent
    if str(local_dir) not in sys.path:
        sys.path.append(str(local_dir))
    from flow_matching_unet_model import ConditionalUnet1D

def _load_diffusion_transformer_class():
    module_path = Path(__file__).resolve().parents[2] / "diffusion" / "reach_lift" / "transformer_policy_model.py"
    if not module_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("reach_lift_transformer_policy_model_for_flow_train", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load transformer module from {module_path}")
    spec.loader.exec_module(module)
    return module.ConditionalTransformer1D


def _load_diffusion_unet_class():
    module_path = Path(__file__).resolve().parents[2] / "diffusion" / "reach_lift" / "diffusion_policy_unet.py"
    spec = importlib.util.spec_from_file_location("reach_lift_diffusion_policy_unet_for_flow_train", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diffusion U-Net module from {module_path}")
    spec.loader.exec_module(module)
    return module.DiffusionPolicyConditionalUnet1D


DiffusionPolicyConditionalUnet1D = _load_diffusion_unet_class()
ConditionalTransformer1D = _load_diffusion_transformer_class()

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(os.environ.get("TRAIN_DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent.parent.parent


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"


class Config:
    data_root = REPO_ROOT / "datas" / "reach_lift_data1"
    obs_dir = data_root / "obs"
    act_dir = data_root / "actions"

    chunk_size = int(os.environ.get("CHUNK_SIZE", "8"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1024"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "150"))
    learning_rate = env_float("LEARNING_RATE", 2e-4)
    weight_decay = env_float("WEIGHT_DECAY", 1e-6)
    num_workers = int(os.environ.get("NUM_WORKERS", "16"))
    cache_episodes = os.environ.get("CACHE_EPISODES", "0") == "1"
    max_train_windows_per_epoch = int(os.environ.get("MAX_TRAIN_WINDOWS_PER_EPOCH", "0"))
    max_val_windows = int(os.environ.get("MAX_VAL_WINDOWS", "0"))
    max_stats_episodes = int(os.environ.get("MAX_STATS_EPISODES", "0"))
    batch_log_every = int(os.environ.get("BATCH_LOG_EVERY", "50"))
    action_mode = os.environ.get("ACTION_MODE", "lr")
    use_lr_scheduler = env_bool("USE_LR_SCHEDULER", True)
    lr_warmup_epochs = env_int("LR_WARMUP_EPOCHS", 5)
    min_learning_rate_scale = env_float("MIN_LEARNING_RATE_SCALE", 0.1)

    flow_steps = 100
    checkpoint_save_every = 10

    model_arch = os.environ.get("MODEL_ARCH", "stanford_dp")
    down_dims = (192, 384, 768)
    time_embed_dim = 384
    kernel_size = 5
    n_groups = 8
    cond_predict_scale = True
    transformer_dim = env_int("TRANSFORMER_DIM", 384)
    transformer_heads = env_int("TRANSFORMER_HEADS", 8)
    transformer_layers = env_int("TRANSFORMER_LAYERS", 8)
    transformer_ff_dim = env_int("TRANSFORMER_FF_DIM", 1536)
    transformer_dropout = env_float("TRANSFORMER_DROPOUT", 0.1)

    raw_obs_dim = 35
    processed_obs_frame_dim = 34 if action_mode == "lr" else 31
    obs_history_steps = 4
    processed_obs_dim = processed_obs_frame_dim * obs_history_steps
    raw_action_dim = 7
    processed_action_dim = 8 if action_mode == "lr" else 7

    grasp_close_threshold = env_float("GRASP_CLOSE_THRESHOLD", 0.013)
    lift_height_threshold = env_float("LIFT_HEIGHT_THRESHOLD", 0.02)
    pregrasp_lookback = env_int("PREGRASP_LOOKBACK", 12)
    grasp_window_weight = env_float("GRASP_WINDOW_WEIGHT", 8.0)
    lift_window_weight = env_float("LIFT_WINDOW_WEIGHT", 12.0)
    pregrasp_window_weight = env_float("PREGRASP_WINDOW_WEIGHT", 16.0)
    grasp_step_weight = env_float("GRASP_STEP_WEIGHT", 10.0)
    lift_step_weight = env_float("LIFT_STEP_WEIGHT", 14.0)
    pregrasp_step_weight = env_float("PREGRASP_STEP_WEIGHT", 20.0)
    gripper_loss_weight = env_float("GRIPPER_LOSS_WEIGHT", 8.0)
    oversample_critical_windows = os.environ.get("OVERSAMPLE_CRITICAL_WINDOWS", "1") == "1"
    grasp_sample_weight = env_float("GRASP_SAMPLE_WEIGHT", 5.0)
    lift_sample_weight = env_float("LIFT_SAMPLE_WEIGHT", 10.0)
    pregrasp_sample_weight = env_float("PREGRASP_SAMPLE_WEIGHT", 8.0)
    friendly_gripper_actions = env_bool("FRIENDLY_GRIPPER_ACTIONS", False)
    friendly_lift_delta_threshold = env_float("FRIENDLY_LIFT_DELTA_THRESHOLD", 0.04)
    friendly_gripper_close_value = env_float("FRIENDLY_GRIPPER_CLOSE_VALUE", 0.0)
    friendly_gripper_open_value = env_float("FRIENDLY_GRIPPER_OPEN_VALUE", 0.025)
    friendly_gripper_preclose_steps = env_int("FRIENDLY_GRIPPER_PRECLOSE_STEPS", 0)
    friendly_gripper_hold_after_lift_steps = env_int("FRIENDLY_GRIPPER_HOLD_AFTER_LIFT_STEPS", 16)
    friendly_gripper_hold_to_episode_end = env_bool("FRIENDLY_GRIPPER_HOLD_TO_EPISODE_END", True)
    friendly_force_open_before_close = env_bool("FRIENDLY_FORCE_OPEN_BEFORE_CLOSE", True)
    filter_successful_lift_episodes = env_bool("FILTER_SUCCESSFUL_LIFT_EPISODES", False)
    trim_terminal_reset_actions = env_bool("TRIM_TERMINAL_RESET_ACTIONS", False)
    trim_after_success = env_bool("TRIM_AFTER_SUCCESS", False)
    post_success_keep_steps = env_int("POST_SUCCESS_KEEP_STEPS", 0)
    terminal_reset_atol = env_float("TERMINAL_RESET_ATOL", 1e-4)
    reconstruct_grasp_lift_windows = env_bool("RECONSTRUCT_GRASP_LIFT_WINDOWS", False)
    reconstruct_grasp_mode = os.environ.get("RECONSTRUCT_GRASP_MODE", "close")
    reconstruct_lift_start_delta = env_float("RECONSTRUCT_LIFT_START_DELTA", 0.005)
    reconstruct_lift_confirm_delta = env_float("RECONSTRUCT_LIFT_CONFIRM_DELTA", 0.01)
    reconstruct_lift_confirm_steps = env_int("RECONSTRUCT_LIFT_CONFIRM_STEPS", 6)
    reconstruct_grasp_pre_steps = env_int("RECONSTRUCT_GRASP_PRE_STEPS", 24)
    reconstruct_grasp_post_close_steps = env_int("RECONSTRUCT_GRASP_POST_CLOSE_STEPS", 10)
    grasp_crop_sample_weight = env_float("GRASP_CROP_SAMPLE_WEIGHT", 30.0)
    lift_crop_sample_weight = env_float("LIFT_CROP_SAMPLE_WEIGHT", 40.0)
    grasp_crop_step_weight = env_float("GRASP_CROP_STEP_WEIGHT", 24.0)
    lift_crop_step_weight = env_float("LIFT_CROP_STEP_WEIGHT", 32.0)
    grasp_crop_window_weight = env_float("GRASP_CROP_WINDOW_WEIGHT", 24.0)
    lift_crop_window_weight = env_float("LIFT_CROP_WINDOW_WEIGHT", 32.0)

    use_ema = True
    ema_decay = 0.999
    inference_noise_mode = "zeros"

    if model_arch == "transformer":
        default_save_dir = FILE_DIR / (
            "reach_lift_flow_match_transformer_lr_matched_8chunk"
            if chunk_size == 8
            else "reach_lift_flow_match_transformer_lr_matched"
        )
    else:
        default_save_dir = FILE_DIR / (
            (
                "reach_lift_flow_match_lr_matched_dp_backbone_8chunk"
                if chunk_size == 8
                else "reach_lift_flow_match_lr_matched_dp_backbone"
            )
            if action_mode == "lr"
            else (
                "reach_lift_flow_match_matched_dp_backbone_8chunk"
                if chunk_size == 8
                else "reach_lift_flow_match_matched_dp_backbone"
            )
        )
    save_dir = Path(os.environ.get("SAVE_DIR_OVERRIDE", default_save_dir))
    checkpoint_filename = os.environ.get(
        "CHECKPOINT_FILENAME",
        "reach_lift_flow_match_transformer.pt"
        if model_arch == "transformer"
        else "reach_lift_flow_match_matched_dp_backbone.pt",
    )
    stats_filename = os.environ.get(
        "STATS_FILENAME",
        "reach_lift_flow_match_transformer_stats.npz"
        if model_arch == "transformer"
        else "reach_lift_flow_match_matched_dp_backbone_stats.npz",
    )
    checkpoint_path = save_dir / checkpoint_filename
    stats_path = save_dir / stats_filename
    log_txt_path = save_dir / "train_log.txt"
    log_md_path = save_dir / "train_log.md"

config = Config()
config.save_dir.mkdir(parents=True, exist_ok=True)

dataset = None
model = None
ema_helper = None
ema_model = None
optimizer = None

INITIAL_ACTION_FROM_CFG_SETTINGS = np.array(
    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025, 0.025],
    dtype=np.float32,
)
if config.action_mode != "lr":
    INITIAL_ACTION_FROM_CFG_SETTINGS = INITIAL_ACTION_FROM_CFG_SETTINGS[:7]

RAW_RESET_ACTION_FROM_CFG_SETTINGS = np.array(
    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.0, 0.025, 0.025],
    dtype=np.float32,
)


def action_is_terminal_reset(raw_action: np.ndarray) -> np.ndarray:
    cols = min(raw_action.shape[-1], RAW_RESET_ACTION_FROM_CFG_SETTINGS.shape[0])
    return np.all(
        np.isclose(
            raw_action[..., :cols],
            RAW_RESET_ACTION_FROM_CFG_SETTINGS[:cols],
            atol=config.terminal_reset_atol,
            rtol=0.0,
        ),
        axis=-1,
    )


def compute_trimmed_length(raw_obs: np.ndarray, raw_action: np.ndarray, n: int) -> int:
    n = min(int(n), len(raw_obs), len(raw_action))
    end = n
    if n <= 0:
        return 0

    if config.trim_after_success:
        obj_z = np.asarray(raw_obs[:n, 20], dtype=np.float32)
        success_mask = (obj_z - obj_z[0]) >= config.friendly_lift_delta_threshold
        if np.any(success_mask):
            first_success_idx = int(np.flatnonzero(success_mask)[0])
            end = min(end, first_success_idx + max(0, config.post_success_keep_steps))

    if config.trim_terminal_reset_actions and end > 0:
        reset_mask = action_is_terminal_reset(np.asarray(raw_action[:end], dtype=np.float32))
        while end > 0 and bool(reset_mask[end - 1]):
            end -= 1

    return max(0, int(end))


def compress_robot_state(raw_state: np.ndarray) -> np.ndarray:
    if config.action_mode == "lr":
        return np.concatenate([raw_state[:, :6], raw_state[:, 7:9]], axis=1).astype(np.float32)
    return np.concatenate([raw_state[:, :6], raw_state[:, 7:9].mean(axis=1, keepdims=True)], axis=1).astype(np.float32)

def build_prev_joint_history(raw_action: np.ndarray) -> np.ndarray:
    if raw_action.shape[1] == config.processed_action_dim:
        processed = raw_action.astype(np.float32)
    elif raw_action.shape[1] >= 9:
        processed = compress_robot_state(raw_action)
    else:
        raise ValueError(f"Unsupported raw action dim: {raw_action.shape[1]}")

    history = np.empty_like(processed, dtype=np.float32)
    history[0] = INITIAL_ACTION_FROM_CFG_SETTINGS
    if len(processed) > 1:
        history[1:] = processed[:-1]
    return history

def preprocess_reach_obs(raw_obs: np.ndarray, prev_joint_raw: np.ndarray) -> np.ndarray:
    if raw_obs.shape[0] != prev_joint_raw.shape[0]:
        n = min(raw_obs.shape[0], prev_joint_raw.shape[0])
        raw_obs = raw_obs[:n]
        prev_joint_raw = prev_joint_raw[:n]

    joint_pos = raw_obs[:, 0:9]
    joint_vel = raw_obs[:, 9:18]
    object_position = raw_obs[:, 18:21]
    pose_command = raw_obs[:, 21:28]

    joint_pos_processed = compress_robot_state(joint_pos)
    joint_vel_processed = compress_robot_state(joint_vel)
    return np.concatenate(
        [joint_pos_processed, joint_vel_processed, object_position, pose_command, prev_joint_raw],
        axis=1,
    ).astype(np.float32)


def build_obs_history(processed_obs: np.ndarray, history_steps: int) -> np.ndarray:
    if processed_obs.ndim != 2:
        raise ValueError(f"Expected processed_obs with shape [T, D], got {processed_obs.shape}")
    if history_steps <= 1:
        return processed_obs.astype(np.float32)

    num_steps, feat_dim = processed_obs.shape
    history = np.empty((num_steps, history_steps, feat_dim), dtype=np.float32)
    for step in range(num_steps):
        start = max(0, step - history_steps + 1)
        window = processed_obs[start : step + 1]
        history[step, -window.shape[0] :] = window
        if window.shape[0] < history_steps:
            history[step, : history_steps - window.shape[0]] = window[0]
    return history.reshape(num_steps, history_steps * feat_dim).astype(np.float32)


def preprocess_reach_action(raw_action: np.ndarray) -> np.ndarray:
    if raw_action.shape[1] == config.processed_action_dim:
        return raw_action.astype(np.float32)
    if raw_action.shape[1] >= 9:
        return compress_robot_state(raw_action)
    raise ValueError(f"Unsupported raw action dim: {raw_action.shape[1]}")


def get_gripper_columns(raw_action: np.ndarray) -> list[int]:
    if raw_action.shape[1] >= 9:
        return [7, 8]
    if raw_action.shape[1] == 8:
        return [6, 7]
    if raw_action.shape[1] == 7:
        return [6]
    raise ValueError(f"Unsupported raw action dim for gripper columns: {raw_action.shape[1]}")


def make_friendly_reach_lift_actions(raw_obs: np.ndarray, raw_action: np.ndarray) -> np.ndarray:
    action = raw_action.astype(np.float32, copy=False)
    if not config.friendly_gripper_actions:
        return action
    n = min(len(raw_obs), len(action))
    if n == 0:
        return action

    obj_z = raw_obs[:n, 20]
    lift_mask = (obj_z - obj_z[0]) >= config.friendly_lift_delta_threshold
    if not np.any(lift_mask):
        return action

    gripper = get_episode_gripper(action[:n])
    close_mask = gripper <= config.grasp_close_threshold
    if not np.any(close_mask):
        return action

    first_close_idx = int(np.flatnonzero(close_mask)[0])
    first_lift_idx = int(np.flatnonzero(lift_mask)[0])
    close_start = max(0, first_close_idx - config.friendly_gripper_preclose_steps)
    if config.friendly_gripper_hold_to_episode_end:
        close_end = n
    else:
        close_end = min(n, max(first_lift_idx + config.friendly_gripper_hold_after_lift_steps + 1, close_start + 1))

    friendly = action.copy()
    gripper_cols = get_gripper_columns(friendly)
    if config.friendly_force_open_before_close and close_start > 0:
        friendly[:close_start, gripper_cols] = config.friendly_gripper_open_value
    friendly[close_start:close_end, gripper_cols] = config.friendly_gripper_close_value
    return friendly


def build_chunk_weight_targets(
    obs_slice: np.ndarray,
    act_slice: np.ndarray,
    *,
    close_threshold: float,
    lift_height_threshold: float,
    grasp_window_weight: float,
    lift_window_weight: float,
    pregrasp_window_weight: float,
    grasp_step_weight: float,
    lift_step_weight: float,
    pregrasp_step_weight: float,
    pregrasp_lookback: int,
) -> tuple[np.ndarray, np.float32]:
    num_steps = act_slice.shape[0]
    step_weight = np.ones((num_steps,), dtype=np.float32)
    window_weight = 1.0

    if act_slice.shape[1] >= 9:
        gripper = 0.5 * (act_slice[:, 7] + act_slice[:, 8])
    elif act_slice.shape[1] == 8:
        gripper = 0.5 * (act_slice[:, 6] + act_slice[:, 7])
    elif act_slice.shape[1] == 7:
        gripper = act_slice[:, 6]
    else:
        raise ValueError(f"Unsupported action dim for weighting: {act_slice.shape[1]}")

    close_mask = gripper <= close_threshold
    if np.any(close_mask):
        first_close_idx = int(np.argmax(close_mask))
        pregrasp_start = max(0, first_close_idx - pregrasp_lookback)
        step_weight[pregrasp_start : first_close_idx + 1] = np.maximum(
            step_weight[pregrasp_start : first_close_idx + 1],
            pregrasp_step_weight,
        )
        step_weight[close_mask] = np.maximum(step_weight[close_mask], grasp_step_weight)
        window_weight = max(window_weight, pregrasp_window_weight)
        window_weight = max(window_weight, grasp_window_weight)

    obj_z = obs_slice[:, 20]
    lift_delta = obj_z - obj_z[0]
    lift_mask = lift_delta >= lift_height_threshold
    if np.any(lift_mask):
        step_weight[lift_mask] = np.maximum(step_weight[lift_mask], lift_step_weight)
        window_weight = max(window_weight, lift_window_weight)

    return step_weight, np.float32(window_weight)


def get_episode_gripper(raw_action: np.ndarray) -> np.ndarray:
    if raw_action.shape[1] >= 9:
        return 0.5 * (raw_action[:, 7] + raw_action[:, 8])
    if raw_action.shape[1] == 8:
        return 0.5 * (raw_action[:, 6] + raw_action[:, 7])
    if raw_action.shape[1] == 7:
        return raw_action[:, 6]
    raise ValueError(f"Unsupported raw action dim for sampling: {raw_action.shape[1]}")


def find_grasp_lift_segments(raw_obs: np.ndarray, raw_action: np.ndarray) -> dict[str, int] | None:
    """Split a successful episode into grasp-before-lift and lift-after-z-rises segments."""
    n = min(len(raw_obs), len(raw_action))
    if n <= 1:
        return None

    obj_z = np.asarray(raw_obs[:n, 20], dtype=np.float32)
    lift_delta = obj_z - obj_z[0]
    success_candidates = np.flatnonzero(lift_delta >= config.friendly_lift_delta_threshold)
    if len(success_candidates) == 0:
        return None
    success_idx = int(success_candidates[0])

    lift_start_idx = success_idx
    start_delta = min(config.reconstruct_lift_start_delta, config.friendly_lift_delta_threshold)
    confirm_delta = min(config.reconstruct_lift_confirm_delta, config.friendly_lift_delta_threshold)
    confirm_steps = max(1, config.reconstruct_lift_confirm_steps)
    for idx in range(1, success_idx + 1):
        if lift_delta[idx] < start_delta:
            continue
        lookahead_end = min(n, idx + confirm_steps + 1)
        if float(np.max(lift_delta[idx:lookahead_end])) >= confirm_delta:
            lift_start_idx = idx
            break

    gripper = get_episode_gripper(np.asarray(raw_action[:n], dtype=np.float32))
    close_candidates = np.flatnonzero(gripper[: lift_start_idx + 1] <= config.grasp_close_threshold)
    if len(close_candidates) > 0:
        close_idx = int(close_candidates[0])
    else:
        close_idx = max(0, lift_start_idx - config.pregrasp_lookback)

    if config.reconstruct_grasp_mode == "lift_pre":
        grasp_start = max(0, lift_start_idx - config.reconstruct_grasp_pre_steps)
        grasp_end = lift_start_idx
    else:
        grasp_start = max(0, close_idx - config.reconstruct_grasp_pre_steps)
        grasp_end = min(lift_start_idx, close_idx + config.reconstruct_grasp_post_close_steps + 1)
        if grasp_end <= grasp_start:
            grasp_start = max(0, lift_start_idx - config.reconstruct_grasp_pre_steps)
            grasp_end = lift_start_idx
    if grasp_end <= grasp_start:
        grasp_end = min(n, grasp_start + 1)

    lift_end = min(n, success_idx + 1)
    if lift_end <= lift_start_idx:
        lift_end = min(n, lift_start_idx + 1)

    return {
        "grasp_start": int(grasp_start),
        "grasp_end": int(grasp_end),
        "lift_start": int(lift_start_idx),
        "lift_end": int(lift_end),
        "success_idx": int(success_idx),
        "close_idx": int(close_idx),
    }


def build_grasp_lift_phase_weight_targets(
    raw_obs: np.ndarray,
    raw_action: np.ndarray,
    local_start: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.float32]:
    step_weight = np.ones((chunk_size,), dtype=np.float32)
    window_weight = np.float32(1.0)
    if not config.reconstruct_grasp_lift_windows:
        return step_weight, window_weight

    segments = find_grasp_lift_segments(raw_obs, raw_action)
    if segments is None:
        return step_weight, window_weight

    chunk_indices = np.arange(local_start, local_start + chunk_size)
    grasp_mask = (chunk_indices >= segments["grasp_start"]) & (chunk_indices < segments["grasp_end"])
    lift_mask = (chunk_indices >= segments["lift_start"]) & (chunk_indices < segments["lift_end"])
    if np.any(grasp_mask):
        step_weight[grasp_mask] = np.maximum(step_weight[grasp_mask], config.grasp_crop_step_weight)
        window_weight = np.float32(max(float(window_weight), config.grasp_crop_window_weight))
    if np.any(lift_mask):
        step_weight[lift_mask] = np.maximum(step_weight[lift_mask], config.lift_crop_step_weight)
        window_weight = np.float32(max(float(window_weight), config.lift_crop_window_weight))
    return step_weight, window_weight


def has_lift_success(raw_obs: np.ndarray) -> bool:
    if len(raw_obs) == 0:
        return False
    obj_z = raw_obs[:, 20]
    return bool(np.max(obj_z - obj_z[0]) >= config.friendly_lift_delta_threshold)


def filter_lift_success_episodes(episodes):
    if not config.filter_successful_lift_episodes:
        return episodes

    cache_tag = (
        f"delta{config.friendly_lift_delta_threshold:g}"
        f"_trimr{int(config.trim_terminal_reset_actions)}"
        f"_trims{int(config.trim_after_success)}_keep{config.post_success_keep_steps}"
        f"_atol{config.terminal_reset_atol:g}"
    ).replace(".", "p")
    cache_path = config.data_root / f"lift_success_episode_flags_{cache_tag}.npz"
    flags = None
    if cache_path.exists() and os.environ.get("REBUILD_LIFT_SUCCESS_FLAGS", "0") != "1":
        cached = np.load(cache_path)
        if int(cached["episode_count"]) == len(episodes):
            flags = cached["flags"].astype(bool)
            print(f"[INFO] Loaded lift success cache: {cache_path}")

    if flags is None:
        flags_list = []
        for obs_path, _, n in tqdm(episodes, desc="Filtering lift-success episodes"):
            raw_obs = np.asarray(np.load(obs_path)[:n], dtype=np.float32)
            flags_list.append(has_lift_success(raw_obs))
        flags = np.asarray(flags_list, dtype=bool)
        np.savez(cache_path, flags=flags, episode_count=np.asarray(len(episodes), dtype=np.int64))
        print(f"[INFO] Saved lift success cache: {cache_path}")

    filtered = [episode for episode, keep in zip(episodes, flags) if keep]
    print(f"[INFO] Lift-success filter kept {len(filtered)}/{len(episodes)} episodes")
    return filtered


def compute_critical_window_sample_weights(episodes, chunk_size: int) -> np.ndarray:
    """Give grasp/pregrasp/lift windows more chances to appear in training batches."""
    cache_tag = (
        f"N{chunk_size}_close{config.grasp_close_threshold:g}_lift{config.lift_height_threshold:g}"
        f"_lookback{config.pregrasp_lookback}_pre{config.pregrasp_sample_weight:g}"
        f"_grasp{config.grasp_sample_weight:g}_liftw{config.lift_sample_weight:g}"
        f"_friendly{int(config.friendly_gripper_actions)}"
        f"_trimr{int(config.trim_terminal_reset_actions)}"
        f"_trims{int(config.trim_after_success)}_keep{config.post_success_keep_steps}"
        f"_recon{int(config.reconstruct_grasp_lift_windows)}"
        f"_mode{config.reconstruct_grasp_mode}"
        f"_ls{config.reconstruct_lift_start_delta:g}_lc{config.reconstruct_lift_confirm_delta:g}"
        f"_gp{config.reconstruct_grasp_pre_steps}_gpost{config.reconstruct_grasp_post_close_steps}"
        f"_gw{config.grasp_crop_sample_weight:g}_lw{config.lift_crop_sample_weight:g}"
    ).replace(".", "p")
    cache_path = config.data_root / f"critical_sampler_weights_{cache_tag}.npz"
    if cache_path.exists() and os.environ.get("REBUILD_SAMPLER_WEIGHTS", "0") != "1":
        cached = np.load(cache_path)
        sample_weights = cached["sample_weights"].astype(np.float32)
        if int(cached["episode_count"]) == len(episodes):
            print(f"[INFO] Loaded sampler weight cache: {cache_path}")
            return sample_weights

    weights = [None] * len(episodes)
    ordered_episodes = sorted(enumerate(episodes), key=lambda item: str(item[1][0]))
    for original_idx, (obs_path, act_path, n) in tqdm(ordered_episodes, desc="Computing sampler weights"):
        raw_obs = np.asarray(np.load(obs_path)[:n], dtype=np.float32)
        raw_act = np.asarray(np.load(act_path)[:n], dtype=np.float32)
        n = min(len(raw_obs), len(raw_act), n)
        if n < chunk_size:
            continue
        raw_obs = raw_obs[:n]
        raw_act = make_friendly_reach_lift_actions(raw_obs, raw_act[:n])

        step_priority = np.ones((n,), dtype=np.float32)
        gripper = get_episode_gripper(raw_act)
        close_mask = gripper <= config.grasp_close_threshold
        if np.any(close_mask):
            first_close_idx = int(np.flatnonzero(close_mask)[0])
            pregrasp_start = max(0, first_close_idx - config.pregrasp_lookback)
            step_priority[pregrasp_start : first_close_idx + 1] = np.maximum(
                step_priority[pregrasp_start : first_close_idx + 1],
                config.pregrasp_sample_weight,
            )
            step_priority[close_mask] = np.maximum(step_priority[close_mask], config.grasp_sample_weight)

        obj_z = raw_obs[:, 20]
        lift_mask = (obj_z - obj_z[0]) >= config.lift_height_threshold
        if np.any(lift_mask):
            step_priority[lift_mask] = np.maximum(step_priority[lift_mask], config.lift_sample_weight)

        if config.reconstruct_grasp_lift_windows:
            segments = find_grasp_lift_segments(raw_obs, raw_act)
            if segments is not None:
                step_priority[segments["grasp_start"] : segments["grasp_end"]] = np.maximum(
                    step_priority[segments["grasp_start"] : segments["grasp_end"]],
                    config.grasp_crop_sample_weight,
                )
                step_priority[segments["lift_start"] : segments["lift_end"]] = np.maximum(
                    step_priority[segments["lift_start"] : segments["lift_end"]],
                    config.lift_crop_sample_weight,
                )

        valid_len = n - chunk_size + 1
        episode_weights = np.empty((valid_len,), dtype=np.float32)
        for local_start in range(valid_len):
            episode_weights[local_start] = float(step_priority[local_start : local_start + chunk_size].max())
        weights[original_idx] = episode_weights

    if not weights:
        return np.ones((0,), dtype=np.float32)
    if any(weight is None for weight in weights):
        raise RuntimeError("Failed to compute all sampler weights")
    sample_weights = np.concatenate(weights).astype(np.float32)
    np.savez(
        cache_path,
        sample_weights=sample_weights,
        episode_count=np.asarray(len(episodes), dtype=np.int64),
    )
    print(f"[INFO] Saved sampler weight cache: {cache_path}")
    return sample_weights


def compute_window_weights(n: int, chunk_size: int) -> np.ndarray:
    t = np.arange(n)
    start_min = np.maximum(0, t - chunk_size + 1)
    start_max = np.minimum(t, n - chunk_size)
    w = start_max - start_min + 1
    return np.maximum(w, 0).astype(np.float64)


def infer_npy_rows_from_size(path: Path, cols: int) -> int | None:
    """Fast path for simple float32 2D .npy files with a 128-byte header."""
    payload_bytes = path.stat().st_size - 128
    row_bytes = cols * np.dtype(np.float32).itemsize
    if payload_bytes >= 0 and payload_bytes % row_bytes == 0:
        return payload_bytes // row_bytes
    return None


def build_episode_index(obs_dir: Path, act_dir: Path, chunk_size: int):
    cache_tag = f"raw_obs{config.raw_obs_dim}_act9"
    if config.trim_terminal_reset_actions or config.trim_after_success:
        cache_tag += (
            f"_trimr{int(config.trim_terminal_reset_actions)}"
            f"_trims{int(config.trim_after_success)}"
            f"_keep{config.post_success_keep_steps}"
            f"_delta{config.friendly_lift_delta_threshold:g}"
            f"_atol{config.terminal_reset_atol:g}"
        ).replace(".", "p")
    cache_path = config.data_root / f"episode_index_{cache_tag}.npz"
    if cache_path.exists() and os.environ.get("REBUILD_EPISODE_INDEX", "0") != "1":
        cached = np.load(cache_path, allow_pickle=True)
        obs_paths = cached["obs_paths"]
        act_paths = cached["act_paths"]
        lengths = cached["lengths"].astype(np.int64)
        episodes = [
            (Path(obs_path), Path(act_path), int(n))
            for obs_path, act_path, n in zip(obs_paths, act_paths, lengths)
            if int(n) >= chunk_size
        ]
        print(f"[INFO] Loaded episode index cache: {cache_path} episodes={len(episodes)}")
        return episodes

    obs_paths = sorted(obs_dir.glob("obs_ep_*.npy"))
    episodes = []
    for obs_path in tqdm(obs_paths, desc="Indexing episodes"):
        act_path = act_dir / obs_path.name.replace("obs_", "act_")
        if not act_path.exists():
            continue
        obs_n = infer_npy_rows_from_size(obs_path, config.raw_obs_dim)
        act_n = infer_npy_rows_from_size(act_path, 9)
        if obs_n is None:
            obs_n = len(np.load(obs_path, mmap_mode="r"))
        if act_n is None:
            act_n = len(np.load(act_path, mmap_mode="r"))
        n = min(int(obs_n), int(act_n))
        if config.trim_terminal_reset_actions or config.trim_after_success:
            raw_obs = np.load(obs_path, mmap_mode="r")
            raw_act = np.load(act_path, mmap_mode="r")
            n = compute_trimmed_length(raw_obs, raw_act, n)
        episodes.append((obs_path, act_path, n))

    np.savez(
        cache_path,
        obs_paths=np.asarray([str(obs_path) for obs_path, _, _ in episodes], dtype=object),
        act_paths=np.asarray([str(act_path) for _, act_path, _ in episodes], dtype=object),
        lengths=np.asarray([n for _, _, n in episodes], dtype=np.int64),
    )
    filtered = [(obs_path, act_path, n) for obs_path, act_path, n in episodes if n >= chunk_size]
    print(f"[INFO] Saved episode index cache: {cache_path} episodes={len(filtered)}")
    return filtered



def compute_stats(episodes, chunk_size: int):
    obs_sum = np.zeros((config.processed_obs_dim,), dtype=np.float64)
    obs_sq = np.zeros((config.processed_obs_dim,), dtype=np.float64)
    obs_count = 0

    act_sum = np.zeros((config.processed_action_dim,), dtype=np.float64)
    act_sq = np.zeros((config.processed_action_dim,), dtype=np.float64)
    act_weight_sum = 0.0

    for obs_path, act_path, n in tqdm(episodes, desc="Computing stats"):
        raw_obs = np.load(obs_path)
        raw_act = np.load(act_path)
        n = min(len(raw_obs), len(raw_act), int(n))
        n = compute_trimmed_length(raw_obs, raw_act, n)
        if n < chunk_size:
            continue
        raw_obs = raw_obs[:n]
        raw_act = make_friendly_reach_lift_actions(raw_obs, raw_act[:n])

        proc_prev_joint = build_prev_joint_history(raw_act)
        proc_obs = preprocess_reach_obs(raw_obs, proc_prev_joint)
        proc_obs = build_obs_history(proc_obs, config.obs_history_steps)
        proc_act = preprocess_reach_action(raw_act)

        valid_len = n - chunk_size + 1
        proc_obs = proc_obs[:valid_len]

        obs_sum += proc_obs.sum(axis=0)
        obs_sq += (proc_obs ** 2).sum(axis=0)
        obs_count += valid_len

        weights = compute_window_weights(n, chunk_size)[:, None]
        act_sum += (proc_act * weights).sum(axis=0)
        act_sq += ((proc_act ** 2) * weights).sum(axis=0)
        act_weight_sum += float(weights.sum())

    obs_mean = obs_sum / max(obs_count, 1)
    obs_var = obs_sq / max(obs_count, 1) - obs_mean ** 2
    obs_std = np.sqrt(np.maximum(obs_var, 1e-12)) + 1e-6

    act_mean = act_sum / max(act_weight_sum, 1)
    act_var = act_sq / max(act_weight_sum, 1) - act_mean ** 2
    act_std = np.sqrt(np.maximum(act_var, 1e-12)) + 1e-6

    return (
        obs_mean.astype(np.float32),
        obs_std.astype(np.float32),
        act_mean.astype(np.float32),
        act_std.astype(np.float32),
    )


class ReachChunkStreamingDataset(Dataset):
    def __init__(
        self,
        episodes,
        chunk_size: int,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_mean: np.ndarray,
        act_std: np.ndarray,
    ):
        self.episodes = episodes
        self.chunk_size = chunk_size
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = act_std.astype(np.float32)

        self.window_counts = np.asarray([n - chunk_size + 1 for _, _, n in episodes], dtype=np.int64)
        self.cum_windows = np.cumsum(self.window_counts)
        self.total_windows = int(self.cum_windows[-1]) if len(self.cum_windows) > 0 else 0

        self._cache_ep = None
        self._cache_obs_history = None
        self._cache_obs_raw = None
        self._cache_act_raw = None
        self._cache_act_processed = None
        self._episode_cache = None

        if config.cache_episodes:
            self._episode_cache = [
                self._preprocess_episode(obs_path, act_path, n, precompute_weights=True)
                for obs_path, act_path, n in tqdm(self.episodes, desc="Caching episodes")
            ]

    def __len__(self):
        return self.total_windows

    def _preprocess_episode(self, obs_path: Path, act_path: Path, n_hint: int, *, precompute_weights: bool):
        obs = np.load(obs_path)
        act = np.load(act_path)
        n = min(len(obs), len(act), n_hint)
        n = compute_trimmed_length(obs, act, n)
        obs = np.asarray(obs[:n], dtype=np.float32)
        obs = obs[:n]
        act = make_friendly_reach_lift_actions(obs, np.asarray(act[:n], dtype=np.float32))
        prev_joint = build_prev_joint_history(act)
        obs_history = build_obs_history(preprocess_reach_obs(obs, prev_joint), config.obs_history_steps)
        act_processed = preprocess_reach_action(act)

        if not precompute_weights:
            return obs_history, obs, act, act_processed

        valid_len = n - self.chunk_size + 1
        step_weights = np.empty((valid_len, self.chunk_size), dtype=np.float32)
        window_weights = np.empty((valid_len,), dtype=np.float32)
        for local_start in range(valid_len):
            step_weight, window_weight = build_chunk_weight_targets(
                obs[local_start : local_start + self.chunk_size],
                act[local_start : local_start + self.chunk_size],
                close_threshold=config.grasp_close_threshold,
                lift_height_threshold=config.lift_height_threshold,
                grasp_window_weight=config.grasp_window_weight,
                lift_window_weight=config.lift_window_weight,
                pregrasp_window_weight=config.pregrasp_window_weight,
                grasp_step_weight=config.grasp_step_weight,
                lift_step_weight=config.lift_step_weight,
                pregrasp_step_weight=config.pregrasp_step_weight,
                pregrasp_lookback=config.pregrasp_lookback,
            )
            phase_step_weight, phase_window_weight = build_grasp_lift_phase_weight_targets(
                obs,
                act,
                local_start,
                self.chunk_size,
            )
            step_weight = np.maximum(step_weight, phase_step_weight)
            window_weight = np.float32(max(float(window_weight), float(phase_window_weight)))
            step_weights[local_start] = step_weight
            window_weights[local_start] = window_weight

        return obs_history, act_processed, step_weights, window_weights

    def _load_episode(self, ep_idx: int):
        if self._episode_cache is not None:
            return self._episode_cache[ep_idx]
        if self._cache_ep == ep_idx:
            return self._cache_obs_history, self._cache_obs_raw, self._cache_act_raw, self._cache_act_processed
        obs_path, act_path, _ = self.episodes[ep_idx]
        obs_history, obs, act, act_processed = self._preprocess_episode(obs_path, act_path, self.episodes[ep_idx][2], precompute_weights=False)
        self._cache_ep = ep_idx
        self._cache_obs_history = obs_history
        self._cache_obs_raw = obs
        self._cache_act_raw = act
        self._cache_act_processed = act_processed
        return obs_history, obs, act, act_processed

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        ep_idx = int(np.searchsorted(self.cum_windows, idx, side="right"))
        prev = 0 if ep_idx == 0 else self.cum_windows[ep_idx - 1]
        local_start = int(idx - prev)

        episode_data = self._load_episode(ep_idx)
        if self._episode_cache is not None:
            obs_history, act_processed, step_weights, window_weights = episode_data
            step_weight = step_weights[local_start]
            window_weight = window_weights[local_start]
        else:
            obs_history, obs_raw, act_raw, act_processed = episode_data
            obs_chunk_raw = np.asarray(obs_raw[local_start : local_start + self.chunk_size])
            act_slice = np.asarray(act_raw[local_start : local_start + self.chunk_size])
            step_weight_1d, window_weight = build_chunk_weight_targets(
                obs_chunk_raw,
                act_slice,
                close_threshold=config.grasp_close_threshold,
                lift_height_threshold=config.lift_height_threshold,
                grasp_window_weight=config.grasp_window_weight,
                lift_window_weight=config.lift_window_weight,
                pregrasp_window_weight=config.pregrasp_window_weight,
                grasp_step_weight=config.grasp_step_weight,
                lift_step_weight=config.lift_step_weight,
                pregrasp_step_weight=config.pregrasp_step_weight,
                pregrasp_lookback=config.pregrasp_lookback,
            )
            phase_step_weight, phase_window_weight = build_grasp_lift_phase_weight_targets(
                obs_raw,
                act_raw,
                local_start,
                self.chunk_size,
            )
            step_weight_1d = np.maximum(step_weight_1d, phase_step_weight)
            window_weight = np.float32(max(float(window_weight), float(phase_window_weight)))
            step_weight = step_weight_1d

        obs = obs_history[local_start]
        act = act_processed[local_start : local_start + self.chunk_size]

        obs = (obs - self.obs_mean) / self.obs_std
        act = (act - self.act_mean) / self.act_std

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(act.astype(np.float32)),
            torch.from_numpy(step_weight.astype(np.float32)),
            torch.tensor(window_weight, dtype=torch.float32),
        )


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.copy_(self.shadow[name])


def compute_flow_target(action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(action_chunk)
    t = torch.rand(action_chunk.size(0), device=action_chunk.device)
    t_view = t.view(-1, 1, 1)
    x_t = t_view * action_chunk + (1.0 - t_view) * noise
    target_velocity = action_chunk - noise
    return x_t, t, target_velocity


def compute_flow_loss(
    pred: torch.Tensor,
    target_velocity: torch.Tensor,
    step_weight: torch.Tensor,
    window_weight: torch.Tensor,
) -> torch.Tensor:
    dim_weight = torch.ones((1, pred.shape[1], 1), device=pred.device, dtype=pred.dtype)
    gripper_dims = 2 if config.action_mode == "lr" else 1
    dim_weight[:, -gripper_dims:, :] = config.gripper_loss_weight
    loss_map = F.mse_loss(pred, target_velocity, reduction="none")
    loss_map = (loss_map * dim_weight).sum(dim=1) / dim_weight.sum(dim=1)
    return ((loss_map * step_weight).mean(dim=1) * window_weight).mean()


def compute_unweighted_velocity_mse(pred: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target_velocity)


def compute_gripper_velocity_mse(pred: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred[:, -1:, :], target_velocity[:, -1:, :])


def initialize_log_files(resume_checkpoint: str | None = None) -> None:
    if resume_checkpoint:
        with config.log_txt_path.open("a", encoding="utf-8") as f:
            f.write(f"resume_from={resume_checkpoint}\n")
        with config.log_md_path.open("a", encoding="utf-8") as f:
            f.write(f"\nResumed from `{resume_checkpoint}`.\n\n")
        return

    config.log_txt_path.write_text(f"Flow Matching {config.model_arch} training log\n", encoding="utf-8")
    config.log_md_path.write_text(
        f"# Flow Matching {config.model_arch} Training Log\n\n"
        "| Epoch | Train Weighted | Train MSE | Train Gripper MSE | Val Weighted | Val MSE | Val Gripper MSE | LR |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        encoding="utf-8",
    )


def append_epoch_log(
    epoch: int,
    train_loss: float,
    train_mse: float,
    train_gripper_mse: float,
    val_loss: float,
    val_mse: float,
    val_gripper_mse: float,
    learning_rate: float,
) -> None:
    with config.log_txt_path.open("a", encoding="utf-8") as f:
        f.write(
            " ".join(
                [
                    f"epoch={epoch:03d}",
                    f"train_weighted={train_loss:.6f}",
                    f"train_mse={train_mse:.6f}",
                    f"train_gripper_mse={train_gripper_mse:.6f}",
                    f"val_weighted={val_loss:.6f}",
                    f"val_mse={val_mse:.6f}",
                    f"val_gripper_mse={val_gripper_mse:.6f}",
                    f"lr={learning_rate:.8f}",
                ]
            )
            + "\n"
        )

    with config.log_md_path.open("a", encoding="utf-8") as f:
        f.write(
            f"| {epoch:03d} | {train_loss:.6f} | {train_mse:.6f} | {train_gripper_mse:.6f} | "
            f"{val_loss:.6f} | {val_mse:.6f} | {val_gripper_mse:.6f} | {learning_rate:.8f} |\n"
        )


def append_final_log(final_val_loss: float, final_val_mse: float, final_val_gripper_mse: float, best_val: float) -> None:
    with config.log_txt_path.open("a", encoding="utf-8") as f:
        f.write(f"final_val_loss={final_val_loss:.6f}\n")
        f.write(f"final_val_mse={final_val_mse:.6f}\n")
        f.write(f"final_val_gripper_mse={final_val_gripper_mse:.6f}\n")
        f.write(f"best_val={best_val:.6f}\n")

    with config.log_md_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"- final_val_loss: {final_val_loss:.6f}\n")
        f.write(f"- final_val_mse: {final_val_mse:.6f}\n")
        f.write(f"- final_val_gripper_mse: {final_val_gripper_mse:.6f}\n")
        f.write(f"- best_val: {best_val:.6f}\n")


def save_checkpoint(path: Path, best_val: float, epoch: int) -> None:
    config_dict = {
        name: getattr(config, name)
        for name in dir(config)
        if not name.startswith("_") and not callable(getattr(config, name))
    }
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict() if ema_model is not None else None,
            "config": config_dict,
            "model_variant": (
                "reach_lift_flow_joint_hist_lr_balanced_v6"
                if config.action_mode == "lr"
                else "reach_lift_flow_joint_hist_matched_balanced_v5"
            ),
            "best_val": best_val,
        },
        path,
    )


def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, float, float]:
    model.eval()
    weighted_losses = []
    velocity_mse_losses = []
    gripper_mse_losses = []
    with torch.no_grad():
        for obs, action_chunk, step_weight, window_weight in loader:
            obs = obs.to(device)
            action_chunk = action_chunk.to(device).permute(0, 2, 1)
            step_weight = step_weight.to(device)
            window_weight = window_weight.to(device)

            x_t, t, target_velocity = compute_flow_target(action_chunk)
            pred_velocity = model(x_t, t, obs)
            weighted_losses.append(compute_flow_loss(pred_velocity, target_velocity, step_weight, window_weight).item())
            velocity_mse_losses.append(compute_unweighted_velocity_mse(pred_velocity, target_velocity).item())
            gripper_mse_losses.append(compute_gripper_velocity_mse(pred_velocity, target_velocity).item())
    if not weighted_losses:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(weighted_losses)),
        float(np.mean(velocity_mse_losses)),
        float(np.mean(gripper_mse_losses)),
    )


def build_lr_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_epochs = max(1, min(config.lr_warmup_epochs, total_epochs))
    min_scale = config.min_learning_rate_scale

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_scale + (1.0 - min_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def sample_action_chunk(model: nn.Module, obs_batch: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
    model.eval()
    num_steps = config.flow_steps if num_steps is None else num_steps
    if config.inference_noise_mode == "zeros":
        x = torch.zeros(obs_batch.size(0), config.processed_action_dim, config.chunk_size, device=device)
    else:
        x = torch.randn(obs_batch.size(0), config.processed_action_dim, config.chunk_size, device=device)
    dt = 1.0 / float(num_steps)

    for i in range(num_steps):
        t = torch.full((obs_batch.size(0),), i * dt, device=device)
        velocity = model(x, t, obs_batch)
        x = x + velocity * dt

    return x.permute(0, 2, 1)


def main() -> None:
    global dataset, model, ema_helper, ema_model, optimizer

    episodes = filter_lift_success_episodes(build_episode_index(config.obs_dir, config.act_dir, config.chunk_size))

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(episodes))
    train_count = int(len(episodes) * 0.95)
    train_episodes = [episodes[i] for i in perm[:train_count]]
    val_episodes = [episodes[i] for i in perm[train_count:]]

    if config.stats_path.exists() and os.environ.get("RECOMPUTE_STATS", "0") != "1":
        stats = np.load(config.stats_path)
        obs_mean = stats["obs_mean"].astype(np.float32)
        obs_std = stats["obs_std"].astype(np.float32)
        act_mean = stats["act_mean"].astype(np.float32)
        act_std = stats["act_std"].astype(np.float32)
    else:
        reuse_stats_path = os.environ.get("REUSE_STATS_PATH")
        if reuse_stats_path:
            reuse_stats = np.load(Path(reuse_stats_path).expanduser())
            obs_mean = reuse_stats["obs_mean"].astype(np.float32)
            obs_std = reuse_stats["obs_std"].astype(np.float32)
            act_mean = reuse_stats["act_mean"].astype(np.float32)
            act_std = reuse_stats["act_std"].astype(np.float32)
        else:
            stats_episodes = train_episodes
            if 0 < config.max_stats_episodes < len(train_episodes):
                stats_indices = rng.choice(len(train_episodes), size=config.max_stats_episodes, replace=False)
                stats_episodes = [train_episodes[i] for i in stats_indices]
                print(f"[INFO] Limited stats episodes to {len(stats_episodes)}/{len(train_episodes)}")
            obs_mean, obs_std, act_mean, act_std = compute_stats(stats_episodes, config.chunk_size)
        np.savez(config.stats_path, obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std)

    train_dataset = ReachChunkStreamingDataset(
        train_episodes,
        config.chunk_size,
        obs_mean,
        obs_std,
        act_mean,
        act_std,
    )
    val_dataset = ReachChunkStreamingDataset(
        val_episodes,
        config.chunk_size,
        obs_mean,
        obs_std,
        act_mean,
        act_std,
    )

    subset_rng = np.random.default_rng(123)
    train_sampler = None
    train_shuffle = True
    if config.oversample_critical_windows:
        sample_weights = compute_critical_window_sample_weights(train_episodes, config.chunk_size)
        if len(sample_weights) != len(train_dataset):
            raise RuntimeError(
                f"Sampler weight count mismatch: weights={len(sample_weights)} dataset={len(train_dataset)}"
            )
        train_samples_per_epoch = (
            config.max_train_windows_per_epoch
            if 0 < config.max_train_windows_per_epoch < len(train_dataset)
            else len(train_dataset)
        )
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=train_samples_per_epoch,
            replacement=True,
            generator=torch.Generator().manual_seed(123),
        )
        train_shuffle = False
        print(
            "[INFO] Oversampling critical windows "
            f"critical={(sample_weights > 1.0).sum()}/{len(sample_weights)} "
            f"samples_per_epoch={train_samples_per_epoch} "
            f"pregrasp={config.pregrasp_sample_weight} grasp={config.grasp_sample_weight} lift={config.lift_sample_weight}"
        )
    elif 0 < config.max_train_windows_per_epoch < len(train_dataset):
        indices = subset_rng.choice(len(train_dataset), size=config.max_train_windows_per_epoch, replace=False)
        train_dataset = Subset(train_dataset, np.sort(indices).tolist())
        train_shuffle = False
        print(f"[INFO] Limited train windows per epoch to {config.max_train_windows_per_epoch}")
    if 0 < config.max_val_windows < len(val_dataset):
        indices = subset_rng.choice(len(val_dataset), size=config.max_val_windows, replace=False)
        val_dataset = Subset(val_dataset, indices.tolist())
        print(f"[INFO] Limited validation windows to {config.max_val_windows}")

    dataset = train_dataset

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=train_shuffle, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    if config.model_arch == "transformer":
        if ConditionalTransformer1D is None:
            raise RuntimeError("Transformer model code was removed. Use MODEL_ARCH=stanford_dp for U-Net training.")
        model = ConditionalTransformer1D(
            action_dim=config.processed_action_dim,
            obs_dim=config.processed_obs_dim,
            horizon=config.chunk_size,
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            num_layers=config.transformer_layers,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.transformer_dropout,
        ).to(device)
    else:
        model = DiffusionPolicyConditionalUnet1D(
            input_dim=config.processed_action_dim,
            global_cond_dim=config.processed_obs_dim,
            diffusion_step_embed_dim=config.time_embed_dim,
            down_dims=tuple(config.down_dims),
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale,
        ).to(device)
    ema_helper = EMA(model, config.ema_decay) if config.use_ema else None
    ema_model = copy.deepcopy(model) if config.use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    lr_scheduler = build_lr_scheduler(optimizer, config.num_epochs) if config.use_lr_scheduler else None

    best_val = float("inf")
    start_epoch = 1
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT")
    if resume_checkpoint:
        resume_path = Path(resume_checkpoint).expanduser()
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if ema_model is not None and checkpoint.get("ema_state_dict") is not None:
            ema_model.load_state_dict(checkpoint["ema_state_dict"])
        best_val = float(checkpoint.get("best_val", best_val))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[INFO] Resumed from {resume_path} at epoch {start_epoch} best_val={best_val:.6f}")

    initialize_log_files(resume_checkpoint)

    for epoch in range(start_epoch, config.num_epochs + 1):
        model.train()
        train_losses = []
        train_velocity_mse_losses = []
        train_gripper_mse_losses = []
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.num_epochs}",
            disable=os.environ.get("TQDM_DISABLE", "0") == "1",
        )

        for batch_idx, (obs, action_chunk, step_weight, window_weight) in enumerate(pbar, start=1):
            obs = obs.to(device)
            action_chunk = action_chunk.to(device).permute(0, 2, 1)
            step_weight = step_weight.to(device)
            window_weight = window_weight.to(device)

            x_t, t, target_velocity = compute_flow_target(action_chunk)
            pred_velocity = model(x_t, t, obs)
            loss = compute_flow_loss(pred_velocity, target_velocity, step_weight, window_weight)
            velocity_mse = compute_unweighted_velocity_mse(pred_velocity, target_velocity)
            gripper_mse = compute_gripper_velocity_mse(pred_velocity, target_velocity)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema_helper is not None:
                ema_helper.update(model)
                ema_helper.copy_to(ema_model)

            train_losses.append(loss.item())
            train_velocity_mse_losses.append(velocity_mse.item())
            train_gripper_mse_losses.append(gripper_mse.item())
            pbar.set_postfix(
                train_weighted=f"{np.mean(train_losses):.6f}",
                train_mse=f"{np.mean(train_velocity_mse_losses):.6f}",
                train_gripper=f"{np.mean(train_gripper_mse_losses):.6f}",
            )
            if config.batch_log_every > 0 and batch_idx % config.batch_log_every == 0:
                progress_msg = (
                    f"epoch={epoch:03d} batch={batch_idx}/{len(train_loader)} "
                    f"train_weighted_running={float(np.mean(train_losses)):.6f} "
                    f"train_mse_running={float(np.mean(train_velocity_mse_losses)):.6f} "
                    f"train_gripper_mse_running={float(np.mean(train_gripper_mse_losses)):.6f}"
                )
                print(progress_msg)
                with config.log_txt_path.open("a", encoding="utf-8") as f:
                    f.write(progress_msg + "\n")

        eval_model = ema_model if ema_model is not None else model
        val_loss, val_mse, val_gripper_mse = evaluate(eval_model, val_loader)
        epoch_train_loss = float(np.mean(train_losses))
        epoch_train_mse = float(np.mean(train_velocity_mse_losses))
        epoch_train_gripper_mse = float(np.mean(train_gripper_mse_losses))
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"epoch={epoch:03d} "
            f"train_weighted={epoch_train_loss:.6f} train_mse={epoch_train_mse:.6f} "
            f"train_gripper_mse={epoch_train_gripper_mse:.6f} "
            f"val_weighted={val_loss:.6f} val_mse={val_mse:.6f} val_gripper_mse={val_gripper_mse:.6f} "
            f"lr={current_lr:.8f}"
        )
        append_epoch_log(
            epoch,
            epoch_train_loss,
            epoch_train_mse,
            epoch_train_gripper_mse,
            val_loss,
            val_mse,
            val_gripper_mse,
            current_lr,
        )

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(config.checkpoint_path, best_val, epoch)

        if epoch % config.checkpoint_save_every == 0:
            periodic_checkpoint_path = config.save_dir / (
                f"reach_lift_flow_match_transformer_epoch_{epoch:03d}.pt"
                if config.model_arch == "transformer"
                else f"reach_lift_flow_match_matched_dp_backbone_epoch_{epoch:03d}.pt"
            )
            save_checkpoint(periodic_checkpoint_path, best_val, epoch)

        if lr_scheduler is not None:
            lr_scheduler.step()

    final_eval_model = ema_model if ema_model is not None else model
    final_val_loss, final_val_mse, final_val_gripper_mse = evaluate(final_eval_model, val_loader)
    print(
        f"final_val_loss={final_val_loss:.6f} "
        f"final_val_mse={final_val_mse:.6f} "
        f"final_val_gripper_mse={final_val_gripper_mse:.6f}"
    )
    append_final_log(final_val_loss, final_val_mse, final_val_gripper_mse, best_val)

    batch_obs, _, _, _ = next(iter(val_loader))
    batch_obs = batch_obs[:4].to(device)
    pred_chunks = sample_action_chunk(final_eval_model, batch_obs)
    act_mean_t = torch.from_numpy(act_mean).to(device)
    act_std_t = torch.from_numpy(act_std).to(device)
    pred_chunks_denorm = pred_chunks * act_std_t + act_mean_t
    print(f"preview_pred_shape={tuple(pred_chunks_denorm.shape)}")


if __name__ == "__main__":
    main()
