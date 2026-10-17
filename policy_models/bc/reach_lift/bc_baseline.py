"""
Behavior Cloning baseline for the PPO trajectory datasets.

This file contains everything needed for a fair BC baseline:
  - dataset indexing and DP/FMP-compatible preprocessing
  - a deterministic action-chunk U-Net policy using the DP/FMP backbone
  - training with plain/weighted MSE
  - Isaac python-policy wrapper via load_policy(...)
  - an eval command helper that launches imitation_play_fixed_left_right.py

Typical training:
PYTHONPATH=$PWD \
CHUNK_SIZE=8 \
NUM_EPOCHS=150 \
BATCH_SIZE=8192 \
ACTION_MODE=mean \
TRAIN_DEVICE=cuda:0 \
SAVE_DIR_OVERRIDE=./policy_models/bc/reach_lift/reach_lift_bc_unet_8chunk \
CHECKPOINT_FILENAME=reach_lift_bc_unet.pt \
STATS_FILENAME=reach_lift_bc_stats.npz \
python ./policy_models/bc/reach_lift/bc_baseline.py train

Typical evaluation:
TERM=xterm PYTHONPATH=$PWD isaaclab_4 -p ./launch/imitation_play_fixed_left_right.py \
  --task reach_lift_imitation \
  --policy_type python \
  --policy_path ./policy_models/bc/reach_lift/bc_baseline.py \
  --policy_checkpoint ./policy_models/bc/reach_lift/reach_lift_bc_unet_8chunk/reach_lift_bc_unet.pt \
  --policy_stats ./policy_models/bc/reach_lift/reach_lift_bc_unet_8chunk/reach_lift_bc_stats.npz \
  --num_envs 100 \
  --output_horizon 8 \
  --execute_horizon 8 \
  --episode_length_s 50 \
  --success_z 1.14 \
  --episodes 100 \
  --seed 50 \
  --device cuda:0 \
  --disable_predicted_path_viz \
  --headless
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import subprocess
import sys
import time
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm exists in the project env.
    tqdm = None

# Isaac's importlib loader executes python policy files without inserting the
# module into sys.modules first. dataclasses expects it to exist.
if __name__ not in sys.modules:
    _module = types.ModuleType(__name__)
    _module.__dict__.update(globals())
    sys.modules[__name__] = _module

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parents[2]
DIFFUSION_UNET_PATH = REPO_ROOT / "policy_models" / "diffusion" / "reach_lift" / "diffusion_policy_unet.py"
DEFAULT_INITIAL_ACTION_9 = np.array(
    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025, 0.025, 0.025],
    dtype=np.float32,
)


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0") == "1"


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    dims = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not dims:
        raise ValueError("HIDDEN_DIMS must contain at least one layer size.")
    return dims


def parse_int_tuple(text: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise ValueError("Expected a comma-separated integer tuple.")
    return values


def load_diffusion_unet_class():
    spec = importlib.util.spec_from_file_location("bc_diffusion_policy_unet", DIFFUSION_UNET_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load DP/FMP U-Net backbone from {DIFFUSION_UNET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DiffusionPolicyConditionalUnet1D


DiffusionPolicyConditionalUnet1D = load_diffusion_unet_class()


@dataclass
class Config:
    task_family: str = os.environ.get("TASK_FAMILY", "reach_lift")
    data_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "DATA_ROOT",
                str(
                    REPO_ROOT
                    / "datas"
                    / ("reach_data1" if os.environ.get("TASK_FAMILY", "reach_lift") == "reach" else "reach_lift_data1")
                ),
            )
        )
    )
    action_mode: str = os.environ.get("ACTION_MODE", "mean")
    chunk_size: int = env_int("CHUNK_SIZE", 8)
    obs_history_steps: int = field(
        default_factory=lambda: env_int(
            "OBS_HISTORY_STEPS",
            1 if os.environ.get("TASK_FAMILY", "reach_lift") == "reach" else 4,
        )
    )

    batch_size: int = env_int("BATCH_SIZE", 8192)
    num_epochs: int = env_int("NUM_EPOCHS", 150)
    learning_rate: float = env_float("LEARNING_RATE", 2e-4)
    weight_decay: float = env_float("WEIGHT_DECAY", 1e-6)
    num_workers: int = env_int("NUM_WORKERS", 16)
    train_device: str = os.environ.get(
        "TRAIN_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    seed: int = env_int("SEED", 42)
    batch_log_every: int = env_int("BATCH_LOG_EVERY", 50)
    checkpoint_save_every: int = env_int("CHECKPOINT_SAVE_EVERY", 10)
    max_train_windows_per_epoch: int = env_int("MAX_TRAIN_WINDOWS_PER_EPOCH", 0)
    max_val_windows: int = env_int("MAX_VAL_WINDOWS", 100000)
    max_stats_episodes: int = env_int("MAX_STATS_EPISODES", 0)
    cache_episodes: bool = env_bool("CACHE_EPISODES", False)
    rebuild_episode_index: bool = env_bool("REBUILD_EPISODE_INDEX", False)
    recompute_stats: bool = env_bool("RECOMPUTE_STATS", False)
    reuse_stats_path: str = os.environ.get("REUSE_STATS_PATH", "")

    model_arch: str = os.environ.get("MODEL_ARCH", "unet")
    down_dims: tuple[int, ...] = field(
        default_factory=lambda: parse_int_tuple(
            os.environ.get(
                "DOWN_DIMS",
                "192,384,768" if os.environ.get("TASK_FAMILY", "reach_lift") == "reach_lift" else "128,256,512",
            )
        )
    )
    time_embed_dim: int = env_int(
        "TIME_EMBED_DIM",
        384 if os.environ.get("TASK_FAMILY", "reach_lift") == "reach_lift" else 256,
    )
    kernel_size: int = env_int("KERNEL_SIZE", 5)
    n_groups: int = env_int("N_GROUPS", 8)
    cond_predict_scale: bool = env_bool("COND_PREDICT_SCALE", True)

    use_loss_weights: bool = env_bool("USE_LOSS_WEIGHTS", True)
    oversample_critical_windows: bool = env_bool("OVERSAMPLE_CRITICAL_WINDOWS", True)
    grasp_close_threshold: float = env_float("GRASP_CLOSE_THRESHOLD", 0.013)
    lift_height_threshold: float = env_float("LIFT_HEIGHT_THRESHOLD", 0.02)
    pregrasp_lookback: int = env_int("PREGRASP_LOOKBACK", 12)
    grasp_window_weight: float = env_float("GRASP_WINDOW_WEIGHT", 8.0)
    lift_window_weight: float = env_float("LIFT_WINDOW_WEIGHT", 12.0)
    pregrasp_window_weight: float = env_float("PREGRASP_WINDOW_WEIGHT", 16.0)
    grasp_step_weight: float = env_float("GRASP_STEP_WEIGHT", 10.0)
    lift_step_weight: float = env_float("LIFT_STEP_WEIGHT", 14.0)
    pregrasp_step_weight: float = env_float("PREGRASP_STEP_WEIGHT", 20.0)
    gripper_loss_weight: float = env_float("GRIPPER_LOSS_WEIGHT", 8.0)
    grasp_sample_weight: float = env_float("GRASP_SAMPLE_WEIGHT", 5.0)
    lift_sample_weight: float = env_float("LIFT_SAMPLE_WEIGHT", 10.0)
    pregrasp_sample_weight: float = env_float("PREGRASP_SAMPLE_WEIGHT", 8.0)

    trim_terminal_reset_actions: bool = env_bool("TRIM_TERMINAL_RESET_ACTIONS", False)
    terminal_reset_atol: float = env_float("TERMINAL_RESET_ATOL", 1e-4)
    trim_after_success: bool = env_bool("TRIM_AFTER_SUCCESS", False)
    post_success_keep_steps: int = env_int("POST_SUCCESS_KEEP_STEPS", 0)
    success_z_for_trim: float = env_float("SUCCESS_Z_FOR_TRIM", 1.14)

    hidden_dims: tuple[int, ...] = field(
        default_factory=lambda: parse_hidden_dims(os.environ.get("HIDDEN_DIMS", "1024,1024,512"))
    )
    dropout: float = env_float("DROPOUT", 0.05)
    grad_clip_norm: float = env_float("GRAD_CLIP_NORM", 1.0)
    use_layer_norm: bool = env_bool("USE_LAYER_NORM", True)

    @property
    def obs_dir(self) -> Path:
        obs_dir = self.data_root / "obs"
        if obs_dir.exists():
            return obs_dir
        if any(self.data_root.glob("obs_ep_*.npy")):
            return self.data_root
        return obs_dir

    @property
    def act_dir(self) -> Path:
        act_dir = self.data_root / "actions"
        if act_dir.exists():
            return act_dir
        if any(self.data_root.glob("act_ep_*.npy")):
            return self.data_root
        return act_dir

    @property
    def raw_obs_dim(self) -> int:
        if self.task_family == "reach_lift":
            return 35
        if self.task_family == "reach":
            return 25
        raise ValueError("TASK_FAMILY must be reach_lift or reach.")

    @property
    def processed_action_dim(self) -> int:
        return 8 if self.action_mode == "lr" else 7

    @property
    def processed_obs_frame_dim(self) -> int:
        if self.task_family == "reach_lift":
            return 34 if self.action_mode == "lr" else 31
        return 31 if self.action_mode == "lr" else 28

    @property
    def processed_obs_dim(self) -> int:
        return self.processed_obs_frame_dim * self.obs_history_steps

    @property
    def save_dir(self) -> Path:
        default_name = f"{self.task_family}_bc_{self.model_arch}_{self.chunk_size}chunk"
        return Path(os.environ.get("SAVE_DIR_OVERRIDE", str(FILE_DIR / default_name)))

    @property
    def checkpoint_path(self) -> Path:
        return self.save_dir / os.environ.get("CHECKPOINT_FILENAME", f"{self.task_family}_bc_{self.model_arch}.pt")

    @property
    def stats_path(self) -> Path:
        return self.save_dir / os.environ.get("STATS_FILENAME", f"{self.task_family}_bc_stats.npz")

    @property
    def log_txt_path(self) -> Path:
        return self.save_dir / "train_log.txt"

    @property
    def log_md_path(self) -> Path:
        return self.save_dir / "train_log.md"

def progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)

def initial_action(action_mode: str) -> np.ndarray:
    if action_mode == "lr":
        return np.concatenate([DEFAULT_INITIAL_ACTION_9[:6], DEFAULT_INITIAL_ACTION_9[7:9]]).astype(np.float32)
    return np.concatenate([DEFAULT_INITIAL_ACTION_9[:6], DEFAULT_INITIAL_ACTION_9[7:9].mean(keepdims=True)]).astype(np.float32)

def read_npy_rows(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])

def compress_robot_state(raw_state: np.ndarray, action_mode: str) -> np.ndarray:
    if raw_state.shape[1] < 9:
        raise ValueError(f"Expected robot state dim >= 9, got {raw_state.shape[1]}")
    if action_mode == "lr":
        return np.concatenate([raw_state[:, :6], raw_state[:, 7:9]], axis=1).astype(np.float32)
    return np.concatenate([raw_state[:, :6], raw_state[:, 7:9].mean(axis=1, keepdims=True)], axis=1).astype(np.float32)

def preprocess_action(raw_action: np.ndarray, action_mode: str) -> np.ndarray:
    raw_action = np.asarray(raw_action, dtype=np.float32)
    target_dim = 8 if action_mode == "lr" else 7
    if raw_action.shape[1] == target_dim:
        return raw_action.astype(np.float32)
    if raw_action.shape[1] >= 9:
        return compress_robot_state(raw_action, action_mode)
    raise ValueError(f"Unsupported raw action dim: {raw_action.shape[1]}")

def build_prev_action_history(raw_action: np.ndarray, action_mode: str) -> np.ndarray:
    processed = preprocess_action(raw_action, action_mode)
    history = np.empty_like(processed, dtype=np.float32)
    history[0] = initial_action(action_mode)
    if len(processed) > 1:
        history[1:] = processed[:-1]
    return history

def preprocess_obs(raw_obs: np.ndarray, prev_action: np.ndarray, cfg: Config) -> np.ndarray:
    if raw_obs.shape[0] != prev_action.shape[0]:
        n = min(raw_obs.shape[0], prev_action.shape[0])
        raw_obs = raw_obs[:n]
        prev_action = prev_action[:n]

    joint_pos = raw_obs[:, 0:9]
    joint_vel = raw_obs[:, 9:18]
    joint_pos_processed = compress_robot_state(joint_pos, cfg.action_mode)
    joint_vel_processed = compress_robot_state(joint_vel, cfg.action_mode)

    if cfg.task_family == "reach_lift":
        if raw_obs.shape[1] < 28:
            raise ValueError(f"reach_lift obs requires dim >= 28, got {raw_obs.shape[1]}")
        object_pos = raw_obs[:, 18:21]
        pose_command = raw_obs[:, 21:28]
        parts = [joint_pos_processed, joint_vel_processed, object_pos, pose_command, prev_action]
    else:
        if raw_obs.shape[1] < 25:
            raise ValueError(f"reach obs requires dim >= 25, got {raw_obs.shape[1]}")
        pose_command = raw_obs[:, 18:25]
        parts = [joint_pos_processed, joint_vel_processed, pose_command, prev_action]

    return np.concatenate(parts, axis=1).astype(np.float32)

def build_obs_history(processed_obs: np.ndarray, history_steps: int) -> np.ndarray:
    if history_steps <= 1:
        return processed_obs.astype(np.float32)
    num_steps, feat_dim = processed_obs.shape
    history = np.empty((num_steps, history_steps, feat_dim), dtype=np.float32)
    for step in range(num_steps):
        start = max(0, step - history_steps + 1)
        window = processed_obs[start : step + 1]
        history[step, -len(window) :] = window
        if len(window) < history_steps:
            history[step, : history_steps - len(window)] = window[0]
    return history.reshape(num_steps, history_steps * feat_dim).astype(np.float32)

def compute_trimmed_length(raw_obs: np.ndarray, raw_action: np.ndarray, n_hint: int, cfg: Config) -> int:
    n = min(int(n_hint), len(raw_obs), len(raw_action))
    if n <= 0:
        return 0

    if cfg.trim_terminal_reset_actions and n > 2:
        init = DEFAULT_INITIAL_ACTION_9[: raw_action.shape[1]]
        while n > 1 and np.allclose(raw_action[n - 1], init, atol=cfg.terminal_reset_atol):
            n -= 1

    if cfg.trim_after_success and cfg.task_family == "reach_lift" and raw_obs.shape[1] >= 21:
        heights = raw_obs[:n, 20]
        success_indices = np.flatnonzero(heights >= cfg.success_z_for_trim)
        if len(success_indices) > 0:
            n = min(n, int(success_indices[0]) + 1 + cfg.post_success_keep_steps)

    return n

def build_phase_masks(obs: np.ndarray, action: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(obs), len(action))
    pregrasp = np.zeros((n,), dtype=bool)
    grasp = np.zeros((n,), dtype=bool)
    lift = np.zeros((n,), dtype=bool)
    if n == 0 or cfg.task_family != "reach_lift" or obs.shape[1] < 21:
        return pregrasp, grasp, lift

    obj_z = obs[:n, 20]
    z_delta = obj_z - float(obj_z[0])
    lift = z_delta >= cfg.lift_height_threshold

    processed_action = preprocess_action(action[:n], cfg.action_mode)
    grip = processed_action[:, -1]
    grasp = grip <= cfg.grasp_close_threshold

    close_indices = np.flatnonzero(grasp)
    if len(close_indices) > 0:
        first_close = int(close_indices[0])
        start = max(0, first_close - cfg.pregrasp_lookback)
        pregrasp[start:first_close] = True
    return pregrasp, grasp, lift

def build_chunk_weights(obs_chunk: np.ndarray, act_chunk: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.float32]:
    step_weight = np.ones((len(obs_chunk),), dtype=np.float32)
    window_weight = np.float32(1.0)
    pregrasp, grasp, lift = build_phase_masks(obs_chunk, act_chunk, cfg)

    if pregrasp.any():
        step_weight[pregrasp] = np.maximum(step_weight[pregrasp], cfg.pregrasp_step_weight)
        window_weight = np.float32(max(float(window_weight), cfg.pregrasp_window_weight))
    if grasp.any():
        step_weight[grasp] = np.maximum(step_weight[grasp], cfg.grasp_step_weight)
        window_weight = np.float32(max(float(window_weight), cfg.grasp_window_weight))
    if lift.any():
        step_weight[lift] = np.maximum(step_weight[lift], cfg.lift_step_weight)
        window_weight = np.float32(max(float(window_weight), cfg.lift_window_weight))
    return step_weight, window_weight

def build_episode_index(cfg: Config):
    cache_tag = (
        f"bc_{cfg.task_family}_raw_obs{cfg.raw_obs_dim}_act9_N{cfg.chunk_size}"
        f"_trimr{int(cfg.trim_terminal_reset_actions)}_trims{int(cfg.trim_after_success)}"
    )
    cache_path = cfg.data_root / f"episode_index_{cache_tag}.npz"
    if cache_path.exists() and not cfg.rebuild_episode_index:
        cached = np.load(cache_path, allow_pickle=True)
        episodes = [
            (Path(obs_path), Path(act_path), int(n))
            for obs_path, act_path, n in zip(cached["obs_paths"], cached["act_paths"], cached["lengths"])
            if int(n) >= cfg.chunk_size
        ]
        missing_paths = any((not obs_path.exists()) or (not act_path.exists()) for obs_path, act_path, _ in episodes[:32])
        if episodes and not missing_paths:
            print(f"[INFO] Loaded episode index cache: {cache_path} episodes={len(episodes)}")
            return episodes
        reason = "empty" if not episodes else "stale paths"
        print(f"[WARN] Ignoring {reason} episode index cache: {cache_path}")

    obs_paths = sorted(cfg.obs_dir.glob("obs_ep_*.npy"))
    episodes = []
    for obs_path in progress(obs_paths, desc="Indexing episodes"):
        act_path = cfg.act_dir / obs_path.name.replace("obs_", "act_")
        if not act_path.exists():
            continue
        obs_n = read_npy_rows(obs_path)
        act_n = read_npy_rows(act_path)
        n = min(obs_n, act_n)
        if cfg.trim_terminal_reset_actions or cfg.trim_after_success:
            raw_obs = np.load(obs_path, mmap_mode="r")
            raw_act = np.load(act_path, mmap_mode="r")
            n = compute_trimmed_length(raw_obs, raw_act, n, cfg)
        if n >= cfg.chunk_size:
            episodes.append((obs_path, act_path, int(n)))

    np.savez(
        cache_path,
        obs_paths=np.asarray([str(x[0]) for x in episodes], dtype=object),
        act_paths=np.asarray([str(x[1]) for x in episodes], dtype=object),
        lengths=np.asarray([x[2] for x in episodes], dtype=np.int64),
    )
    print(f"[INFO] Saved episode index cache: {cache_path} episodes={len(episodes)}")
    return episodes


def compute_stats(episodes, cfg: Config):
    obs_sum = np.zeros((cfg.processed_obs_dim,), dtype=np.float64)
    obs_sq = np.zeros((cfg.processed_obs_dim,), dtype=np.float64)
    obs_count = 0
    act_sum = np.zeros((cfg.processed_action_dim,), dtype=np.float64)
    act_sq = np.zeros((cfg.processed_action_dim,), dtype=np.float64)
    act_count = 0

    for obs_path, act_path, n_hint in progress(episodes, desc="Computing stats"):
        raw_obs = np.load(obs_path)
        raw_act = np.load(act_path)
        n = compute_trimmed_length(raw_obs, raw_act, n_hint, cfg)
        if n < cfg.chunk_size:
            continue
        raw_obs = np.asarray(raw_obs[:n], dtype=np.float32)
        raw_act = np.asarray(raw_act[:n], dtype=np.float32)
        prev_action = build_prev_action_history(raw_act, cfg.action_mode)
        obs_history = build_obs_history(preprocess_obs(raw_obs, prev_action, cfg), cfg.obs_history_steps)
        act = preprocess_action(raw_act, cfg.action_mode)

        valid_len = n - cfg.chunk_size + 1
        obs_valid = obs_history[:valid_len]
        obs_sum += obs_valid.sum(axis=0)
        obs_sq += (obs_valid ** 2).sum(axis=0)
        obs_count += len(obs_valid)

        act_sum += act.sum(axis=0)
        act_sq += (act ** 2).sum(axis=0)
        act_count += len(act)

    obs_mean = obs_sum / max(obs_count, 1)
    obs_var = obs_sq / max(obs_count, 1) - obs_mean ** 2
    obs_std = np.sqrt(np.maximum(obs_var, 1e-12)) + 1e-6
    act_mean = act_sum / max(act_count, 1)
    act_var = act_sq / max(act_count, 1) - act_mean ** 2
    act_std = np.sqrt(np.maximum(act_var, 1e-12)) + 1e-6
    return obs_mean.astype(np.float32), obs_std.astype(np.float32), act_mean.astype(np.float32), act_std.astype(np.float32)


class ActionChunkDataset(Dataset):
    def __init__(self, episodes, cfg: Config, obs_mean, obs_std, act_mean, act_std):
        self.episodes = episodes
        self.cfg = cfg
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)
        self.act_mean = act_mean.astype(np.float32)
        self.act_std = act_std.astype(np.float32)
        self.window_counts = np.asarray([n - cfg.chunk_size + 1 for _, _, n in episodes], dtype=np.int64)
        self.cum_windows = np.cumsum(self.window_counts)
        self.total_windows = int(self.cum_windows[-1]) if len(self.cum_windows) else 0
        self._cache_ep = None
        self._cache_data = None
        self._episode_cache = None
        if cfg.cache_episodes:
            self._episode_cache = [
                self._preprocess_episode(obs_path, act_path, n)
                for obs_path, act_path, n in progress(episodes, desc="Caching episodes")
            ]

    def __len__(self):
        return self.total_windows

    def _preprocess_episode(self, obs_path: Path, act_path: Path, n_hint: int):
        raw_obs = np.load(obs_path)
        raw_act = np.load(act_path)
        n = compute_trimmed_length(raw_obs, raw_act, n_hint, self.cfg)
        raw_obs = np.asarray(raw_obs[:n], dtype=np.float32)
        raw_act = np.asarray(raw_act[:n], dtype=np.float32)
        prev_action = build_prev_action_history(raw_act, self.cfg.action_mode)
        obs_history = build_obs_history(preprocess_obs(raw_obs, prev_action, self.cfg), self.cfg.obs_history_steps)
        act = preprocess_action(raw_act, self.cfg.action_mode)
        return obs_history, raw_obs, raw_act, act

    def _load_episode(self, ep_idx: int):
        if self._episode_cache is not None:
            return self._episode_cache[ep_idx]
        if self._cache_ep == ep_idx:
            return self._cache_data
        obs_path, act_path, n = self.episodes[ep_idx]
        self._cache_data = self._preprocess_episode(obs_path, act_path, n)
        self._cache_ep = ep_idx
        return self._cache_data

    def __getitem__(self, idx: int):
        ep_idx = int(np.searchsorted(self.cum_windows, idx, side="right"))
        prev = 0 if ep_idx == 0 else int(self.cum_windows[ep_idx - 1])
        start = int(idx - prev)

        obs_history, raw_obs, raw_act, act = self._load_episode(ep_idx)
        obs = (obs_history[start] - self.obs_mean) / self.obs_std
        action_chunk = act[start : start + self.cfg.chunk_size]
        action_chunk = (action_chunk - self.act_mean) / self.act_std

        step_weight, window_weight = build_chunk_weights(
            raw_obs[start : start + self.cfg.chunk_size],
            raw_act[start : start + self.cfg.chunk_size],
            self.cfg,
        )
        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(action_chunk.astype(np.float32)),
            torch.from_numpy(step_weight.astype(np.float32)),
            torch.tensor(float(window_weight), dtype=torch.float32),
        )


def compute_critical_sample_weights(episodes, cfg: Config) -> np.ndarray:
    cache_tag = (
        f"bc_{cfg.task_family}_N{cfg.chunk_size}_mode{cfg.action_mode}"
        f"_close{cfg.grasp_close_threshold:g}_lift{cfg.lift_height_threshold:g}"
        f"_lookback{cfg.pregrasp_lookback}_pre{cfg.pregrasp_sample_weight:g}"
        f"_grasp{cfg.grasp_sample_weight:g}_liftw{cfg.lift_sample_weight:g}"
        f"_trimr{int(cfg.trim_terminal_reset_actions)}"
        f"_trims{int(cfg.trim_after_success)}_keep{cfg.post_success_keep_steps}"
    ).replace(".", "p")
    cache_path = cfg.data_root / f"critical_sampler_weights_{cache_tag}.npz"
    expected_windows = int(sum(max(0, n - cfg.chunk_size + 1) for _, _, n in episodes))
    if cache_path.exists() and not env_bool("REBUILD_SAMPLER_WEIGHTS", False):
        cached = np.load(cache_path)
        sample_weights = cached["sample_weights"].astype(np.float32)
        episode_count = int(cached["episode_count"])
        if episode_count == len(episodes) and len(sample_weights) == expected_windows:
            print(f"[INFO] Loaded sampler weight cache: {cache_path}")
            return sample_weights
        print(
            "[WARN] Ignoring stale sampler weight cache: "
            f"{cache_path} episodes={episode_count}/{len(episodes)} "
            f"weights={len(sample_weights)}/{expected_windows}"
        )

    weights: list[np.ndarray | None] = [None] * len(episodes)
    ordered_episodes = sorted(enumerate(episodes), key=lambda item: str(item[1][0]))
    for original_idx, (obs_path, act_path, n_hint) in progress(ordered_episodes, desc="Computing sampler weights"):
        raw_obs_mmap = np.load(obs_path, mmap_mode="r")
        raw_act_mmap = np.load(act_path, mmap_mode="r")
        n = compute_trimmed_length(raw_obs_mmap, raw_act_mmap, n_hint, cfg)
        if n < cfg.chunk_size:
            weights[original_idx] = np.zeros((0,), dtype=np.float32)
            continue
        raw_obs = np.asarray(raw_obs_mmap[:n], dtype=np.float32)
        raw_act = np.asarray(raw_act_mmap[:n], dtype=np.float32)
        valid_len = n - cfg.chunk_size + 1

        step_priority = np.ones((n,), dtype=np.float32)
        processed_action = preprocess_action(raw_act, cfg.action_mode)
        gripper = processed_action[:, 6] if processed_action.shape[1] > 6 else raw_act[:, -1]
        close_mask = gripper <= cfg.grasp_close_threshold
        if np.any(close_mask):
            first_close_idx = int(np.flatnonzero(close_mask)[0])
            pregrasp_start = max(0, first_close_idx - cfg.pregrasp_lookback)
            step_priority[pregrasp_start : first_close_idx + 1] = np.maximum(
                step_priority[pregrasp_start : first_close_idx + 1],
                cfg.pregrasp_sample_weight,
            )
            step_priority[close_mask] = np.maximum(step_priority[close_mask], cfg.grasp_sample_weight)

        obj_z = raw_obs[:, 20]
        lift_mask = (obj_z - obj_z[0]) >= cfg.lift_height_threshold
        if np.any(lift_mask):
            step_priority[lift_mask] = np.maximum(step_priority[lift_mask], cfg.lift_sample_weight)

        ep_weights = np.empty((valid_len,), dtype=np.float32)
        for start in range(valid_len):
            ep_weights[start] = float(step_priority[start : start + cfg.chunk_size].max())
        weights[original_idx] = ep_weights

    if any(weight is None for weight in weights):
        raise RuntimeError("Failed to compute all BC sampler weights")
    sample_weights = np.concatenate(weights).astype(np.float32) if weights else np.zeros((0,), dtype=np.float32)
    np.savez(
        cache_path,
        sample_weights=sample_weights,
        episode_count=np.asarray(len(episodes), dtype=np.int64),
    )
    print(f"[INFO] Saved sampler weight cache: {cache_path}")
    return sample_weights


class BCChunkMLP(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (1024, 1024, 512),
        dropout: float = 0.05,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = self.net(obs)
        return out.view(obs.shape[0], self.chunk_size, self.action_dim)


class DeterministicBCUnet1D(nn.Module):
    """DP/FMP U-Net backbone used as a deterministic action-chunk regressor."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        chunk_size: int,
        down_dims: tuple[int, ...] = (192, 384, 768),
        time_embed_dim: int = 384,
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.unet = DiffusionPolicyConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=time_embed_dim,
            down_dims=tuple(down_dims),
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        x = torch.zeros(batch, self.action_dim, self.chunk_size, device=obs.device, dtype=obs.dtype)
        t = torch.zeros(batch, device=obs.device, dtype=obs.dtype)
        pred = self.unet(x, t, obs)
        return pred.permute(0, 2, 1)


def build_model(cfg: Config, model_type: str | None = None) -> nn.Module:
    model_type = (model_type or cfg.model_arch).lower()
    if model_type in {"unet", "bc_unet", "deterministic_unet", "stanford_dp"}:
        return DeterministicBCUnet1D(
            cfg.processed_obs_dim,
            cfg.processed_action_dim,
            cfg.chunk_size,
            down_dims=tuple(cfg.down_dims),
            time_embed_dim=cfg.time_embed_dim,
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
            cond_predict_scale=cfg.cond_predict_scale,
        )
    if model_type in {"mlp", "bc_mlp"}:
        return BCChunkMLP(
            cfg.processed_obs_dim,
            cfg.processed_action_dim,
            cfg.chunk_size,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
    raise ValueError(f"Unsupported MODEL_ARCH={model_type!r}. Use unet or mlp.")


def build_model_from_checkpoint_config(cfg_dict: dict) -> nn.Module:
    model_type = str(cfg_dict.get("model_arch", cfg_dict.get("model_type", "unet"))).lower()
    obs_dim = int(cfg_dict.get("processed_obs_dim", 124))
    act_dim = int(cfg_dict.get("processed_action_dim", 7))
    chunk_size = int(cfg_dict.get("chunk_size", 8))
    if model_type in {"unet", "bc_unet", "deterministic_unet", "stanford_dp"}:
        return DeterministicBCUnet1D(
            obs_dim,
            act_dim,
            chunk_size,
            down_dims=tuple(cfg_dict.get("down_dims", (192, 384, 768))),
            time_embed_dim=int(cfg_dict.get("time_embed_dim", 384)),
            kernel_size=int(cfg_dict.get("kernel_size", 5)),
            n_groups=int(cfg_dict.get("n_groups", 8)),
            cond_predict_scale=bool(cfg_dict.get("cond_predict_scale", True)),
        )
    if model_type in {"mlp", "bc_mlp"}:
        return BCChunkMLP(
            obs_dim,
            act_dim,
            chunk_size,
            hidden_dims=tuple(cfg_dict.get("hidden_dims", (1024, 1024, 512))),
            dropout=float(cfg_dict.get("dropout", 0.0)),
            use_layer_norm=bool(cfg_dict.get("use_layer_norm", True)),
        )
    raise ValueError(f"Unsupported checkpoint model type: {model_type!r}")

def compute_losses(pred: torch.Tensor, target: torch.Tensor, step_weight: torch.Tensor, window_weight: torch.Tensor, cfg: Config):
    raw = F.mse_loss(pred, target, reduction="none")
    mse = raw.mean()
    gripper_mse = raw[:, :, -1:].mean()
    if not cfg.use_loss_weights:
        return mse, mse, gripper_mse

    dim_weight = torch.ones((pred.shape[-1],), device=pred.device, dtype=pred.dtype)
    dim_weight[-1] = cfg.gripper_loss_weight
    weighted = raw * dim_weight.view(1, 1, -1)
    weighted = weighted.mean(dim=2) * step_weight
    weighted = weighted.mean(dim=1) * window_weight
    return weighted.mean(), mse, gripper_mse

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: Config, device: torch.device):
    model.eval()
    weighted_losses = []
    mse_losses = []
    gripper_losses = []
    for obs, action_chunk, step_weight, window_weight in loader:
        obs = obs.to(device)
        action_chunk = action_chunk.to(device)
        step_weight = step_weight.to(device)
        window_weight = window_weight.to(device)
        pred = model(obs)
        weighted, mse, grip = compute_losses(pred, action_chunk, step_weight, window_weight, cfg)
        weighted_losses.append(float(weighted.item()))
        mse_losses.append(float(mse.item()))
        gripper_losses.append(float(grip.item()))
    return (
        float(np.mean(weighted_losses)) if weighted_losses else math.inf,
        float(np.mean(mse_losses)) if mse_losses else math.inf,
        float(np.mean(gripper_losses)) if gripper_losses else math.inf,
    )


def save_checkpoint(path: Path, model: nn.Module, cfg: Config, epoch: int, best_val: float, optimizer=None):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "model_type": f"bc_{cfg.model_arch}",
        "model_variant": f"{cfg.task_family}_bc_{cfg.model_arch}_chunk_{cfg.action_mode}_v1",
        "config": {
            **asdict(cfg),
            "data_root": str(cfg.data_root),
            "save_dir": str(cfg.save_dir),
            "model_arch": cfg.model_arch,
            "down_dims": tuple(cfg.down_dims),
            "time_embed_dim": cfg.time_embed_dim,
            "kernel_size": cfg.kernel_size,
            "n_groups": cfg.n_groups,
            "cond_predict_scale": cfg.cond_predict_scale,
            "processed_obs_dim": cfg.processed_obs_dim,
            "processed_obs_frame_dim": cfg.processed_obs_frame_dim,
            "processed_action_dim": cfg.processed_action_dim,
            "raw_obs_dim": cfg.raw_obs_dim,
        },
    }
    torch.save(ckpt, path)

def train_main():
    cfg = Config()
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.train_device)

    print(f"[INFO] task_family={cfg.task_family} data_root={cfg.data_root}")
    print(f"[INFO] action_mode={cfg.action_mode} N={cfg.chunk_size} obs_dim={cfg.processed_obs_dim} act_dim={cfg.processed_action_dim}")
    print(
        "[INFO] model_arch="
        f"{cfg.model_arch} down_dims={tuple(cfg.down_dims)} time_embed_dim={cfg.time_embed_dim} "
        "deterministic=1 noise=0 denoising=0 flow_integration=0"
    )
    print(f"[INFO] epochs={cfg.num_epochs} batch={cfg.batch_size} device={device}")
    print(f"[INFO] save_dir={cfg.save_dir}")

    episodes = build_episode_index(cfg)
    if not episodes:
        raise RuntimeError(f"No episodes found in {cfg.data_root}")

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(episodes))
    train_count = int(len(episodes) * 0.95)
    train_episodes = [episodes[i] for i in perm[:train_count]]
    val_episodes = [episodes[i] for i in perm[train_count:]]

    if cfg.stats_path.exists() and not cfg.recompute_stats:
        stats = np.load(cfg.stats_path)
        obs_mean = stats["obs_mean"].astype(np.float32)
        obs_std = stats["obs_std"].astype(np.float32)
        act_mean = stats["act_mean"].astype(np.float32)
        act_std = stats["act_std"].astype(np.float32)
        print(f"[INFO] Loaded stats: {cfg.stats_path}")
    elif cfg.reuse_stats_path:
        stats = np.load(Path(cfg.reuse_stats_path).expanduser())
        obs_mean = stats["obs_mean"].astype(np.float32)
        obs_std = stats["obs_std"].astype(np.float32)
        act_mean = stats["act_mean"].astype(np.float32)
        act_std = stats["act_std"].astype(np.float32)
        np.savez(cfg.stats_path, obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std)
        print(f"[INFO] Reused stats: {cfg.reuse_stats_path}")
    else:
        stats_episodes = train_episodes
        if 0 < cfg.max_stats_episodes < len(train_episodes):
            stats_indices = rng.choice(len(train_episodes), size=cfg.max_stats_episodes, replace=False)
            stats_episodes = [train_episodes[i] for i in stats_indices]
            print(f"[INFO] Limited stats episodes to {len(stats_episodes)}/{len(train_episodes)}")
        obs_mean, obs_std, act_mean, act_std = compute_stats(stats_episodes, cfg)
        np.savez(cfg.stats_path, obs_mean=obs_mean, obs_std=obs_std, act_mean=act_mean, act_std=act_std)
        print(f"[INFO] Saved stats: {cfg.stats_path}")

    train_dataset = ActionChunkDataset(train_episodes, cfg, obs_mean, obs_std, act_mean, act_std)
    val_dataset = ActionChunkDataset(val_episodes, cfg, obs_mean, obs_std, act_mean, act_std)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(f"Empty dataset split: train={len(train_dataset)} val={len(val_dataset)}")

    train_sampler = None
    train_shuffle = True
    if cfg.oversample_critical_windows:
        sample_weights = compute_critical_sample_weights(train_episodes, cfg)
        if len(sample_weights) != len(train_dataset):
            raise RuntimeError(f"Sampler weight mismatch: weights={len(sample_weights)} dataset={len(train_dataset)}")
        samples_per_epoch = (
            cfg.max_train_windows_per_epoch
            if 0 < cfg.max_train_windows_per_epoch < len(train_dataset)
            else len(train_dataset)
        )
        train_sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=samples_per_epoch,
            replacement=True,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
        train_shuffle = False
        print(f"[INFO] Oversampling critical windows samples_per_epoch={samples_per_epoch}")
    elif 0 < cfg.max_train_windows_per_epoch < len(train_dataset):
        indices = np.random.default_rng(cfg.seed + 1).choice(len(train_dataset), size=cfg.max_train_windows_per_epoch, replace=False)
        train_dataset = Subset(train_dataset, np.sort(indices).tolist())
        train_shuffle = False
        print(f"[INFO] Limited train windows per epoch to {len(train_dataset)}")

    if 0 < cfg.max_val_windows < len(val_dataset):
        indices = np.random.default_rng(cfg.seed + 2).choice(len(val_dataset), size=cfg.max_val_windows, replace=False)
        val_dataset = Subset(val_dataset, indices.tolist())
        print(f"[INFO] Limited validation windows to {len(val_dataset)}")

    loader_kwargs = {"batch_size": cfg.batch_size, "num_workers": cfg.num_workers, "pin_memory": False}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=train_shuffle, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_val = math.inf
    start_epoch = 1
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT", "")
    if resume_checkpoint:
        checkpoint = torch.load(Path(resume_checkpoint).expanduser(), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val = float(checkpoint.get("best_val", best_val))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"[INFO] Resumed from {resume_checkpoint} at epoch {start_epoch}")

    cfg.log_txt_path.write_text("BC baseline training log\n", encoding="utf-8")
    cfg.log_md_path.write_text(
        "| Epoch | Train weighted | Train MSE | Train gripper MSE | Val weighted | Val MSE | Val gripper MSE | LR |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        encoding="utf-8",
    )

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        model.train()
        train_weighted = []
        train_mse = []
        train_grip = []
        t0 = time.time()
        pbar = progress(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", disable=env_bool("TQDM_DISABLE", False))
        for batch_idx, (obs, action_chunk, step_weight, window_weight) in enumerate(pbar, start=1):
            obs = obs.to(device)
            action_chunk = action_chunk.to(device)
            step_weight = step_weight.to(device)
            window_weight = window_weight.to(device)

            pred = model(obs)
            loss, mse, grip = compute_losses(pred, action_chunk, step_weight, window_weight, cfg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()

            train_weighted.append(float(loss.item()))
            train_mse.append(float(mse.item()))
            train_grip.append(float(grip.item()))
            
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(weighted=f"{np.mean(train_weighted):.6f}", mse=f"{np.mean(train_mse):.6f}")
            if cfg.batch_log_every > 0 and batch_idx % cfg.batch_log_every == 0:
                msg = (
                    f"epoch={epoch:03d} batch={batch_idx}/{len(train_loader)} "
                    f"train_weighted_running={np.mean(train_weighted):.6f} "
                    f"train_mse_running={np.mean(train_mse):.6f} "
                    f"train_gripper_mse_running={np.mean(train_grip):.6f}"
                )
                print(msg)
                with cfg.log_txt_path.open("a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        val_weighted, val_mse, val_grip = evaluate(model, val_loader, cfg, device)
        epoch_msg = (
            f"epoch={epoch:03d} train_weighted={np.mean(train_weighted):.6f} "
            f"train_mse={np.mean(train_mse):.6f} train_gripper_mse={np.mean(train_grip):.6f} "
            f"val_weighted={val_weighted:.6f} val_mse={val_mse:.6f} val_gripper_mse={val_grip:.6f} "
            f"lr={optimizer.param_groups[0]['lr']:.8f} elapsed_s={time.time() - t0:.1f}"
        )
        print(epoch_msg)
        with cfg.log_txt_path.open("a", encoding="utf-8") as f:
            f.write(epoch_msg + "\n")
        with cfg.log_md_path.open("a", encoding="utf-8") as f:
            f.write(
                f"| {epoch} | {np.mean(train_weighted):.6f} | {np.mean(train_mse):.6f} | "
                f"{np.mean(train_grip):.6f} | {val_weighted:.6f} | {val_mse:.6f} | "
                f"{val_grip:.6f} | {optimizer.param_groups[0]['lr']:.8f} |\n"
            )

        if val_weighted < best_val:
            best_val = val_weighted
            save_checkpoint(cfg.checkpoint_path, model, cfg, epoch, best_val, optimizer)
            print(f"[INFO] Saved best checkpoint: {cfg.checkpoint_path} best_val={best_val:.6f}")
        if cfg.checkpoint_save_every > 0 and epoch % cfg.checkpoint_save_every == 0:
            periodic_path = cfg.save_dir / f"{cfg.task_family}_bc_{cfg.model_arch}_epoch_{epoch:03d}.pt"
            save_checkpoint(periodic_path, model, cfg, epoch, best_val, optimizer)

    save_checkpoint(cfg.save_dir / f"{cfg.task_family}_bc_{cfg.model_arch}_last.pt", model, cfg, cfg.num_epochs, best_val, optimizer)
    print(f"[DONE] best_val={best_val:.6f} checkpoint={cfg.checkpoint_path}")

def _compress_robot_state_torch(raw_state: torch.Tensor, action_mode: str) -> torch.Tensor:
    if action_mode == "lr":
        return torch.cat([raw_state[:, :6], raw_state[:, 7:9]], dim=1)
    return torch.cat([raw_state[:, :6], raw_state[:, 7:9].mean(dim=1, keepdim=True)], dim=1)

def _preprocess_obs_torch(obs: torch.Tensor, prev_action: torch.Tensor, cfg: dict) -> torch.Tensor:
    action_mode = str(cfg.get("action_mode", "mean"))
    task_family = str(cfg.get("task_family", "reach_lift"))
    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    joint_pos_processed = _compress_robot_state_torch(joint_pos, action_mode)
    joint_vel_processed = _compress_robot_state_torch(joint_vel, action_mode)
    if task_family == "reach_lift":
        object_pos = obs[:, 18:21]
        pose_command = obs[:, 21:28]
        return torch.cat([joint_pos_processed, joint_vel_processed, object_pos, pose_command, prev_action], dim=1)
    pose_command = obs[:, 18:25]
    return torch.cat([joint_pos_processed, joint_vel_processed, pose_command, prev_action], dim=1)


def _stack_obs_history_torch(current_frame: torch.Tensor, history: torch.Tensor | None, history_steps: int):
    batch, feat_dim = current_frame.shape
    if history_steps <= 1:
        return None, current_frame
    if history is None or history.shape != (batch, history_steps, feat_dim):
        history = current_frame.unsqueeze(1).repeat(1, history_steps, 1)
    else:
        history = torch.cat([history[:, 1:, :], current_frame.unsqueeze(1)], dim=1)
    return history, history.reshape(batch, history_steps * feat_dim)


class Policy:
    def __init__(self, model, obs_mean, obs_std, act_mean, act_std, cfg: dict, device):
        self.model = model
        self.obs_mean = torch.as_tensor(obs_mean, device=device, dtype=torch.float32)
        self.obs_std = torch.as_tensor(obs_std, device=device, dtype=torch.float32)
        self.act_mean = torch.as_tensor(act_mean, device=device, dtype=torch.float32)
        self.act_std = torch.as_tensor(act_std, device=device, dtype=torch.float32)
        self.cfg = cfg
        self.device = device
        self.chunk_size = int(cfg["chunk_size"])
        self.action_mode = str(cfg.get("action_mode", "mean"))
        self.obs_history_steps = int(cfg.get("obs_history_steps", 4))
        self.prev_action = None
        self.obs_history = None
        self.model.eval()

    def reset(self):
        self.prev_action = None
        self.obs_history = None

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor, output_horizon: int | None = None, execute_horizon: int | None = None):
        obs = obs.to(self.device, dtype=torch.float32)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        batch = obs.shape[0]
        if self.prev_action is None or self.prev_action.shape[0] != batch:
            init = initial_action(self.action_mode)
            self.prev_action = torch.as_tensor(init, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)

        if obs.shape[-1] != self.obs_mean.shape[-1]:
            current_frame = _preprocess_obs_torch(obs, self.prev_action, self.cfg)
            self.obs_history, obs = _stack_obs_history_torch(current_frame, self.obs_history, self.obs_history_steps)
        obs = (obs - self.obs_mean) / self.obs_std
        action_chunks = self.model(obs) * self.act_std.view(1, 1, -1) + self.act_mean.view(1, 1, -1)

        if output_horizon is not None and int(output_horizon) != action_chunks.shape[1]:
            output_horizon = int(output_horizon)
            if output_horizon < action_chunks.shape[1]:
                action_chunks = action_chunks[:, :output_horizon, :]
            else:
                pad = output_horizon - action_chunks.shape[1]
                action_chunks = torch.cat([action_chunks, action_chunks[:, -1:, :].repeat(1, pad, 1)], dim=1)

        executed_steps = int(execute_horizon) if execute_horizon is not None else 1
        executed_steps = max(1, min(executed_steps, action_chunks.shape[1]))
        self.prev_action = action_chunks[:, executed_steps - 1, :].detach()
        return action_chunks

def load_policy(device: str | torch.device = "cpu", output_horizon: int | None = None, execute_horizon: int | None = None):
    del output_horizon, execute_horizon
    device = torch.device(device)
    default_ckpt = FILE_DIR / "reach_lift_bc_unet_8chunk" / "reach_lift_bc_unet.pt"
    default_stats = FILE_DIR / "reach_lift_bc_unet_8chunk" / "reach_lift_bc_stats.npz"
    ckpt_path = Path(os.environ.get("POLICY_CHECKPOINT", str(default_ckpt))).expanduser().resolve()
    stats_path = Path(os.environ.get("POLICY_STATS", str(default_stats))).expanduser().resolve()
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    if "model_arch" not in cfg:
        cfg["model_arch"] = str(ckpt.get("model_type", "bc_mlp")).replace("bc_", "")
    model = build_model_from_checkpoint_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stats = np.load(stats_path)
    return Policy(
        model,
        stats["obs_mean"].astype(np.float32),
        stats["obs_std"].astype(np.float32),
        stats["act_mean"].astype(np.float32),
        stats["act_std"].astype(np.float32),
        cfg,
        device,
    )

def eval_main(argv: list[str]):
    parser = argparse.ArgumentParser(description="Launch Isaac eval for this BC policy.")
    parser.add_argument("--task", default="reach_lift_imitation")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--stats", default="")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode-length-s", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--success-dist", type=float, default=0.02)
    parser.add_argument("--success-z", type=float, default=1.14)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-path", default="")
    parser.add_argument("--isaac-cmd", default=os.environ.get("ISAACLAB_CMD", "isaaclab_4"))
    args = parser.parse_args(argv)

    cfg = Config()
    checkpoint = args.checkpoint or str(cfg.checkpoint_path)
    stats = args.stats or str(cfg.stats_path)
    cmd = [
        args.isaac_cmd,
        "-p",
        "./launch/imitation_play_fixed_left_right.py",
        "--kit_args=--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error",
        "--task",
        args.task,
        "--policy_type",
        "python",
        "--policy_path",
        str(Path(__file__).resolve()),
        "--policy_checkpoint",
        checkpoint,
        "--policy_stats",
        stats,
        "--num_envs",
        str(args.num_envs),
        "--output_horizon",
        str(args.n),
        "--execute_horizon",
        str(args.k),
        "--episode_length_s",
        str(args.episode_length_s),
        "--joint_arrival_tolerance",
        "0.015",
        "--max_action_hold_steps",
        "15",
        "--action_smoothing_alpha",
        "0.8",
        "--next_action_lookahead_margin",
        "0.004",
        "--replan_lookahead_steps",
        "0",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--disable_predicted_path_viz",
    ]
    if "reach_lift" in args.task:
        cmd.extend(["--success_z", str(args.success_z)])
    else:
        cmd.extend(["--success_dist", str(args.success_dist)])
    if args.save_path:
        cmd.extend(["--save_path", args.save_path])
    if args.headless:
        cmd.append("--headless")

    env = os.environ.copy()
    env["TERM"] = env.get("TERM", "xterm")
    env["PYTHONPATH"] = str(REPO_ROOT)
    print("[INFO] Running eval command:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode)

def info_main():
    cfg = Config()
    print(f"task_family={cfg.task_family}")
    print(f"data_root={cfg.data_root}")
    print(f"obs_dir={cfg.obs_dir}")
    print(f"act_dir={cfg.act_dir}")
    print(f"chunk_size={cfg.chunk_size}")
    print(f"action_mode={cfg.action_mode}")
    print(f"model_arch={cfg.model_arch}")
    print(f"down_dims={tuple(cfg.down_dims)}")
    print(f"time_embed_dim={cfg.time_embed_dim}")
    print(f"processed_obs_dim={cfg.processed_obs_dim}")
    print(f"processed_action_dim={cfg.processed_action_dim}")
    print(f"save_dir={cfg.save_dir}")
    print(f"checkpoint={cfg.checkpoint_path}")
    print(f"stats={cfg.stats_path}")

def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "train"
    if command == "train":
        train_main()
    elif command == "eval":
        eval_main(sys.argv[2:])
    elif command == "info":
        info_main()
    else:
        raise SystemExit(f"Unknown command: {command}. Use train, eval, or info.")

if __name__ == "__main__":
    main()
