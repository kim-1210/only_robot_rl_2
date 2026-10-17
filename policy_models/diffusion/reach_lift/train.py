import math
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .diffusion_policy_unet import DiffusionPolicyConditionalUnet1D
except ImportError:
    from diffusion_policy_unet import DiffusionPolicyConditionalUnet1D

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent.parent.parent

# ---
# Code cell 3
class Config:
    data_root = REPO_ROOT / "datas" / "reach_lift_data1"
    obs_dir = data_root / "obs"
    act_dir = data_root / "actions"

    chunk_size = 8
    batch_size = 1024
    num_epochs = 500
    learning_rate = 1.5e-4
    weight_decay = 1e-6
    num_workers = 16

    diffusion_steps = 100
    beta_start = 1e-4
    beta_end = 2e-2
    schedule = "cosine"

    model_arch = "stanford_dp"
    down_dims = (192, 384, 768)
    time_embed_dim = 384
    kernel_size = 5
    n_groups = 8
    cond_predict_scale = True

    raw_obs_dim = 35
    processed_obs_frame_dim = 31
    obs_history_steps = 4
    processed_obs_dim = processed_obs_frame_dim * obs_history_steps
    raw_action_dim = 7
    processed_action_dim = 7

    grasp_close_threshold = 0.013
    lift_height_threshold = 0.02
    pregrasp_lookback = 8
    grasp_window_weight = 4.0
    lift_window_weight = 5.0
    pregrasp_window_weight = 10.0
    grasp_step_weight = 5.0
    lift_step_weight = 6.0
    pregrasp_step_weight = 12.0
    gripper_loss_weight = 4.0

    use_ema = True
    ema_decay = 0.999
    checkpoint_save_every = 10
    inference_noise_mode = "zeros"

    save_dir = FILE_DIR / "reach_lift_diffusion_unet_artifacts_balanced_v3_8chunk"
    checkpoint_path = save_dir / "reach_lift_diffusion_unet_balanced_v3.pt"
    stats_path = save_dir / "reach_lift_diffusion_balanced_v3_stats.npz"
    log_txt_path = save_dir / "train_log.txt"
    log_md_path = save_dir / "train_log.md"

config = Config()
config.save_dir.mkdir(parents=True, exist_ok=True)
config

# ---
# Code cell 4
INITIAL_ACTION_FROM_CFG_SETTINGS = np.array(
    [
        -1.57,  # base
        0.0,    # shoulder
        0.7,    # elbow
        0.87,   # wrist1
        1.57,   # wrist2
        1.57,   # wrist3
        0.025,  # gripper joint at reset
    ],
    dtype=np.float32,
)


def compress_robot_state(raw_state: np.ndarray) -> np.ndarray:
    """Keep the 6 arm joints and collapse the two finger joints into one scalar."""
    return np.concatenate([raw_state[:, :6], raw_state[:, 7:9].mean(axis=1, keepdims=True)], axis=1).astype(np.float32)


def build_prev_joint_history(raw_action: np.ndarray) -> np.ndarray:
    """Provide the previous realized joint state, seeding the first step from the robot reset pose."""
    if raw_action.shape[1] == 7:
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
    """Raw reach_lift obs layout: [joint_pos_rel(9), joint_vel_rel(9), object_pos(3), pose_command(7), last_action(7)]."""
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
    """Stack a short left-padded observation history to improve grasp commitment."""
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
    """Policy action layout: [6 arm commands, gripper command]."""
    if raw_action.shape[1] == 7:
        return raw_action.astype(np.float32)
    if raw_action.shape[1] >= 9:
        return compress_robot_state(raw_action)
    raise ValueError(f"Unsupported raw action dim: {raw_action.shape[1]}")


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
    """Build per-step and per-window weights for grasp/lift-critical windows."""
    num_steps = act_slice.shape[0]
    step_weight = np.ones((num_steps,), dtype=np.float32)
    window_weight = 1.0

    if act_slice.shape[1] >= 9:
        gripper = 0.5 * (act_slice[:, 7] + act_slice[:, 8])
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


def compute_window_weights(n: int, chunk_size: int) -> np.ndarray:
    t = np.arange(n)
    start_min = np.maximum(0, t - chunk_size + 1)
    start_max = np.minimum(t, n - chunk_size)
    w = start_max - start_min + 1
    return np.maximum(w, 0).astype(np.float64)


def build_episode_index(obs_dir: Path, act_dir: Path, chunk_size: int):
    obs_paths = sorted(obs_dir.glob("obs_ep_*.npy"))
    episodes = []
    for obs_path in tqdm(obs_paths, desc="Indexing episodes"):
        act_path = act_dir / obs_path.name.replace("obs_", "act_")
        if not act_path.exists():
            continue
        obs = np.load(obs_path, mmap_mode="r")
        act = np.load(act_path, mmap_mode="r")
        n = min(len(obs), len(act))
        if n < chunk_size:
            continue
        episodes.append((obs_path, act_path, n))
    return episodes


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
        n = min(len(raw_obs), len(raw_act))
        if n < chunk_size:
            continue
        raw_obs = raw_obs[:n]
        raw_act = raw_act[:n]

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


episodes = build_episode_index(config.obs_dir, config.act_dir, config.chunk_size)

rng = np.random.default_rng(42)
perm = rng.permutation(len(episodes))
train_count = int(len(episodes) * 0.95)
train_episodes = [episodes[i] for i in perm[:train_count]]
val_episodes = [episodes[i] for i in perm[train_count:]]

obs_mean, obs_std, act_mean, act_std = compute_stats(train_episodes, config.chunk_size)

np.savez(
    config.stats_path,
    obs_mean=obs_mean,
    obs_std=obs_std,
    act_mean=act_mean,
    act_std=act_std,
)

len(episodes), len(train_episodes), len(val_episodes)

# ---
# Code cell 5
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

    def __len__(self):
        return self.total_windows

    def _load_episode(self, ep_idx: int):
        if self._cache_ep == ep_idx:
            return self._cache_obs_history, self._cache_obs_raw, self._cache_act_raw, self._cache_act_processed
        obs_path, act_path, _ = self.episodes[ep_idx]
        obs = np.load(obs_path)
        act = np.load(act_path)
        n = min(len(obs), len(act))
        obs = np.asarray(obs[:n], dtype=np.float32)
        act = np.asarray(act[:n], dtype=np.float32)
        prev_joint = build_prev_joint_history(act)
        obs_history = build_obs_history(preprocess_reach_obs(obs, prev_joint), config.obs_history_steps)
        act_processed = preprocess_reach_action(act)
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

        obs_history, obs_raw, act_raw, act_processed = self._load_episode(ep_idx)
        obs_chunk_raw = np.asarray(obs_raw[local_start : local_start + self.chunk_size])
        act_slice = np.asarray(act_raw[local_start : local_start + self.chunk_size])
        obs = obs_history[local_start]
        act = act_processed[local_start : local_start + self.chunk_size]
        step_weight, window_weight = build_chunk_weight_targets(
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

        obs = (obs - self.obs_mean) / self.obs_std
        act = (act - self.act_mean) / self.act_std

        return (
            torch.from_numpy(obs.astype(np.float32)),
            torch.from_numpy(act.astype(np.float32)),
            torch.from_numpy(step_weight.astype(np.float32)),
            torch.tensor(window_weight, dtype=torch.float32),
        )


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

dataset = train_dataset

loader_kwargs = {
    "batch_size": config.batch_size,
    "num_workers": config.num_workers,
    "pin_memory": False,
}
if config.num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    **loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    **loader_kwargs,
)

len(train_dataset), len(val_dataset)

# ---
# Code cell 6
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -scale)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, action_dim: int, obs_dim: int, base_channels: int, time_embed_dim: int, obs_embed_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.obs_mlp = nn.Sequential(
            nn.Linear(obs_dim, obs_embed_dim),
            nn.SiLU(),
            nn.Linear(obs_embed_dim, obs_embed_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(time_embed_dim + obs_embed_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels),
        )

        self.input_proj = nn.Conv1d(action_dim, base_channels, kernel_size=3, padding=1)
        self.down1 = ResBlock1D(base_channels, base_channels, base_channels)
        self.downsample1 = nn.Conv1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.down2 = ResBlock1D(base_channels, base_channels * 2, base_channels)
        self.downsample2 = nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)

        self.mid1 = ResBlock1D(base_channels * 2, base_channels * 2, base_channels)
        self.mid2 = ResBlock1D(base_channels * 2, base_channels * 2, base_channels)

        self.upsample1 = nn.ConvTranspose1d(base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock1D(base_channels * 4, base_channels, base_channels)
        self.upsample2 = nn.ConvTranspose1d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock1D(base_channels * 2, base_channels, base_channels)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, action_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        obs_emb = self.obs_mlp(obs)
        cond = self.cond_mlp(torch.cat([time_emb, obs_emb], dim=-1))

        x0 = self.input_proj(x)
        d1 = self.down1(x0, cond)
        d2 = self.down2(self.downsample1(d1), cond)
        mid = self.mid2(self.mid1(self.downsample2(d2), cond), cond)

        u1 = self.upsample1(mid)
        u1 = self.up1(torch.cat([u1, d2], dim=1), cond)
        u2 = self.upsample2(u1)
        u2 = self.up2(torch.cat([u2, d1], dim=1), cond)
        return self.output_proj(u2)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps) / num_steps
    alpha_bars = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5) ** 2
    alpha_bars = alpha_bars / alpha_bars[0]
    betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
    return torch.clamp(betas, 1e-5, 0.999)


class DiffusionScheduler:
    def __init__(self, num_steps: int, beta_start: float, beta_end: float, device: torch.device, schedule: str = "linear"):
        self.num_steps = num_steps
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_steps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_steps)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t_index: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t_index].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def predict_x0(self, x_t: torch.Tensor, t_index: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t_index].view(-1, 1, 1)
        return (x_t - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)

    def ddim_step(self, x_t: torch.Tensor, t_index: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        x0 = self.predict_x0(x_t, t_index, pred_noise).clamp(-5.0, 5.0)
        prev_t_index = torch.clamp(t_index - 1, min=0)
        alpha_bar_prev = self.alpha_bars[prev_t_index].view(-1, 1, 1)
        x_prev = torch.sqrt(alpha_bar_prev) * x0 + torch.sqrt(1.0 - alpha_bar_prev) * pred_noise
        done_mask = (t_index == 0).view(-1, 1, 1)
        return torch.where(done_mask, x0, x_prev)


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


model = DiffusionPolicyConditionalUnet1D(
    input_dim=config.processed_action_dim,
    global_cond_dim=config.processed_obs_dim,
    diffusion_step_embed_dim=config.time_embed_dim,
    down_dims=tuple(config.down_dims),
    kernel_size=config.kernel_size,
    n_groups=config.n_groups,
    cond_predict_scale=config.cond_predict_scale,
).to(device)

scheduler = DiffusionScheduler(
    config.diffusion_steps,
    config.beta_start,
    config.beta_end,
    device,
    schedule=config.schedule,
)

ema_helper = EMA(model, config.ema_decay) if config.use_ema else None
ema_model = copy.deepcopy(model) if config.use_ema else None
model

# ---
# Code cell 7
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def compute_diffusion_loss(
    pred: torch.Tensor,
    target_noise: torch.Tensor,
    step_weight: torch.Tensor,
    window_weight: torch.Tensor,
) -> torch.Tensor:
    dim_weight = torch.ones((1, pred.shape[1], 1), device=pred.device, dtype=pred.dtype)
    dim_weight[:, -1, :] = config.gripper_loss_weight
    loss_map = F.mse_loss(pred, target_noise, reduction="none")
    loss_map = (loss_map * dim_weight).sum(dim=1) / dim_weight.sum(dim=1)
    return ((loss_map * step_weight).mean(dim=1) * window_weight).mean()


def initialize_log_files() -> None:
    config.log_txt_path.write_text("Diffusion U-Net training log\n", encoding="utf-8")
    config.log_md_path.write_text(
        "# Diffusion U-Net Training Log\n\n"
        "| Epoch | Train Loss | Val Loss |\n"
        "| --- | ---: | ---: |\n",
        encoding="utf-8",
    )


def append_epoch_log(epoch: int, train_loss: float, val_loss: float) -> None:
    with config.log_txt_path.open("a", encoding="utf-8") as f:
        f.write(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}\n")

    with config.log_md_path.open("a", encoding="utf-8") as f:
        f.write(f"| {epoch:03d} | {train_loss:.6f} | {val_loss:.6f} |\n")


def append_final_log(final_val_loss: float, best_val: float) -> None:
    with config.log_txt_path.open("a", encoding="utf-8") as f:
        f.write(f"final_val_loss={final_val_loss:.6f}\n")
        f.write(f"best_val={best_val:.6f}\n")

    with config.log_md_path.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"- final_val_loss: {final_val_loss:.6f}\n")
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
            "model_variant": "reach_lift_joint_hist_balanced_v3",
            "best_val": best_val,
        },
        path,
    )


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for obs, action_chunk, step_weight, window_weight in loader:
            obs = obs.to(device)
            action_chunk = action_chunk.to(device).permute(0, 2, 1)
            step_weight = step_weight.to(device)
            window_weight = window_weight.to(device)

            t_index = torch.randint(0, scheduler.num_steps, (obs.size(0),), device=device)
            t = t_index.float() / max(scheduler.num_steps - 1, 1)
            noise = torch.randn_like(action_chunk)
            noisy = scheduler.q_sample(action_chunk, t_index, noise)
            pred = model(noisy, t, obs)
            losses.append(compute_diffusion_loss(pred, noise, step_weight, window_weight).item())
    return float(np.mean(losses)) if losses else float("nan")


best_val = float("inf")
history = []
initialize_log_files()

for epoch in range(1, config.num_epochs + 1):
    model.train()
    train_losses = []
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")

    for obs, action_chunk, step_weight, window_weight in pbar:
        obs = obs.to(device)
        action_chunk = action_chunk.to(device).permute(0, 2, 1)
        step_weight = step_weight.to(device)
        window_weight = window_weight.to(device)

        t_index = torch.randint(0, scheduler.num_steps, (obs.size(0),), device=device)
        t = t_index.float() / max(scheduler.num_steps - 1, 1)
        noise = torch.randn_like(action_chunk)
        noisy = scheduler.q_sample(action_chunk, t_index, noise)

        pred = model(noisy, t, obs)
        loss = compute_diffusion_loss(pred, noise, step_weight, window_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if ema_helper is not None:
            ema_helper.update(model)
            ema_helper.copy_to(ema_model)

        train_losses.append(loss.item())
        pbar.set_postfix(train_loss=f"{np.mean(train_losses):.6f}")

    eval_model = ema_model if ema_model is not None else model
    val_loss = evaluate(eval_model, val_loader)
    epoch_train_loss = float(np.mean(train_losses))
    history.append((epoch, epoch_train_loss, val_loss))
    print(f"epoch={epoch:03d} train_loss={epoch_train_loss:.6f} val_loss={val_loss:.6f}")
    append_epoch_log(epoch, epoch_train_loss, val_loss)

    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(config.checkpoint_path, best_val, epoch)

    if epoch % config.checkpoint_save_every == 0:
        periodic_checkpoint_path = config.save_dir / f"reach_lift_diffusion_unet_epoch_{epoch:03d}.pt"
        save_checkpoint(periodic_checkpoint_path, best_val, epoch)

best_val

# ---
# Code cell 8
@torch.no_grad()
def sample_action_chunk(model: nn.Module, obs_batch: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
    model.eval()
    num_steps = scheduler.num_steps if num_steps is None else num_steps
    if config.inference_noise_mode == "zeros":
        x = torch.zeros(obs_batch.size(0), config.processed_action_dim, config.chunk_size, device=device)
    else:
        x = torch.randn(obs_batch.size(0), config.processed_action_dim, config.chunk_size, device=device)

    for step in reversed(range(num_steps)):
        t_index = torch.full((obs_batch.size(0),), step, device=device, dtype=torch.long)
        t = t_index.float() / max(num_steps - 1, 1)
        pred_noise = model(x, t, obs_batch)
        x = scheduler.ddim_step(x, t_index, pred_noise)

    return x.permute(0, 2, 1)


# Final validation after training
final_eval_model = ema_model if ema_model is not None else model
final_val_loss = evaluate(final_eval_model, val_loader)
print(f"final_val_loss={final_val_loss:.6f}")
append_final_log(final_val_loss, best_val)


eval_model = ema_model if ema_model is not None else model
batch_obs, batch_target, _, _ = next(iter(val_loader))
batch_obs = batch_obs[:4].to(device)
pred_chunks = sample_action_chunk(eval_model, batch_obs)

act_mean = torch.from_numpy(dataset.act_mean).to(device)
act_std = torch.from_numpy(dataset.act_std).to(device)
pred_chunks_denorm = pred_chunks * act_std + act_mean

pred_chunks_denorm.shape
