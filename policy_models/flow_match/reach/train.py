import copy
import os
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .flow_matching_unet_model import ConditionalUnet1D
except ImportError:
    from flow_matching_unet_model import ConditionalUnet1D


def _load_diffusion_unet_class():
    module_path = Path(__file__).resolve().parents[2] / "diffusion" / "reach" / "diffusion_policy_unet.py"
    spec = importlib.util.spec_from_file_location("reach_diffusion_policy_unet_for_flow_train", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diffusion U-Net module from {module_path}")
    spec.loader.exec_module(module)
    return module.DiffusionPolicyConditionalUnet1D


DiffusionPolicyConditionalUnet1D = _load_diffusion_unet_class()

torch.manual_seed(42)
np.random.seed(42)

device = torch.device(os.environ.get("TRAIN_DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu")

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parent.parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


class Config:
    data_root = _env_path("DATA_ROOT", REPO_ROOT / "datas" / "reach_data1")
    obs_dir = _env_path("OBS_DIR", data_root / "obs")
    act_dir = _env_path("ACT_DIR", data_root / "actions")

    chunk_size = _env_int("CHUNK_SIZE", 16)
    batch_size = _env_int("BATCH_SIZE", 4096)
    num_epochs = _env_int("NUM_EPOCHS", 150)
    learning_rate = _env_float("LEARNING_RATE", 1e-4)
    weight_decay = _env_float("WEIGHT_DECAY", 1e-6)
    num_workers = _env_int("NUM_WORKERS", 16)

    flow_steps = _env_int("FLOW_STEPS", 100)
    checkpoint_save_every = _env_int("CHECKPOINT_SAVE_EVERY", 10)

    model_arch = os.environ.get("MODEL_ARCH", "stanford_dp")
    down_dims = tuple(int(x) for x in os.environ.get("DOWN_DIMS", "128,256,512").split(","))
    kernel_size = _env_int("KERNEL_SIZE", 5)
    n_groups = _env_int("N_GROUPS", 8)
    cond_predict_scale = _env_bool("COND_PREDICT_SCALE", True)
    base_channels = _env_int("BASE_CHANNELS", 128)
    time_embed_dim = _env_int("TIME_EMBED_DIM", 256 if model_arch == "stanford_dp" else 128)
    obs_embed_dim = _env_int("OBS_EMBED_DIM", 128)

    raw_obs_dim = 25
    processed_obs_dim = 28
    raw_action_dim = 7
    processed_action_dim = 7

    use_ema = _env_bool("USE_EMA", True)
    ema_decay = _env_float("EMA_DECAY", 0.999)

    save_dir = _env_path("SAVE_DIR_OVERRIDE", FILE_DIR / "reach_flow_match_artifacts")
    checkpoint_path = save_dir / os.environ.get("CHECKPOINT_FILENAME", "reach_flow_match.pt")
    stats_path = save_dir / os.environ.get("STATS_FILENAME", "reach_flow_match_stats.npz")
    log_txt_path = save_dir / "train_log.txt"
    log_md_path = save_dir / "train_log.md"
    tqdm_disable = _env_bool("TQDM_DISABLE", False)


config = Config()
if not config.obs_dir.exists() and any(config.data_root.glob("obs_ep_*.npy")):
    config.obs_dir = config.data_root
if not config.act_dir.exists() and any(config.data_root.glob("act_ep_*.npy")):
    config.act_dir = config.data_root
config.save_dir.mkdir(parents=True, exist_ok=True)

dataset = None
model = None
ema_helper = None
ema_model = None
optimizer = None

INITIAL_ACTION_FROM_CFG_SETTINGS = np.array(
    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025],
    dtype=np.float32,
)


def compress_robot_state(raw_state: np.ndarray) -> np.ndarray:
    return np.concatenate([raw_state[:, :6], raw_state[:, 7:9].mean(axis=1, keepdims=True)], axis=1).astype(np.float32)


def build_prev_joint_history(raw_action: np.ndarray) -> np.ndarray:
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
    if raw_obs.shape[0] != prev_joint_raw.shape[0]:
        n = min(raw_obs.shape[0], prev_joint_raw.shape[0])
        raw_obs = raw_obs[:n]
        prev_joint_raw = prev_joint_raw[:n]

    joint_pos = raw_obs[:, 0:9]
    joint_vel = raw_obs[:, 9:18]
    pose_command = raw_obs[:, 18:25]

    joint_pos_processed = compress_robot_state(joint_pos)
    joint_vel_processed = compress_robot_state(joint_vel)
    return np.concatenate(
        [joint_pos_processed, joint_vel_processed, pose_command, prev_joint_raw],
        axis=1,
    ).astype(np.float32)


def preprocess_reach_action(raw_action: np.ndarray) -> np.ndarray:
    if raw_action.shape[1] == 7:
        return raw_action.astype(np.float32)
    if raw_action.shape[1] >= 9:
        return compress_robot_state(raw_action)
    raise ValueError(f"Unsupported raw action dim: {raw_action.shape[1]}")


def compute_window_weights(n: int, chunk_size: int) -> np.ndarray:
    t = np.arange(n)
    start_min = np.maximum(0, t - chunk_size + 1)
    start_max = np.minimum(t, n - chunk_size)
    w = start_max - start_min + 1
    return np.maximum(w, 0).astype(np.float64)


def build_episode_index(obs_dir: Path, act_dir: Path, chunk_size: int):
    obs_paths = sorted(obs_dir.glob("obs_ep_*.npy"))
    episodes = []
    for obs_path in tqdm(obs_paths, desc="Indexing episodes", disable=config.tqdm_disable):
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

    for obs_path, act_path, n in tqdm(episodes, desc="Computing stats", disable=config.tqdm_disable):
        raw_obs = np.load(obs_path)
        raw_act = np.load(act_path)
        n = min(len(raw_obs), len(raw_act))
        if n < chunk_size:
            continue
        raw_obs = raw_obs[:n]
        raw_act = raw_act[:n]

        proc_prev_joint = build_prev_joint_history(raw_act)
        proc_obs = preprocess_reach_obs(raw_obs, proc_prev_joint)
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
        self._cache_obs = None
        self._cache_act = None

    def __len__(self):
        return self.total_windows

    def _load_episode(self, ep_idx: int):
        if self._cache_ep == ep_idx:
            return self._cache_obs, self._cache_act
        obs_path, act_path, _ = self.episodes[ep_idx]
        obs = np.load(obs_path, mmap_mode="r")
        act = np.load(act_path, mmap_mode="r")
        self._cache_ep = ep_idx
        self._cache_obs = obs
        self._cache_act = act
        return obs, act

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_windows:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.total_windows}")

        ep_idx = int(np.searchsorted(self.cum_windows, idx, side="right"))
        prev = 0 if ep_idx == 0 else self.cum_windows[ep_idx - 1]
        local_start = int(idx - prev)

        obs_raw, act_raw = self._load_episode(ep_idx)
        obs_slice = np.asarray(obs_raw[local_start : local_start + 1])
        act_slice = np.asarray(act_raw[local_start : local_start + self.chunk_size])
        prev_joint_slice = np.asarray(
            INITIAL_ACTION_FROM_CFG_SETTINGS[None, :]
            if local_start == 0
            else preprocess_reach_action(np.asarray(act_raw[local_start - 1 : local_start]))
        )

        obs = preprocess_reach_obs(obs_slice, prev_joint_slice)[0]
        act = preprocess_reach_action(act_slice)

        obs = (obs - self.obs_mean) / self.obs_std
        act = (act - self.act_mean) / self.act_std

        return torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(act.astype(np.float32))


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


def initialize_log_files() -> None:
    config.log_txt_path.write_text(f"Flow Matching {config.model_arch} training log\n", encoding="utf-8")
    config.log_md_path.write_text(
        f"# Flow Matching {config.model_arch} Training Log\n\n"
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
            "model_variant": "reach_flow_joint_autoreg_v1",
            "best_val": best_val,
        },
        path,
    )


def compute_flow_target(action_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise = torch.randn_like(action_chunk)
    t = torch.rand(action_chunk.size(0), device=action_chunk.device)
    t_view = t.view(-1, 1, 1)
    x_t = t_view * action_chunk + (1.0 - t_view) * noise
    target_velocity = action_chunk - noise
    return x_t, t, target_velocity


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for obs, action_chunk in loader:
            obs = obs.to(device)
            action_chunk = action_chunk.to(device).permute(0, 2, 1)
            x_t, t, target_velocity = compute_flow_target(action_chunk)
            pred_velocity = model(x_t, t, obs)
            losses.append(F.mse_loss(pred_velocity, target_velocity).item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def sample_action_chunk(model: nn.Module, obs_batch: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
    model.eval()
    num_steps = config.flow_steps if num_steps is None else num_steps
    x = torch.randn(obs_batch.size(0), config.processed_action_dim, config.chunk_size, device=device)
    dt = 1.0 / float(num_steps)

    for i in range(num_steps):
        t = torch.full((obs_batch.size(0),), i * dt, device=device)
        velocity = model(x, t, obs_batch)
        x = x + velocity * dt

    return x.permute(0, 2, 1)


def main() -> None:
    global dataset, model, ema_helper, ema_model, optimizer

    print(f"[INFO] data_root={config.data_root}")
    print(f"[INFO] chunk_size={config.chunk_size} batch_size={config.batch_size} epochs={config.num_epochs}")
    print(f"[INFO] model_arch={config.model_arch}")
    print(f"[INFO] device={device} save_dir={config.save_dir}")

    episodes = build_episode_index(config.obs_dir, config.act_dir, config.chunk_size)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(episodes))
    train_count = int(len(episodes) * 0.95)
    train_episodes = [episodes[i] for i in perm[:train_count]]
    val_episodes = [episodes[i] for i in perm[train_count:]]

    obs_mean, obs_std, act_mean, act_std = compute_stats(train_episodes, config.chunk_size)
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
    dataset = train_dataset

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    if config.model_arch == "stanford_dp":
        model = DiffusionPolicyConditionalUnet1D(
            input_dim=config.processed_action_dim,
            global_cond_dim=config.processed_obs_dim,
            diffusion_step_embed_dim=config.time_embed_dim,
            down_dims=tuple(config.down_dims),
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale,
        ).to(device)
    elif config.model_arch == "simple_unet":
        model = ConditionalUnet1D(
            action_dim=config.processed_action_dim,
            obs_dim=config.processed_obs_dim,
            base_channels=config.base_channels,
            time_embed_dim=config.time_embed_dim,
            obs_embed_dim=config.obs_embed_dim,
        ).to(device)
    else:
        raise ValueError(f"Unsupported MODEL_ARCH={config.model_arch!r}. Use 'stanford_dp' or 'simple_unet'.")
    ema_helper = EMA(model, config.ema_decay) if config.use_ema else None
    ema_model = copy.deepcopy(model) if config.use_ema else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val = float("inf")
    initialize_log_files()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}", disable=config.tqdm_disable)

        for obs, action_chunk in pbar:
            obs = obs.to(device)
            action_chunk = action_chunk.to(device).permute(0, 2, 1)

            x_t, t, target_velocity = compute_flow_target(action_chunk)
            pred_velocity = model(x_t, t, obs)
            loss = F.mse_loss(pred_velocity, target_velocity)

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
        print(f"epoch={epoch:03d} train_loss={epoch_train_loss:.6f} val_loss={val_loss:.6f}")
        append_epoch_log(epoch, epoch_train_loss, val_loss)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(config.checkpoint_path, best_val, epoch)

        if epoch % config.checkpoint_save_every == 0:
            periodic_checkpoint_path = config.save_dir / f"reach_flow_match_epoch_{epoch:03d}.pt"
            save_checkpoint(periodic_checkpoint_path, best_val, epoch)

    final_eval_model = ema_model if ema_model is not None else model
    final_val_loss = evaluate(final_eval_model, val_loader)
    print(f"final_val_loss={final_val_loss:.6f}")
    append_final_log(final_val_loss, best_val)

    batch_obs, _, = next(iter(val_loader))
    batch_obs = batch_obs[:4].to(device)
    pred_chunks = sample_action_chunk(final_eval_model, batch_obs)
    act_mean_t = torch.from_numpy(dataset.act_mean).to(device)
    act_std_t = torch.from_numpy(dataset.act_std).to(device)
    pred_chunks_denorm = pred_chunks * act_std_t + act_mean_t
    print(f"preview_pred_shape={tuple(pred_chunks_denorm.shape)}")


if __name__ == "__main__":
    main()
