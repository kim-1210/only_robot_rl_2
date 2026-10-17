import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

try:
    from .diffusion_policy_unet import DiffusionPolicyConditionalUnet1D
except ImportError:
    from diffusion_policy_unet import DiffusionPolicyConditionalUnet1D


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


def _compress_legacy_pair(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :7], x[:, 7:9].mean(dim=1, keepdim=True)], dim=1)


def _compress_robot_state(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :6], x[:, 7:9].mean(dim=1, keepdim=True)], dim=1)


def _is_reach_lift_joint_hist_variant(model_variant: str | None) -> bool:
    if model_variant is None:
        return False
    return str(model_variant).startswith("reach_lift_joint_hist")


def _expand_legacy_processed_action(action: torch.Tensor) -> torch.Tensor:
    return torch.cat([action[:, :7], action[:, 7:8], action[:, 7:8]], dim=1)


def _preprocess_obs_tensor_legacy(obs: torch.Tensor, prev_action_raw: torch.Tensor, processed_obs_dim: int) -> torch.Tensor:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] == processed_obs_dim:
        return obs
    if obs.shape[-1] < 25:
        raise ValueError(f"Expected obs dim >= 25, got {obs.shape[-1]}")

    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    object_position = obs[:, 18:21]
    target_object_position = obs[:, 21:25]

    joint_pos_processed = _compress_legacy_pair(joint_pos)
    joint_vel_processed = _compress_legacy_pair(joint_vel)
    processed_last_action = _compress_legacy_pair(prev_action_raw)

    return torch.cat(
        [joint_pos_processed, joint_vel_processed, object_position, target_object_position, processed_last_action],
        dim=1,
    )


def _preprocess_obs_tensor_v2(obs: torch.Tensor, processed_obs_dim: int) -> torch.Tensor:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] == processed_obs_dim:
        return obs
    if obs.shape[-1] < 35:
        raise ValueError(f"Expected obs dim >= 35, got {obs.shape[-1]}")

    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    object_position = obs[:, 18:21]
    pose_command = obs[:, 21:28]
    last_action = obs[:, 28:35]

    return torch.cat(
        [_compress_robot_state(joint_pos), _compress_robot_state(joint_vel), object_position, pose_command, last_action],
        dim=1,
    )


def _preprocess_obs_tensor_v3(obs: torch.Tensor, prev_joint_raw: torch.Tensor, processed_obs_dim: int) -> torch.Tensor:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] == processed_obs_dim:
        return obs
    if obs.shape[-1] < 28:
        raise ValueError(f"Expected obs dim >= 28, got {obs.shape[-1]}")

    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    object_position = obs[:, 18:21]
    pose_command = obs[:, 21:28]

    return torch.cat(
        [_compress_robot_state(joint_pos), _compress_robot_state(joint_vel), object_position, pose_command, prev_joint_raw],
        dim=1,
    )


def _preprocess_obs_tensor_v3_frame(obs: torch.Tensor, prev_joint_raw: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] < 28:
        raise ValueError(f"Expected obs dim >= 28, got {obs.shape[-1]}")

    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    object_position = obs[:, 18:21]
    pose_command = obs[:, 21:28]
    return torch.cat(
        [_compress_robot_state(joint_pos), _compress_robot_state(joint_vel), object_position, pose_command, prev_joint_raw],
        dim=1,
    )


def _preprocess_obs_tensor_reach(obs: torch.Tensor, prev_joint_raw: torch.Tensor, processed_obs_dim: int) -> torch.Tensor:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] == processed_obs_dim:
        return obs
    if obs.shape[-1] < 25:
        raise ValueError(f"Expected obs dim >= 25, got {obs.shape[-1]}")

    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    pose_command = obs[:, 18:25]
    return torch.cat(
        [_compress_robot_state(joint_pos), _compress_robot_state(joint_vel), pose_command, prev_joint_raw],
        dim=1,
    )


def _stack_obs_history_frames(
    current_frame: torch.Tensor,
    history: torch.Tensor | None,
    history_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if current_frame.ndim != 2:
        raise ValueError(f"Expected current_frame with shape [B, D], got {tuple(current_frame.shape)}")
    batch, feat_dim = current_frame.shape
    if history_steps <= 1:
        return current_frame.unsqueeze(1), current_frame

    if history is None or history.shape[0] != batch or history.shape[1] != history_steps or history.shape[2] != feat_dim:
        history = current_frame.unsqueeze(1).repeat(1, history_steps, 1)
    else:
        history = torch.cat([history[:, 1:, :], current_frame.unsqueeze(1)], dim=1)

    return history, history.reshape(batch, history_steps * feat_dim)


def _legacy_action_to_env_action(action_chunks: torch.Tensor) -> torch.Tensor:
    env_action = torch.cat([action_chunks[:, :, :6], action_chunks[:, :, 7:8]], dim=-1)
    env_action[:, :, -1] = torch.where(env_action[:, :, -1] <= 0.0125, -1.0, 1.0)
    return env_action


def _infer_model_config(cfg: dict, state_dict: dict, ckpt_model_variant: str | None = None) -> dict:
    inferred = dict(cfg)
    model_arch = inferred.get("model_arch")
    if model_arch is None:
        model_arch = "stanford_dp" if "diffusion_step_encoder.1.weight" in state_dict else "simple_unet"
    inferred["model_arch"] = model_arch

    if model_arch == "stanford_dp":
        inferred.setdefault("processed_obs_dim", state_dict["global_cond_encoder.0.weight"].shape[1])
        inferred.setdefault("processed_action_dim", state_dict["final_conv.1.weight"].shape[0])
        inferred.setdefault("time_embed_dim", state_dict["diffusion_step_encoder.1.weight"].shape[1])
        down_dims = []
        idx = 0
        while f"down_modules.{idx}.0.blocks.0.block.0.weight" in state_dict:
            down_dims.append(state_dict[f"down_modules.{idx}.0.blocks.0.block.0.weight"].shape[0])
            idx += 1
        inferred.setdefault("down_dims", tuple(down_dims) if down_dims else (128, 256, 512))
        inferred.setdefault("kernel_size", state_dict["final_conv.0.block.0.weight"].shape[-1])
        inferred.setdefault("n_groups", 8)
        inferred.setdefault("cond_predict_scale", True)
    else:
        inferred.setdefault("processed_obs_dim", state_dict["obs_mlp.0.weight"].shape[1])
        inferred.setdefault("processed_action_dim", state_dict["output_proj.2.weight"].shape[0])
        inferred.setdefault("obs_embed_dim", state_dict["obs_mlp.0.weight"].shape[0])
        inferred.setdefault("time_embed_dim", state_dict["time_mlp.1.weight"].shape[0])
        inferred.setdefault("base_channels", state_dict["input_proj.weight"].shape[0])
    inferred.setdefault("chunk_size", 16)
    inferred.setdefault("diffusion_steps", 100)
    inferred.setdefault("beta_start", 1e-4)
    inferred.setdefault("beta_end", 2e-2)
    inferred.setdefault("schedule", "cosine")
    inferred.setdefault("inference_noise_mode", "randn")
    inferred.setdefault(
        "model_variant",
        ckpt_model_variant
        or (
            "legacy_v1"
            if int(inferred["processed_action_dim"]) == 8
            else (
                "reach_lift_joint_hist_v1"
                if int(inferred["processed_obs_dim"]) > 31 and int(inferred["processed_obs_dim"]) % 31 == 0
                else (
                    "reach_lift_joint_autoreg_v3"
                    if int(inferred["processed_obs_dim"]) == 31
                    else ("reach_joint_autoreg_v1" if int(inferred["processed_obs_dim"]) == 28 else "reach_lift_v2")
                )
            )
        ),
    )
    if _is_reach_lift_joint_hist_variant(inferred["model_variant"]):
        inferred.setdefault("processed_obs_frame_dim", 31)
        inferred.setdefault("obs_history_steps", int(inferred["processed_obs_dim"]) // int(inferred["processed_obs_frame_dim"]))
    return inferred


class Policy:
    def __init__(self, model: nn.Module, scheduler: DiffusionScheduler, obs_mean: np.ndarray, obs_std: np.ndarray,
                 act_mean: np.ndarray, act_std: np.ndarray, chunk_size: int, device: torch.device,
                 model_variant: str, obs_history_steps: int = 1, processed_obs_frame_dim: int | None = None,
                 inference_noise_mode: str = "randn"):
        self.model = model
        self.scheduler = scheduler
        self.obs_mean = torch.from_numpy(obs_mean).to(device)
        self.obs_std = torch.from_numpy(obs_std).to(device)
        self.act_mean = torch.from_numpy(act_mean).to(device)
        self.act_std = torch.from_numpy(act_std).to(device)
        self.chunk_size = chunk_size
        self.device = device
        self.model_variant = model_variant
        self.obs_history_steps = max(1, int(obs_history_steps))
        self.processed_obs_frame_dim = (
            int(processed_obs_frame_dim)
            if processed_obs_frame_dim is not None
            else int(self.obs_mean.shape[-1] // self.obs_history_steps)
        )
        self.inference_noise_mode = inference_noise_mode
        self.prev_action_raw = None
        self.obs_history = None
        self.model.eval()

    def reset(self) -> None:
        self.prev_action_raw = None
        self.obs_history = None

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor, output_horizon: int | None = None, execute_horizon: int | None = None):
        obs = obs.to(self.device, dtype=torch.float32)
        batch = obs.shape[0] if obs.ndim > 1 else 1
        if self.model_variant == "legacy_v1":
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.0, 0.025, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            obs = _preprocess_obs_tensor_legacy(obs, self.prev_action_raw, self.obs_mean.shape[-1])
        elif self.model_variant == "reach_lift_joint_autoreg_v3":
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            obs = _preprocess_obs_tensor_v3(obs, self.prev_action_raw, self.obs_mean.shape[-1])
        elif _is_reach_lift_joint_hist_variant(self.model_variant):
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            if obs.shape[-1] == self.obs_mean.shape[-1]:
                processed_obs = obs
            else:
                current_frame = _preprocess_obs_tensor_v3_frame(obs, self.prev_action_raw)
                self.obs_history, processed_obs = _stack_obs_history_frames(
                    current_frame,
                    self.obs_history,
                    self.obs_history_steps,
                )
                obs = processed_obs
        elif self.model_variant == "reach_joint_autoreg_v1":
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            obs = _preprocess_obs_tensor_reach(obs, self.prev_action_raw, self.obs_mean.shape[-1])
        else:
            obs = _preprocess_obs_tensor_v2(obs, self.obs_mean.shape[-1])
        obs = (obs - self.obs_mean) / self.obs_std

        if self.inference_noise_mode == "zeros":
            x = torch.zeros(batch, self.act_mean.shape[-1], self.chunk_size, device=self.device)
        else:
            x = torch.randn(batch, self.act_mean.shape[-1], self.chunk_size, device=self.device)

        for step in reversed(range(self.scheduler.num_steps)):
            t_index = torch.full((batch,), step, device=self.device, dtype=torch.long)
            t = t_index.float() / max(self.scheduler.num_steps - 1, 1)
            pred_noise = self.model(x, t, obs)
            x = self.scheduler.ddim_step(x, t_index, pred_noise)

        action_chunks = x.permute(0, 2, 1)
        action_chunks = action_chunks * self.act_std + self.act_mean
        raw_action_chunks = action_chunks
        if self.model_variant == "legacy_v1":
            action_chunks = _legacy_action_to_env_action(action_chunks)

        if output_horizon is not None and output_horizon != self.chunk_size:
            if output_horizon < self.chunk_size:
                action_chunks = action_chunks[:, :output_horizon, :]
            else:
                pad = output_horizon - self.chunk_size
                last = action_chunks[:, -1:, :].repeat(1, pad, 1)
                action_chunks = torch.cat([action_chunks, last], dim=1)

        if self.model_variant == "legacy_v1":
            executed_steps = execute_horizon if execute_horizon is not None else 1
            executed_steps = max(1, min(executed_steps, raw_action_chunks.shape[1]))
            self.prev_action_raw = _expand_legacy_processed_action(raw_action_chunks[:, executed_steps - 1, :]).detach()
        elif self.model_variant == "reach_lift_joint_autoreg_v3" or _is_reach_lift_joint_hist_variant(self.model_variant):
            executed_steps = execute_horizon if execute_horizon is not None else 1
            executed_steps = max(1, min(executed_steps, raw_action_chunks.shape[1]))
            self.prev_action_raw = raw_action_chunks[:, executed_steps - 1, :].detach()
        elif self.model_variant == "reach_joint_autoreg_v1":
            executed_steps = execute_horizon if execute_horizon is not None else 1
            executed_steps = max(1, min(executed_steps, raw_action_chunks.shape[1]))
            self.prev_action_raw = raw_action_chunks[:, executed_steps - 1, :].detach()

        return action_chunks


def load_policy(device: str | torch.device = "cpu", output_horizon: int | None = None, execute_horizon: int | None = None):
    device = torch.device(device)
    default_ckpt = Path(__file__).resolve().parent / "reach_diffusion_unet_artifacts" / "reach_diffusion_unet.pt"
    default_stats = Path(__file__).resolve().parent / "reach_diffusion_unet_artifacts" / "reach_diffusion_stats.npz"

    ckpt_path = Path(os.environ.get("POLICY_CHECKPOINT", default_ckpt)).expanduser().resolve()
    stats_path = Path(os.environ.get("POLICY_STATS", default_stats)).expanduser().resolve()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise RuntimeError("Checkpoint missing model_state_dict.")
    cfg = _infer_model_config(ckpt.get("config", {}), state_dict, ckpt.get("model_variant"))

    obs_dim = int(cfg["processed_obs_dim"])
    act_dim = int(cfg["processed_action_dim"])
    chunk_size = int(cfg["chunk_size"])
    model_arch = cfg["model_arch"]
    diffusion_steps = int(cfg["diffusion_steps"])
    beta_start = float(cfg["beta_start"])
    beta_end = float(cfg["beta_end"])
    schedule = cfg["schedule"]
    model_variant = cfg["model_variant"]
    obs_history_steps = int(cfg.get("obs_history_steps", 1))
    processed_obs_frame_dim = int(cfg.get("processed_obs_frame_dim", obs_dim))
    inference_noise_mode = str(cfg.get("inference_noise_mode", "randn"))

    if model_arch == "stanford_dp":
        model = DiffusionPolicyConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=int(cfg["time_embed_dim"]),
            down_dims=tuple(int(x) for x in cfg["down_dims"]),
            kernel_size=int(cfg["kernel_size"]),
            n_groups=int(cfg["n_groups"]),
            cond_predict_scale=bool(cfg["cond_predict_scale"]),
        ).to(device)
    else:
        base_channels = int(cfg["base_channels"])
        time_embed_dim = int(cfg["time_embed_dim"])
        obs_embed_dim = int(cfg["obs_embed_dim"])
        model = ConditionalUnet1D(act_dim, obs_dim, base_channels, time_embed_dim, obs_embed_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    stats = np.load(stats_path)
    obs_mean = stats["obs_mean"].astype(np.float32)
    obs_std = stats["obs_std"].astype(np.float32)
    act_mean = stats["act_mean"].astype(np.float32)
    act_std = stats["act_std"].astype(np.float32)

    scheduler = DiffusionScheduler(diffusion_steps, beta_start, beta_end, device, schedule=schedule)

    return Policy(
        model,
        scheduler,
        obs_mean,
        obs_std,
        act_mean,
        act_std,
        chunk_size,
        device,
        model_variant,
        obs_history_steps=obs_history_steps,
        processed_obs_frame_dim=processed_obs_frame_dim,
        inference_noise_mode=inference_noise_mode,
    )
