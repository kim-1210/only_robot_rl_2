import os
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

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
    spec = importlib.util.spec_from_file_location("reach_lift_transformer_policy_model_for_flow", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load transformer module from {module_path}")
    spec.loader.exec_module(module)
    return module.ConditionalTransformer1D


def _load_diffusion_unet_class():
    module_path = Path(__file__).resolve().parents[2] / "diffusion" / "reach_lift" / "diffusion_policy_unet.py"
    spec = importlib.util.spec_from_file_location("reach_lift_diffusion_policy_unet_for_flow", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load diffusion U-Net module from {module_path}")
    spec.loader.exec_module(module)
    return module.DiffusionPolicyConditionalUnet1D


DiffusionPolicyConditionalUnet1D = _load_diffusion_unet_class()
ConditionalTransformer1D = _load_diffusion_transformer_class()


def _compress_robot_state(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :6], x[:, 7:9].mean(dim=1, keepdim=True)], dim=1)


def _compress_robot_state_lr(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :6], x[:, 7:9]], dim=1)


def _is_reach_lift_flow_joint_hist_lr_variant(model_variant: str | None) -> bool:
    if model_variant is None:
        return False
    return str(model_variant).startswith("reach_lift_flow_joint_hist")


def _is_reach_lift_flow_joint_hist_matched_variant(model_variant: str | None) -> bool:
    if model_variant is None:
        return False
    return str(model_variant).startswith("reach_lift_flow_joint_hist_matched")


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


def _preprocess_obs_tensor_reach_lift_frame_lr(obs: torch.Tensor, prev_joint_raw: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 1:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] < 28:
        raise ValueError(f"Expected obs dim >= 28, got {obs.shape[-1]}")

    joint_pos = obs[:, 0:9]
    joint_vel = obs[:, 9:18]
    object_position = obs[:, 18:21]
    pose_command = obs[:, 21:28]
    return torch.cat(
        [_compress_robot_state_lr(joint_pos), _compress_robot_state_lr(joint_vel), object_position, pose_command, prev_joint_raw],
        dim=1,
    )


def _preprocess_obs_tensor_reach_lift_frame_matched(obs: torch.Tensor, prev_joint_raw: torch.Tensor) -> torch.Tensor:
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


def _infer_model_config(cfg: dict, state_dict: dict, ckpt_model_variant: str | None = None) -> dict:
    inferred = dict(cfg)
    model_arch = inferred.get("model_arch")
    if model_arch is None:
        model_arch = "stanford_dp" if "diffusion_step_encoder.1.weight" in state_dict else "simple_unet"
    inferred["model_arch"] = model_arch

    if model_arch == "transformer":
        inferred.setdefault("processed_obs_dim", state_dict["obs_encoder.0.weight"].shape[1])
        inferred.setdefault("processed_action_dim", state_dict["action_in.weight"].shape[1])
        inferred.setdefault("chunk_size", state_dict["pos_emb"].shape[1])
        inferred.setdefault("transformer_dim", state_dict["action_in.weight"].shape[0])
        inferred.setdefault("transformer_heads", 8)
        layer_ids = {
            int(key.split(".")[2])
            for key in state_dict
            if key.startswith("encoder.layers.") and key.split(".")[2].isdigit()
        }
        inferred.setdefault("transformer_layers", max(layer_ids) + 1 if layer_ids else 8)
        inferred.setdefault("transformer_ff_dim", state_dict["encoder.layers.0.linear1.weight"].shape[0])
        inferred.setdefault("transformer_dropout", 0.1)
    elif model_arch == "stanford_dp":
        inferred.setdefault("processed_obs_dim", state_dict["global_cond_encoder.0.weight"].shape[1])
        inferred.setdefault("processed_action_dim", state_dict["final_conv.1.weight"].shape[0])
        inferred.setdefault("time_embed_dim", state_dict["diffusion_step_encoder.1.weight"].shape[1])
        down_dims = []
        idx = 0
        while f"down_modules.{idx}.0.blocks.0.block.0.weight" in state_dict:
            down_dims.append(state_dict[f"down_modules.{idx}.0.blocks.0.block.0.weight"].shape[0])
            idx += 1
        inferred.setdefault("down_dims", tuple(down_dims) if down_dims else (192, 384, 768))
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
    inferred.setdefault("flow_steps", 50)
    inferred.setdefault("inference_noise_mode", "randn")
    inferred.setdefault(
        "model_variant",
        ckpt_model_variant
        or (
            "reach_lift_flow_joint_hist_matched_v1"
            if int(inferred["processed_action_dim"]) == 7 and int(inferred["processed_obs_dim"]) > 31
            else "reach_lift_flow_joint_hist_lr_v1"
            if int(inferred["processed_action_dim"]) == 8 and int(inferred["processed_obs_dim"]) > 34
            else "reach_flow_joint_autoreg_v1"
        ),
    )
    if _is_reach_lift_flow_joint_hist_matched_variant(inferred["model_variant"]):
        inferred.setdefault("processed_obs_frame_dim", 31)
        inferred.setdefault("obs_history_steps", int(inferred["processed_obs_dim"]) // int(inferred["processed_obs_frame_dim"]))
    elif _is_reach_lift_flow_joint_hist_lr_variant(inferred["model_variant"]):
        inferred.setdefault("processed_obs_frame_dim", 34)
        inferred.setdefault("obs_history_steps", int(inferred["processed_obs_dim"]) // int(inferred["processed_obs_frame_dim"]))
    else:
        inferred.setdefault("obs_history_steps", 1)
        inferred.setdefault("processed_obs_frame_dim", int(inferred["processed_obs_dim"]))
    return inferred


class Policy:
    def __init__(
        self,
        model: nn.Module,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_mean: np.ndarray,
        act_std: np.ndarray,
        chunk_size: int,
        device: torch.device,
        flow_steps: int,
        model_variant: str,
        obs_history_steps: int = 1,
        processed_obs_frame_dim: int | None = None,
        inference_noise_mode: str = "randn",
    ):
        self.model = model
        self.obs_mean = torch.from_numpy(obs_mean).to(device)
        self.obs_std = torch.from_numpy(obs_std).to(device)
        self.act_mean = torch.from_numpy(act_mean).to(device)
        self.act_std = torch.from_numpy(act_std).to(device)
        self.chunk_size = chunk_size
        self.device = device
        self.flow_steps = flow_steps
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

        if _is_reach_lift_flow_joint_hist_matched_variant(self.model_variant):
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            if obs.shape[-1] != self.obs_mean.shape[-1]:
                current_frame = _preprocess_obs_tensor_reach_lift_frame_matched(obs, self.prev_action_raw)
                self.obs_history, obs = _stack_obs_history_frames(
                    current_frame,
                    self.obs_history,
                    self.obs_history_steps,
                )
        elif _is_reach_lift_flow_joint_hist_lr_variant(self.model_variant):
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            if obs.shape[-1] != self.obs_mean.shape[-1]:
                current_frame = _preprocess_obs_tensor_reach_lift_frame_lr(obs, self.prev_action_raw)
                self.obs_history, obs = _stack_obs_history_frames(
                    current_frame,
                    self.obs_history,
                    self.obs_history_steps,
                )
        else:
            if self.prev_action_raw is None or self.prev_action_raw.shape[0] != batch:
                self.prev_action_raw = torch.tensor(
                    [-1.57, 0.0, 0.7, 0.87, 1.57, 1.57, 0.025],
                    device=self.device,
                    dtype=torch.float32,
                ).unsqueeze(0).repeat(batch, 1)
            obs = _preprocess_obs_tensor_reach(obs, self.prev_action_raw, self.obs_mean.shape[-1])

        obs = (obs - self.obs_mean) / self.obs_std

        if self.inference_noise_mode == "zeros":
            x = torch.zeros(batch, self.act_mean.shape[-1], self.chunk_size, device=self.device)
        else:
            x = torch.randn(batch, self.act_mean.shape[-1], self.chunk_size, device=self.device)
        dt = 1.0 / float(self.flow_steps)

        for i in range(self.flow_steps):
            t = torch.full((batch,), i * dt, device=self.device)
            velocity = self.model(x, t, obs)
            x = x + velocity * dt

        action_chunks = x.permute(0, 2, 1)
        action_chunks = action_chunks * self.act_std + self.act_mean

        if output_horizon is not None and output_horizon != self.chunk_size:
            if output_horizon < self.chunk_size:
                action_chunks = action_chunks[:, :output_horizon, :]
            else:
                pad = output_horizon - self.chunk_size
                last = action_chunks[:, -1:, :].repeat(1, pad, 1)
                action_chunks = torch.cat([action_chunks, last], dim=1)

        executed_steps = execute_horizon if execute_horizon is not None else 1
        executed_steps = max(1, min(executed_steps, action_chunks.shape[1]))
        self.prev_action_raw = action_chunks[:, executed_steps - 1, :].detach()

        return action_chunks


def load_policy(device: str | torch.device = "cpu", output_horizon: int | None = None, execute_horizon: int | None = None):
    del output_horizon, execute_horizon

    device = torch.device(device)
    default_ckpt = Path(__file__).resolve().parent / "reach_lift_flow_match_artifacts_v2" / "reach_lift_flow_match.pt"
    default_stats = Path(__file__).resolve().parent / "reach_lift_flow_match_artifacts_v2" / "reach_lift_flow_match_stats.npz"

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
    flow_steps = int(cfg["flow_steps"])
    model_variant = str(cfg["model_variant"])
    obs_history_steps = int(cfg.get("obs_history_steps", 1))
    processed_obs_frame_dim = int(cfg.get("processed_obs_frame_dim", obs_dim))
    inference_noise_mode = str(cfg.get("inference_noise_mode", "randn"))

    if cfg.get("model_arch") == "transformer":
        if ConditionalTransformer1D is None:
            raise RuntimeError("Transformer model code was removed. Use a U-Net flow-matching checkpoint.")
        model = ConditionalTransformer1D(
            action_dim=act_dim,
            obs_dim=obs_dim,
            horizon=chunk_size,
            d_model=int(cfg["transformer_dim"]),
            nhead=int(cfg["transformer_heads"]),
            num_layers=int(cfg["transformer_layers"]),
            dim_feedforward=int(cfg["transformer_ff_dim"]),
            dropout=float(cfg.get("transformer_dropout", 0.1)),
        ).to(device)
    elif cfg.get("model_arch") == "stanford_dp":
        model = DiffusionPolicyConditionalUnet1D(
            input_dim=act_dim,
            global_cond_dim=obs_dim,
            diffusion_step_embed_dim=int(cfg["time_embed_dim"]),
            down_dims=tuple(cfg.get("down_dims", (192, 384, 768))),
            kernel_size=int(cfg.get("kernel_size", 5)),
            n_groups=int(cfg.get("n_groups", 8)),
            cond_predict_scale=bool(cfg.get("cond_predict_scale", True)),
        ).to(device)
    else:
        model = ConditionalUnet1D(
            act_dim,
            obs_dim,
            int(cfg["base_channels"]),
            int(cfg["time_embed_dim"]),
            int(cfg["obs_embed_dim"]),
        ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    stats = np.load(stats_path)
    obs_mean = stats["obs_mean"].astype(np.float32)
    obs_std = stats["obs_std"].astype(np.float32)
    act_mean = stats["act_mean"].astype(np.float32)
    act_std = stats["act_std"].astype(np.float32)

    return Policy(
        model,
        obs_mean,
        obs_std,
        act_mean,
        act_std,
        chunk_size,
        device,
        flow_steps,
        model_variant,
        obs_history_steps=obs_history_steps,
        processed_obs_frame_dim=processed_obs_frame_dim,
        inference_noise_mode=inference_noise_mode,
    )
