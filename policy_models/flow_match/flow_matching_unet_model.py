import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
