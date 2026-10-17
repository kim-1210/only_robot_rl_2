from __future__ import annotations

import torch


def no_termination(num_envs: int, device: torch.device | str) -> torch.Tensor:
    return torch.zeros(num_envs, dtype=torch.bool, device=device)
