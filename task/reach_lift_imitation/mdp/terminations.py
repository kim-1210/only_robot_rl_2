from __future__ import annotations

import torch


def object_below_minimum_height(root_pos_w: torch.Tensor, minimum_height: float) -> torch.Tensor:
    return root_pos_w[:, 2] < minimum_height


def object_at_or_above_height(root_pos_w: torch.Tensor, height: float) -> torch.Tensor:
    return root_pos_w[:, 2] >= height
