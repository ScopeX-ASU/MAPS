"""
Date: 2024-11-24 16:54:54
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 16:57:46
FilePath: /MAPS/core/ai4pde/models/layers/pos_encoding.py
"""

from functools import lru_cache

import torch

__all__ = ["get_linear_pos_enc"]


@lru_cache(maxsize=16)
def get_linear_pos_enc(shape, device):
    ## shape is the spatial dimensions only, do not include batch size or channels
    grids = torch.meshgrid(*[torch.arange(0, size, device=device) for size in shape])
    mesh = torch.stack(grids[::-1], dim=0).unsqueeze(0)  # [1, 2, h, w] real
    return mesh
