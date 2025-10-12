"""
Date: 2024-11-24 15:30:30
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 15:31:13
FilePath: /MAPS/core/ai4pde/models/layers/fourier_feature.py
"""

import numpy as np
import torch
from torch import nn


class LearnableFourierFeatures(nn.Module):
    def __init__(self, pos_dim, f_dim, h_dim, d_dim, g_dim=1, gamma=1.0):
        super(LearnableFourierFeatures, self).__init__()
        assert (
            f_dim % 2 == 0
        ), "number of fourier feature dimensions must be divisible by 2."
        assert (
            d_dim % g_dim == 0
        ), "number of D dimension must be divisible by the number of G dimension."
        enc_f_dim = int(f_dim / 2)
        dg_dim = int(d_dim / g_dim)
        self.Wr = nn.Parameter(torch.randn([enc_f_dim, pos_dim]) * (gamma**2))
        self.mlp = nn.Sequential(
            nn.Linear(f_dim, h_dim), nn.GELU(), nn.Linear(h_dim, dg_dim)
        )
        self.div_term = np.sqrt(f_dim)

    def forward(self, pos):
        # input pos dim: (B L G M)
        # output dim: (B L D)
        # L stands for sequence length. all dimensions must be flattened to a single dimension.
        XWr = torch.matmul(pos, self.Wr.T)
        F = torch.cat([torch.cos(XWr), torch.sin(XWr)], dim=-1) / self.div_term
        Y = self.mlp(F)
        # pos_enc = rearrange(Y, "b l g d -> b l (g d)")
        pos_enc = Y.flatten(-1, -2)

        return pos_enc
