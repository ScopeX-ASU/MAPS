"""
Date: 2024-03-25 20:29:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-25 20:30:40
FilePath: /pyutility/pyutils/loss/dual_focal.py
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["dual_focal_loss", "DualFocalLoss"]


def dual_focal_loss(
    input: Tensor, target: Tensor, gamma: float = 5, reduction: str = "mean"
) -> Tensor:
    if input.dim() > 2:
        input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
        input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
        input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
    target = target.view(-1, 1)

    logp_k = F.log_softmax(input, dim=1)
    softmax_logits = logp_k.exp()
    logp_k = logp_k.gather(1, target)
    logp_k = logp_k.view(-1)
    p_k = logp_k.exp()  # p_k: probility at target label
    p_j_mask = (
        torch.lt(softmax_logits, p_k.reshape(p_k.shape[0], 1)) * 1
    )  # mask all logit larger and equal than p_k
    p_j = torch.topk(p_j_mask * softmax_logits, 1)[0].squeeze()

    loss = -1 * (1 - p_k + p_j) ** gamma * logp_k

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class DualFocalLoss(torch.nn.modules.loss._Loss):
    """https://arxiv.org/abs/2305.13665
    Dual Focal Loss for Calibration
    """

    def __init__(self, gamma=5):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        return dual_focal_loss(input, target, self.gamma, self.reduction)
