"""
Date: 2024-03-25 20:19:34
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-25 20:19:35
FilePath: /pyutility/pyutils/loss/alpha_divergence.py
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["f_divergence", "adaptive_loss", "AdaptiveLossSoft"]


def f_divergence(
    q_logits: Tensor, p_logits: Tensor, alpha: float, iw_clip: float = 1e3
) -> Tuple[Tensor, Tensor]:
    """https://github.com/facebookresearch/AlphaNet/blob/master/loss_ops.py
    title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
    author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
    """
    assert isinstance(alpha, float)
    q_prob = F.softmax(q_logits.data, dim=1)
    p_prob = F.softmax(p_logits.data, dim=1)
    # gradient is only backpropagated here
    q_log_prob = F.log_softmax(q_logits, dim=1)

    importance_ratio = p_prob / q_prob
    if abs(alpha) < 1e-3:
        importance_ratio.clamp_(0, iw_clip).log_()
        f = -importance_ratio
        f_base = 0
        rho_f = importance_ratio.sub_(1)
    elif abs(alpha - 1.0) < 1e-3:
        # f = importance_ratio * importance_ratio.log()
        f = importance_ratio.log().mul_(importance_ratio)
        f_base = 0
        rho_f = importance_ratio
    else:
        iw_alpha = torch.pow(importance_ratio, alpha).clamp_(0, iw_clip)
        f_base = 1.0 / alpha / (alpha - 1.0)
        # f = iw_alpha / alpha / (alpha - 1.0)
        f = iw_alpha.mul(f_base)
        # f_base = 1.0 / alpha / (alpha - 1.0)
        rho_f = iw_alpha.div(alpha).add_(f_base)

    # loss = torch.sum(q_prob * (f - f_base), dim=1)
    loss = f.sub_(f_base).mul_(q_prob).sum(dim=1)
    grad_loss = -q_prob.mul_(rho_f).mul(q_log_prob).sum(dim=1)
    return loss, grad_loss


def adaptive_loss(
    output: Tensor,
    target: Tensor,
    alpha_min: float,
    alpha_max: float,
    iw_clip: float = 1e3,
    reduction: str = "mean",
) -> Tensor:
    loss_left, grad_loss_left = f_divergence(output, target, alpha_min, iw_clip=iw_clip)
    loss_right, grad_loss_right = f_divergence(
        output, target, alpha_max, iw_clip=iw_clip
    )

    ind = torch.gt(loss_left, loss_right).float()
    loss = ind * grad_loss_left + (1.0 - ind) * grad_loss_right

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class AdaptiveLossSoft(torch.nn.modules.loss._Loss):
    """https://github.com/facebookresearch/AlphaNet/blob/master/loss_ops.py
    title={AlphaNet: Improved Training of Supernet with Alpha-Divergence},
    author={Wang, Dilin and Gong, Chengyue and Li, Meng and Liu, Qiang and Chandra, Vikas},
    """

    def __init__(
        self, alpha_min: float, alpha_max: float, iw_clip: float = 1e3
    ) -> None:
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.iw_clip = iw_clip

    def forward(
        self,
        output: Tensor,
        target: Tensor,
        alpha_min: Optional[float] = None,
        alpha_max: Optional[float] = None,
    ) -> Tensor:
        alpha_min = alpha_min or self.alpha_min
        alpha_max = alpha_max or self.alpha_max

        return adaptive_loss(
            output,
            target,
            alpha_min,
            alpha_max,
            iw_clip=self.iw_clip,
            reduction=self.reduction,
        )
