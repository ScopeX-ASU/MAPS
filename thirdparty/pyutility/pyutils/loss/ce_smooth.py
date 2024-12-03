from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["ce_smooth_loss", "CrossEntropyLossSmooth"]


def ce_smooth_loss(output, target, eps, reduction="mean"):
    n_class = output.size(1)
    one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
    target = one_hot.mul_(1 - eps).add_(eps / n_class)
    output_log_prob = F.log_softmax(output, dim=1)
    target.unsqueeze_(1)
    output_log_prob = output_log_prob.unsqueeze(2)
    loss = -torch.bmm(target, output_log_prob)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.eps = label_smoothing

    """cross entropy loss with label smoothing """

    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return ce_smooth_loss(output, target, self.eps, self.reduction)
