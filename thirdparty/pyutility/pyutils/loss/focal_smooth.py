from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["focal_smooth_loss", "FocalLossWithSmoothing"]


def _estimate_difficulty_level(logits, label, num_classes, gamma):
    """
    :param logits:
    :param label:
    :return:
    """
    one_hot_key = torch.nn.functional.one_hot(label, num_classes=num_classes)
    if len(one_hot_key.shape) == 4:
        one_hot_key = one_hot_key.permute(0, 3, 1, 2)
    if one_hot_key.device != logits.device:
        one_hot_key = one_hot_key.to(logits.device)
    pt = one_hot_key * torch.nn.functional.softmax(logits, -1)
    difficulty_level = torch.pow(1 - pt, gamma)
    return difficulty_level


def _estimate_difficulty_level_binary(logits, label, num_classes, alpha, gamma):
    """
    :param logits:
    :param label:
    :return:
    """
    one_hot_key = torch.nn.functional.one_hot(label, num_classes=num_classes)
    if len(one_hot_key.shape) == 4:
        one_hot_key = one_hot_key.permute(0, 3, 1, 2)
    if one_hot_key.device != logits.device:
        one_hot_key = one_hot_key.to(logits.device)
    pt = one_hot_key * torch.nn.functional.softmax(logits, -1)
    # pred = logits.data.max(1)[1]
    # fn = (label == 1) & (pred == 0)
    if alpha is not None:
        neg = label == 1
        alpha = torch.where(neg, 1 + alpha, 1 - alpha)
        difficulty_level = alpha.unsqueeze(1) * torch.pow(1 - pt, gamma)
    else:
        difficulty_level = torch.pow(1 - pt, gamma)

    return difficulty_level


def focal_smooth_loss(
    logits, label, num_classes, gamma, lb_smooth, ignore_index, alpha, reduction="mean"
):
    """
    :param logits: (batch_size, class, height, width)
    :param label:
    :return:
    """
    logits = logits.float()
    if num_classes == 2:
        difficulty_level = _estimate_difficulty_level_binary(
            logits, label, num_classes, alpha, gamma
        )
    else:
        difficulty_level = _estimate_difficulty_level(logits, label, num_classes, gamma)

    with torch.no_grad():
        label = label.clone().detach()
        if ignore_index is not None:
            ignore = label.eq(ignore_index)
            label[ignore] = 0
        lb_pos, lb_neg = 1.0 - lb_smooth, lb_smooth / (num_classes - 1)
        lb_one_hot = (
            torch.empty_like(logits)
            .fill_(lb_neg)
            .scatter_(1, label.unsqueeze(1), lb_pos)
            .detach()
        )
    logs = torch.nn.functional.log_softmax(logits, dim=-1)
    loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
    if ignore_index is not None:
        loss[ignore] = 0
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


class FocalLossWithSmoothing(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        num_classes: int,
        gamma: int = 2,
        lb_smooth: float = 0.1,
        ignore_index: int = None,
        alpha: float = None,
    ):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super().__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._ignore_index = ignore_index
        self._log_softmax = torch.nn.LogSoftmax(dim=-1)
        self._alpha = alpha

        if self._num_classes <= 1:
            raise ValueError("The number of classes must be 2 or higher")
        if self._gamma < 0:
            raise ValueError("Gamma must be 0 or higher")
        if self._alpha is not None:
            if self._alpha < 0 or self._alpha > 1:
                raise ValueError("Alpha must be 0 < alpha < 1")

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        return focal_smooth_loss(
            logits,
            label,
            self._num_classes,
            self._gamma,
            self._lb_smooth,
            self._ignore_index,
            self._alpha,
            reduction=self.reduction,
        )
