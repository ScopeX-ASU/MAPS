"""
Date: 2024-03-26 13:44:08
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-26 13:45:29
FilePath: /pyutility/pyutils/loss/kd.py
"""

"""
Date: 2024-03-26 13:35:41
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-26 13:35:54
FilePath: /SparseTeMPO/home/jiaqigu/projects/pyutility/pyutils/loss/kl_mixed.py
"""

"""
Date: 2024-03-25 20:12:06
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-25 20:13:39
FilePath: /pyutility/pyutils/loss/kl_mixed.py
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import normalize

__all__ = ["kd_loss", "KDLoss", "KLLossMixed"]


def kd_loss(
    y,
    teacher_scores=None,
    labels=None,
    T: float = 2,
    ce_weight: float = 0.1,
    kd_weight: float = 9.0,
    logit_stand: bool = False,
):
    loss = 0
    if ce_weight > 0 and labels is not None:
        loss = loss + ce_weight * F.cross_entropy(y, labels)

    if kd_weight > 0 and teacher_scores is not None:
        y = normalize(y) if logit_stand else y
        teacher_scores = normalize(teacher_scores) if logit_stand else teacher_scores
        p = F.log_softmax(y / T, dim=1)
        q = F.softmax(teacher_scores / T, dim=1)
        l_kl = F.kl_div(p, q, reduction="sum") * (T**2) / y.shape[0]
        loss = loss + kd_weight * l_kl
    return loss


class KDLoss(torch.nn.modules.loss._Loss):
    """
    description: Knowledge distillation loss function combines hard target loss and soft target loss, [https://github.com/szagoruyko/attention-transfer/blob/master/utils.py#L10]
    y {tensor.Tensor} Model output logits from the student model
    teacher_scores {tensor.Tensor} Model output logits from the teacher model
    labels {tensor.LongTensor} Hard labels, Ground truth
    T {scalar, Optional} temperature of the softmax function to make the teacher score softer. Default is 6 when accuracy is high, e.g., >95%. Typical value 1~20 [https://zhuanlan.zhihu.com/p/83456418]
    alpha {scalar, Optional} interpolation between hard and soft target loss. Default set to 0.9 for distillation mode. When hard target loss is very small (alpha=0.9), gets the best results. [https://zhuanlan.zhihu.com/p/102038521]
    return loss {tensor.Tensor} loss function
    """

    def __init__(
        self,
        T: float = 2.0,
        ce_weight: float = 0.1,
        kd_weight: float = 9,
        logit_stand=False,
    ) -> None:
        """
        Args:
            T (float, optional): Temperature for softmax. Defaults to 6.0.
            alpha (float, optional): weighting factor for soft loss from teacher. Defaults to 0.9.
        """
        super().__init__()
        self.T = T
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight
        self.logit_stand = logit_stand

    def forward(
        self,
        y: Tensor,
        teacher_scores: Tensor,
        labels: Tensor,
    ) -> Tensor:

        return kd_loss(
            y,
            teacher_scores,
            labels,
            self.T,
            self.ce_weight,
            self.kd_weight,
            self.logit_stand,
        )


## deprecated
KLLossMixed = KDLoss
