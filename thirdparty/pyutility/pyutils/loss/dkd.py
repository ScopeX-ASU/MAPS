"""
Date: 2024-03-25 20:04:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-25 20:04:23
FilePath: /pyutility/pyutils/loss/decoupled_kd.py
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import normalize

__all__ = ["dkd_loss", "DKDLoss"]


def dkd_loss(
    logits_student_in,
    logits_teacher_in=None,
    target=None,
    ce_weight: float = 1.0,
    kd_alpha: float = 1.0,
    kd_beta: float = 1.0,
    temperature: float = 2.0,
    logit_stand: bool = False,
):
    loss = 0

    if ce_weight > 0 and target is not None:
        loss = loss + ce_weight * F.cross_entropy(logits_student_in, target)

    if kd_alpha > 0 and kd_beta > 0 and logits_teacher_in is not None:
        logits_student = (
            normalize(logits_student_in) if logit_stand else logits_student_in
        )
        logits_teacher = (
            normalize(logits_teacher_in) if logit_stand else logits_teacher_in
        )

        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = _get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction="sum")
            * (temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="sum")
            * (temperature**2)
            / target.shape[0]
        )
        loss = loss + kd_alpha * tckd_loss + kd_beta * nckd_loss
    return loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKDLoss(torch.nn.modules.loss._Loss):
    """
    description: Decoupled Knowledge Distillation(CVPR 2022), [https://github.com/sunshangquan/logit-standardization-KD/blob/master/mdistiller/distillers/DKD.py]
    y {tensor.Tensor} Model output logits from the student model
    teacher_scores {tensor.Tensor} Model output logits from the teacher model
    labels {tensor.LongTensor} Hard labels, Ground truth
    T {scalar, Optional} temperature of the softmax function to make the teacher score softer. Default is 6 when accuracy is high, e.g., >95%. Typical value 1~20 [https://zhuanlan.zhihu.com/p/83456418]
    return loss {tensor.Tensor} loss function
    """

    def __init__(
        self,
        T: float = 2.0,
        alpha: float = 1,
        beta: float = 1,
        logit_stand: bool = True,
    ) -> None:
        """
        Args:
            T (float, optional): Temperature for softmax. Defaults to 2.0.
            alpha (float, optional): weighting factor for soft loss from teacher. Defaults to 1.
            beta (float, optional): weighting factor for hard loss. Defaults to 1.
            typically the weights for this DKD loss is set to 9, while CE loss weight is set to 1.
        """
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.logit_stand = logit_stand

    def forward(
        self,
        y: Tensor,
        teacher_scores: Tensor,
        labels: Tensor,
    ) -> Tensor:

        return dkd_loss(
            y, teacher_scores, labels, self.alpha, self.beta, self.T, self.logit_stand
        )
