"""
Date: 2024-03-25 20:33:11
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-25 20:33:11
FilePath: /pyutility/pyutils/loss/utils.py
"""

import torch


def normalize(logit):
    stdv, mean = torch.std_mean(logit, dim=-1, keepdim=True)
    return (logit - mean) / (1e-7 + stdv)
