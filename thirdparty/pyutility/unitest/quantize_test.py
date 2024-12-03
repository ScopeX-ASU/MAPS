"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 02:10:59
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 02:10:59
"""

import torch
import numpy as np
from pyutils.quantize import *
from pyutils.quant.lsq import ActQuantizer_LSQ, WeightQuantizer_LSQ
from pyutils.general import TimerCtx

import matplotlib.pyplot as plt


def test_input_quant():
    bit = 8
    device = torch.device("cuda")
    qfn = input_quantize_fn(bit, alg="normal", device=device)
    qfn.train()
    for _ in range(10):
        x = torch.randn(640, device=device, requires_grad=True).clamp(0)
        x_q = qfn(x)

        print(x[:4])
        print(x_q[:4])
        print(qfn.scale, qfn.zero_point)
    print("test")
    qfn.eval()
    for _ in range(4):
        x = torch.randn(640, device=device).clamp_(0)
        x_q = qfn(x)

        print(x[:4])
        print(x_q[:4])
        print(qfn.scale, qfn.zero_point)


def test_lsq_quant():
    w_q_config = {
        "nbits": 8,
        "mode": "kernel_wise",
        "offset": False,
        "signed": True,
    }
    act_q_config = {
        "nbits": 8,
        "mode": "kernel_wise",
        "offset": True,
        "signed": False,
    }
    device = torch.device("cuda")
    weight_quantizer = WeightQuantizer_LSQ(2, device=device, **w_q_config)
    input_quantizer = ActQuantizer_LSQ(2, device=device, **act_q_config)
    x = torch.randn(1, 2, 2, 2, device=device).relu()
    x.requires_grad = True
    x_q = input_quantizer(x)
    w = torch.randn(2, 2, 3, 3, device=device, requires_grad=True)
    w_q = weight_quantizer(w)
    print(x, x_q)
    x_q.sum().backward()
    print(x.grad)
    print(w, w_q)
    w_q.sum().backward()
    print(w.grad)
    print(weight_quantizer.alpha.grad)


if __name__ == "__main__":
    # test_input_quant()
    test_lsq_quant()
