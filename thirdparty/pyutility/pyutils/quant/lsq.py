"""
Description:
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-12-14 22:43:29
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-12-14 23:29:16
"""

import torch
from torch import nn
from torch.nn.parameter import Parameter

from pyutils.general import logger

__all__ = ["ActQuantizer_LSQ", "WeightQuantizer_LSQ"]


class GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_outputs):
        scale = ctx.scale
        grad_inputs = grad_outputs * scale
        return grad_inputs, None


grad_scale = GradScale.apply


def get_sparsity_mask(param, sparsity):
    bottomk, _ = torch.topk(
        param.abs().view(-1), int(sparsity * param.numel()), largest=False, sorted=True
    )
    threshold = bottomk.data[
        -1
    ]  # This is the largest element from the group of elements that we prune away
    return torch.gt(torch.abs(param), threshold).type(param.type())


class RoundPass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_inputs = grad_outputs.clone()
        return grad_inputs


round_pass = RoundPass.apply


def get_default_kwargs_q(kwargs_q):
    default = {
        "nbits": 32,
        "mode": "layer_wise",
        "signed": True,
        "offset": True,
    }
    default.update(kwargs_q)
    return default


class ActQuantizer_LSQ(nn.Module):
    def __init__(self, in_features: int, device="cuda:0", **kwargs_q):
        super().__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits = kwargs_q["nbits"]
        if self.nbits <= 0:  # no need to enable quantize
            self.register_parameter("alpha", None)
            return
        self.q_mode = kwargs_q["mode"]
        self.offset = kwargs_q["offset"]
        self.zero_point = None
        self.device = device
        if self.q_mode == "kernel_wise":
            self.alpha = Parameter(torch.empty(in_features, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.empty(in_features, device=device))
                torch.nn.init.zeros_(self.zero_point)
        else:
            self.alpha = Parameter(torch.empty(1, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.tensor([0.0], device=device))

        self.register_buffer("init_state", torch.zeros(1))
        self.register_buffer("signed", torch.zeros(1))

    def update_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q["nbits"] = self.nbits = nbits
        self._compute_quant_range()

    def _compute_quant_range(self):
        if self.signed == 1:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**self.nbits - 1

    def extra_repr(self):
        if self.alpha is None:
            return "fake"
        return "{}".format(self.kwargs_q)

    def _initialize_state(self, x):
        logger.info(
            f"LSQ Act quantizer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}"
        )
        if self.q_mode == "kernel_wise":
            logger.info(f"Scale dimension: {self.alpha.shape}")
        # choose implementation from https://github.com/YanjingLi0202/Q-ViT/blob/main/Quant.py
        if (
            x.data.min() < -1e-5
        ):  # there are significant negative values we will use signed representation
            self.signed.data.fill_(1)
        self._compute_quant_range()
        self.alpha.data.copy_(x.data.abs().mean().mul_(2 / self.Qp**0.5))
        if self.offset:
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9
                + 0.1 * (x.data.min() - self.alpha.data * self.Qn)
            )
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            self._initialize_state(x)

        assert self.init_state == 1

        g = 1.0 / (x.data.numel() * self.Qp) ** 0.5

        self.alpha.data.clamp_(min=1e-4)

        alpha = grad_scale(self.alpha, g)  # scale alpha's gradient by g

        if len(x.shape) == 2:  # linear layer
            alpha = alpha.unsqueeze(0)
        elif len(x.shape) == 4:  # conv layer
            alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        else:
            raise NotImplementedError

        if self.offset:
            zero_point = (
                self.zero_point.round() - self.zero_point
            ).detach() + self.zero_point
            zero_point = grad_scale(zero_point, g)
            zero_point = (
                zero_point.unsqueeze(0)
                if len(x.shape) == 2
                else zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            )
            x = round_pass((x / alpha + zero_point).clamp(self.Qn, self.Qp))
            x = (x - zero_point) * alpha
        else:
            x = round_pass((x / alpha).clamp(self.Qn, self.Qp)).mul(alpha)

        return x


class WeightQuantizer_LSQ(nn.Module):
    def __init__(self, out_features: int, device="cuda:0", **kwargs_q):
        super().__init__()
        self.kwargs_q = get_default_kwargs_q(kwargs_q)
        self.nbits = kwargs_q["nbits"]
        if self.nbits <= 0:  # no need to enable quantize
            self.register_parameter("alpha", None)
            return
        self.q_mode = kwargs_q["mode"]
        self.offset = kwargs_q["offset"]
        self.zero_point = None
        self.device = device
        if self.q_mode == "kernel_wise":
            self.alpha = Parameter(torch.empty(out_features, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.empty(out_features, device=device))
                torch.nn.init.zeros_(self.zero_point)
        else:
            self.alpha = Parameter(torch.empty(1, device=device))
            if self.offset:
                self.zero_point = Parameter(torch.tensor([0.0], device=device))

        self.register_buffer("init_state", torch.zeros(1))
        self.register_buffer("signed", torch.tensor([kwargs_q["signed"]]))

    def update_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.kwargs_q["nbits"] = self.nbits = nbits
        self._compute_quant_range()

    def _compute_quant_range(self):
        if self.signed == 1:
            self.Qn = -(2 ** (self.nbits - 1))
            self.Qp = 2 ** (self.nbits - 1) - 1
        else:
            self.Qn = 0
            self.Qp = 2**self.nbits - 1

    def extra_repr(self):
        if self.alpha is None:
            return "fake"
        return "{}".format(self.kwargs_q)

    def _initialize_state(self, x):
        logger.info(
            f"LSQ Weight quantizer: (mode: {self.q_mode}): initialize weight scale for int{self.nbits} quantization with offset: {self.offset}"
        )
        if self.q_mode == "kernel_wise":
            logger.info(f"Scale dimension: {self.alpha.shape}")

        self._compute_quant_range()
        self.alpha.data.copy_(x.data.abs().mean().mul_(2 / self.Qp**0.5))
        if self.offset:
            self.zero_point.data.copy_(
                self.zero_point.data * 0.9
                + 0.1 * (x.data.min() - self.alpha.data * self.Qn)
            )
        self.init_state.fill_(1)

    def forward(self, x):
        if self.alpha is None:
            return x

        if self.training and self.init_state == 0:
            self._initialize_state(x)

        assert self.init_state == 1

        g = 1.0 / (x.data.numel() * self.Qp) ** 0.5

        self.alpha.data.clamp_(min=1e-4)

        alpha = grad_scale(self.alpha, g)  # scale alpha's gradient by g

        # if len(x.shape) == 2:  # linear layer
        #     alpha = alpha[..., None]
        # elif len(x.shape) == 4:  # conv layer
        #     alpha = alpha[..., None, None, None]
        # else:
        #     raise NotImplementedError
        alpha = alpha.view(alpha.shape + (1,) * (len(x.shape) - len(alpha.shape)))

        if self.offset:
            zero_point = round_pass(self.zero_point)
            zero_point = grad_scale(zero_point, g)
            zero_point = (
                zero_point[..., None]
                if len(x.shape) == 2
                else zero_point[..., None, None, None]
            )
            x = round_pass((x / alpha + zero_point).clamp(self.Qn, self.Qp))
            x = (x - zero_point) * alpha
        else:
            x = round_pass((x / alpha).clamp(self.Qn, self.Qp)).mul(alpha)

        return x
