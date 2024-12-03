"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-06-06 02:45:41
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-06-06 02:45:41
"""

import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "FilterResponseNorm1d",
    "FilterResponseNorm2d",
    "FilterResponseNorm3d",
    "MABN2d",
    "CenConv2d",
]


class FilterResponseNormNd(nn.Module):
    def __init__(self, ndim, num_features, eps=1e-6, learnable_eps=False):
        """
        github.com/gupta-abhay/pytorch-frn/blob/master/frn.py
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], "FilterResponseNorm only supports 3d, 4d or 5d inputs."
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm1d, self).__init__(3, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm2d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(4, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm3d(FilterResponseNormNd):
    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm3d, self).__init__(5, num_features, eps=eps, learnable_eps=learnable_eps)


class BatchNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        running_var,
        eps,
        momentum,
        buffer_x2,
        buffer_gz,
        iters,
        buffer_size,
        warmup_iters,
    ):
        ctx.eps = eps
        ctx.buffer_size = buffer_size
        current_iter = iters.item()
        ctx.current_iter = current_iter
        ctx.warmup_iters = warmup_iters

        N, C, H, W = x.size()
        x2 = (x * x).mean(dim=3).mean(dim=2).mean(dim=0)

        buffer_x2[current_iter % buffer_size].copy_(x2)

        if current_iter <= buffer_size or current_iter < warmup_iters:
            var = x2.view(1, C, 1, 1)
        else:
            var = buffer_x2.mean(dim=0).view(1, C, 1, 1)

        z = x / (var + eps).sqrt()
        r = (var + eps).sqrt() / (running_var + eps).sqrt()

        if current_iter <= max(1000, warmup_iters):
            r = torch.clamp(r, 1, 1)
        else:
            r = torch.clamp(r, 1 / 5, 5)

        y = r * z

        ctx.save_for_backward(z, var, weight, buffer_gz, r)

        running_var.copy_(momentum * running_var + (1 - momentum) * var)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        buffer_size = ctx.buffer_size
        current_iter = ctx.current_iter
        warmup_iters = ctx.warmup_iters

        N, C, H, W = grad_output.size()
        z, var, weight, buffer_gz, r = ctx.saved_variables

        y = r * z
        g = grad_output * weight.view(1, C, 1, 1)
        g = g * r
        gz = (g * z).mean(dim=3).mean(dim=2).mean(dim=0)

        buffer_gz[current_iter % buffer_size].copy_(gz)

        if current_iter <= buffer_size or current_iter < warmup_iters:
            mean_gz = gz.view(1, C, 1, 1)
        else:
            mean_gz = buffer_gz.mean(dim=0).view(1, C, 1, 1)

        gx = 1.0 / torch.sqrt(var + eps) * (g - z * mean_gz)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MABN2d(nn.Module):
    """
    Formal definition of MABN layer and CenConv2d(Conv2d layer with weight
    centralization). Users can use MABN by directly replacing nn.BatchNorm2d
    and nn.Conv2d with MABN and CenConv2d respectively.
    https://github.com/megvii-model/MABN/blob/master/MABN.py
    Applied MABN over a 4D input as described in the paper
    `Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization`
    Args:
        channels: :math:`C` from an expected input of size :math:`(N, C, H, W)`
        B: the real batch size per GPU.
        real_B: The batch size you want to simulate. It must be divisible by B.
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_var computation.
            It should be in the limit of :math`(0, 1)`.
            Default: 0.98
        warmup_iters: number of iterations before using moving average statistics
            to normalize input.
            Default: 100

    """

    def __init__(self, channels, B, real_B, eps=1e-5, momentum=0.98, warmup_iters=100):
        super(MABN2d, self).__init__()
        assert real_B % B == 0
        self.buffer_size = real_B // B
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.register_buffer("iters", torch.zeros(1).type(torch.LongTensor))
        self.register_buffer("buffer_x2", torch.zeros(self.buffer_size, channels))
        self.register_buffer("buffer_gz", torch.zeros(self.buffer_size, channels))

        self.eps = eps
        self.momentum = momentum
        self.warmup_iters = warmup_iters

    def forward(self, x):
        if self.training:
            self.iters.copy_(self.iters + 1)
            x = BatchNormFunction.apply(
                x,
                self.weight,
                self.bias,
                self.running_var,
                self.eps,
                self.momentum,
                self.buffer_x2,
                self.buffer_gz,
                self.iters,
                self.buffer_size,
                self.warmup_iters,
            )
            return x
        else:
            N, C, H, W = x.size()
            var = self.running_var.view(1, C, 1, 1)
            x = x / (var + self.eps).sqrt()

        return self.weight.view(1, C, 1, 1) * x + self.bias.view(1, C, 1, 1)


class CenConv2d(nn.Module):
    """
    https://github.com/megvii-model/MABN/blob/master/MABN.py
    Conv2d layer with Weight Centralization.
    The args is exactly same as torch.nn.Conv2d. It's suggested to set bias=False when
    using CenConv2d with MABN.
    """

    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=False
    ):
        super(CenConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.randn(out_planes, in_planes // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_planes))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
