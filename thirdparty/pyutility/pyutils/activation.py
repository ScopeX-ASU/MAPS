'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-05-19 03:51:51
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-06 23:59:07
'''

import logging

import torch
from torch import nn
from torch.functional import Tensor

logger = logging.Logger(__file__)

__all__ = [
    "Swish",
    "ReLUN",
    "CReLU",
    "SiREN",
]


@torch.jit.script
def swish_fwd(x: Tensor) -> Tensor:
    # return x.mul(torch.sigmoid(x))
    return torch.sigmoid(x).mul_(x)


@torch.jit.script
def swish_bwd(x: Tensor, grad_output: Tensor) -> Tensor:
    x_sigmoid = torch.sigmoid(x)
    # return grad_output * (x_sigmoid * (1. + x * (1. - x_sigmoid)))
    output = (1-x_sigmoid).mul_(x).add_(1).mul_(x_sigmoid)
    del x_sigmoid
    return output.mul_(grad_output)


class SwishJitImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        x = ctx.saved_tensors[0]
        return swish_bwd(x, grad_output)


class Swish(nn.Module):
    """
    Swish activation function from Google
    """

    def __init__(self, inplace: bool = True, memory_efficient: bool = True) -> None:
        super(Swish, self).__init__()
        self.inplace = inplace
        self.swish = self.memory_efficient_swish if memory_efficient else self.original_swish

    def original_swish(self, x, inplace: bool = False) -> Tensor:
        return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())

    def memory_efficient_swish(self, x, inplace: bool = False) -> Tensor:
        return SwishJitImplementation.apply(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.swish(x, self.inplace)


class ReLUN(nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLUN}(x) = \min(\max(0,x), N)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLUN(N)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, N: float, inplace: bool = False) -> None:
        super().__init__(0., N, inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class CReLU(nn.Module):
    """ Complex ReLU which applies ReLU on real and imag individually
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        assert torch.is_complex(
            x), f"CReLU only supports complex-valued input tensor, but got {type(x)}"
        if self.inplace:
            x.real.relu_()
            x.imag.relu_()
            return x
        else:
            return torch.complex(torch.relu(x.real), torch.relu(x.imag))


class SiREN(nn.Module):
    """
    Sinusoidal activation function
    Implicit Neural Representations with Periodic Activation Functions, NeurIPS 2020
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.inplace:
            return x.cos_()
        else:
            return x.cos()
