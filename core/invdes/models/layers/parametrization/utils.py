"""
Date: 2024-10-05 02:05:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-10-05 02:05:18
FilePath: /Metasurface-Opt/core/models/parametrization/utils.py
"""

import torch
from torch import Tensor, nn


class ClipLayer(nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, upper_limit=None, lower_limit=None) -> Tensor:
        if self.mode == "lower_limit":
            assert upper_limit is None and lower_limit is not None
            x = ApplyLowerLimit.apply(x, lower_limit)
        elif self.mode == "upper_limit":
            assert upper_limit is not None and lower_limit is None
            x = ApplyUpperLimit.apply(x, upper_limit)
        elif self.mode == "both":
            assert upper_limit is not None and lower_limit is not None
            x = ApplyBothLimit.apply(x, upper_limit, lower_limit)
        else:
            raise ValueError("Invalid mode")
        return x

    def extra_repr(self) -> str:
        return f"mode={self.mode}"


class BinaryProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permittivity: Tensor, T_bny: float, T_threshold: float):
        ctx.T_bny = T_bny
        ctx.T_threshold = T_threshold
        ctx.save_for_backward(permittivity)
        result = (torch.tanh((0.5 - permittivity) / T_bny) + 1) / 2
        return result

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # if T_bny is larger than T_threshold, then use the automatic differentiation of the tanh function
        # if the T_bny is smaller than T_threshold, then use the gradient as if T_bny is T_threshold
        T_bny = ctx.T_bny
        T_threshold = ctx.T_threshold
        (permittivity,) = ctx.saved_tensors

        if T_bny > T_threshold:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_bny) ** 2)
                / T_bny
            )
        else:
            grad = (
                -grad_output
                * (1 - torch.tanh((0.5 - permittivity) / T_threshold) ** 2)
                / T_threshold
            )

        return grad, None, None


class BinaryProjectionLayer(nn.Module):
    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, x: Tensor, t_bny: float) -> Tensor:
        permittivity = BinaryProjection.apply(x, t_bny, self.threshold)
        return permittivity

    def extra_repr(self) -> str:
        return f"threshold={self.threshold}"


class ApplyLowerLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lower_limit):
        ctx.save_for_backward(x)
        ctx.lower_limit = lower_limit
        return torch.maximum(x, lower_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > lower_limit, propagate grad_output normally
        # If x <= lower_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for lower_limit since it does not require gradients


class ApplyUpperLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        return torch.minimum(x, upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
        )  # None for upper_limit since it does not require gradients


class ApplyBothLimit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, upper_limit, lower_limit):
        ctx.save_for_backward(x)
        ctx.upper_limit = upper_limit
        ctx.lower_limit = lower_limit
        return torch.minimum(torch.maximum(x, lower_limit), upper_limit)

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        (x,) = ctx.saved_tensors
        upper_limit = ctx.upper_limit
        lower_limit = ctx.lower_limit

        # Compute gradient
        # If x > upper_limit, propagate grad_output normally
        # If x <= upper_limit, you can still propagate grad_output
        grad_input = torch.ones_like(x) * grad_output  # Propagate gradients fully

        return (
            grad_input,
            None,
            None,
        )  # None for upper_limit and lower_limit since they do not require gradients


class HeavisideProjectionLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, beta, eta, fw_threshold, bw_threshold):
        ctx.save_for_backward(x, eta)
        ctx.bw_threshold = bw_threshold
        ctx.beta = beta
        if (
            beta < fw_threshold
        ):  # over a large number we will treat this as a pure binary projection
            return (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )
        else:
            return torch.where(
                x < eta,
                torch.tensor(0, dtype=torch.float32, device=x.device),
                torch.tensor(1, dtype=torch.float32, device=x.device),
            )

    @staticmethod
    def backward(ctx, grad_output):
        x, eta = ctx.saved_tensors
        bw_threshold = ctx.bw_threshold
        beta = ctx.beta
        if beta > bw_threshold:
            grad = (
                grad_output
                * (bw_threshold * (1 - (torch.tanh(bw_threshold * (x - eta))) ** 2))
                / (
                    torch.tanh(bw_threshold * eta)
                    + torch.tanh(bw_threshold * (1 - eta))
                )
            )
            denominator = torch.tanh(bw_threshold * eta) + torch.tanh(
                bw_threshold * (1 - eta)
            )
            denominator_grad_eta = bw_threshold * (
                1 - (torch.tanh(bw_threshold * eta)) ** 2
            ) - bw_threshold * (1 - (torch.tanh(bw_threshold * (1 - eta))) ** 2)
            nominator = torch.tanh(bw_threshold * eta) + torch.tanh(
                bw_threshold * (x - eta)
            )
            nominator_grad_eta = bw_threshold * (
                1 - (torch.tanh(bw_threshold * eta)) ** 2
            ) - bw_threshold * (1 - (torch.tanh(bw_threshold * (x - eta))) ** 2)
        else:
            grad = (
                grad_output
                * (beta * (1 - (torch.tanh(beta * (x - eta))) ** 2))
                / (torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta)))
            )
            denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            denominator_grad_eta = beta * (1 - (torch.tanh(beta * eta)) ** 2) - beta * (
                1 - (torch.tanh(beta * (1 - eta))) ** 2
            )
            nominator = torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))
            nominator_grad_eta = beta * (1 - (torch.tanh(beta * eta)) ** 2) - beta * (
                1 - (torch.tanh(beta * (x - eta))) ** 2
            )
        grad_eta = (
            grad_output
            * (denominator * nominator_grad_eta - nominator * denominator_grad_eta)
            / (denominator**2)
        )

        return grad, None, grad_eta, None, None


class HeavisideProjection(nn.Module):
    def __init__(
        self, fw_threshold: float = 200, bw_threshold: float = 80, mode: str = "regular"
    ) -> None:
        super().__init__()
        self.fw_threshold = fw_threshold  # leave threshold here for future use STE
        self.bw_threshold = bw_threshold
        self.mode = mode
        self.clip_layer = ClipLayer(mode="both")
        self.upper_limit = torch.tensor(1.0)
        self.lower_limit = torch.tensor(0.0)

    def forward(self, x: Tensor, beta: Tensor, eta: Tensor) -> Tensor:
        if self.mode == "regular":
            x = (torch.tanh(beta * eta) + torch.tanh(beta * (x - eta))) / (
                torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            )
            ### heaviside cannot guanratee the value is between 0 and 1
            if self.upper_limit.device != x.device:
                self.upper_limit = self.upper_limit.to(x.device)
                self.lower_limit = self.lower_limit.to(x.device)

        elif self.mode.lower() == "ste":
            x = HeavisideProjectionLayer.apply(
                x, beta, eta, self.fw_threshold, self.bw_threshold
            )  # STE
        x = self.clip_layer(
            x, upper_limit=self.upper_limit, lower_limit=self.lower_limit
        )
        return x
