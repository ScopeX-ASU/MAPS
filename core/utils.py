import logging
import math
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.fft
import torch.optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .models.layers.utils import LevelSetInterp, get_eps
from ceviche.constants import *
from torch import Tensor
from torch_sparse import spmm
from typing import Callable, List, Tuple
from ceviche.utils import get_entries_indices
from core.models.layers.utils import get_eigenmode_coefficients, Slice


if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


def plot_fields(fields: Tensor, ground_truth: Tensor, cmap: str = "magma", filepath: str = './figs/fields.png', **kwargs):
    # the field is of shape (batch, 6, x, y)
    fields = fields.reshape(fields.shape[0], -1, 2, fields.shape[-2], fields.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
    fields = torch.view_as_complex(fields)
    ground_truth = ground_truth.reshape(ground_truth.shape[0], -1, 2, ground_truth.shape[-2], ground_truth.shape[-1]).permute(0, 1, 3, 4, 2).contiguous()
    ground_truth = torch.view_as_complex(ground_truth)
    fig, ax = plt.subplots(3, ground_truth.shape[1], figsize=(15, 10), squeeze=False)

    field_name = ['Hx', 'Hy', 'Ez']
    for idx, field in enumerate(field_name):
        v_range = max(torch.abs(fields[0, idx]).max(), torch.abs(ground_truth[0, idx]).max()).item()
        # Plot predicted fields in the first row
        im_pred = ax[0, idx].imshow(torch.abs(fields[0, idx]).cpu().numpy(), vmin=0, vmax=v_range, cmap=cmap)
        ax[0, idx].set_title(f"Predicted Field {field}")
        fig.colorbar(im_pred, ax=ax[0, idx])
        
        # Plot ground truth fields in the second row
        im_gt = ax[1, idx].imshow(torch.abs(ground_truth[0, idx]).cpu().numpy(), vmin=0, vmax=v_range, cmap=cmap)
        ax[1, idx].set_title(f"Ground Truth {field}")
        fig.colorbar(im_gt, ax=ax[1, idx])

        # Plot the difference between the predicted and ground truth fields in the third row
        im_err = ax[2, idx].imshow(torch.abs(fields[0, idx] - ground_truth[0, idx]).cpu().numpy(), cmap=cmap)
        ax[2, idx].set_title(f"Error {field}")
        fig.colorbar(im_err, ax=ax[2, idx])

    # Save the figure with high resolution
    plt.savefig(filepath, dpi=300)
    plt.close()
    # fig, ax = plt.subplots(2, ground_truth.shape[1], figsize=(15, 10), squeeze=False)
    # for i in range(ground_truth.shape[1]):
    #     ax[0, i].imshow(torch.abs(fields[0, i]).cpu().numpy(), cmap=cmap)
    #     ax[0, i].set_title(f"Field {i}")
    #     ax[1, i].imshow(torch.abs(ground_truth[0, i]).cpu().numpy(), cmap=cmap)
    #     ax[1, i].set_title(f"Ground Truth {i}")
    # plt.savefig(filepath, dpi=300)
    # plt.close()

def resize_to_targt_size(image: Tensor, size: Tuple[int, int]) -> Tensor:
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    return F.interpolate(image, size=size, mode='bilinear', align_corners=False).squeeze()

class DAdaptAdam(torch.optim.Optimizer):
    r"""
    Implements Adam with D-Adaptation automatic step-sizes.
    Leave LR set to 1 unless you encounter instability.

    To scale the learning rate differently for each layer, set the 'layer_scale'
    for each parameter group. Increase (or decrease) from its default value of 1.0
    to increase (or decrease) the learning rate for that layer relative to the
    other layers.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int):
            Log using print every k steps, default 0 (no logging).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        log_every=0,
        decouple=False,
        use_bias_correction=False,
        d0=1e-6,
        growth_rate=float("inf"),
        fsdp_in_use=False,
    ):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple:
            print("Using decoupled weight decay")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            k=0,
            layer_scale=1.0,
            numerator_weighted=0.0,
            log_every=log_every,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            fsdp_in_use=fsdp_in_use,
        )
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        sk_l1 = 0.0

        group = self.param_groups[0]
        use_bias_correction = group["use_bias_correction"]
        numerator_weighted = group["numerator_weighted"]
        beta1, beta2 = group["betas"]
        k = group["k"]

        d = group["d"]
        lr = max(group["lr"] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1

        dlr = d * lr * bias_correction

        growth_rate = group["growth_rate"]
        decouple = group["decouple"]
        log_every = group["log_every"]
        fsdp_in_use = group["fsdp_in_use"]

        sqrt_beta2 = beta2 ** (0.5)

        numerator_acum = 0.0

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]
            group_lr = group["lr"]
            r = group["layer_scale"]

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    "Setting different lr values in different parameter groups "
                    "is only supported for values of 0. To scale the learning "
                    "rate differently for each layer, set the 'layer_scale' value instead."
                )

            for p in group["params"]:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                s = state["s"]

                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += (
                        r
                        * dlr
                        * torch.dot(grad.flatten(), s.div(denom).flatten()).item()
                    )

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=r * dlr * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr * (1 - sqrt_beta2))
                    sk_l1 += r * s.abs().sum().item()

            ######

        numerator_weighted = (
            sqrt_beta2 * numerator_weighted + (1 - sqrt_beta2) * numerator_acum
        )
        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have sk_l1 > 0 (unless \|g\|=0)
        if sk_l1 == 0:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = numerator_weighted
                dist_tensor[1] = sk_l1
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_numerator_weighted = dist_tensor[0]
                global_sk_l1 = dist_tensor[1]
            else:
                global_numerator_weighted = numerator_weighted
                global_sk_l1 = sk_l1

            d_hat = global_numerator_weighted / ((1 - sqrt_beta2) * global_sk_l1)
            d = max(d, min(d_hat, d * growth_rate))

        if log_every > 0 and k % log_every == 0:
            logging.info(
                f"lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_l1={global_sk_l1:1.1e} numerator_weighted={global_numerator_weighted:1.1e}"
            )

        for group in self.param_groups:
            group["numerator_weighted"] = numerator_weighted
            group["d"] = d

            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                denom = exp_avg_sq.sqrt().add_(eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group["k"] = k + 1

        return loss


class DeterministicCtx:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.random_state = None
        self.numpy_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        # Save the current states
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_random_state = torch.cuda.get_rng_state()

        # Set deterministic behavior based on the seed
        set_torch_deterministic(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the saved states
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_random_state)


def set_torch_deterministic(seed: int = 0) -> None:
    seed = int(seed) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)


def numerical_gradient_2d(phi, d_size):
    grad_x = torch.zeros_like(phi)
    grad_y = torch.zeros_like(phi)

    # Compute the gradient along the x direction (rows)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if i == 0:
                grad_x[i, j] = (phi[i + 1, j] - phi[i, j]) / d_size
            elif i == phi.shape[0] - 1:
                grad_x[i, j] = (phi[i, j] - phi[i - 1, j]) / d_size
            else:
                grad_x[i, j] = (phi[i + 1, j] - phi[i - 1, j]) / (2 * d_size)

    # Compute the gradient along the y direction (columns)
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if j == 0:
                grad_y[i, j] = (phi[i, j + 1] - phi[i, j]) / d_size
            elif j == phi.shape[1] - 1:
                grad_y[i, j] = (phi[i, j] - phi[i, j - 1]) / d_size
            else:
                grad_y[i, j] = (phi[i, j + 1] - phi[i, j - 1]) / (2 * d_size)

    return grad_x, grad_y


# Auxiliary function to calculate first and second order partial derivatives.
def ls_derivatives(phi, d_size):
    SC = 1e-12

    # First-order derivatives
    phi_x, phi_y = numerical_gradient_2d(phi, d_size)
    phi_x += SC
    phi_y += SC

    # Second-order derivatives
    phi_2x_x, phi_2x_y = numerical_gradient_2d(phi_x, d_size)
    phi_2y_x, phi_2y_y = numerical_gradient_2d(phi_y, d_size)

    phi_xx = phi_2x_x
    phi_xy = phi_2x_y
    phi_yy = phi_2y_y

    return phi_x, phi_y, phi_xx, phi_xy, phi_yy


# Minimum gap size fabrication constraint integrand calculation.
# The "beta" parameter relax the constraint near the zero plane.
class fab_penalty_ls_gap(torch.nn.Module):
    def __init__(self, beta=1, min_feature_size=1):
        super(fab_penalty_ls_gap, self).__init__()
        self.beta = beta
        self.min_feature_size = min_feature_size

    def forward(self, data):
        params = data["params"]
        x_rho = data["x_rho"]
        y_rho = data["y_rho"]
        x_phi = data["x_phi"]
        y_phi = data["y_phi"]
        nx_phi = data["nx_phi"]
        ny_phi = data["ny_phi"]
        rho_size = data["rho_size"]
        grid_size = data["grid_size"]
        # Get the level set surface.
        phi_model = LevelSetInterp(x0=x_rho, y0=y_rho, z0=params, sigma=rho_size)
        phi = phi_model.get_ls(x1=x_phi, y1=y_phi)
        phi = torch.reshape(phi, (nx_phi, ny_phi))

        phi = torch.cat((phi, phi.flip(1)), dim=1)

        # Calculates their derivatives.
        phi_x, phi_y, phi_xx, phi_xy, phi_yy = ls_derivatives(phi, grid_size)

        # Calculates the gap penalty over the level set grid.
        pi_d = np.pi / (1.3 * self.min_feature_size)
        phi_v = torch.maximum(torch.sqrt(phi_x**2 + phi_y**2), torch.tensor(1e-8))
        phi_vv = (
            phi_x**2 * phi_xx + 2 * phi_x * phi_y * phi_xy + phi_y**2 * phi_yy
        ) / phi_v**2
        return torch.nansum(
            torch.maximum(
                (
                    torch.abs(phi_vv) / (pi_d * torch.abs(phi) + self.beta * phi_v)
                    - pi_d
                ),
                torch.tensor(0),
            )
            * grid_size**2
        )


# Minimum radius of curvature fabrication constraint integrand calculation.
# The "alpha" parameter controls its relative weight to the gap penalty.
# The "sharpness" parameter controls the smoothness of the surface near the zero-contour.
# def fab_penalty_ls_curve(params,
#                          alpha=1,
#                          sharpness = 1,
#                          min_feature_size=min_feature_size,
#                          grid_size=ls_grid_size):
class fab_penalty_ls_curve(torch.nn.Module):
    def __init__(self, alpha=1, min_feature_size=1):
        super(fab_penalty_ls_curve, self).__init__()
        self.alpha = alpha
        self.min_feature_size = min_feature_size

    def forward(self, data):
        params = data["params"]
        x_rho = data["x_rho"]
        y_rho = data["y_rho"]
        x_phi = data["x_phi"]
        y_phi = data["y_phi"]
        nx_rho = data["nx_rho"]
        ny_rho = data["ny_rho"]
        nx_phi = data["nx_phi"]
        ny_phi = data["ny_phi"]
        rho_size = data["rho_size"]
        grid_size = data["grid_size"]
        # Get the permittivity surface and calculates their derivatives.
        eps = get_eps(
            params, x_rho, y_rho, x_phi, y_phi, rho_size, nx_rho, ny_rho, nx_phi, ny_phi
        )
        eps = torch.cat((eps, eps.flip(1)), dim=1)
        eps_x, eps_y, eps_xx, eps_xy, eps_yy = ls_derivatives(eps, grid_size)

        # Calculates the curvature penalty over the permittivity grid.
        pi_d = np.pi / (1.1 * self.min_feature_size)
        eps_v = torch.maximum(
            torch.sqrt(eps_x**2 + eps_y**2), torch.tensor(1e-32**1 / 6)
        )
        k = (
            eps_x**2 * eps_yy - 2 * eps_x * eps_y * eps_xy + eps_y**2 * eps_xx
        ) / eps_v**3
        curve_const = torch.abs(k * torch.arctan(eps_v / eps)) - pi_d
        return torch.nansum(
            self.alpha * torch.maximum(curve_const, torch.tensor(0)) * grid_size**2
        )


def padding_to_tiles(x, tile_size):
    """
    Pads the input tensor to a size that is a multiple of the tile size.
    the input x should be a 2D tensor with shape x_dim, y_dim
    """
    pad_x = tile_size - x.size(0) % tile_size
    pad_y = tile_size - x.size(1) % tile_size
    pady_0 = pad_y // 2
    pady_1 = pad_y - pady_0
    padx_0 = pad_x // 2
    padx_1 = pad_x - padx_0
    if pad_x > 0 or pad_y > 0:
        x = torch.nn.functional.pad(x, (pady_0, pady_1, padx_0, padx_1))
    return x, pady_0, pady_1, padx_0, padx_1


def rip_padding(eps, pady_0, pady_1, padx_0, padx_1):
    """
    Removes the padding from the input tensor.
    """
    return eps[padx_0:-padx_1, pady_0:-pady_1]


class ComplexL1Loss(torch.nn.MSELoss):
    def __init__(self, norm=False) -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        """Complex L1 loss between the predicted electric field and target field

        Args:
            x (Tensor): Predicted complex-valued frequency-domain full electric field
            target (Tensor): Ground truth complex-valued electric field intensity

        Returns:
            Tensor: loss
        """
        if self.norm:
            diff = torch.view_as_real(x - target)
            return diff.norm(p=1, dim=[1, 2, 3, 4]).div(torch.view_as_real(target).norm(p=1, dim=[1, 2, 3, 4])).mean()
        return F.l1_loss(torch.view_as_real(x), torch.view_as_real(target))

class NormalizedMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(NormalizedMSELoss, self).__init__()

        self.reduce = reduce

    def forward(self, x, y, mask):
        one_mask = mask != 0

        error_energy = torch.norm((x - y) * one_mask, p=2, dim=(-1, -2))
        field_energy = torch.norm(y * one_mask, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy / field_energy).mean()


class NL2NormLoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(NL2NormLoss, self).__init__()

        self.reduce = reduce

    def forward(self, x, y, mask):
        one_mask = mask != 0

        error_energy = torch.norm((x - y) * one_mask, p=2, dim=(-1, -2))
        field_energy = torch.norm(y * one_mask, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy / field_energy).mean()


class maskedNMSELoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, reduce="mean", weighted_frames=10, weight=1, if_spatial_mask=True
    ):
        super(maskedNMSELoss, self).__init__()

        self.reduce = reduce
        self.weighted_frames = weighted_frames
        self.weight = weight
        self.if_spatial_mask = if_spatial_mask

    def forward(self, x, y, mask, num_iters=1):
        frame_mask = None
        ones_mask = mask != 0
        if self.weighted_frames != 0:
            assert x.shape[-3] // num_iters >= 10
            assert x.shape[-3] % num_iters == 0
            frame_mask = torch.ones((1, x.shape[1], 1, 1)).to(x.device)
            single_prediction_len = x.shape[-3] // num_iters
            for i in range(num_iters):
                frame_mask[
                    :,
                    -single_prediction_len * (i - 1)
                    - self.weighted_frames : -single_prediction_len * (i - 1),
                    :,
                    :,
                ] = 3
        assert (
            (frame_mask is not None) or self.if_spatial_mask
        ), "if mask NMSE, either frame_mask or spatial_mask should be True"
        if self.if_spatial_mask:
            error_energy = torch.norm((x - y) * mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * mask, p=2, dim=(-1, -2))
        else:
            error_energy = torch.norm((x - y) * ones_mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * ones_mask, p=2, dim=(-1, -2))
        if frame_mask is not None:
            return (((error_energy / field_energy) * frame_mask).mean(dim=(-1))).mean()
        else:
            return ((error_energy / field_energy).mean(dim=(-1))).mean()


class maskedNL2NormLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self, reduce="mean", weighted_frames=10, weight=1, if_spatial_mask=True
    ):
        super(maskedNL2NormLoss, self).__init__()

        self.reduce = reduce
        self.weighted_frames = weighted_frames
        self.weight = weight
        self.if_spatial_mask = if_spatial_mask

    def forward(self, x, y, mask, num_iters=1):
        frame_mask = None
        ones_mask = mask != 0
        if self.weighted_frames != 0:
            assert x.shape[-3] // num_iters >= 10
            assert x.shape[-3] % num_iters == 0
            frame_mask = torch.ones((1, x.shape[1], 1, 1)).to(x.device)
            single_prediction_len = x.shape[-3] // num_iters
            for i in range(num_iters):
                frame_mask[
                    :,
                    -single_prediction_len * (i - 1)
                    - self.weighted_frames : -single_prediction_len * (i - 1),
                    :,
                    :,
                ] = 3
        assert (
            (frame_mask is not None) or self.if_spatial_mask
        ), "if mask nl2norm, either frame_mask or spatial_mask should be True"
        if self.if_spatial_mask:
            error_energy = torch.norm((x - y) * mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * mask, p=2, dim=(-1, -2))
        else:
            error_energy = torch.norm((x - y) * ones_mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y * ones_mask, p=2, dim=(-1, -2))
        if frame_mask is not None:
            return (((error_energy / field_energy) * frame_mask).mean(dim=(-1))).mean()
        else:
            return ((error_energy / field_energy).mean(dim=(-1))).mean()


def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x


def print_stat(x, dist=False):
    total_number = None
    distribution = None
    if dist:
        total_number = x.numel()
        distribution = torch.histc(x, bins=10, min=float(x.min()), max=float(x.max()))
    if isinstance(x, torch.Tensor):
        if torch.is_complex(x):
            x = x.abs()
        print(
            f"min = {x.min().data.item():-15g} max = {x.max().data.item():-15g} mean = {x.mean().data.item():-15g} std = {x.std().data.item():-15g}\n total num = {total_number} distribution = {distribution}"
        )
    elif isinstance(x, np.ndarray):
        if np.iscomplexobj(x):
            x = np.abs(x)
        print(
            f"min = {np.min(x):-15g} max = {np.max(x):-15g} mean = {np.mean(x):-15g} std = {np.std(x):-15g}"
        )


class TemperatureScheduler:
    def __init__(self, initial_T, final_T, total_steps):
        self.initial_T = initial_T
        self.final_T = final_T
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        cos_inner = (math.pi * self.current_step) / self.total_steps
        cos_out = math.cos(cos_inner) + 1
        self.current_T = self.final_T + 0.5 * (self.initial_T - self.final_T) * cos_out
        return self.current_T

    def get_temperature(self):
        return self.current_T


class SharpnessScheduler:
    def __init__(self, initial_sharp, final_sharp, total_steps):
        self.initial_sharp = initial_sharp
        self.final_sharp = final_sharp
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        cos_inner = (math.pi * self.current_step) / self.total_steps
        cos_out = math.cos(cos_inner) + 1
        self.current_sharp = (
            self.final_sharp + 0.5 * (self.initial_sharp - self.final_sharp) * cos_out
        )
        return self.current_sharp

    def get_sharpness(self):
        return self.current_sharp


class ResolutionScheduler:
    def __init__(self, initial_res, final_res, total_steps):
        self.initial_res = initial_res
        self.final_res = final_res
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step > self.total_steps:
            self.current_step = self.total_steps
        self.current_res = (
            self.initial_res
            + round(
                (self.final_res - self.initial_res)
                / self.total_steps
                * self.current_step
                / 10
            )
            * 10
        )
        return self.current_res

    def get_resolution(self):
        return self.current_res


class DistanceLoss(torch.nn.modules.loss._Loss):
    def __init__(self, min_distance=0.15):
        super(DistanceLoss, self).__init__()
        self.min_distance = min_distance

    def forward(self, hole_position):
        hole_position = torch.flatten(hole_position, start_dim=0, end_dim=1)
        distance = torch.zeros(hole_position.shape[0], hole_position.shape[0])
        for i in range(hole_position.shape[0]):
            for j in range(hole_position.shape[0]):
                distance[i, j] = torch.norm(
                    hole_position[i][:-1] - hole_position[j][:-1], p=2
                )
        distance_penalty = distance - self.min_distance
        distance_penalty = distance_penalty * (distance_penalty < 0)
        distance_penalty = distance_penalty.sum()
        distance_penalty = -1 * distance_penalty
        return distance_penalty


class AspectRatioLoss(torch.nn.modules.loss._Loss):
    def __init__(self, aspect_ratio=1):
        super(AspectRatioLoss, self).__init__()
        self.aspect_ratio = aspect_ratio

    def forward(self, input):
        height = input["height"]
        width = input["width"]
        period = input["period"]
        min_distance = height * self.aspect_ratio
        width_penalty = width - min_distance
        width_penalty = torch.minimum(
            width_penalty, torch.tensor(0.0, device=width.device)
        )
        width_penalty = width_penalty.abs().sum()

        # Compute gaps between consecutive widths across the batch
        gap = period - (width[:-1] / 2) - (width[1:] / 2)

        # Compute the gap penalty
        gap_penalty = gap - min_distance
        gap_penalty = torch.minimum(gap_penalty, torch.tensor(0.0, device=width.device))
        gap_penalty = gap_penalty.abs().sum()

        return gap_penalty + width_penalty


def padding_to_tiles(x, tile_size):
    """
    Pads the input tensor to a size that is a multiple of the tile size.
    the input x should be a 2D tensor with shape x_dim, y_dim
    """
    pad_x = tile_size - x.size(0) % tile_size
    pad_y = tile_size - x.size(1) % tile_size
    pady_0 = pad_y // 2
    pady_1 = pad_y - pady_0
    padx_0 = pad_x // 2
    padx_1 = pad_x - padx_0
    if pad_x > 0 or pad_y > 0:
        x = torch.nn.functional.pad(x, (pady_0, pady_1, padx_0, padx_1))
    return x, pady_0, pady_1, padx_0, padx_1

def rip_padding(eps, pady_0, pady_1, padx_0, padx_1):
    """
    Removes the padding from the input tensor.
    """
    return eps[padx_0:-padx_1, pady_0:-pady_1]

# class MaxwellResidualLoss(torch.nn.modules.loss._Loss):
#     def __init__(
#         self,
#         wl_cen: float = 1.55,
#         wl_width: float = 0,
#         n_wl: int = 1,
#         size_average=None,
#         reduce=None,
#         reduction: str = "mean",
#     ):
#         super().__init__(size_average, reduce, reduction)
#         self.wl_list = torch.linspace(
#             wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl
#         )
#         self.omegas = 2 * np.pi * C_0 / (self.wl_list * 1e-6)
#         self.As = (None, None)
#         self.bs = {}
#         self.sim = None

#     def make_A(self, eps_r: torch.Tensor, wl_list: List[float]):
#         ## eps_r: [bs, h, w] real tensor
#         ## wl_list: [n_wl] list of wls in um, support spectral bundling
#         eps_vec = eps_r.flatten(1)
#         tot_entries_list = []
#         tot_indices_list = []
#         for eps_v in eps_vec:
#             entries_list = []
#             indices_list = []
#             for wl in wl_list:
#                 self.sim.omega = 2 * np.pi * C_0 / (wl.item() * 1e-6)
#                 self.sim._setup_derivatives() # reset the derivatives since we changed the omega
#                 entries_a, indices_a = self.sim._make_A(
#                     eps_v
#                 )  # return scipy sparse indices and values
#                 entries_list.append(torch.from_numpy(entries_a) if isinstance(entries_a, np.ndarray) else entries_a)
#                 indices_list.append(torch.from_numpy(indices_a) if isinstance(indices_a, np.ndarray) else indices_a)
#             entries_list = torch.stack(entries_list, 0)
#             indices_list = torch.stack(indices_list, 0)
#             tot_entries_list.append(entries_list)
#             tot_indices_list.append(indices_list)
#         tot_entries_list = torch.stack(tot_entries_list, 0).to(
#             eps_r.device
#         )  # [bs, n_wl, 5*h*w]
#         tot_indices_list = (
#             torch.stack(tot_indices_list, 0).to(eps_r.device).long()
#         )  # [bs, n_wl, 2, 5*h*w]
#         self.As = (tot_entries_list, tot_indices_list)
#         return self.As
    
#     def make_sim(self, eps_r: torch.Tensor):
#         from core.models.fdfd.fdfd import fdfd_ez, My_fdfd_ez
#         # self.sim = fdfd_ez(
#         #     omega=2 * np.pi * C_0 / (1.55 * 1e-6), # this is just a init wl value, later will update this
#         #     dL=2e-8,
#         #     eps_r=eps_r[0],
#         #     npml=(50, 50),
#         # )
#         self.sim = My_fdfd_ez(
#             omega=2 * np.pi * C_0 / (1.55 * 1e-6), # this is just a init wl value, later will update this
#             dL=2e-8,
#             eps_r=eps_r[0],
#             npml=(50, 50),
#         )

#     def forward(self, Ez: Tensor, eps_r: Tensor, source: Tensor, target_size, As):
#         Ez = Ez[:, :2, :, :]
#         Ez = resize_to_targt_size(Ez, target_size).permute(0, 2, 3, 1).contiguous()
#         Ez = torch.view_as_complex(Ez) # convert Ez to the required complex format
#         eps_r = resize_to_targt_size(eps_r, target_size)
#         source = torch.view_as_real(source).permute(0, 3, 1, 2) # B, 2, H, W
#         source = resize_to_targt_size(source, target_size).permute(0, 2, 3, 1).contiguous()
#         source = torch.view_as_complex(source) # convert source to the required complex format

#         # there is only one omega in this case
#         Ez = Ez.unsqueeze(1)
#         source = source.unsqueeze(1)

#         ## Ez: [bs, n_wl, h, w] complex tensor
#         ## eps_r: [bs, h, w] real tensor
#         ## source: [bs, n_wl, h, w] complex tensor, source in sim.solve(source), not b, b = 1j * omega * source

#         # step 0: build self.sim
#         if self.sim is None:
#             self.make_sim(eps_r=eps_r**2)
#         # step 1: make A
#         print("begin to make A ... ", flush=True)
#         self.make_A(eps_r**2, self.wl_list)
#         print("finish making A :)", flush=True)

#         # step 2: calculate loss
#         entries, indices = self.As
#         lhs = []
#         if self.omegas.device != source.device:
#             self.omegas = self.omegas.to(source.device)
#         for i in range(Ez.shape[0]): # loop over samples in a batch
#             for j in range(Ez.shape[1]): # loop over different wavelengths
#                 ez = Ez[i, j].flatten()
#                 omega = 2 * np.pi * C_0 / (self.wl_list[j] * 1e-6)
#                 entries = As[f'A-wl-{self.wl_list[j]}-entries_a'][i]
#                 indices = As[f'A-wl-{self.wl_list[j]}-indices_a'][i]
#                 b = source[i, j].flatten() * (1j * omega)
#                 A_by_e = spmm(
#                     indices[i, j],
#                     entries[i, j],
#                     m=ez.shape[0],
#                     n=ez.shape[0],
#                     matrix=ez[:, None],
#                 )[:, 0]
#                 lhs.append(A_by_e)
#         lhs = torch.stack(lhs, 0)  # [bs*n_wl, h*w]
#         b = (
#             (source * (1j * self.omegas[None, :, None, None])).flatten(0, 1).flatten(1)
#         )  # [bs*n_wl, h*w]
#         difference = lhs - b
#         difference = torch.view_as_real(difference).double()
#         b = torch.view_as_real(b).double()
#         # print("this is the l2 norm of the b ", torch.norm(b, p=2, dim=(-2, -1)), flush=True) # ~e+22
#         loss = (torch.norm(difference, p=2, dim=(-2, -1)) / (torch.norm(b, p=2, dim=(-2, -1)) + 1e-6)).mean()
#         return loss


class MaxwellResidualLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        wl_cen: float = 1.55,
        wl_width: float = 0,
        n_wl: int = 1,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ):
        super().__init__(size_average, reduce, reduction)
        self.wl_list = torch.linspace(
            wl_cen - wl_width / 2, wl_cen + wl_width / 2, n_wl
        )
        self.omegas = 2 * np.pi * C_0 / (self.wl_list * 1e-6)

    def forward(self, Ez: Tensor, eps_r: Tensor, source: Tensor, target_size, As):
        Ez = Ez[:, :2, :, :]
        Ez = Ez.permute(0, 2, 3, 1).contiguous()
        Ez = torch.view_as_complex(Ez) # convert Ez to the required complex format
        source = torch.view_as_real(source).permute(0, 3, 1, 2) # B, 2, H, W
        source = source.permute(0, 2, 3, 1).contiguous()
        source = torch.view_as_complex(source) # convert source to the required complex format

        # there is only one omega in this case
        Ez = Ez.unsqueeze(1)
        source = source.unsqueeze(1)

        ## Ez: [bs, n_wl, h, w] complex tensor
        ## eps_r: [bs, h, w] real tensor
        ## source: [bs, n_wl, h, w] complex tensor, source in sim.solve(source), not b, b = 1j * omega * source

        # step 2: calculate loss
        lhs = []
        if self.omegas.device != source.device:
            self.omegas = self.omegas.to(source.device)
        for i in range(Ez.shape[0]): # loop over samples in a batch
            for j in range(Ez.shape[1]): # loop over different wavelengths
                ez = Ez[i, j].flatten()
                omega = 2 * np.pi * C_0 / (self.wl_list[j] * 1e-6)
                wl = round(self.wl_list[j].item()*100)/100
                entries = As[f'A-wl-{wl}-entries_a'][i]
                indices = As[f'A-wl-{wl}-indices_a'][i]
                b = source[i, j].flatten() * (1j * omega)
                A_by_e = spmm(
                    indices,
                    entries,
                    m=ez.shape[0],
                    n=ez.shape[0],
                    matrix=ez[:, None],
                )[:, 0]
                lhs.append(A_by_e)
        lhs = torch.stack(lhs, 0)  # [bs*n_wl, h*w]
        b = (
            (source * (1j * self.omegas[None, :, None, None])).flatten(0, 1).flatten(1)
        )  # [bs*n_wl, h*w]
        difference = lhs - b
        difference = torch.view_as_real(difference).double()
        b = torch.view_as_real(b).double()
        # print("this is the l2 norm of the b ", torch.norm(b, p=2, dim=(-2, -1)), flush=True) # ~e+22
        loss = (torch.norm(difference, p=2, dim=(-2, -1)) / (torch.norm(b, p=2, dim=(-2, -1)) + 1e-6)).mean()
        return loss

class SParamLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduce="mean"):
        super(SParamLoss, self).__init__()

        self.reduce = reduce

    def forward(
        self, 
        fields, # bs, 3, H, W, complex
        ht_m,
        et_m,
        monitor_slices,
        target_SParam,
    ):
        # Step 1: Resize all the fields to the target size
        Ez = fields[:, -2:, :, :].permute(0, 2, 3, 1).contiguous()
        Ez = torch.view_as_complex(Ez) # convert Ez to the required complex format

        Hx = fields[:, :2, :, :].permute(0, 2, 3, 1).contiguous()
        Hx = torch.view_as_complex(Hx)

        Hy = fields[:, 2:4, :, :].permute(0, 2, 3, 1).contiguous()
        Hy = torch.view_as_complex(Hy)

        monitor_slices_x = monitor_slices["port_slice-out_port_1_x"]
        monitor_slices_y = monitor_slices["port_slice-out_port_1_y"]

        # print("this is the monitor_slices_x", monitor_slices_x, flush=True)
        # print("this is the monitor_slices_y", monitor_slices_y, flush=True)
        ## Hx, Hy, Ez: [bs, h, w] complex tensor
        # Stpe 2: Calculate the S-parameters
        batch_size = Ez.shape[0]
        s_params = []
        for i in range(batch_size):
            monitor_slice = Slice(
                y=monitor_slices_y[i],
                x=torch.arange(
                    monitor_slices_x[i][0],
                    monitor_slices_x[i][1],
                ).to(monitor_slices_y[i].device),
            )
            # ht_m and et_m are lists
            s_p, s_m = get_eigenmode_coefficients(
                hx=Hx[i],
                hy=Hy[i],
                ez=Ez[i],
                ht_m=ht_m[i],
                et_m=et_m[i],
                monitor=monitor_slice,
                grid_step=1/50,
                direction="y",
                autograd=True,
                energy=True,
            )
            s_params.append(torch.tensor([s_p, s_m], device=Ez.device))
        s_params = torch.stack(s_params, 0)
        s_params_diff = s_params - target_SParam
        # Step 3: Calculate the loss
        # print("this is the l2 norm of the target_SParam ", torch.norm(target_SParam, p=2, dim=-1), flush=True) ~e-9
        loss = (torch.norm(s_params_diff, p=2, dim=-1) / (torch.norm(target_SParam, p=2, dim=-1) + 1e-12)).mean()
        return loss
    
class GradientLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduce="mean"):
        super(GradientLoss, self).__init__()

        self.reduce = reduce

    def forward(
        self, 
        forward_fields,
        backward_fields,
        adjoint_fields,  
        backward_adjoint_fields,
        target_gradient,
        gradient_multiplier,
        dr_mask = None,
    ):
        # forward_fields_ez = forward_fields[:, -2:, :, :] # the forward fields has three components, we only need the Ez component
        # forward_fields_ez = torch.view_as_complex(forward_fields_ez.permute(0, 2, 3, 1).contiguous())
        # backward_fields_ez = backward_fields[:, -2:, :, :]
        # backward_fields_ez = torch.view_as_complex(backward_fields_ez.permute(0, 2, 3, 1).contiguous())
        # adjoint_fields = torch.view_as_complex(adjoint_fields.permute(0, 2, 3, 1).contiguous()) # adjoint fields only Ez 
        # backward_adjoint_fields = torch.view_as_complex(backward_adjoint_fields.permute(0, 2, 3, 1).contiguous()) # adjoint fields only Ez
        # gradient = -(adjoint_fields*forward_fields_ez).real
        # backward_gradient = -(backward_adjoint_fields*backward_fields_ez).real
        # batch_size = gradient.shape[0]
        # for i in range(batch_size):
        #     gradient[i] = gradient[i] / gradient_multiplier["field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"][i]
        #     backward_gradient[i] = backward_gradient[i] / gradient_multiplier["field_adj_normalizer-wl-1.55-port-out_port_1-mode-1"][i]
        # # Step 0: build one_mask from dr_mask
        # ## This is not correct
        # # need to build a design region mask whose size shold be b, H, W
        # if dr_mask is not None:
        #     dr_masks = []
        #     for i in range(batch_size):
        #         mask = torch.zeros_like(gradient[i]).to(gradient.device)
        #         for key, value in dr_mask.items():
        #             if key.endswith("x_start"):
        #                 x_start = value[i]
        #             elif key.endswith("x_stop"):
        #                 x_stop = value[i]
        #             elif key.endswith("y_start"):
        #                 y_start = value[i]
        #             elif key.endswith("y_stop"):
        #                 y_stop = value[i]
        #             else:
        #                 raise ValueError(f"Invalid key: {key}")
        #         mask[x_start:x_stop, y_start:y_stop] = 1
        #         dr_masks.append(mask)
        #     dr_masks = torch.stack(dr_masks, 0)
        # else:
        #     dr_masks = torch.ones_like(gradient)
        
        # x = - EPSILON_0 * (2 * torch.pi * C_0 / (1.55 * 1e-6))**2 * (gradient + backward_gradient)
        # y = target_gradient
        # error_energy = torch.norm((x - y) * dr_masks, p=2, dim=(-1, -2))
        # field_energy = torch.norm(y * dr_masks, p=2, dim=(-1, -2)) + 1e-6
        # return (error_energy / field_energy).mean()
        forward_fields_ez = forward_fields[:, -2:, :, :] # the forward fields has three components, we only need the Ez component
        forward_fields_ez = torch.view_as_complex(forward_fields_ez.permute(0, 2, 3, 1).contiguous())
        adjoint_fields = torch.view_as_complex(adjoint_fields.permute(0, 2, 3, 1).contiguous()) # adjoint fields only Ez 
        gradient = -(adjoint_fields*forward_fields_ez).real
        batch_size = gradient.shape[0]
        for i in range(batch_size):
            gradient[i] = gradient[i] / gradient_multiplier["field_adj_normalizer-wl-1.55-port-in_port_1-mode-1"][i]
        # Step 0: build one_mask from dr_mask
        ## This is not correct
        # need to build a design region mask whose size shold be b, H, W
        if dr_mask is not None:
            dr_masks = []
            for i in range(batch_size):
                mask = torch.zeros_like(gradient[i]).to(gradient.device)
                for key, value in dr_mask.items():
                    if key.endswith("x_start"):
                        x_start = value[i]
                    elif key.endswith("x_stop"):
                        x_stop = value[i]
                    elif key.endswith("y_start"):
                        y_start = value[i]
                    elif key.endswith("y_stop"):
                        y_stop = value[i]
                    else:
                        raise ValueError(f"Invalid key: {key}")
                mask[x_start:x_stop, y_start:y_stop] = 1
                dr_masks.append(mask)
            dr_masks = torch.stack(dr_masks, 0)
        else:
            dr_masks = torch.ones_like(gradient)
        
        x = - EPSILON_0 * (2 * torch.pi * C_0 / (1.55 * 1e-6))**2 * (gradient)
        y = target_gradient
        error_energy = torch.norm((x - y) * dr_masks, p=2, dim=(-1, -2))
        field_energy = torch.norm(y * dr_masks, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy / field_energy).mean()