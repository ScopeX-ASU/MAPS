import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def build_effective_psf(
    eta: float = 0.55,
    sig_ax: float = 25.0,
    sig_ay: float = 25.0,
    sig_bx: float = 70.0,
    sig_by: float = 70.0,
    step: float = 2.0,
    radius: float = 500.0,
) -> torch.Tensor:
    """Construct the effective PSF kernel using torch tensors."""
    w1 = 1.0 / (1.0 + eta)
    w2 = eta / (1.0 + eta)

    xs = torch.arange(-radius, radius + step, step, dtype=torch.float32)
    ys = torch.arange(-radius, radius + step, step, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    def gauss2d(x: torch.Tensor, y: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
        coeff = 1.0 / (2.0 * math.pi * sx * sy)
        exponent = -(x.square() / (2.0 * sx * sx) + y.square() / (2.0 * sy * sy))
        return coeff * torch.exp(exponent)

    kernel = w1 * gauss2d(xx, yy, sig_ax, sig_ay) + w2 * gauss2d(xx, yy, sig_bx, sig_by)
    kernel /= kernel.sum()
    return kernel