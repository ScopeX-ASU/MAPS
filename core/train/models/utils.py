"""
Date: 2024-11-24 15:28:23
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-11-24 19:34:45
FilePath: /MAPS/core/train/models/utils.py
"""

"""
Date: 2024-08-24 21:37:48
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-24 21:40:35
FilePath: /Metasurface-Opt/core/models/utils.py
"""

import math
import os

import numpy as np
import torch
from pyutils.general import ensure_dir
from torch import Tensor
from torch.nn.utils.rnn import pad_packed_sequence
from autograd import numpy as npa
from core.utils import Slice
__all__ = [
    "conv_output_size",
    "get_last_n_frames",
    "plot_permittivity",
    "nparray_as_real",
    "from_Ez_to_Hx_Hy",
    "get_grid",
    "poynting_vector",
]

def conv_output_size(in_size, kernel_size, padding=0, stride=1, dilation=1):
    return math.floor(
        (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    )

def get_last_n_frames(packed_sequence, n=100):
    # Unpack the sequence
    padded_sequences, lengths = pad_packed_sequence(packed_sequence, batch_first=True)
    last_frames = []

    for i, length in enumerate(lengths):
        # Calculate the start index for slicing
        start = max(length - n, 0)
        # Extract up to the last n frames
        last_n = padded_sequences[i, start:length, :]
        last_frames.append(last_n)
    last_frames = torch.stack(last_frames, dim=0)
    return last_frames

def plot_permittivity(
    permittivity: Tensor, filepath: str = "./figs/permittivity_default.png"
):
    import matplotlib.pyplot as plt

    dir_path = os.path.dirname(filepath)
    ensure_dir(dir_path)

    fig, ax = plt.subplots()
    # Plot the permittivity data and capture the image
    cax = ax.imshow(permittivity.data.cpu().numpy(), cmap="viridis")

    # Add color bar next to the plot
    fig.colorbar(cax, ax=ax)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filepath, dpi=300)

def nparray_as_real(data):
    return np.stack((data.real, data.imag), axis=-1)

def from_Ez_to_Hx_Hy(sim, eps: Tensor, Ez: Tensor, channel_first: bool = True) -> None:
    # sim: Simulation solver
    # eps b, h, w
    # Ez b, 2, h, w
    if not torch.is_complex(Ez):
        is_complex = False
        if channel_first:
            Ez = Ez.permute(0, 2, 3, 1).contiguous()
        Ez = torch.view_as_complex(Ez)
    else:
        is_complex = True

    ## Ez = [bs, h, w] complex
    Hx = []
    Hy = []
    for i in range(Ez.size(0)):
        sim.eps_r = eps[i]
        Hx_vec, Hy_vec = sim._Ez_to_Hx_Hy(Ez[i].flatten())
        Hx.append(torch.view_as_real(Hx_vec.reshape(Ez[i].shape)))
        Hy.append(torch.view_as_real(Hy_vec.reshape(Ez[i].shape)))
    Hx = torch.stack(Hx, 0)  # [bs, h, w, 2]
    Hy = torch.stack(Hy, 0)  # [bs, h, w, 2]
    if is_complex:
        Hx = torch.view_as_complex(Hx)
        Hy = torch.view_as_complex(Hy)
    elif channel_first:
        Hx = Hx.permute(0, 3, 1, 2).contiguous()
        Hy = Hy.permute(0, 3, 1, 2).contiguous()

    return Hx, Hy

def get_grid(shape, dl):
    # dl in um
    # computes the coordinates in the grid

    (Nx, Ny) = shape
    # if Ny % 2 == 0:
    #     Ny -= 1
    # coordinate vectors
    x_coord = np.linspace(-(Nx - 1) / 2 * dl, (Nx - 1) / 2 * dl, Nx)
    y_coord = np.linspace(-(Ny - 1) / 2 * dl, (Ny - 1) / 2 * dl, Ny)

    # x and y coordinate arrays
    xs, ys = np.meshgrid(x_coord, y_coord, indexing="ij")
    return (xs, ys)

def poynting_vector(
    Hx, Hy, Ez, grid_step, monitor=None, direction="x+", autograd=False
):
    if autograd:
        conj = npa.conj
        real = npa.real
        sum = npa.sum
    else:
        conj = np.conj
        real = np.real
        sum = np.sum
    if isinstance(monitor, (Slice, np.ndarray)):
        Hx = Hx[monitor]
        Hy = Hy[monitor]
        Ez_conj = conj(Ez[monitor])

    if direction == "x+":
        P = sum(real(Ez_conj * Hy)) * (-grid_step)
    elif direction == "x-":
        P = sum(real(Ez_conj * Hy)) * grid_step
    elif direction == "y+":
        P = sum(real(Ez_conj * Hx)) * grid_step
    elif direction == "y-":
        P = sum(real(Ez_conj * Hx)) * (-grid_step)
    else:
        raise ValueError("Invalid direction")
    return P

