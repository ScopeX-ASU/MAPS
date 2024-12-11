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
