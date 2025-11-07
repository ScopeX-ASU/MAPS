import numpy as np
import torch


@torch.compile
def binarize(device, eta=0.5, beta=np.inf):
    num = np.tanh(beta * eta) + torch.tanh(beta * (device - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1 - eta))
    device = num / den
    return device


def binarize_hard(device, eta=0.5):
    device[device < eta] = 0
    device[device >= eta] = 1
    return device


def trim(device):
    x, y = np.nonzero(device)
    xl, xr = x.min(), x.max()
    yl, yr = y.min(), y.max()
    return device[xl : xr + 1, yl : yr + 1]


def pad(device, slice_length, fctr=1):
    pady = (
        slice_length * np.ceil(device.shape[0] / slice_length) - device.shape[0]
    ) / 2 + slice_length * (fctr - 1) / 2
    padx = (
        slice_length * np.ceil(device.shape[1] / slice_length) - device.shape[1]
    ) / 2 + slice_length * (fctr - 1) / 2
    device = np.pad(
        device,
        [
            (int(np.ceil(pady)), int(np.floor(pady))),
            (int(np.ceil(padx)), int(np.floor(padx))),
        ],
        mode="constant",
    )
    return device
