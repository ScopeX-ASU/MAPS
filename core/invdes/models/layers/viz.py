"""
Date: 1969-12-31 17:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-12-15 00:47:25
FilePath: /MAPS/core/invdes/models/layers/viz.py
"""

import matplotlib.pylab as plt
import matplotlib.pyplot as plot
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from core.utils import print_stat

""" Utilities for plotting and visualization """


def real(
    val, outline=None, ax=None, cbar=False, cmap="RdBu", outline_alpha=0.5, font_size=9
):
    """Plots the real part of 'val', optionally overlaying an outline of 'outline'"""

    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    vmax = np.abs(val).max()
    h = ax.imshow(np.real(val.T), cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax)

    if outline is not None:
        ax.contour(outline.T, 0, colors="k", alpha=outline_alpha)

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    if cbar:
        add_colorbar(h, font_size=font_size)

    return ax


def add_colorbar(mappable, font_size=9):
    last_axes = plot.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.17)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=font_size)
    plot.sca(last_axes)
    return cbar


def abs(
    val,
    outline=None,
    ax=None,
    cbar=False,
    cmap="magma",
    outline_alpha=0.5,
    outline_val=None,
    alpha=1,
    font_size=9,
):
    """Plots the absolute value of 'val', optionally overlaying an outline of 'outline'"""
    if ax is None:
        fig, ax = plt.subplots(1, 1, constrained_layout=True)
    # print("this is the shape of val", val.shape, flush=True)
    vmax = np.abs(val).max()
    h = ax.imshow(
        np.abs(val.T), cmap=cmap, origin="lower", vmin=0, vmax=vmax, alpha=alpha
    )

    if outline_val is None and outline is not None:
        outline_val = 0.5 * (outline.min() + outline.max())
    if outline is not None:
        ax.contour(outline.T, [outline_val], colors="w", alpha=outline_alpha)

    ax.set_ylabel("y")
    ax.set_xlabel("x")
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.15)
    # if cbar:
    #     plt.colorbar(h, cax=cax)
    if cbar:
        add_colorbar(h, font_size=font_size)

    return ax
