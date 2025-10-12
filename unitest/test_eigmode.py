"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-10 22:15:58
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-10 22:19:25
"""

import copy

# add angler to path (not necessary if pip installed)
import sys

import matplotlib.pylab as plt
import numpy as np
import torch

# import the main simulation and optimization classes
from angler import Simulation

# define the similation constants
lambda0 = 1.55e-6  # free space wavelength (m)
c0 = 299792458  # speed of light in vacuum (m/s)
omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
dl = 0.1  # grid size (L0) um
NPML = [10, 10]  # number of pml grid points on x and y borders
pol = "Hz"  # polarization (either 'Hz' or 'Ez')
source_amp = 1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
ext_wg = 10  # length os extended input waveguide (um)
wg = 0.4  # waveguide width (um)
port1 = 2.5  # input waveguide center location (um)
port2 = 5.6  # input waveguide center location (um)
mode_center_1 = 2.5
mode_center_2 = 5.7
mode_width = 0.5
pads = ((14, 38, 2, 3.5), (14, 38, 4, 5.5))  # pad location (xl,xh,yl,yh) (um)
eps_range = (11.9, 12.3)  # tuning range of Si permittivity

# define permittivity of three port system
# eps_r, design_region = N_port(N, L, H, w, d, l, spc, dl, NPML, eps_m)
device = torch.device("cuda:0")
wavelength, epsilon, _ = torch.load("./data/mmi2x2/processed/training.pt")
print(wavelength.shape)
eps_r = epsilon[0, 0].t().numpy().real
print(eps_r.shape)
# print(eps_r[1, :])
# exit(0)
## add waveguide
eps_si = eps_r.max()
eps_sio2 = eps_r.min()

## binarize permittivities on SiO2 and Si boundary
# eps_r[eps_r > (eps_si + eps_sio2) / 2] = eps_si
# eps_r[eps_r <= (eps_si + eps_sio2) / 2] = eps_sio2

## extension of input waveguide
eps_i = np.zeros((int(ext_wg / dl), eps_r.shape[1])) + eps_sio2
start, end = round((port1 - wg / 2) / dl), round((port1 + wg / 2) / dl) - 1
print(start, end)
eps_i[:, round((port1 - wg / 2) / dl) : round((port1 + wg / 2) / dl) - 1] = eps_si
eps_i[:, [start - 1, end]] = 6.177296

start, end = round((port2 - wg / 2) / dl) + 1, round((port2 + wg / 2) / dl)
print(start, end)
eps_i[:, round((port2 - wg / 2) / dl) + 1 : round((port2 + wg / 2) / dl)] = eps_si
eps_i[:, [start - 1, end]] = 6.177296
eps_r = np.concatenate([eps_i, eps_r], axis=0)
(Nx, Ny) = eps_r.shape

n_points = 5


simulation = Simulation(omega, eps_r, dl, NPML, pol)

# plot the permittivity distribution
# simulation.plt_eps(outline=False)
# plt.savefig("angler_mmi_eps.png", dpi=300)
# exit(0)

l = 0.5
ny = mode_center_2

# set the input waveguide modal source
simulation.add_mode(
    neff=np.sqrt(eps_si),
    direction_normal="x",
    center=[NPML[0] + int(l / 2 / dl), ny / dl],
    width=int(2 * mode_width / dl),
    scale=source_amp,
)
center = [NPML[0] + int(l / 2 / dl), ny / dl]
width = int(2 * mode_width / dl)
simulation.setup_modes()
print(
    simulation.src[
        int(center[0]) : int(center[0] + 1),
        int(center[1] - width // 2) : int(center[1] + width // 2),
    ]
)
