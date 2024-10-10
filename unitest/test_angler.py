'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-27 01:21:54
LastEditors: JeremieMelo jqgu@utexas.edu
LastEditTime: 2023-09-12 16:54:11
'''
import numpy as np
import matplotlib.pylab as plt
import copy
from pyutils.general import TimerCtx
from pyutils.torch_train import set_torch_deterministic
# add angler to path (not necessary if pip installed)
import sys
import torch

from core.utils import CurlLoss, TMPropagateFieldLoss
sys.path.append("..")

# import the main simulation and optimization classes
from angler import Simulation, Optimization

# import some structure generators
from angler.structures import three_port, two_port
from angler.derivatives import unpack_derivs

# define the similation constants
lambda0 = 1.55e-6              # free space wavelength (m)
c0 = 3e8                    # speed of light in vacuum (m/s)
omega = 2*np.pi*c0/lambda0  # angular frequency (2pi/s)
dl = 0.1                    # grid size (L0)
NPML = [10, 10]             # number of pml grid points on x and y borders
# NPML = [0, 0]             # number of pml grid points on x and y borders
pol = 'Hz'                  # polarization (either 'Hz' or 'Ez')
source_amp = 1e-9           # amplitude of modal source (make around 1 for nonlinear effects)

# define material constants
n_index = 2.44              # refractive index
eps_m = n_index**2          # relative permittivity

# define permittivity of three port system
# eps_r, design_region = N_port(N, L, H, w, d, l, spc, dl, NPML, eps_m)
device = torch.device("cuda:0")
# wavelength, epsilon, field = torch.load("./data/mmi2x2/processed/training.pt")
from data.mmi.device_shape import mmi_3x3_L_random
set_torch_deterministic(0)
device = mmi_3x3_L_random()
eps_r = device.epsilon_map
# eps_r = epsilon[0,0].t().numpy().real
print(eps_r.shape)
## add waveguide
eps_si = eps_r.max()
eps_sio2 = eps_r.min()
print(eps_si, eps_sio2)
extra_l = 100
eps_i = np.zeros((extra_l, eps_r.shape[1])) + eps_sio2
wg = 4
eps_i[:,23:23+wg] = eps_si
eps_i[:,54:54+wg] = eps_si
eps_r = np.concatenate([eps_i, eps_r], axis=0)


ds = 1
eps_r = eps_r[:, ::ds]
# eps_r[eps_r>8] = eps_si
# eps_r[eps_r<7.999] = eps_sio2
name = "angler_mmi_simu_0.1.png"
# eps_r[250:450, 30:51] = eps_si+0.1
(Nx, Ny) = eps_r.shape
# nx, ny = int(Nx/2), int(Ny/2)            # halfway grid points

# make a new simulation object
simulation = Simulation(omega, eps_r, dl, NPML, pol)

print("Computed a domain with {} grids in x and {} grids in y".format(Nx,Ny))
print("The simulation has {} grids per free space wavelength".format(int(lambda0/dl/simulation.L0)))

# plot the permittivity distribution
simulation.plt_eps(outline=False)
plt.savefig("angler_mmi_eps.png")

N = 2
l = 2
d = 10
w = dl * wg / ds
# ny = 59 / ds
ny = 55.5 / ds

# set the input waveguide modal source
simulation.add_mode(neff=np.sqrt(eps_si), direction_normal='x', center=[NPML[0]+int(l/2/dl), ny], width=int(2*w/dl), scale=source_amp)
simulation.setup_modes()

# set source and solve for electromagnetic fields
with TimerCtx() as t:
    (Ex, Ey, Hz) = simulation.solve_fields(solver="scipy")
print(f"Solve field cost: {t.interval} s")
simulation.fields["Hz"] = Ey
simulation.plt_abs(outline=True, cbar=True)
# simulation.plt_re()
plt.savefig(name)
exit(0)

print(Ex.shape)
loss = CurlLoss(grid_step = dl, wavelength=lambda0*1e6)
print(loss(torch.from_numpy(np.stack([Ex, Ey, Hz], axis=0)), torch.from_numpy(eps_r).to(torch.cfloat)))
ex, ey = loss.Hy_to_E(torch.from_numpy(Hz).to(torch.cfloat).unsqueeze(0).unsqueeze(0), epsilon_r=torch.from_numpy(eps_r).to(torch.cfloat).unsqueeze(0).unsqueeze(0))
mag_Ex = np.mean(np.abs(Ex)**2)
mag_Ey = np.mean(np.abs(Ey)**2)
mag_Hz = np.mean(np.abs(Hz)**2)
mag_ex = np.mean(np.abs(ex[0,0].numpy())**2)
mag_ey = np.mean(np.abs(ey[0,0].numpy())**2)
print(mag_Ex, mag_ex)
simulation.fields["Hz"] = Ey
simulation.plt_abs(outline=True, cbar=True)
# simulation.plt_re()
plt.savefig(name)
simulation.fields["Hz"] = ex[0,0].numpy()
simulation.plt_abs(outline=True, cbar=True)
# simulation.plt_re()
# plt.savefig("angler_mmi_gen.png")
(Dyb, Dxb, Dxf, Dyf) = unpack_derivs(simulation.derivs)
# print(Dyb.todense().shape)
print([i for i in Dyb.todense()[10001].flatten().tolist()[0] if abs(i) > 0])
