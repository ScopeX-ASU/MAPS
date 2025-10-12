"""
this code is just the example in meep tutorial for frequency-domain solver

I modified this code to verify the focal picture see if it look like Yu's result
"""

import h5py
import matplotlib.pyplot as plt
import meep as mp
import numpy
import numpy as np
import torch
from numpy import linalg as LA

n = 3.48
pad_x = 2
pad_y = 1
dpml = 2

res = 50

aperture = 20
height_sub = dpml + pad_x + 2
height_ridge = 0.75
period = 0.3
f_min = 60
wl_cen = 0.85
alpha = 0.5
wl_width = 0.1

sx = 2 * (pad_x + dpml) + 2 + 1 + f_min + 20
sy = 2 * (dpml + pad_y) + aperture
cell_size = mp.Vector3(sx, sy)

boundary = [
    mp.PML(2, direction=mp.X),
    mp.PML(2, direction=mp.Y),
]

nonpml_vol = mp.Volume(mp.Vector3(), size=mp.Vector3(sx - 2 * dpml, sy - 2 * dpml))

geometry = []

sub = mp.Block(
    center=mp.Vector3(-sx / 2 + height_sub / 2, 0),
    size=mp.Vector3(height_sub, aperture, 0),
    material=mp.Medium(index=n),
)

geometry.append(sub)

widths = torch.zeros(122)
with h5py.File("core/models/Si_metalens1D_for_850nm_FL60um.mat", "r") as f:
    widths = f["Si_width"][: len(f["Si_width"]) // 2 + 1, 0]
    if isinstance(widths, numpy.ndarray):
        widths = torch.tensor(widths)
        widths = widths.flip(0)
    widths = widths * 1e6

for i in range(len(widths)):
    geometry.append(
        mp.Block(
            center=mp.Vector3(-0.5 * sx + height_sub + 0.5 * height_ridge, period * i),
            size=mp.Vector3(height_ridge, widths[i], mp.inf),
            material=mp.Medium(index=n),
        )
    )
    geometry.append(
        mp.Block(
            center=mp.Vector3(-0.5 * sx + height_sub + 0.5 * height_ridge, -period * i),
            size=mp.Vector3(height_ridge, widths[i], mp.inf),
            material=mp.Medium(index=n),
        )
    )

fcen = 1 / wl_cen  # pulse center frequency
## alpha from 1/3 to 1/2
fwidth = (
    3 * alpha * (1 / (wl_cen - wl_width / 2) - 1 / (wl_cen + wl_width / 2))
)  # pulse frequency width
src_center = (-0.5 * sx + height_sub / 2 + 0.5, 0)
src = [
    mp.EigenModeSource(
        src=mp.GaussianSource(fcen, fwidth=fwidth),
        center=mp.Vector3(*src_center),
        size=mp.Vector3(y=aperture),
        eig_match_freq=True,
    )
]

sim = mp.Simulation(
    resolution=res,
    cell_size=mp.Vector3(*cell_size),
    boundary_layers=boundary,
    geometry=geometry,
    sources=src,
    default_material=mp.Medium(index=1),
    force_all_components=True,
)

dft_obj = sim.add_dft_fields([mp.Ez], fcen, 0, 1, where=nonpml_vol)

sim.run(until=200)

eps_data = sim.get_array(vol=nonpml_vol, component=mp.Dielectric)
ez_data = np.real(sim.get_dft_array(dft_obj, mp.Ez, 0))

plt.figure()
plt.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
plt.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
plt.axis("off")
plt.savefig("unitest/dft_field_initial_design.png")
