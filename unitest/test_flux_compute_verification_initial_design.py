'''
this code is used to verify the flux computation of the initial design

first a normalize run should be performed to get the incident flux
then, run a second simulation and compute the following fields:
calculate the reflected flux
farfield with a very big monitor, calculate it using inner function, the flux should be close to the incident flux - reflected flux
farfield with a very big monitor, calculate it using pouyting flux, the flux should be close to the incident flux - reflected flux
'''

import meep as mp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import h5py
import torch
import numpy

n = 3.48
pad_x = 2
pad_y = 1
dpml = 2
nfreq = 20

res = 50

aperture = 20
height_sub = dpml+pad_x+2
height_ridge = 0.75
period = 0.3
f_min = 60
wl_cen = 0.85
alpha = 0.5
wl_width = 0.1

sample_step = f_min / 300

sx = 2*(pad_x+dpml) + 2 + 1 # 2 is in the height of the sub, 1 is the max height of the ridge
sy = 2*(dpml+pad_y) + aperture
cell_size = mp.Vector3(sx, sy)

boundary = [
    mp.PML(2, direction=mp.X),
    mp.PML(2, direction=mp.Y),
]

nonpml_vol = mp.Volume(mp.Vector3(), size=mp.Vector3(sx-2*dpml,sy-2*dpml))

geometry = []

sub = mp.Block(center=mp.Vector3(-sx/2 + height_sub/2,0),
                size=mp.Vector3(height_sub, aperture, 0),
                material=mp.Medium(index=n))

geometry.append(sub)

with h5py.File("core/models/Si_metalens1D_for_850nm_FL60um.mat", 'r') as f:
    widths = f["Si_width"][: len(f["Si_width"])//2 + 1, 0]
    if isinstance(widths, numpy.ndarray):
        widths = torch.tensor(widths)
        widths = widths.flip(0)
    widths = widths * 1e6

for i in range(len(widths)):
    geometry.append(mp.Block(center=mp.Vector3(-0.5*sx+height_sub+0.5*height_ridge, period*i),
                             size=mp.Vector3(height_ridge, widths[i], mp.inf),
                             material=mp.Medium(index=n)))
    geometry.append(mp.Block(center=mp.Vector3(-0.5*sx+height_sub+0.5*height_ridge, -period*i),
                             size=mp.Vector3(height_ridge, widths[i], mp.inf),
                             material=mp.Medium(index=n)))

fcen = 1 / wl_cen  # pulse center frequency
## alpha from 1/3 to 1/2
fwidth = (
    3 * alpha * (1 / (wl_cen - wl_width / 2) - 1 / (wl_cen + wl_width / 2))
)  # pulse frequency width
src_center = (-0.5*sx+height_sub/2+0.5, 0)
src = [
    mp.EigenModeSource(
                    src=mp.GaussianSource(fcen, fwidth=fwidth),
                    center=mp.Vector3(*src_center),
                    size=mp.Vector3(y=aperture),
                    eig_match_freq=True,
                    eig_kpoint=mp.Vector3(1, 0),
                )
]

norm_sim = mp.Simulation(
    resolution=res,
    cell_size=mp.Vector3(*cell_size),
    boundary_layers=boundary,
    geometry=[],
    sources=src,
    default_material=mp.Medium(index=n),
    force_all_components=True,
)

flux_box = norm_sim.add_flux(fcen, fwidth, nfreq,
    mp.FluxRegion(
        center=mp.Vector3(-0.5*sx+height_sub/2+1, 0),
        size=mp.Vector3(0, aperture),
    )
)
norm_sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-3))
incident_power = numpy.array(mp.get_fluxes(flux_box)).sum() # this should be a list nf elements
print("this is the incident power: ", incident_power)
# ax = norm_sim.plot2D()
# plt.savefig('unitest/incident_power_norm_run.png', dpi =1000)

sim = mp.Simulation(
    resolution=res,
    cell_size=mp.Vector3(*cell_size),
    boundary_layers=boundary,
    geometry=geometry,
    sources=src,
    default_material=mp.Medium(index=1),
    force_all_components=True,
)

refl_fr = mp.FluxRegion(center=mp.Vector3(-0.5*sx+height_sub/2,0,0), size=mp.Vector3(0,aperture,0))
refl = sim.add_flux(fcen, fwidth, nfreq, refl_fr)

tran_fr = mp.FluxRegion(center=mp.Vector3(-0.5*sx+height_sub+height_ridge+0.5,0), size=mp.Vector3(0,aperture,0))
tran = sim.add_flux(fcen, fwidth, nfreq, tran_fr)

top = mp.FluxRegion(center=mp.Vector3(-0.5*sx+height_sub+height_ridge+0.25,aperture/2), size=mp.Vector3(0.5,0,0))
bot = mp.FluxRegion(center=mp.Vector3(-0.5*sx+height_sub+height_ridge+0.25,-aperture/2), size=mp.Vector3(0.5,0,0))
top_flux = sim.add_flux(fcen, fwidth, nfreq, top)
bot_flux = sim.add_flux(fcen, fwidth, nfreq, bot)

nearfield_box = sim.add_near2far(fcen, fwidth, nfreq,
                                 mp.Near2FarRegion(center=mp.Vector3(-0.5*sx+height_sub+height_ridge+0.5,0),
                                                   size=mp.Vector3(0,aperture,0)),
                                 mp.Near2FarRegion(center=mp.Vector3(-0.5*sx+height_sub+height_ridge+0.25,aperture/2),
                                                   size=mp.Vector3(0.5,0,0)),
                                 mp.Near2FarRegion(center=mp.Vector3(-0.5*sx+height_sub+height_ridge+0.25,-aperture/2),
                                                   size=mp.Vector3(0.5,0,0),
                                                   weight=-1))

sim.run(until_after_sources=mp.stop_when_dft_decayed(1e-3))

reflected_flux = numpy.array(mp.get_fluxes(refl)).sum()
transmitted_flux = numpy.array(mp.get_fluxes(tran)).sum()
top_flux = numpy.array(mp.get_fluxes(top_flux)).sum()
bot_flux = numpy.array(mp.get_fluxes(bot_flux)).sum()

res_ff = 100/f_min

far_flux_box = (numpy.array(nearfield_box.flux(mp.Y,
                                   mp.Volume(center=mp.Vector3(f_min/2, f_min),
                                             size=mp.Vector3(f_min, 0, 0)),
                                   res_ff)) -
                numpy.array(nearfield_box.flux(mp.Y,
                                   mp.Volume(center=mp.Vector3(f_min/2, -f_min),
                                             size=mp.Vector3(f_min, 0, 0)),
                                   res_ff)) +
                numpy.array(nearfield_box.flux(mp.X,
                                   mp.Volume(center=mp.Vector3(f_min),
                                             size=mp.Vector3(y=2*f_min)),
                                   res_ff)))

E_field = []
H_field = []
for i in range(int(f_min // sample_step)):
    ff = sim.get_farfield(nearfield_box,
                          mp.Vector3(i * sample_step, f_min))
    E_field.append(numpy.array(ff).reshape(nfreq, 6)[:, :3])
    H_field.append(numpy.array(ff).reshape(nfreq, 6)[:, 3:])

for j in range(int(f_min // sample_step)):
    ff = sim.get_farfield(nearfield_box,
                          mp.Vector3(i * sample_step, -f_min))
    E_field.append(numpy.array(ff).reshape(nfreq, 6)[:, :3])
    H_field.append(numpy.array(ff).reshape(nfreq, 6)[:, 3:])

for k in range(int(f_min // sample_step) - 1):
    ff1 = sim.get_farfield(nearfield_box,
                          mp.Vector3(f_min, k * sample_step))
    E_field.append(numpy.array(ff1).reshape(nfreq, 6)[:, :3])
    H_field.append(numpy.array(ff1).reshape(nfreq, 6)[:, 3:])
    if k != 0:
        ff2 = sim.get_farfield(nearfield_box,
                        mp.Vector3(f_min, -k * sample_step))
        E_field.append(numpy.array(ff2).reshape(nfreq, 6)[:, :3])
        H_field.append(numpy.array(ff2).reshape(nfreq, 6)[:, 3:])

E_field = numpy.stack(E_field, axis=0)
H_field = numpy.stack(H_field, axis=0)

print("this is the shape of the E field: ", E_field.shape)
print("this is the shape of the H field: ", H_field.shape)

Px = np.real(np.conj(E_field[:, :, 1]) * H_field[:, :, 2] - np.conj(E_field[:, :, 2]) * H_field[:, :, 1])
Py = np.real(np.conj(E_field[:, :, 2]) * H_field[:, :, 0] - np.conj(E_field[:, :, 0]) * H_field[:, :, 2])
Pr = np.sqrt(np.square(Px) + np.square(Py))

far_flux_discret_int = np.sum(Pr) * 4 * f_min /len(Pr)


print("this is the reflected power: ", reflected_flux)
print("this is the transmitted power: ", transmitted_flux)
print("this is the top power: ", top_flux)
print("this is the bot power: ", bot_flux)

print("this is the farfield power: ", far_flux_box.sum())
print("this is the farfield power using numerical int: ", far_flux_discret_int)

print("this is the error: ", incident_power - (-1) * reflected_flux - transmitted_flux - top_flux - (-1) * bot_flux)