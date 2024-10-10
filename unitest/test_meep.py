# %%
'''
Description: 
Author: Jiaqi Gu (jiaqigu@asu.edu)
Date: 2023-10-03 12:12:40
LastEditors: ScopeX-ASU jiaqigu@asu.edu
LastEditTime: 2023-10-07 18:02:18
'''
import meep as mp
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import Video

# n = 3.4  # index of waveguide
n = 2.45  # index of waveguide, 1550 nm, TE0 mode, 220nm thinkness, 500 nm width
w = 0.5  # width of waveguide
r = 5  # inner radius of ring
wg_gap = 0.15
pad = 1.5  # padding between waveguide and edge of PML
dpml = 2  # thickness of PML
sxy = 2 * (r + w + pad + dpml)  # cell size

# %%
c1 = mp.Cylinder(radius=r + w, material=mp.Medium(index=n))
c2 = mp.Cylinder(radius=r)
wg_ycenter = -(r + w + wg_gap + w / 2)
c3 = mp.Block(
        mp.Vector3(mp.inf, w, mp.inf),
        center=mp.Vector3(y=wg_ycenter),
        material=mp.Medium(index=n),
    )

# %%
fcen = 1/1.55  # pulse center frequency
# df = fcen - 0.05  # pulse frequency width
df = 1/1.5 - 1/1.6  # pulse frequency width
src = mp.EigenModeSource(
            src=mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(-sxy/2+dpml, wg_ycenter),
            size=(0, w, 0),
            eig_match_freq=True,
            eig_parity=mp.ODD_Z + mp.EVEN_Y,
        )



# %%
sim = mp.Simulation(
    cell_size=mp.Vector3(sxy, sxy),
    boundary_layers=[mp.PML(dpml)],
    geometry=[c3],
    sources=[src],
    resolution=10,
)

nfreq = 400  # number of frequencies at which to compute flux
# transmitted flux
tran_fr = mp.FluxRegion(
    center=mp.Vector3(0.5 * sxy - dpml, wg_ycenter, 0), size=mp.Vector3(0, 2 * w, 0)
)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# plt.figure(dpi=150)
# sim.plot2D()
# plt.show()

pt = mp.Vector3(0.5 * sxy - dpml - 0.5, wg_ycenter)

f = plt.figure(dpi=150)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
sim.run(mp.at_every(0.5, Animate), until=80)
plt.close()
straight_tran_flux = mp.get_fluxes(tran)

# Process the animation and view it
filename = "media/ring_simple_wg.mp4"
Animate.to_mp4(20, filename)
# Animate.to_gif(5, filename)
Video(filename)

# for normalization run, save flux fields data for reflection plane

# %%
sim.reset_meep()
sim = mp.Simulation(
    cell_size=mp.Vector3(sxy, sxy),
    geometry=[c1, c2, c3],
    sources=[src],
    resolution=10,
    boundary_layers=[mp.PML(dpml)],
)
nfreq = 400  # number of frequencies at which to compute flux
# transmitted flux
inp_fr = mp.FluxRegion(
    center=mp.Vector3(-0.5 * sxy + dpml + 0.5, wg_ycenter, 0), size=mp.Vector3(0, 2 * w, 0)
)
tran_fr = mp.FluxRegion(
    center=mp.Vector3(0.5 * sxy - dpml, wg_ycenter, 0), size=mp.Vector3(0, 2 * w, 0)
)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)
inp = sim.add_flux(fcen, df, nfreq, inp_fr)

plt.figure(dpi=150)
sim.plot2D()
plt.show()

# %%
pt = mp.Vector3(sxy/2-dpml, wg_ycenter)
f = plt.figure(dpi=150)
Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)

sim.run(
    mp.at_beginning(Animate),
    # mp.at_beginning(mp.output_epsilon),
    mp.after_sources(mp.Harminv(mp.Ez, pt, fcen, df)),
    mp.at_every(0.3, Animate), # 0.3 times unit = 0.3*20 = 6 timestep = 6 * 0.16666 fs = 1 fs
    # mp.at_every(0.6, Animate), # 0.6 times unit = 0.6*20 = 12 timestep = 12 * 0.16666 fs = 2 fs
    until_after_sources=2000,
    # until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(r+0.2), 1e-3),
)
ring_tran_flux = mp.get_fluxes(tran)
ring_inp_flux = mp.get_fluxes(inp)

flux_freqs = mp.get_flux_freqs(tran)
# Close the animator's working frame
plt.close()

# Process the animation and view it
filename = "media/ring_simple.mp4"
Animate.to_mp4(20, filename)
# Animate.to_gif(5, filename)
Video(filename)


# %%
from matplotlib import markers


wl = []
Rs = []
Ts = []
Is = []
Os = []
print(flux_freqs)
print(ring_tran_flux)
print(ring_inp_flux)
for i in range(nfreq):
    wl = np.append(wl, 1 / flux_freqs[i])
    Os = np.append(Os, ring_tran_flux[i])
    Is = np.append(Is, ring_inp_flux[i])
    # Ts = np.append(Ts, ring_tran_flux[i] / ring_inp_flux[i])
    Ts = np.append(Ts, ring_tran_flux[i] / ring_inp_flux[i])

if mp.am_master():
    # plt.figure(dpi=150, figsize=(9,3))
    xlim = [1.5, 1.6]
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    # plt.plot(wl, Ts, "ro-", label="transmittance")
    axes[0].plot(wl, Is, "go-", label="input", markersize=1)
    axes[0].set(xlim=xlim, ylim=[0, 800])
    axes[0].legend(loc="upper right")

    axes[1].plot(wl, Os, "bo-", label="output", markersize=1)
    axes[1].set(xlim=xlim, ylim=[0, 800])
    axes[1].legend(loc="upper right")

    axes[2].plot(wl, Ts, "go-", label="transmission", markersize=1)
    axes[2].set(xlim=xlim, ylim=[0, 2])
    axes[2].legend(loc="upper right")
    # plt.axis([1.4, 1.6, 0, 150])
    plt.xlabel("wavelength (μm)")
    # plt.legend(loc="upper right")
    plt.show()

# %%


# %%



