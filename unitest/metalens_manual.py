"""
Date: 2025-01-04 03:19:57
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-04 03:50:42
FilePath: /MAPS/unitest/metalens_manual.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

data = h5py.File(
    "./core/invdes/initialization/results_Si_metalens1D_for 850nm_FL30um.mat"
)
ind = 1
E = np.array(data["Ey"][:]).view(np.complex128)[ind].squeeze()  # 2D array
E = E.T

plt.imshow(E.real, cmap="RdBu")
plt.colorbar()
plt.savefig(f"./unitest/original_field_{ind}.jpg", dpi=300)
plt.close()

ys = data["y"][:].squeeze()  # 1D array
zs = data["z"][:].squeeze()  # 1D array

### resample in uniform grid


y = np.linspace(ys[0], ys[-1], 4000)
z = np.linspace(zs[0], zs[-1], 16000)
f_real = RegularGridInterpolator((ys, zs), E.real)
f_imag = RegularGridInterpolator((ys, zs), E.imag)
Y, Z = np.meshgrid(y, z, indexing="ij")
indices = np.stack([Y, Z], axis=-1)
E_real = f_real(indices)
### plot the original field f
fig, ax = plt.subplots(figsize=(30, 10))
plt.imshow(E_real, cmap="RdBu")
plt.colorbar()
plt.savefig(f"./unitest/resampled_field_{ind}.jpg", dpi=600)
plt.close()

plt.imshow(np.abs(E) ** 2, cmap="magma")
plt.colorbar()
plt.savefig(f"./unitest/resampled_field_int_{ind}.jpg", dpi=600)
plt.close()
