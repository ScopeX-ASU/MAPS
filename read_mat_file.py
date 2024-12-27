import h5py
import matplotlib.pyplot as plt
import torch
import numpy as np

# Open the .mat file
with h5py.File("core/invdes/initialization/results_Si_metalens1D_for 850nm_FL30um.mat", "r") as f:
    # List all variable names in the file
    print("Keys:", list(f.keys()))
    # Load the Ey dataset
    Ey = f['Ey'][:]
    print("Shape of Ey:", Ey.shape)
    print("Dtype of Ey:", Ey.dtype)
    print("Type of Ey:", type(Ey))
    
    # Separate real and imaginary parts
    Ey_real = Ey['real']
    Ey_imag = Ey['imag']
    print("Shape of real part:", Ey_real.shape)
    print("Shape of imaginary part:", Ey_imag.shape)
    
    # Combine into a complex array
    Ey_complex = Ey_real + 1j * Ey_imag
    print("Shape of Ey_complex:", Ey_complex.shape)
    print("Dtype of Ey_complex:", Ey_complex.dtype)

    Ey_complex = torch.tensor(Ey_complex)
    Ey_complex = Ey_complex.squeeze()
    plt.figure()
    plt.imshow((torch.abs(Ey_complex[0][:300, ...])**2).cpu().numpy().T, cmap='magma')
    plt.savefig('Ey_intensity.png', dpi=600)
    plt.close()
    Ey_real = torch.tensor(Ey_real).squeeze()
    plt.figure()
    plt.imshow((Ey_real[0][:300, ...]).cpu().numpy().T, cmap='RdBu')
    plt.savefig('Ey_real.png', dpi=600)
    plt.close()
