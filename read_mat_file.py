import h5py
import matplotlib.pyplot as plt
import torch
import numpy as np

# Open the .mat file
with h5py.File("core/invdes/initialization/results_Si_metalens1D_for_850nm_FL30um.mat", "r") as f:
    # List all variable names in the file
    print("Keys:", list(f.keys()))
    y = f['y'][:]
    z = f['z'][:]
    # Load the Ey dataset
    Ey = f['Ey'][:]
    print("Shape of Ey:", Ey.shape)
    print("Dtype of Ey:", Ey.dtype)
    print("Type of Ey:", type(Ey))
    print("this is the shape of y: ", y.shape, y)
    print("this is the shape of z: ", z.shape, z)
    
    # Separate real and imaginary parts
    Ey_real = Ey['real']
    Ey_imag = Ey['imag']
    print("Shape of real part:", Ey_real.shape)
    print("Shape of imaginary part:", Ey_imag.shape)
    
    # Combine into a complex array
    Ey_complex = Ey_real + 1j * Ey_imag
    print("Shape of Ey_complex:", Ey_complex.shape)
    print("Dtype of Ey_complex:", Ey_complex.dtype)

    Ey_complex = torch.tensor(Ey_complex).squeeze()
    Ey_complex = Ey_complex.squeeze()

    y = torch.tensor(y).squeeze() * 1e6
    z = torch.tensor(z).squeeze() * 1e6 # convert to microns

    intensity = torch.abs(Ey_complex)**2

    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(z[:300], y, intensity[0][:300].T, shading='auto', cmap='magma')

    # Add labels and a colorbar
    plt.xlabel("Y Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Field Intensity Plot on Non-Uniform Mesh")
    plt.colorbar(mesh, label="Intensity")

    # Save the plot as a PNG file
    output_file = "Ey_intensity.png"
    plt.savefig(output_file, dpi=300)  # dpi=300 for high-resolution output
    plt.close()

    Ey_real = torch.tensor(Ey_real).squeeze()
    plt.figure(figsize=(8, 6))
    mesh = plt.pcolormesh(z[:300], y, Ey_real[0][:300].T, shading='auto', cmap='RdBu')
    # Add labels and a colorbar
    plt.xlabel("Y Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Field Real Part Plot on Non-Uniform Mesh")
    plt.colorbar(mesh, label="Real Part")

    # Save the plot as a PNG file
    output_file = "Ey_real.png"
    plt.savefig(output_file, dpi=300)  # dpi=300 for high-resolution output
    plt.close()

    # plt.figure()
    # plt.imshow((torch.abs(Ey_complex[0][:300, ...])**2).cpu().numpy().T, cmap='magma')
    # plt.colorbar()
    # plt.savefig('Ey_intensity.png', dpi=600)
    # plt.close()
    # Ey_real = torch.tensor(Ey_real).squeeze()
    # plt.figure()
    # plt.imshow((Ey_real[0][:300, ...]).cpu().numpy().T, cmap='RdBu')
    # plt.savefig('Ey_real.png', dpi=600)
    # plt.close()
