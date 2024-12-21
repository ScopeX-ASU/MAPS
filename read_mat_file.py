import h5py

# Open the .mat file
with h5py.File("core/invdes/initialization/Si_metalens1D_for_850nm_FL30um_EDOF4.mat", "r") as f:
    # List all variable names in the file
    print("Keys:", list(f.keys()))
    
    # Access a specific variable (replace 'your_variable_name' with the actual name)
    variable = f['Si_width'][:]
    print(variable)
