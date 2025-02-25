nosf_path = "figs/metalens_TF_fsdx-0.3_wl-0.85_p-0.3_mat-Si/transfer_matrix_NOSF.h5"
tfsf_path = "figs/metalens_TF_fsdx-0.3_wl-0.85_p-0.3_mat-Si/transfer_matrix_TFSF.h5"

import h5py
import torch
import numpy

# open the h5py file
with h5py.File(nosf_path, "r") as f:
    # read the data
    data = f["transfer_matrix"][:]
    # convert to torch tensor
    nosf = torch.tensor(data, dtype=torch.complex64)

with h5py.File(tfsf_path, "r") as f:
    # read the data
    data = f["transfer_matrix"][:]
    # convert to torch tensor
    tfsf = torch.tensor(data, dtype=torch.complex64)

# check if the two tensors are equal
assert torch.allclose(nosf, tfsf)
print("The two transfer matrices are equal!")