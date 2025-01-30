'''
read the forward field and adjoint field from the data
compare the calculated gradient and the gt gradient
'''
import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../MAPS")
)
sys.path.insert(0, project_root)
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from core.utils import print_stat

def test_gradient(filepath):
    # open the file
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())
        # print("this is the keys: ", keys)
        # quit()
        # read the forward field and adjoint field
        forward_field = torch.from_numpy(f["field_solutions-wl-1.55-port-in_port_1-mode-1-temp-300"][:]).to("cuda")
        adjoint_field = torch.from_numpy(f["fields_adj-wl-1.55-port-in_port_1-mode-1"][:]).to("cuda")
        # read the gradient
        gradient = torch.from_numpy(f["gradient"][:]).to("cuda")

        forward_field = forward_field[-1, ...].contiguous()
        adjoint_field = adjoint_field[-1, ...].contiguous()

        # calculate the gradient
        calculated_gradient = (-adjoint_field * forward_field).real
        # print_stat(calculated_gradient)
        # print_stat(gradient)
        # quit()
        
        plt.figure()
        plt.imshow(calculated_gradient.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("Calculated Gradient")
        plt.savefig("./figs/grad_0.png")
        plt.close()

        plt.figure()
        plt.imshow(gradient.cpu().detach().numpy(), cmap="RdBu")
        plt.colorbar()
        plt.title("GT Gradient")
        plt.savefig("./figs/grad_gt_0.png")
        plt.close()

if __name__ == "__main__":
    filepath = "/home/pingchua/projects/MAPS/data/fdfd/bending/raw_random/bending_id-3_opt_step_0.h5"
    test_gradient(filepath)