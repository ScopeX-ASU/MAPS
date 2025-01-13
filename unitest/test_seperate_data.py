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

def test_gradient(filepath1, filepath2):
    # open the file
    with h5py.File(filepath1, "r") as f:
        keys = list(f.keys())
        print(keys)
        # read the forward field and adjoint field
        # forward_field = torch.from_numpy(f["field_solutions"][:]).to("cuda")
        # adjoint_field = torch.from_numpy(f["fields_adj"][:]).to("cuda")
        # # read the gradient
        # gradient_gt_1 = torch.from_numpy(f["gradient"][:]).to("cuda")

        # forward_field_1 = forward_field[-1, ...].contiguous()
        # adjoint_field_1 = adjoint_field[-1, ...].contiguous()

        # # calculate the gradient
        # calculated_gradient_1 = (adjoint_field_1 * forward_field_1).real
        # # print_stat(calculated_gradient)
        # # print_stat(gradient)
        # # quit()
        
        # plt.figure()
        # plt.imshow(calculated_gradient_1.cpu().detach().numpy(), cmap="RdBu")
        # plt.colorbar()
        # plt.title("Calculated Gradient")
        # plt.savefig("./figs/grad_1.png")
        # plt.close()

        # plt.figure()
        # plt.imshow(gradient_gt_1.cpu().detach().numpy(), cmap="RdBu")
        # plt.colorbar()
        # plt.title("GT Gradient")
        # plt.savefig("./figs/grad_gt_1.png")
        # plt.close()

    # with h5py.File(filepath2, "r") as f:
    #     keys = list(f.keys())
    #     # read the forward field and adjoint field
    #     forward_field = torch.from_numpy(f["field_solutions"][:]).to("cuda")
    #     adjoint_field = torch.from_numpy(f["fields_adj"][:]).to("cuda")
    #     # read the gradient
    #     gradient_gt_2 = torch.from_numpy(f["gradient"][:]).to("cuda")
    #     total_gradient = torch.from_numpy(f["total_gradient"][:]).to("cuda")

    #     forward_field_2 = forward_field[-1, ...].contiguous()
    #     adjoint_field_2 = adjoint_field[-1, ...].contiguous()

    #     # calculate the gradient
    #     calculated_gradient_2 = (adjoint_field_2 * forward_field_2).real
    #     # print_stat(calculated_gradient)
    #     # print_stat(gradient)
    #     # quit()
        
    #     plt.figure()
    #     plt.imshow(calculated_gradient_2.cpu().detach().numpy(), cmap="RdBu")
    #     plt.colorbar()
    #     plt.title("Calculated Gradient")
    #     plt.savefig("./figs/grad_2.png")
    #     plt.close()

    #     plt.figure()
    #     plt.imshow(gradient_gt_2.cpu().detach().numpy(), cmap="RdBu")
    #     plt.colorbar()
    #     plt.title("GT Gradient")
    #     plt.savefig("./figs/grad_gt_2.png")
    #     plt.close()

    #     plt.figure()
    #     plt.imshow((gradient_gt_1 + gradient_gt_2).cpu().detach().numpy(), cmap="RdBu")
    #     plt.colorbar()
    #     plt.title("Calculated Gradient")
    #     plt.savefig("./figs/grad_sum.png")
    #     plt.close()

    #     plt.figure()
    #     plt.imshow(total_gradient.cpu().detach().numpy(), cmap="RdBu")
    #     plt.colorbar()
    #     plt.title("Calculated Gradient")
    #     plt.savefig("./figs/grad_total.png")
    #     plt.close()

if __name__ == "__main__":
    filepath1 = "/home/pingchua/projects/MAPS/data/fdfd/mdm/raw_opt_traj_ptb/mdm_id-59_opt_step_9_perturbed_0-in_slice_1-1.55-2-300.h5"
    filepath2 = "/home/pingchua/projects/MAPS/data/fdfd/mdm/raw_small/mdm_id-0_opt_step_3-in_slice_1-1.55-2-300.h5"
    test_gradient(filepath1, filepath2)