'''
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-02-21 19:32:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-12-15 17:28:48
'''
from core.builder import make_dataloader
from data.mmi.device_shape import MMI_NxM, mmi_3x3_L
from core.models.pde_base import PDE_NN_BASE
import torch
import matplotlib.pyplot as plt
from core.models import FFNO2d
from angler.simulation import Simulation
from pyutils.torch_train import load_model
def plot_enc():
    N = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=(25.9, 6.1),
        wg_width=(1.1, 1.1),
        port_diff=(6.1 / N, 6.1/N),
        port_len=3,
        border_width=0.25,
        grid_step=0.05,
        NPML=[30, 30],
    )
    w, h = (18.5, 1.2)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    eps = mmi.set_pad_eps([11.85, 11.91, 12.03])
    model = PDE_NN_BASE()
    eps = torch.from_numpy(mmi.trim_pml(eps)).t().to(torch.cfloat)[None, None,...]
    wavelength = torch.tensor([[1.55]])
    grid_step=torch.tensor([[0.05, 0.05]])
    grid = model.get_grid(eps.shape, eps.device, mode="exp", epsilon=eps, wavelength=wavelength, grid_step=grid_step)

    grid_x = torch.view_as_complex(grid[:,:2].permute(0,2,3,1).contiguous())
    grid_y = torch.view_as_complex(grid[:,2:].permute(0,2,3,1).contiguous())
    grid_z = grid_x.add(grid_y)
    # grid_t1 = torch.view_as_complex(torch.view_as_real(grid_x).mul(torch.view_as_real(grid_y)))
    grid_t1 = grid_x.mul(grid_y)
    grid_t2 = grid_x.div(grid_y)
    fig, axes  = plt.subplots(5,1)
    axes[0].imshow(grid[0,0], cmap="gray")
    axes[1].imshow(grid[0,2], cmap="gray")
    axes[2].imshow(grid_z[0].real, cmap="gray")
    axes[3].imshow(grid_t1[0].real, cmap="gray")
    axes[4].imshow(grid_t2[0].real, cmap="gray")
    fig.savefig(f"mmi3x3_posenc.png", dpi=150)


def plot_device():
    N = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=(25.9, 6.1),
        wg_width=(1.1, 1.1),
        port_diff=(6.1 / N, 6.1/N),
        port_len=3,
        taper_len=3,
        taper_width=1.4,
        border_width=0.25,
        grid_step=0.05,
        NPML=[30, 30],
    )
    simulation = Simulation(omega=1,eps_r=mmi.epsilon_map, dl=1, NPML=[30,30], pol="Hz")
    simulation.solve_fields()
    simulation.plt_re()
    plt.savefig(f"mmi3x3_taper.png", dpi=200)


if __name__ == "__main__":
    plot_device()
