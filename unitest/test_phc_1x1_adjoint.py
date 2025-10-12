"""
Date: 2024-06-14 02:38:39
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-14 02:56:48
FilePath: /Robust-Inverse-Design/unitest/test_phc_1x1_adjoint.py
"""

import torch

############ pytorch must be imported later
from core.models.layers.phc_1x1_fdtd import PhC_1x1

eps_sio2 = 1.44**2
eps_si = 3.48**2


def test_phc_adjoint():
    permittivity = torch.randn((201, 201))
    device = PhC_1x1(
        num_in_ports=1,
        num_out_ports=1,
        box_size=[10, 10],
        wg_width=(1.7320508076, 1.7320508076),
        port_len=3,
        taper_width=1.7320508076,
        taper_len=2,
        eps_r=eps_si,
        eps_bg=eps_sio2,
    )
    device.update_permittivity(permittivity)
    device.add_source(0)
    device.create_simulation(
        resolution=20,
        border_width=[0, 1],
        PML=(2, 2),
        record_interval=0.3,
        store_fields=["Ez"],
        until=250,
        stop_when_decay=False,
    )
    device.create_objective(0, 0)
    device.create_optimzation()
    f0, grad = device.obtain_objective_and_gradient()
    print(f0, grad)


if __name__ == "__main__":
    test_phc_adjoint()
