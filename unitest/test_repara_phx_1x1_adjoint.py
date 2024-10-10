'''
Date: 2024-06-14 03:02:52
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-06-14 03:15:08
FilePath: /Robust-Inverse-Design/unitest/test_repara_phx_1x1_adjoint.py
'''
############ pytorch must be imported later
from core.models import Repara_PhC_1x1
import torch


def test_adjoint_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    init_device_cfg = dict(
        num_in_ports=1,
        num_out_ports=1,
        box_size=[19.8, 12],
        wg_width=(1.7320508076, 1.7320508076),
        port_len=3,
        taper_width=1.7320508076,
        taper_len=2,
        eps_r=12.1104,
        eps_bg=2.0736,
    )
    sim_cfg = dict(
        resolution=20,
        border_width=[0, 1],
        PML=(2, 2),
        record_interval=0.3,
        store_fields=["Ez"],
        until=2,
        stop_when_decay=False,
    )
    model = Repara_PhC_1x1(
        device_cfg=init_device_cfg,
        sim_cfg=sim_cfg,
        perturbation=False,
        num_rows_perside=6,
        num_cols=8,
        adjoint_mode="fdtd",
        df=0,
        nf=1,
    ).to(device)
    print(model)
    pos_position, permittivity, fom = model(0.05, 0.01)

    loss = pos_position.sum() + permittivity.sum()
    loss = loss + fom

    loss.backward()
    print(fom)
    print(model.hole_position.grad)


if __name__ == "__main__":
    test_adjoint_grad()
