from ceviche.solvers import _solve_direct
from ceviche.constants import *
from ceviche.utils import make_sparse
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from core.models.layers.utils import (
    Si_eps,
    SiO2_eps,
)
import scipy.sparse.linalg as spl
import time
from torch_sparse import spmm
from core.utils import print_stat
import matplotlib.pyplot as plt
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import cupyx.scipy.sparse.linalg as cpx_linalg
import scipy.sparse as sp

DEFAULT_ITERATIVE_METHOD = 'bicg'

ITERATIVE_METHODS = {
    'bicg': spl.bicg,
    'bicgstab': spl.bicgstab,
    'cg': spl.cg,
    'cgs': spl.cgs,
    'gmres': spl.gmres,
    'lgmres': spl.lgmres,
    'qmr': spl.qmr,
    'gcrotmk': spl.gcrotmk
}

ATOL = 1e-8


def _solve_iterative(A, b, x0=None, iterative_method=DEFAULT_ITERATIVE_METHOD, rtol=1e-5, atol=1e-6):
    # """ Iterative solver """

    # # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    try:
        solver_fn = ITERATIVE_METHODS[iterative_method]
    except:
        raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(iterative_method, ITERATIVE_METHODS))

    # call the solver using scipy's API
    x, info = solver_fn(A, b, x0=x0, rtol=rtol, atol=atol)
    return x

def _solve_iterative_torch(entries_a, indices_a, b, x0=None, iterative_method=DEFAULT_ITERATIVE_METHOD, rtol=1e-5, atol=1e-6):
    # """ Iterative solver """

    # # error checking on the method name (https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
    # try:
    #     solver_fn = ITERATIVE_METHODS[iterative_method]
    # except:
    #     raise ValueError("iterative method {} not found.\n supported methods are:\n {}".format(iterative_method, ITERATIVE_METHODS))

    # # call the solver using scipy's API
    # x, info = solver_fn(A, b, x0=x0, rtol=rtol, atol=atol)
    def obj_fn(indices, entries, x, b):
        residual = b - spmm(
            indices,
            entries,
            m=b.shape[0],
            n=b.shape[0],
            matrix=x[:, None],
        )[:, 0]
        # print(f"this is the l2 norm of the residual: {torch.norm(residual, p=2)}", flush=True)
        # print(f"this is the l2 norm of the b: {torch.norm(b, p=2)}", flush=True)
        return torch.norm(residual, p=2) / torch.norm(b, p=2)
    device = torch.device("cuda:0")
    b = b.to(torch.complex128).to(device)
    if x0 is not None:
        x0 = torch.from_numpy(x0).to(device)
    else:
        x0 = torch.zeros_like(b).to(device)
    if iterative_method == "adam":
        x0 = torch.nn.Parameter(x0)
        optimizer = torch.optim.Adam([x0], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
        residual = obj_fn(indices_a, entries_a, x0, b)
        counter = 0
        print("the init residual is: ", residual, flush=True)
        while residual > rtol:
            optimizer.zero_grad()
            residual = obj_fn(indices_a, entries_a, x0, b)
            residual.backward()
            optimizer.step()
            scheduler.step() 
            if counter % 10 == 0:
                print(f"@ counter {counter}, rtol is {residual}")
            counter += 1
    return x0

def test_Axb_solver(As, b):
    # b = b/torch.max(torch.abs(b))
    wl = 1.55
    entries_a_torch = As[f'A-wl-{wl}-entries_a']
    indices_a_torch = As[f'A-wl-{wl}-indices_a']
    entries_a_np = entries_a_torch.cpu().numpy()
    indices_a_np = indices_a_torch.cpu().numpy()
    b_torch = b
    b_np = b.cpu().numpy()

    regular_A_np = make_sparse(entries_a_np, indices_a_np, (b_np.shape[0], b_np.shape[0]))
    start_time = time.time()
    e_direct_np = _solve_direct(regular_A_np, b_np) # e_direct is the solution from the direct solver
    end_time = time.time()
    plt.figure()
    plt.imshow(np.abs(e_direct_np.reshape((600, 600))))
    plt.colorbar()
    plt.savefig("./figs/e_direct.png")
    plt.close()
    print("this is the time for the direct solver: ", end_time - start_time, flush=True)
    e_direct_torch = torch.from_numpy(e_direct_np).to(b_torch.device).flatten()
    A_by_e = spmm(
        indices_a_torch,
        entries_a_torch,
        m=e_direct_torch.shape[0],
        n=e_direct_torch.shape[0],
        matrix=e_direct_torch[:, None],
    )[:, 0]
    # plot A_by_e
    # A_by_e_np = A_by_e.cpu().numpy()
    # reshape_A_by_e_np = np.reshape(np.abs(A_by_e_np), (600, 600))
    # reshape_b_np = np.reshape(np.abs(b_np), (600, 600))
    # print("this is the shape of the A_by_e_np: ", reshape_A_by_e_np.shape, flush=True)
    # print("this is the shape of the b_np: ", reshape_b_np.shape, flush=True)
    # plt.figure()
    # plt.imshow(reshape_A_by_e_np)
    # plt.colorbar()
    # plt.savefig("./figs/A_by_e.png")
    # plt.close()

    # plt.figure()
    # plt.imshow(reshape_b_np)
    # plt.colorbar()
    # plt.savefig("./figs/b.png")
    # plt.close()
    # quit()
    residual = b - A_by_e
    residual_norm_torch = torch.norm(residual, p=2).double()
    b_norm_torch = torch.norm(b, p=2).double()
    ratio_torch = (residual_norm_torch / b_norm_torch)
    print("this is the residual norm from torch: ", residual_norm_torch, flush=True)
    print("this is the b norm from torch: ", b_norm_torch, flush=True)
    print("this is the ratio of the residual to the b from torch: ", ratio_torch, flush=True)
    
    residual_abs = torch.abs(residual)
    b_abs = torch.abs(b) + 100

    ratio_abs = (residual_abs / b_abs)
    print_stat(ratio_abs)

    # print("this is the max ratio of the residual to the b from torch: ", ratio_abs, flush=True)
    # plt.figure()
    # plt.imshow(ratio_abs.cpu().numpy().reshape((600, 600)))
    # plt.colorbar()
    # plt.savefig("./figs/err_ratio_abs.png")
    # plt.close()

    # # Calculate the L2 norm (Euclidean norm) of the residual
    residual_np = b_np - regular_A_np @ e_direct_np
    residual_norm_np = np.linalg.norm(residual_np, ord=2)
    b_norm_np = np.linalg.norm(b_np, ord=2)
    ratio_np = residual_norm_np / b_norm_np
    print("this is the residual norm from numpy: ", residual_norm_np, flush=True)
    print("this is the b norm from numpy: ", b_norm_np, flush=True)
    print("this is the ratio of the residual to the b from numpy: ", ratio_np, flush=True)
    noisy_x0 = e_direct_np * (np.random.normal(1, 0.1, e_direct_np.shape))
    plt.figure()
    plt.imshow(np.abs(noisy_x0.reshape((600, 600))))
    plt.colorbar()
    plt.savefig("./figs/noisy_x0.png")
    plt.close()
    start_time = time.time()
    e_iterate_np = _solve_iterative(regular_A_np, b_np, x0=None, iterative_method="lgmres", rtol=1e-3, atol=1e-6)
    end_time = time.time()
    print(f"this is the time for the iterative solver {'lgmres'}: ", end_time - start_time, flush=True)

# norm(b - A @ x) <= max(rtol*norm(b), atol)
def test_Axb_solver_torch(As, b):
    # b = b/torch.max(torch.abs(b))
    wl = 1.55
    entries_a_torch = As[f'A-wl-{wl}-entries_a']
    indices_a_torch = As[f'A-wl-{wl}-indices_a']
    entries_a_np = entries_a_torch.cpu().numpy()
    indices_a_np = indices_a_torch.cpu().numpy()
    b_np = b.cpu().numpy()

    regular_A_np = make_sparse(entries_a_np, indices_a_np, (b_np.shape[0], b_np.shape[0]))
    start_time = time.time()
    e_direct_np = _solve_direct(regular_A_np, b_np) # e_direct is the solution from the direct solver
    noisy_x0 = e_direct_np * (np.random.normal(1, 0.1, e_direct_np.shape))
    device = torch.device("cuda:0")
    b = b.to(device)
    start_time = time.time()
    x = _solve_iterative_torch(entries_a_torch, indices_a_torch, b, x0=noisy_x0, iterative_method="gmres", rtol=1e-2, atol=1e-6)
    end_time = time.time()
    print(f"this is the time for the iterative torch solver {'gmres'}: ", end_time - start_time, flush=True)





if __name__ == "__main__":
    with h5py.File("data/fdfd/metacoupler/mfs_raw/metacoupler_id-6_opt_step_9.h5", "r") as f:
        keys = list(f.keys())
        orgion_size = torch.from_numpy(f["eps_map"][()]).float().size()
        # eps_map = resize_to_targt_size(torch.from_numpy(f["eps_map"][()]).float(), (200, 300))
        # gradient = resize_to_targt_size(torch.from_numpy(f["gradient"][()]).float(), (200, 300))
        eps_map = torch.from_numpy(f["eps_map"][()]).float().sqrt() # sqrt the eps_map to get the refractive index
        gradient = torch.from_numpy(f["gradient"][()]).float()
        field_solutions = {}
        s_params = {}
        adj_srcs = {}
        src_profiles = {}
        incident_field = {}
        fields_adj = {}
        field_normalizer = {}
        design_region_mask = {}
        ht_m = {}
        et_m = {}
        monitor_slice = {}
        As = {}
        for key in keys:
            if key.startswith("field_solutions"):
                field = torch.from_numpy(f[key][()])
                field_solutions[key] = field
                # field = torch.view_as_real(field).permute(0, 3, 1, 2)
                # field = resize_to_targt_size(field, (200, 300)).permute(0, 2, 3, 1)
                # field_solutions[key] = torch.view_as_complex(field.contiguous())
            elif key.startswith("s_params"):
                s_params[key] = torch.from_numpy(f[key][()]).float()
            elif key.startswith("adj_src"):
                adjoint_src = torch.from_numpy(f[key][()])
                adj_srcs[key] = adjoint_src
                # adjoint_src = torch.view_as_real(adjoint_src).permute(2, 0, 1)
                # adjoint_src = resize_to_targt_size(adjoint_src, (200, 300)).permute(1, 2, 0)
                # adj_srcs[key] = torch.view_as_complex(adjoint_src.contiguous())
            elif key.startswith("source_profile"):
                source_profile = torch.from_numpy(f[key][()])
                src_profiles[key] = source_profile
                if key == "source_profile-wl-1.55-port-in_port_1-mode-1":
                    mode = source_profile[int(0.4 * source_profile.shape[0] / 2)]
                    mode = mode.unsqueeze(0).repeat(source_profile.shape[0], 1)
                    source_index = int(0.4 * source_profile.shape[0] / 2)
                    resolution = 2e-8
                    epsilon = Si_eps(1.55)
                    lambda_0 = 1.55e-6
                    k = (2 * torch.pi / lambda_0) * torch.sqrt(torch.tensor(epsilon))
                    x_coords = torch.arange(600).float()
                    distances = torch.abs(x_coords - source_index) * resolution
                    phase_shifts = (k * distances).unsqueeze(1)
                    mode = mode * torch.exp(1j * phase_shifts)
                    # mode = torch.view_as_real(mode).permute(2, 0, 1)
                    # mode = resize_to_targt_size(mode, (200, 300)).permute(1, 2, 0)
                    incident_key = key.replace("source_profile", "incident_field")
                    incident_field[incident_key] = mode
                    # incident_field[incident_key] = torch.view_as_complex(mode.contiguous())
                # source_profile = torch.view_as_real(source_profile).permute(2, 0, 1)
                # source_profile = resize_to_targt_size(source_profile, (200, 300)).permute(1, 2, 0)
                # src_profile[key] = torch.view_as_complex(source_profile.contiguous())
            elif key.startswith("fields_adj"):
                field = torch.from_numpy(f[key][()])
                fields_adj[key] = field
                # field = torch.view_as_real(field).permute(0, 3, 1, 2)
                # field = resize_to_targt_size(field, (200, 300)).permute(0, 2, 3, 1)
                # fields_adj[key] = torch.view_as_complex(field.contiguous())
            elif key.startswith("field_adj_normalizer"):
                field_normalizer[key] = torch.from_numpy(f[key][()]).float()
            elif key.startswith("design_region_mask"):
                design_region_mask[key] = int(f[key][()])
            elif key.startswith("ht_m"):
                ht_m[key] = torch.from_numpy(f[key][()])
                ht_m[key+"-origin_size"] = torch.tensor(ht_m[key].shape)
                ht_m[key] = torch.view_as_real(ht_m[key]).permute(1, 0).unsqueeze(0)
                ht_m[key] = F.interpolate(ht_m[key], size=5000, mode='linear', align_corners=True)
                ht_m[key] = ht_m[key].squeeze(0).permute(1, 0).contiguous()
                ht_m[key] = torch.view_as_complex(ht_m[key])
                # print("this is the dtype of the ht_m: ", ht_m[key].dtype, flush=True)
            elif key.startswith("et_m"):
                et_m[key] = torch.from_numpy(f[key][()])
                et_m[key+"-origin_size"] = torch.tensor(et_m[key].shape)
                et_m[key] = torch.view_as_real(et_m[key]).permute(1, 0).unsqueeze(0)
                et_m[key] = F.interpolate(et_m[key], size=5000, mode='linear', align_corners=True)
                et_m[key] = et_m[key].squeeze(0).permute(1, 0).contiguous()
                et_m[key] = torch.view_as_complex(et_m[key])
                # print("this is the dtype of the et_m: ", et_m[key].dtype, flush=True)
            elif key.startswith("A-"):
                As[key] = torch.from_numpy(f[key][()])
            elif key.startswith("port_slice"):
                data = f[key][()]
                if key.endswith("_y"):
                    monitor_slice[key] = torch.tensor([data[0], data[-1] + 1])
                elif key.endswith("_x"):
                    monitor_slice[key] = torch.tensor([data])
                else:
                    monitor_slice[key] = torch.tensor(data)

    device = torch.device("cuda:0")
    for key, A in As.items():
        As[key] = A.to(device, non_blocking=True)
    for key, src_profile in src_profiles.items():
        src_profiles[key] = src_profile.to(device, non_blocking=True)
    source = src_profiles["source_profile-wl-1.55-port-in_port_1-mode-1"]
    omega = 2 * torch.pi * C_0 / (1.55 * 1e-6)
    b = (source * 1j * omega).flatten()
    test_Axb_solver(As, b)