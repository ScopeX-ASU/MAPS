import argparse
import copy
import cv2
from typing import List

import torch
import torch.cuda.amp as amp
import torch.fft
import matplotlib.pyplot as plt
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, Optimizer, Scheduler

from core.utils import (
    rip_padding,
    padding_to_tiles,
    print_stat,
)

from core import builder
from core.NVILT_Share.photonic_model import * 
from core.models.layers import *

eps_sio2 = 1.44**2
eps_si = 3.48**2
air = 1.0**2

def test_phc(
        model,
        mode,
        criterion: Criterion,
        aux_criterions: Criterion,
        lossv: List,
        device: torch.device = torch.device("cuda:0"),
        plot: bool = False,
        grad_scaler=None,
) -> None:
    '''
    model: the photonic crystal device
    mode: which kind of non idealities to test
        temp permittiivity drift
        wavelength drift
        row-wise hole drift
        lithography error
        combined all of above error
    criterion: the loss function
    aux_criterions: the auxiliary loss functions
    lossv: the loss list
    device: the device to run the model
    plot: whether to plot the results
    grad_scaler: the gradient scaler
    '''
    with torch.no_grad():
        model.set_num_superlattice(2)
        output = model(sharpness=configs.sharp_scheduler.final_sharp, T_lse=configs.model.T_lse, device_resolution=configs.model.sim_cfg.resolution, eval_resolution=configs.model.sim_cfg.resolution)

        ax = model.device.opt.simulation.plt_abs(outline=True, cbar=True)
        fig = ax.figure
        fig.savefig('./test_corner_case.png')
        
        if isinstance(output, tuple):
            hole_position = output[0]["hole_position"]
            permittivity = output[0]["permittivity_list"]
            fom = output[0]["eff"]
            vg = output[0]["vg"]
            aux_out = output[1]
        else:
            hole_position = output["hole_position"]
            permittivity = output["permittivity_list"]
            fom = output["eff"]
            vg = output["vg"]
            aux_out = None
        permittivity_list = permittivity
        new_permittivity_list = []
        for i in range(len(permittivity_list)):
            print("this is the shape of permittivity_list[i]: ", permittivity_list[i].shape)
            if i != len(permittivity_list) - 1:
                new_permittivity_list.append(permittivity_list[i][:-1])
            else:
                new_permittivity_list.append(permittivity_list[i])
        original_x, original_y = model.obtain_eps(torch.cat(new_permittivity_list, dim=0), configs.model.sim_cfg.resolution).shape
        print("this is the efficieny of the loaded device: ", fom, flush=True)
        print("this is the original shape of the permittivity: ", original_x, original_y, flush=True)
        # eps = model.obtain_eps() # TODO implement the obtain_eps function (DONE for angler)
        # # normalize the eps to 0-1
        # eps = (eps - eps.min()) / (eps.max() - eps.min())
        # print("this is the state of eps obtained from fdfd", flush=True)
        # print_stat(eps)
        # print("this is the shape of eps: ", eps.shape, flush=True)
        # coupling_region_top = model.obtain_coupling_region_top() # TODO implement the obtain_coupling_region_top function
        # coupling_region_out_top = model.obtain_coupling_region_out_top() # TODO implement the obtain_coupling_region_out_top function
        # displacement = model.obtain_displacement() # TODO implement the obtain_displacement function

        if mode == "lithography_error":
            permittivity_list = model.build_device(T_lse=configs.model.T_lse, sharpness=configs.sharp_scheduler.final_sharp, resolution=310)
            new_permittivity_list = []
            for i in range(len(permittivity_list)):
                print("this is the shape of permittivity_list[i]: ", permittivity_list[i].shape)
                if i != len(permittivity_list) - 1:
                    new_permittivity_list.append(permittivity_list[i][:-1])
                else:
                    new_permittivity_list.append(permittivity_list[i])
            permittivity = torch.cat(new_permittivity_list, dim=0)

            eps = model.obtain_eps(permittivity=permittivity, resolution=310) # TODO implement the obtain_eps function (DONE for angler)
            eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(eps, 620)
            eps = 1 + (3.48**2 - 1) * eps

            print("this is the shape of eps after padding: ", eps.shape)
            print("this is the pady_0: ", pady_0)
            print("this is the pady_1: ", pady_1)
            print("this is the padx_0: ", padx_0)
            print("this is the padx_1: ", padx_1)

            eps_np = eps.numpy()
            # Scale the image to 0-255
            eps_uint8 = (eps_np * 255).astype(np.uint8)
            print("this is the state of eps_uint8 to be saved", flush=True)
            print_stat(eps_uint8)
            # Save using OpenCV
            cv2.imwrite("./eps.png", eps_uint8)

            solver = nvilt_engine_2(image_path='./eps.png', morph = 0, scale_factor=1)
            mask, x_out, x_out_max, x_out_min = solver.nvilt.forward_batch_test(use_morph=False)
            final_image = torch.cat((solver.target_s, mask, x_out), dim=3).cpu().detach().numpy()[0,0,:,:]*255
            print_stat(x_out)
            print_stat(mask)
            print("this is the shape of final_image: ", final_image.shape)
            print("this is the shape of x_out_max: ", x_out_max.shape)
            # plot the x_out using matplotlib
            plt.imsave("./x_out.png", x_out.cpu(), cmap='gray')
            plt.imsave("./mask.png", mask.cpu(), cmap='gray')

            x_out = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)
            x_out_max = rip_padding(x_out_max.squeeze(), pady_0, pady_1, padx_0, padx_1)
            x_out_min = rip_padding(x_out_min.squeeze(), pady_0, pady_1, padx_0, padx_1)

            x_out_norm = torch.nn.functional.interpolate(x_out.unsqueeze(0).unsqueeze(0), size=(original_x, original_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            x_out_max = torch.nn.functional.interpolate(x_out_max.unsqueeze(0).unsqueeze(0), size=(original_x, original_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            x_out_min = torch.nn.functional.interpolate(x_out_min.unsqueeze(0).unsqueeze(0), size=(original_x, original_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)

            test_device_norm = PhC_1x1_fdfd_angler_eff_vg(
                num_in_ports=1,
                num_out_ports=1,
                num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                superlattice_cfg=model.superlattice_cfg,
                coupling_region_cfg=model.coupling_region_cfg,
                boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                port_width=model.port_width,  # in/out wavelength width, um
                port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                taper_len=model.taper_len,  # taper length. um. Default to 0
                eps_r=model.eps_r,  # relative refractive index
                eps_bg=model.eps_bg,  # background refractive index
                a=model.a,  # lattice constant
                r=model.r,  # radius of the holes

                border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            test_device_norm.create_objective(1.55e-6, eval(model.eps_r), x_out_norm, True)
            test_device_norm.create_optimzation()
            result = test_device_norm.obtain_objective()
            print("this is the efficieny of the norm device: ", result)
            ax = test_device_norm.opt.simulation.plt_abs(outline=True, cbar=True)
            fig = ax.figure
            fig.savefig('./test_corner_norm.png')
            test_device_max = PhC_1x1_fdfd_angler_eff_vg(
                num_in_ports=1,
                num_out_ports=1,
                num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                superlattice_cfg=model.superlattice_cfg,
                coupling_region_cfg=model.coupling_region_cfg,
                boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                port_width=model.port_width,  # in/out wavelength width, um
                port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                taper_len=model.taper_len,  # taper length. um. Default to 0
                eps_r=model.eps_r,  # relative refractive index
                eps_bg=model.eps_bg,  # background refractive index
                a=model.a,  # lattice constant
                r=model.r,  # radius of the holes

                border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            test_device_max.create_objective(1.55e-6, eval(model.eps_r), x_out_max, True)
            test_device_max.create_optimzation()
            result = test_device_max.obtain_objective()
            print("this is the efficieny of the max device: ", result)
            ax = test_device_max.opt.simulation.plt_abs(outline=True, cbar=True)
            fig = ax.figure
            fig.savefig('./test_corner_max.png')

            test_device_min = PhC_1x1_fdfd_angler_eff_vg(
                num_in_ports=1,
                num_out_ports=1,
                num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                superlattice_cfg=model.superlattice_cfg,
                coupling_region_cfg=model.coupling_region_cfg,
                boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                port_width=model.port_width,  # in/out wavelength width, um
                port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                taper_len=model.taper_len,  # taper length. um. Default to 0
                eps_r=model.eps_r,  # relative refractive index
                eps_bg=model.eps_bg,  # background refractive index
                a=model.a,  # lattice constant
                r=model.r,  # radius of the holes

                border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            test_device_min.create_objective(1.55e-6, eval(model.eps_r), x_out_min, True)
            test_device_min.create_optimzation()
            result = test_device_min.obtain_objective()
            print("this is the efficieny of the min device: ", result)
            ax = test_device_min.opt.simulation.plt_abs(outline=True, cbar=True)
            fig = ax.figure
            fig.savefig('./test_corner_min.png')


            # print("this is the shape of mask_out: ", mask_out.shape, flush=True)
            # print("this is the shape of mask_out.squeeze(): ", mask_out.squeeze().shape, flush=True)
            # eps_out = rip_padding(mask_out.squeeze(), pady_0, pady_1, padx_0, padx_1)
            # eps_max = rip_padding(mask_max.squeeze(), pady_0, pady_1, padx_0, padx_1)
            # eps_min = rip_padding(mask_min.squeeze(), pady_0, pady_1, padx_0, padx_1)
            # print("this is the shape of eps_out: ", eps_out.shape, flush=True)
            
            # # save the eps_out to a png file
            # plt.imsave("./eps_out.png", eps_out.cpu(), cmap='gray')
            # plt.imsave("./eps_max.png", eps_max.cpu(), cmap='gray')
            # plt.imsave("./eps_min.png", eps_min.cpu(), cmap='gray')

        elif mode == "etching_error":
            eta = model.build_eta('center2edge_descend', 0.6, 0.4, permittivity)
            # plot the eta
            plt.clf()
            eta_plot = torch.transpose(eta, 0, 1).cpu().detach().numpy()
            plt.imshow(eta_plot, cmap='gray')
            plt.colorbar()
            plt.savefig('./eta_center2edge_descend.png')

            projection = HeavisideProjection(0.05)
            permittivity = projection(permittivity, 20, eta)
            test_device = PhC_1x1_fdfd_angler_eff_vg(
                num_in_ports=1,
                num_out_ports=1,
                num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                superlattice_cfg=model.superlattice_cfg,
                coupling_region_cfg=model.coupling_region_cfg,
                boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                port_width=model.port_width,  # in/out wavelength width, um
                port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                taper_len=model.taper_len,  # taper length. um. Default to 0
                eps_r=eval(model.eps_r),  # relative refractive index
                eps_bg=eval(model.eps_bg),  # background refractive index
                a=model.a,  # lattice constant
                r=model.r,  # radius of the holes

                border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            test_device.create_objective(1.55e-6, 3.48, permittivity)
            test_device.create_optimzation()
            result = test_device.obtain_objective()
            print("this is the efficieny of the normal device: ", result)
            ax = model.device.opt.simulation.plt_abs(outline=True, cbar=True)
            fig = ax.figure
            fig.savefig('./center2edge_descend.png')


            # fom_list = []
            # eta = np.linspace(0.4, 0.6, 20)
            # for i in range(len(eta)):
            #     model.set_eta(eta[i])
            #     output = model(configs.sharp_scheduler.lr_max, configs.model.T_lse)
            #     if isinstance(output, tuple):
            #         hole_position = output[0]["hole_position"]
            #         permittivity = output[0]["permittivity_list"]
            #         fom = output[0]["eff"]
            #         vg = output[0]["vg"]
            #         aux_out = output[1]
            #     else:
            #         hole_position = output["hole_position"]
            #         permittivity = output["permittivity_list"]
            #         fom = output["eff"]
            #         vg = output["vg"]
            #         aux_out = None
            #     print("this is the efficieny of the normal device: ", fom)
            #     fom_list.append(fom)

            # plt.clf()
            # plt.plot(eta, fom_list, marker='o', linestyle='-', color='b')

            # # Add labels and title
            # plt.xlabel('eta')
            # plt.ylabel('Efficiency')
            # plt.title('eta vs Efficiency Curve')
            # plt.legend()

            # # Save the plot as a .png file
            # plt.savefig('eta_eff_curve.png')

        elif mode == "lithography_etching_error":
            fom_tensor = torch.zeros(3, 5)
            eta = np.linspace(0.4, 0.6, 5)
            scale_factor = (1 / model.resolution) / (2/620)
            target_x = round(eps.shape[0] * scale_factor)
            target_y = round(eps.shape[1] * scale_factor)
            original_x = eps.shape[0]
            original_y = eps.shape[1]

            scaled_eps = torch.nn.functional.interpolate(eps.unsqueeze(0).unsqueeze(0), size=(target_x, target_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)

            eps, pady_0, pady_1, padx_0, padx_1 = padding_to_tiles(scaled_eps, 620)

            eps_np = eps.numpy()
            # Scale the image to 0-255
            eps_uint8 = (eps_np * 255).astype(np.uint8)
            # Save using OpenCV
            cv2.imwrite("./eps.png", eps_uint8)

            solver = nvilt_engine_2(image_path='./eps.png', morph = 0, scale_factor=1)
            mask, x_out, x_out_max, x_out_min = solver.nvilt.forward_batch_test(use_morph=False)
            final_image = torch.cat((solver.target_s, mask, x_out), dim=3).cpu().detach().numpy()[0,0,:,:]*255

            x_out = rip_padding(x_out.squeeze(), pady_0, pady_1, padx_0, padx_1)
            x_out_max = rip_padding(x_out_max.squeeze(), pady_0, pady_1, padx_0, padx_1)
            x_out_min = rip_padding(x_out_min.squeeze(), pady_0, pady_1, padx_0, padx_1)

            x_out_norm = torch.nn.functional.interpolate(x_out.unsqueeze(0).unsqueeze(0), size=(original_x, original_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            x_out_max = torch.nn.functional.interpolate(x_out_max.unsqueeze(0).unsqueeze(0), size=(original_x, original_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            x_out_min = torch.nn.functional.interpolate(x_out_min.unsqueeze(0).unsqueeze(0), size=(original_x, original_y), mode='bicubic', align_corners=False).squeeze(0).squeeze(0)
            for i in range(5):
                eta_i = model.build_eta('constant', 0.6, 0.4, x_out_norm, eta[i])
                projection = HeavisideProjection(0.05)
                x_out_norm_i = projection(x_out_norm, 20, eta_i)
                test_device_norm = PhC_1x1_fdfd_angler_eff_vg(
                    num_in_ports=1,
                    num_out_ports=1,
                    num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                    superlattice_cfg=model.superlattice_cfg,
                    coupling_region_cfg=model.coupling_region_cfg,
                    boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                    port_width=model.port_width,  # in/out wavelength width, um
                    port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                    taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                    taper_len=model.taper_len,  # taper length. um. Default to 0
                    eps_r=model.eps_r,  # relative refractive index
                    eps_bg=model.eps_bg,  # background refractive index
                    a=model.a,  # lattice constant
                    r=model.r,  # radius of the holes

                    border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                    grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                test_device_norm.create_objective(1.55e-6, eval(model.eps_r), x_out_norm_i, True)
                test_device_norm.create_optimzation()
                result = test_device_norm.obtain_objective()
                fom_tensor[1, i] = result

            for i in range(5):
                eta_i = model.build_eta('constant', 0.6, 0.4, x_out_max, eta[i])
                projection = HeavisideProjection(0.05)
                x_out_max_i = projection(x_out_max, 20, eta_i)
                test_device_max = PhC_1x1_fdfd_angler_eff_vg(
                    num_in_ports=1,
                    num_out_ports=1,
                    num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                    superlattice_cfg=model.superlattice_cfg,
                    coupling_region_cfg=model.coupling_region_cfg,
                    boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                    port_width=model.port_width,  # in/out wavelength width, um
                    port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                    taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                    taper_len=model.taper_len,  # taper length. um. Default to 0
                    eps_r=model.eps_r,  # relative refractive index
                    eps_bg=model.eps_bg,  # background refractive index
                    a=model.a,  # lattice constant
                    r=model.r,  # radius of the holes

                    border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                    grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                test_device_max.create_objective(1.55e-6, eval(model.eps_r), x_out_max_i, True)
                test_device_max.create_optimzation()
                result = test_device_max.obtain_objective()
                fom_tensor[0, i] = result

            for i in range(5):
                eta_i = model.build_eta('constant', 0.6, 0.4, x_out_min, eta[i])
                projection = HeavisideProjection(0.05)
                x_out_min_i = projection(x_out_min, 20, eta_i)
                test_device_min = PhC_1x1_fdfd_angler_eff_vg(
                    num_in_ports=1,
                    num_out_ports=1,
                    num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                    superlattice_cfg=model.superlattice_cfg,
                    coupling_region_cfg=model.coupling_region_cfg,
                    boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                    port_width=model.port_width,  # in/out wavelength width, um
                    port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                    taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                    taper_len=model.taper_len,  # taper length. um. Default to 0
                    eps_r=model.eps_r,  # relative refractive index
                    eps_bg=model.eps_bg,  # background refractive index
                    a=model.a,  # lattice constant
                    r=model.r,  # radius of the holes

                    border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                    grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                test_device_min.create_objective(1.55e-6, eval(model.eps_r), x_out_min_i, True)
                test_device_min.create_optimzation()
                result = test_device_min.obtain_objective()
                fom_tensor[2, i] = result
            plt.clf()
            fig = plt.figure()
            # Plot the 2D tensor as a heatmap
            plt.imshow(fom_tensor.cpu().detach().numpy(), cmap='viridis')
            plt.colorbar()
            plt.xlabel('eta')
            plt.ylabel('Litho cornors')
            plt.title('combined litho corner and eta vs Efficiency')
            plt.savefig('./litho_eta_eff.png')

        elif mode == "temp_wl_error":
            wl = np.linspace(1.55e-6-2.5e-9, 1.55e-6+2.5e-9, 5)
            temp = np.linspace(300-50, 300+50, 5)
            fom = torch.zeros(len(temp), len(wl))
            for i in range(len(temp)):
                for j in range(len(wl)):
                    t = temp[i]
                    refractive_index = 3.48 + 1.86e-4 * (t - 300)
                    eps_r = (refractive_index**2)
                    test_device = PhC_1x1_fdfd_angler_eff_vg(
                        num_in_ports=1,
                        num_out_ports=1,
                        num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                        superlattice_cfg=model.superlattice_cfg,
                        coupling_region_cfg=model.coupling_region_cfg,
                        boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                        port_width=model.port_width,  # in/out wavelength width, um
                        port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                        taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                        taper_len=model.taper_len,  # taper length. um. Default to 0
                        eps_r=eps_r,  # relative refractive index
                        eps_bg=eval(model.eps_bg),  # background refractive index
                        a=model.a,  # lattice constant
                        r=model.r,  # radius of the holes

                        border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                        grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                        NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                    )
                    test_device.create_objective(wl[j], np.sqrt(eps_r), permittivity)
                    test_device.create_optimzation()
                    result = test_device.obtain_objective()
                    print("this is the efficieny of the normal device: ", result)
                    fom[i, j] = result
                    # ax = test_device.opt.simulation.plt_abs(outline=True, cbar=True)
                    # fig = ax.figure
                    # fig.savefig('./test_corner_norm.png')
            plt.clf()
            fig = plt.figure()
            # Plot the 2D tensor as a heatmap
            plt.imshow(fom.cpu().detach().numpy(), cmap='viridis')
            plt.colorbar()
            plt.xlabel('Temperature (K)')
            plt.ylabel('WaveLength (m)')
            plt.title('combined temp and wl vs Efficiency')
            plt.savefig('./temp_wl_eff.png')

        elif mode == 'wavelength_error':
            print("this is the efficieny of the normal device in first forward: ", fom)
            fom_list = []
            wl = np.linspace(1.55e-6-2.5e-9, 1.55e-6+2.5e-9, 20)
            for i in range(len(wl)):
                test_device = copy.deepcopy(model.device)
                test_device.create_objective(wl[i], 3.48, permittivity)
                test_device.create_optimzation()
                result = test_device.obtain_objective()
                print("this is the efficieny of the normal device: ", result)
                fom_list.append(result)
                # ax = test_device.opt.simulation.plt_abs(outline=True, cbar=True)
                # fig = ax.figure
                # fig.savefig('./test_corner_norm.png')
            plt.clf()
            plt.plot(wl, fom_list, marker='o', linestyle='-', color='b')

            # Add labels and title
            plt.xlabel('Wavelength (m)')
            plt.ylabel('Efficiency')
            plt.title('Wavelength vs Efficiency Curve')
            plt.legend()

            # Save the plot as a .png file
            plt.savefig('wl_eff_curve.png')

        elif mode == 'temp_error':
            fom_list = []
            vg_list = []
            print("this is the efficieny of the normal device in first forward: ", fom)
            print("this is the vg of the normal device in first forward: ", vg)
            temp = np.linspace(300-50, 300+50, 20)
            for i in range(len(temp)):
                t = temp[i]
                refractive_index = 3.48 + 1.86e-4 * (t - 300)
                eps_r = (refractive_index**2)
                test_device = PhC_1x1_fdfd_angler_eff_vg(
                    num_in_ports=1,
                    num_out_ports=1,
                    num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                    superlattice_cfg=model.superlattice_cfg,
                    coupling_region_cfg=model.coupling_region_cfg,
                    boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                    port_width=model.port_width,  # in/out wavelength width, um
                    port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                    taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                    taper_len=model.taper_len,  # taper length. um. Default to 0
                    eps_r=eps_r,  # relative refractive index
                    eps_bg=eval(model.eps_bg),  # background refractive index
                    a=model.a,  # lattice constant
                    r=model.r,  # radius of the holes

                    border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                    grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                    NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
                )
                test_device.create_objective(1.55e-6, 3.48, permittivity)
                test_device.create_optimzation()
                result = test_device.obtain_objective()
                print("this is the efficieny of the norm device: ", result)
                fom_list.append(result)
                # ax = test_device.opt.simulation.plt_abs(outline=True, cbar=True)
                # fig = ax.figure
                # fig.savefig('./test_corner_norm.png')


                test_superlattice = PhC_1x1_Superlattice(
                    superlattice_cfg=model.superlattice_cfg,
                    r=model.r/model.a,
                    eps_r=eps_r,
                    eps_bg=eval(model.eps_bg),
                    eps_lower=eval(model.eps_bg),
                    eps_upper=eval(model.eps_bg),
                    cal_mode=model.cal_bd_mode,
                    kz_symmetry="odd",
                )
                hole_position = model.hole_position
                hole_position = hole_position/model.a
                test_superlattice.build_variables(hole_position)
                result = test_superlattice.obtain_objective_and_gradient("need_value")
                print("this is the vg error of the norm device: ", result)
                vg_list.append(result)
                # test_superlattice.plot_superlattice("./test_corner_norm" + "_superlattice.png")
            plt.clf()
            plt.plot(temp, fom_list, marker='o', linestyle='-', color='b', label='My Curve')

            # Add labels and title
            plt.xlabel('Temperature (K)')
            plt.ylabel('Efficiency')
            plt.title('Temp vs Efficiency Curve')
            plt.legend()

            # Save the plot as a .png file
            plt.savefig('temp_eff_curve.png')
            plt.clf()
            plt.plot(temp, vg_list, marker='o', linestyle='-', color='r', label='My Curve')

            # Add labels and title
            plt.xlabel('Temperature (K)')
            plt.ylabel('Ng loss')
            plt.title('Temp vs Ng loss Curve')
            plt.legend()

            # Save the plot as a .png file
            plt.savefig('temp_ng_curve.png')


        elif mode == "rowwise_hole_drift_error":
            displacement_top = model.displacement_top.clone()

            row_purtabation = 0.1*(model.a*3**0.5/2)*torch.ones(
                1,
                model.superlattice_cfg["Ny_opt"],
            )
            displacement_top_max = displacement_top.clone()
            displacement_top_min = displacement_top.clone()
            # print("this is the shape of displacement_top_max[:, :, 1]: ", displacement_top_max[:, :, 1].shape)
            # print("this is the shape of row_purtabation: ", row_purtabation.shape)
            displacement_top_max[:, :, 1] = displacement_top_max[:, :, 1] + row_purtabation
            displacement_top_min[:, :, 1] = displacement_top_min[:, :, 1] - row_purtabation

            displacement_bot_max = displacement_top_max.clone()
            displacement_bot_min = displacement_top_min.clone()
            displacement_bot_max[:, :, 1] = -displacement_bot_max[:, :, 1]
            displacement_bot_min[:, :, 1] = -displacement_bot_min[:, :, 1]
            displacement_max = torch.cat((displacement_top_max, displacement_bot_max), dim=1)
            displacement_min = torch.cat((displacement_top_min, displacement_bot_min), dim=1)

            up_hole_position = model.up_hole_position.clone()
            down_hole_position = up_hole_position.clone()
            down_hole_position[:, :, 1] = -down_hole_position[:, :, 1]
            hole_position = torch.cat((up_hole_position, down_hole_position), dim=1)

            hole_position_max = hole_position + displacement_max
            hole_position_min = hole_position + displacement_min

            X, Y = model.get_meshgrid()  # X [M, N], Y [M, N]

            superlattice_permittivity_max = model.batch_gaussian(
                X, Y, hole_position_max, A=1
            )  # [holes_x, holes_y, M, N]
            superlattice_permittivity_max = configs.model.T_lse * torch.logsumexp(superlattice_permittivity_max / configs.model.T_lse, dim=(0, 1))
            superlattice_permittivity_max = model.binary_projection(superlattice_permittivity_max, eval(configs.lr_scheduler.lr_min))
            permittivity_max_list = [permittivity_list[0]] + [superlattice_permittivity_max] * model.superlattice_cfg["num_superlattice"] + [permittivity_list[-1]]
            new_permittivity_list = []
            for i in range(len(permittivity_max_list)):
                if i != len(permittivity_max_list) - 1:
                    new_permittivity_list.append(permittivity_max_list[i][:-1])
                else:
                    new_permittivity_list.append(permittivity_max_list[i])
            permittivity_max = torch.cat(new_permittivity_list, dim=0)

            superlattice_permittivity_min = model.batch_gaussian(
                X, Y, hole_position_min, A=1
            )  # [holes_x, holes_y, M, N]
            superlattice_permittivity_min = configs.model.T_lse * torch.logsumexp(superlattice_permittivity_min / configs.model.T_lse, dim=(0, 1))
            superlattice_permittivity_min = model.binary_projection(superlattice_permittivity_min, eval(configs.lr_scheduler.lr_min))
            permittivity_min_list = [permittivity_list[0]] + [superlattice_permittivity_min] * model.superlattice_cfg["num_superlattice"] + [permittivity_list[-1]]
            new_permittivity_list = []
            for i in range(len(permittivity_min_list)):
                if i != len(permittivity_min_list) - 1:
                    new_permittivity_list.append(permittivity_min_list[i][:-1])
                else:
                    new_permittivity_list.append(permittivity_min_list[i])
            permittivity_min = torch.cat(new_permittivity_list, dim=0)

            test_device_max = PhC_1x1_fdfd_angler_eff_vg(
                num_in_ports=1,
                num_out_ports=1,
                num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                superlattice_cfg=model.superlattice_cfg,
                coupling_region_cfg=model.coupling_region_cfg,
                boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                port_width=model.port_width,  # in/out wavelength width, um
                port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                taper_len=model.taper_len,  # taper length. um. Default to 0
                eps_r=model.eps_r,  # relative refractive index
                eps_bg=model.eps_bg,  # background refractive index
                a=model.a,  # lattice constant
                r=model.r,  # radius of the holes

                border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            test_device_max.create_objective(1.55e-6, eval(model.eps_r), permittivity_max)
            test_device_max.create_optimzation()
            result = test_device_max.obtain_objective()
            print("this is the efficieny of the max device: ", result)
            ax = test_device_max.opt.simulation.plt_abs(outline=True, cbar=True)
            fig = ax.figure
            fig.savefig('./test_corner_max.png')

            test_superlattice_max = PhC_1x1_Superlattice(
                superlattice_cfg=model.superlattice_cfg,
                r=model.r/model.a,
                eps_r=eval(model.eps_r),
                eps_bg=eval(model.eps_bg),
                eps_lower=eval(model.eps_bg),
                eps_upper=eval(model.eps_bg),
                cal_mode=model.cal_bd_mode,
                kz_symmetry="odd",
            )
            hole_position_max = hole_position_max/model.a
            test_superlattice_max.build_variables(hole_position_max)
            result = test_superlattice_max.obtain_objective_and_gradient("need_value")
            print("this is the vg error of the max device: ", result)
            test_superlattice_max.plot_superlattice("./test_corner_max" + "_superlattice.png")

            test_device_min = PhC_1x1_fdfd_angler_eff_vg(
                num_in_ports=1,
                num_out_ports=1,
                num_superlattice=model.superlattice_cfg['num_superlattice'], # for now 3 is enough for a demo I think
                superlattice_cfg=model.superlattice_cfg,
                coupling_region_cfg=model.coupling_region_cfg,
                boxedge_to_top_row=model.superlattice_cfg['boxedge_to_top_row'],  # distance from the box edge to the top row
                port_width=model.port_width,  # in/out wavelength width, um
                port_len=model.port_len,  # length of in/out waveguide from PML to box. um
                taper_width=model.taper_width,  # taper width near the multi-mode region. um. Default to 0
                taper_len=model.taper_len,  # taper length. um. Default to 0
                eps_r=model.eps_r,  # relative refractive index
                eps_bg=model.eps_bg,  # background refractive index
                a=model.a,  # lattice constant
                r=model.r,  # radius of the holes

                border_width=model.sim_cfg["border_width"][1],  # space between box and PML. um
                grid_step=1/model.resolution,  # isotropic grid step um. is this grid step euqal to the resolution?
                NPML=model.coupling_region_cfg["NPML"],  # PML pixel width. pixel
            )
            test_device_min.create_objective(1.55e-6, eval(model.eps_r), permittivity_min)
            test_device_min.create_optimzation()
            result = test_device_min.obtain_objective()
            print("this is the efficieny of the min device: ", result)
            ax = test_device_min.opt.simulation.plt_abs(outline=True, cbar=True)
            fig = ax.figure
            fig.savefig('./test_corner_min.png')

            test_superlattice_min = PhC_1x1_Superlattice(
                superlattice_cfg=model.superlattice_cfg,
                r=model.r/model.a,
                eps_r=eval(model.eps_r),
                eps_bg=eval(model.eps_bg),
                eps_lower=eval(model.eps_bg),
                eps_upper=eval(model.eps_bg),
                cal_mode=model.cal_bd_mode,
                kz_symmetry="odd",
            )
            hole_position_min = hole_position_min/model.a
            test_superlattice_min.build_variables(hole_position_min)
            result = test_superlattice_min.obtain_objective_and_gradient("need_value")
            print("this is the vg error of the min device: ", result)
            test_superlattice_min.plot_superlattice("./test_corner_min" + "_superlattice.png")

        else:
            raise ValueError(f"mode {mode} is not supported")
        # eff = evaluate_eff(eps, coupling_region_top, coupling_region_out_top, displacement) # TODO implement the evaluate_eff function
        # vg_mse = evaluate_vg(eps, coupling_region_top, coupling_region_out_top, displacement) # TODO implement the evaluate_vg function

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    # parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    # parser.add_argument('--pdb', action='store_true', help='pdb')
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)

    # Extract parameters from model.parameters()
    params_from_parameters = set(p for p in model.parameters() if p.requires_grad)

    # Extract parameters from model.named_parameters()
    params_from_named_parameters = set(p for name, p in model.named_parameters())

    # Check if the two sets are the same
    if params_from_parameters == params_from_named_parameters:
        lg.info("The sets of parameters from model.parameters() and model.named_parameters() are the same.")
    else:
        raise ValueError("The sets of parameters from model.parameters() and model.named_parameters() are different.")

    if model.opt_coupling_method == "level_set":
        # Initialize the parameter groups
        param_groups = [
            {'params': [], 'lr': configs.optimizer.lr_level_set},  # For level-set related parameters
            {'params': [], 'lr': configs.optimizer.lr}  # For other parameters
        ]

        # Loop over all parameters in the model and categorize them
        for name, param in model.named_parameters():
            if name == "coupling_region_top" or name == "coupling_region_out_top":
                param_groups[0]['params'].append(param)
            else:
                param_groups[1]['params'].append(param)

    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    optimizer = builder.make_optimizer(
        # [p for p in model.parameters() if p.requires_grad],
        param_groups,
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    lr_scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)
    sharp_scheduler = builder.make_scheduler(
        optimizer, name="sharpness", config_file=configs.sharp_scheduler
    )

    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    try:
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )
        else:
            lg.info("No checkpoint to restore, output the initial model video")

        test_phc(
            model,
            mode="lithography_error",
            criterion=criterion,
            aux_criterions=aux_criterions,
            lossv=[],
            device=device,
            plot=False,
            grad_scaler=grad_scaler,
        )

    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")