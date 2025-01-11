import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../MAPS")
)
sys.path.insert(0, project_root)

import argparse

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from pyutils.general import logger as lg
from pyutils.general import AverageMeter
from pyutils.torch_train import (
    BestKModelSaver,
    set_torch_deterministic,
    load_model,
)
import matplotlib.pyplot as plt
from core.train import builder
from core.train.models.utils import from_Ez_to_Hx_Hy
from core.train.trainer import PredTrainer
from core.utils import cal_total_field_adj_src_from_fwd_field, cal_fom_from_fwd_field
from core.utils import train_configs as configs
from core.train.trainer import data_preprocess
import numpy as np
import copy

class dual_predictor(nn.Module):
    def __init__(self, model_fwd, model_adj):
        super(dual_predictor, self).__init__()
        self.model_fwd = nn.ModuleDict({
            f"{str(wl).replace('.', 'p')}-{mode}-{temp}-{in_port_name}-{out_port_name}": model
            for (wl, mode, temp, in_port_name, out_port_name), model in model_fwd.items()
        }) # this is now a dictionary of models [wl, mode, temp, in_port_name, out_port_name] -> model # most of the time it should contain at most 2 models
        self.model_adj = nn.ModuleDict({
            f"{str(wl).replace('.', 'p')}-{mode}-{temp}-{in_port_name}-{out_port_name}": model
            for (wl, mode, temp, in_port_name, out_port_name), model in model_adj.items()
        }) # this is now a dictionary of models [wl, mode, temp, in_port_name, out_port_name] -> model # most of the time it should contain at most 2 models

    def forward(
        self, 
        data
    ):
        eps = data["eps_map"]
        src = {}
        adjoint_source = {}
        x_fwd = {}
        forward_field = {}
        adjoint_field = {}
        # ['eps_map', 'adj_srcs', 'gradient', 'field_solutions', 's_params', 'src_profiles', 'fields_adj', 'field_normalizer', 'design_region_mask', 'ht_m', 'et_m', 'monitor_slices', 'As', 'opt_cfg_file_path']
        # this is the key of data htms:  
        # [
        #     'ht_m-wl-1.55-port-in_port_1-mode-1', 
        #     'ht_m-wl-1.55-port-in_port_1-mode-1-origin_size', 
        #     'ht_m-wl-1.55-port-in_port_1-mode-2', 
        #     'ht_m-wl-1.55-port-in_port_1-mode-2-origin_size', 
        #     'ht_m-wl-1.55-port-out_port_1-mode-1', 
        #     'ht_m-wl-1.55-port-out_port_1-mode-1-origin_size', 
        #     'ht_m-wl-1.55-port-out_port_2-mode-2', 
        #     'ht_m-wl-1.55-port-out_port_2-mode-2-origin_size', 
        #     'ht_m-wl-1.55-port-refl_port_1-mode-1', 
        #     'ht_m-wl-1.55-port-refl_port_1-mode-1-origin_size', 
        #     'ht_m-wl-1.55-port-refl_port_1-mode-2', 
        #     'ht_m-wl-1.55-port-refl_port_1-mode-2-origin_size'
        # ]
        # this is the keys in adjoint field:  
        # ['fields_adj-wl-1.55-port-in_port_1-mode-1', 'fields_adj-wl-1.55-port-in_port_1-mode-2']
        # keys in field solutions:
        # ['field_solutions-wl-1.55-port-in_port_1-mode-1-temp-300', 'field_solutions-wl-1.55-port-out_port_1-mode-1-temp-300', 'field_solutions-wl-1.55-port-refl_port_1-mode-1-temp-300']
        for key, model in self.model_fwd.items():
            wl, mode, temp, in_port_name, out_port_name = key.split("-")
            wl, mode, temp = float(wl.replace('p', '.')), int(mode), eval(temp)
            src[(wl, mode, in_port_name)] = data["src_profiles"][f"source_profile-wl-{wl}-port-{in_port_name}-mode-{mode}"]
            x_fwd[(wl, mode, temp, in_port_name, out_port_name)] = model(eps, src[(wl, mode, in_port_name)])
            with torch.enable_grad():
                forward_field[(wl, mode, temp, in_port_name, out_port_name)], adj_source = cal_total_field_adj_src_from_fwd_field(
                    Ez=x_fwd[(wl, mode, temp, in_port_name, out_port_name)],
                    # Ez=data["field_solutions"]["field_solutions-wl-1.55-port-in_port_1-mode-1-temp-300"][:, -2:, ...],
                    eps=eps,
                    ht_ms=data["ht_m"], # this two only used for adjoint field calculation, we don't need it here in forward pass
                    et_ms=data["et_m"],
                    monitors=data["monitor_slices"],
                    pml_mask=model.pml_mask,
                    from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                    return_adj_src=True,
                    sim=model.sim,
                    opt_cfg_file_path=data['opt_cfg_file_path'],
                    wl=wl,
                    mode=mode,
                    temp=temp,
                    in_port_name=in_port_name,
                    out_port_name=out_port_name,
                )
            # the adjoint source calculated with the one that stored in the dataset have a scale factor difference since we want to normalize the adjoint source power to be 1e-8
            adjoint_source[(wl, mode, temp, in_port_name, out_port_name)] = adj_source = adj_source.detach()
            adj_model = self.model_adj[key]
            x_adj = adj_model(eps, adj_source)
            adjoint_field[(wl, mode, temp, in_port_name, out_port_name)], _ = cal_total_field_adj_src_from_fwd_field(
                                            Ez=x_adj,
                                            eps=eps,
                                            ht_ms=data['ht_m'],
                                            et_ms=data['et_m'],
                                            monitors=data['monitor_slices'],
                                            pml_mask=adj_model.pml_mask,
                                            from_Ez_to_Hx_Hy_func=from_Ez_to_Hx_Hy,
                                            return_adj_src=False,
                                            sim=adj_model.sim,
                                            opt_cfg_file_path=data['opt_cfg_file_path'],
                                            wl=wl,
                                            mode=mode,
                                            temp=temp,
                                            in_port_name=in_port_name,
                                            out_port_name=out_port_name,
                                        )
        return {
            "forward_field": forward_field,
            "adjoint_field": adjoint_field,
            "adjoint_source": adjoint_source,
        }
    
def test_grad(
    model,
    test_loader,
    device,
    configs,
):
    model.eval()
    fwd_model = model.model_fwd["1p55-1-300-in_port_1-out_port_1"]
    main_criterion_meter = AverageMeter("gradient_similarity")

    data_counter = 0
    total_data = len(test_loader.dataset)  # Total samples
    num_batches = len(test_loader)  # Number of batches

    iterator = iter(test_loader)
    local_step = 0
    cosine_similarity_lsit = []
    while local_step < num_batches:
        try:
            data = next(iterator)
        except StopIteration:
            iterator = iter(test_loader)
            data = next(iterator)

        data = data_preprocess(data, device)
        # Ensure `eps` requires gradients
        eps = data["eps_map"].clone().detach().to(device).requires_grad_(True)
        # Replace `eps` in the data dictionary with the new tensor
        data["eps_map"] = eps
        with amp.autocast('cuda', enabled=False):
            fwd_field = fwd_model(eps, data["src_profiles"]["source_profile-wl-1.55-port-in_port_1-mode-1"])
            # calculate the figure of merit from forward field
            # '1p55-1-300-in_port_1-out_port_1'
            fom = cal_fom_from_fwd_field(
                    Ez = fwd_field,
                    eps = eps,
                    ht_ms = data["ht_m"],
                    et_ms = data["et_m"],
                    monitors = data["monitor_slices"],
                    sim = fwd_model.sim,
                    opt_cfg_file_path=data['opt_cfg_file_path'],
                    wl=1.55,
                    mode=1,
                    temp=300,
                    in_port_name='in_port_1',
            )
            fom = fom * (-1)
            # print("this is the shape of fom: ", fom.shape)
            # print("this is the fom: ", fom)
            # quit()



        # Backward pass to compute gradients
        grad_outputs = torch.ones_like(fom, device=device)  # Gradient vector for each sample in the batch
        gradients = torch.autograd.grad(
            outputs=fom,
            inputs=eps,
            grad_outputs=grad_outputs,
            retain_graph=True,  # Retain computation graph for potential further use
            create_graph=False,  # Do not create graph for second-order gradients
        )[0]
        # gradients now holds the gradients of `fom` w.r.t. `eps`
        # print(f"Gradients shape: {gradients.shape}")  # Should match `eps` shape
        # print(f"Gradients: {gradients}")
        # quit()
        grad_gt = data["gradient"]
        design_region_mask = data["design_region_mask"]
        # 'design_region_mask-bending_region_x_start', 'design_region_mask-bending_region_x_stop', 'design_region_mask-bending_region_y_start', 'design_region_mask-bending_region_y_stop'
        for i in range(gradients.shape[0]):
            grad_i = gradients[i]
            grad_gt_i = grad_gt[i]

            dr_mask = torch.zeros_like(grad_i, device=device)
            x_start = design_region_mask['design_region_mask-bending_region_x_start'][i]
            x_stop = design_region_mask['design_region_mask-bending_region_x_stop'][i]
            y_start = design_region_mask['design_region_mask-bending_region_y_start'][i]
            y_stop = design_region_mask['design_region_mask-bending_region_y_stop'][i]
            dr_mask[y_start:y_stop, x_start:x_stop] = 1
            # Mask the gradients
            masked_grad_i = grad_i * dr_mask  # Apply the mask to grad_i
            masked_grad_gt_i = grad_gt_i * dr_mask  # Apply the mask to grad_gt_i

            # Flatten the masked gradients to 1D tensors
            masked_grad_i_flat = masked_grad_i[y_start:y_stop, x_start:x_stop].flatten()
            masked_grad_gt_i_flat = masked_grad_gt_i[y_start:y_stop, x_start:x_stop].flatten()

            # plt.figure()
            # plt.imshow(masked_grad_i_flat.reshape(x_stop - x_start, y_stop - y_start).cpu().numpy(), cmap='RdYlBu')
            # plt.colorbar()
            # plt.title("norm_grad_i")
            # plt.savefig(f"./figs/grad_{i}.png")
            # plt.close()

            # plt.figure()
            # plt.imshow(masked_grad_gt_i_flat.reshape(x_stop - x_start, y_stop - y_start).cpu().numpy(), cmap='RdYlBu')
            # plt.colorbar()
            # plt.title("norm_grad_gt_i")
            # plt.savefig(f"./figs/grad_gt_{i}.png")
            # plt.close()
            # quit()

            # Compute cosine similarity
            cosine_similarity = F.cosine_similarity(
                masked_grad_i_flat.unsqueeze(0),  # Add batch dimension
                masked_grad_gt_i_flat.unsqueeze(0),  # Add batch dimension
                dim=1,  # Compute similarity across the feature dimension
            )

            # If norms are zero, the result will be NaN, so handle that case
            if torch.isnan(cosine_similarity).any():
                cosine_similarity = torch.tensor(0.0, device=device)

            cosine_similarity_lsit.append(cosine_similarity)
        local_step += 1
    
    cosine_similarity = torch.stack(cosine_similarity_lsit).mean()
    print(f"Mean cosine similarity: {cosine_similarity.item()}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.update(opts)
    # Convert tuple strings to actual tuples
    if hasattr(configs.model_fwd, "mode_list"):
        if configs.model_fwd.type != "FNO2d":
            assert hasattr(configs.model_fwd, "kernel_list"), "kernel_list should be defined if mode_list is defined"
            configs['model_fwd']['mode_list'] = [(60, 60)] * len(configs['model_fwd']['kernel_list'])
        else:
            configs['model_fwd']['mode_list'] = [(60, 60)] * 4
    if hasattr(configs.model_adj, "mode_list"):
        if configs.model_adj.type != "FNO2d":
            assert hasattr(configs.model_adj, "kernel_list"), "kernel_list should be defined if mode_list is defined"
            configs['model_adj']['mode_list'] = [(60, 60)] * len(configs['model_adj']['kernel_list'])
        else:
            configs['model_adj']['mode_list'] = [(60, 60)] * 4
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
        print("cuda is available and set to device: ", device, flush=True)
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False
    print("this is the config: \n", configs, flush=True)
    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))
    model_fwd = {}
    assert len(configs.model_fwd.temp) == len(configs.model_fwd.mode) == len(configs.model_fwd.wl) == len(configs.model_fwd.in_out_port_name), "temp, mode, wl, in_out_port_name should have the same length"
    for i in range(len(configs.model_fwd.temp)):
        model_cfg = copy.deepcopy(configs.model_fwd)
        model_cfg.temp = temp = model_cfg.temp[i]
        model_cfg.mode = mode = model_cfg.mode[i]
        model_cfg.wl = wl = model_cfg.wl[i]
        model_cfg.in_port_name = in_port_name = model_cfg.in_out_port_name[i][0]
        model_cfg.out_port_name = out_port_name = model_cfg.in_out_port_name[i][1]
        model_fwd[(wl, mode, temp, in_port_name, out_port_name)] = builder.make_model(device=device, **model_cfg)
    # model_fwd = builder.make_model(device=device, **configs.model_fwd)
    print("this is the model: \n", model_fwd, flush=True)
    
    model_adj = {}
    assert len(configs.model_adj.temp) == len(configs.model_adj.mode) == len(configs.model_adj.wl) == len(configs.model_adj.in_out_port_name), "temp, mode, wl, in_out_port_name should have the same length"
    for i in range(len(configs.model_adj.temp)):
        model_cfg = copy.deepcopy(configs.model_adj)
        model_cfg.temp = temp = model_cfg.temp[i]
        model_cfg.mode = mode = model_cfg.mode[i]
        model_cfg.wl = wl = model_cfg.wl[i]
        model_cfg.in_port_name = in_port_name = model_cfg.in_out_port_name[i][0]
        model_cfg.out_port_name = out_port_name = model_cfg.in_out_port_name[i][1]
        model_adj[(wl, mode, temp, in_port_name, out_port_name)] = builder.make_model(device=device, **model_cfg)
    # model_adj = builder.make_model(device=device, **configs.model_adj)
    print("this is the model: \n", model_adj, flush=True)
    # model_adj = builder.make_model(
    #     device=device,
    #     **configs.model_adj
    # )

    model = dual_predictor(model_fwd, model_adj)
    train_loader, validation_loader, test_loader = builder.make_dataloader()
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer, config_file=configs.lr_scheduler)
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    print("aux criterions used in training: ", aux_criterions, flush=True)

    log_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.log_criterion.items()
        if float(config.weight) > 0
    }
    print("log criterions used to monitor performance: ", log_criterions, flush=True)

    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))

    model_name = 'dual_predictor'
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    lg.info(f"Current fwd NN checkpoint: {checkpoint}")

    # load model:
    if (
        int(configs.checkpoint.resume)
        and len(configs.checkpoint.restore_checkpoint) > 0
    ):
        load_model(
            model,
            configs.checkpoint.restore_checkpoint,
            ignore_size_mismatch=int(configs.checkpoint.no_linear),
        )

    trainer = PredTrainer(
        data_loaders={
            "train": train_loader,
            "val": validation_loader,
            "test": test_loader,
        }, 
        model=model, 
        criterion=criterion,
        aux_criterion=aux_criterions,
        log_criterion=log_criterions, 
        optimizer=optimizer, 
        scheduler=scheduler, 
        saver=saver,
        grad_scaler=grad_scaler,
        device=device, 
    )
    # trainer.train(
    #     data_loader=test_loader,
    #     task='test',
    #     epoch=0,
    # )
    # quit()
    assert configs.dataset.data_dir == "raw_test"
    test_grad(
        model=model,
        test_loader=test_loader,
        device=device,
        configs=configs,
    )

if __name__ == "__main__":
    main()
