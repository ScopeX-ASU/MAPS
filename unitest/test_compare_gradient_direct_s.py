import os
import sys

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../MAPS"))
sys.path.insert(0, project_root)

import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import BestKModelSaver, load_model, set_torch_deterministic

from core.train import builder
from core.train.models.utils import from_Ez_to_Hx_Hy
from core.train.trainer import PredTrainer, data_preprocess
from core.utils import cal_fom_from_fields, cal_total_field_adj_src_from_fwd_field
from core.utils import train_configs as configs


class fwd_predictor(nn.Module):
    def __init__(self, model_fwd):
        super(fwd_predictor, self).__init__()
        self.model_fwd = model_fwd  # this is now a dictionary of models [wl, mode, temp, in_port_name, out_port_name] -> model # most of the time it should contain at most 2 models

    def forward(
        self,
        data,
        epoch=1,
    ):
        # return_dict = {
        #     "eps_map": eps_map,
        #     "adj_src": adj_src,
        #     "gradient": gradient,
        #     "fwd_field": fwd_field,
        #     "s_params": s_params,
        #     "src_profile": src_profile,
        #     "adj_field": adj_field,
        #     "field_normalizer": field_adj_normalizer,
        #     "design_region_mask": design_region_mask,
        #     "ht_m": ht_m,
        #     "et_m": et_m,
        #     "monitor_slices": monitor_slices,
        #     "A": A,
        #     "opt_cfg_file_path": opt_cfg_file_path,
        #     "input_slice": input_slice,
        #     "wavelength": wavelength,
        #     "mode": mode,
        #     "temp": temp,
        # }
        eps = data["eps_map"]
        src = {}
        wl = data["wavelength"]
        mode = data["mode"]
        temp = data["temp"]
        in_slice_name = data["input_slice"]
        src = data["src_profile"]
        fwd_model_output = self.model_fwd(
            eps,
            src,
            monitor_slices=data["monitor_slices"],
            monitor_slice_list=None,
            in_slice_name=in_slice_name,
            wl=wl,
            temp=temp,
        )
        if isinstance(fwd_model_output, tuple):
            assert (
                len(fwd_model_output) == 2
            ), "fwd_model_output should be a tuple of length 2"
            fwd_Ez_field = fwd_model_output[0]
            s_params = fwd_model_output[1]
        with torch.enable_grad():
            fwd_field, _, _ = cal_total_field_adj_src_from_fwd_field(
                Ez4adj=data["fwd_field"][:, -2:, ...],
                Ez4fullfield=fwd_Ez_field,
                # Ez=data["fwd_field"][:, -2:, ...],
                eps=eps,
                ht_ms=data[
                    "ht_m"
                ],  # this two only used for adjoint field calculation, we don't need it here in forward pass
                et_ms=data["et_m"],
                monitors=data["monitor_slices"],
                pml_mask=self.model_fwd.pml_mask,
                return_adj_src=False,
                sim=self.model_fwd.sim,
                opt_cfg_file_path=data["opt_cfg_file_path"],
                wl=wl,
                mode=mode,
                temp=temp,
                src_in_slice_name=in_slice_name,
            )
        return {
            "forward_field": fwd_field,
            "s_params": (
                s_params if len(s_params) > 0 else None
            ),  # only pass s_params if we have s-param head to predict s_params
            "adjoint_field": None,
            "adjoint_source": None,
        }


def test_grad(
    model,
    test_loader,
    device,
    configs,
):
    model.eval()
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
        with amp.autocast("cuda", enabled=False):
            output = model(data)
            fwd_field = output["forward_field"]
            if output["s_params"] is not None:
                s_params = output["s_params"]
                fom = -s_params[:, 0] + 0.1 * s_params[:, 1]
            else:
                # calculate the figure of merit from forward field
                print(
                    "this is the key of the model_fwd: ",
                    list(model.model_fwd.keys()),
                    flush=True,
                )
                quit()

        # Backward pass to compute gradients
        grad_outputs = torch.ones_like(
            fom, device=device
        )  # Gradient vector for each sample in the batch
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
            x_start = design_region_mask["design_region_mask-bending_region_x_start"][i]
            x_stop = design_region_mask["design_region_mask-bending_region_x_stop"][i]
            y_start = design_region_mask["design_region_mask-bending_region_y_start"][i]
            y_stop = design_region_mask["design_region_mask-bending_region_y_stop"][i]
            dr_mask[y_start:y_stop, x_start:x_stop] = 1
            # Mask the gradients
            masked_grad_i = grad_i * dr_mask  # Apply the mask to grad_i
            masked_grad_gt_i = grad_gt_i * dr_mask  # Apply the mask to grad_gt_i

            if local_step == 1 and i == 0:
                cal_vmax = (
                    masked_grad_i[
                        x_start:x_stop,
                        y_start:y_stop,
                    ]
                    .abs()
                    .max()
                    .item()
                )

                plt.figure()
                plt.imshow(
                    np.rot90(
                        masked_grad_i[
                            x_start:x_stop,
                            y_start:y_stop,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    cmap="RdBu",
                    vmin=-cal_vmax,
                    vmax=cal_vmax,
                )
                plt.colorbar()
                plt.title("gradients")
                plt.savefig(
                    f"./figs/gradients_ad_S_{configs.model_fwd.type}.png", dpi=300
                )
                plt.close()

                gt_vmax = (
                    masked_grad_gt_i[
                        x_start:x_stop,
                        y_start:y_stop,
                    ]
                    .abs()
                    .max()
                    .item()
                )

                plt.figure()
                plt.imshow(
                    np.rot90(
                        masked_grad_gt_i[
                            x_start:x_stop,
                            y_start:y_stop,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    ),
                    cmap="RdBu",
                    vmin=-gt_vmax,
                    vmax=gt_vmax,
                )
                plt.colorbar()
                plt.title("grad_gt")
                plt.savefig(
                    f"./figs/grad_gt_ad_S_{configs.model_fwd.type}.png", dpi=300
                )
                plt.close()

            # Flatten the masked gradients to 1D tensors
            masked_grad_i_flat = masked_grad_i[y_start:y_stop, x_start:x_stop].flatten()
            masked_grad_gt_i_flat = masked_grad_gt_i[
                y_start:y_stop, x_start:x_stop
            ].flatten()

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
    if hasattr(configs.model_fwd, "mode_list"):
        if configs.model_fwd.type != "FNO2d":
            assert hasattr(
                configs.model_fwd, "kernel_list"
            ), "kernel_list should be defined if mode_list is defined"
            configs["model_fwd"]["mode_list"] = [(50, 50)] * len(
                configs["model_fwd"]["kernel_list"]
            )
        else:
            configs["model_fwd"]["mode_list"] = [(50, 50)] * 4
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

    configs.model_fwd.device = device
    model_fwd = builder.make_model(**configs.model_fwd)
    print("this is the model: \n", model_fwd, flush=True)

    model = fwd_predictor(model_fwd)
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

    model_name = "fwd_predictor"
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
    test_grad(
        model=model,
        test_loader=test_loader,
        device=device,
        configs=configs,
    )


if __name__ == "__main__":
    main()
