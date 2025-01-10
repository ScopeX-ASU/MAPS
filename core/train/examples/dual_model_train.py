import sys
import os

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../MAPS"))
sys.path.insert(0, project_root)

from core.utils import train_configs as configs
import torch
import torch.nn as nn
import torch.amp as amp
from pyutils.general import logger as lg
import argparse
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from core.train import builder
from core.utils import cal_total_field_adj_src_from_fwd_field
from core.train.models.utils import from_Ez_to_Hx_Hy
from core.train.trainer import PredTrainer
import copy
import matplotlib.pyplot as plt

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
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
    lg.info(f"Number of NN parameters: {count_parameters(model)}")

    model_name = 'dual_predictor'
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"
    lg.info(f"Current fwd NN checkpoint: {checkpoint}")

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

    for epoch in range(1, int(configs.run.n_epochs) + 1):
        trainer.train(
            data_loader=train_loader,
            task='train',
            epoch=epoch,
        )
        trainer.train(
            data_loader=validation_loader,
            task='val',
            epoch=epoch,
        )
        if epoch > int(configs.run.n_epochs) - 21:
            trainer.train(
                data_loader=test_loader,
                task='test',
                epoch=epoch,
            )
            trainer.save_model(
                epoch=epoch,
                checkpoint_path=checkpoint
            )

if __name__ == "__main__":
    main()