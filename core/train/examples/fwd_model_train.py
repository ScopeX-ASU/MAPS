import argparse

import torch
import torch.amp as amp
import torch.nn as nn
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    set_torch_deterministic,
)

from core.train import builder
from core.train.trainer import PredTrainer
from core.utils import cal_total_field_adj_src_from_fwd_field
from core.utils import train_configs as configs

# Add the project root to sys.path
# project_root = os.path.abspath(
#     os.path.join(os.path.dirname(__file__), "../../../../MAPS")
# )
# sys.path.insert(0, project_root)


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
    lg.info(f"Number of NN parameters: {count_parameters(model)}")

    model_name = "dual_predictor"
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
    # trainer.single_batch_check()
    # quit()
    for epoch in range(1, int(configs.run.n_epochs) + 1):
        trainer.train(
            data_loader=train_loader,
            task="train",
            epoch=epoch,
        )
        trainer.train(
            data_loader=validation_loader,
            task="val",
            epoch=epoch,
        )
        if epoch > int(configs.run.n_epochs) - 21:
            trainer.train(
                data_loader=test_loader,
                task="test",
                epoch=epoch,
            )
            trainer.save_model(epoch=epoch, checkpoint_path=checkpoint)


if __name__ == "__main__":
    main()
