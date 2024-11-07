import random
from typing import Tuple

import torch
import torch.nn as nn
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from torch.utils.data import Sampler

from core.models import *
from core.datasets import *

from .utils import (
    DAdaptAdam,
    DistanceLoss,
    NL2NormLoss,
    NormalizedMSELoss,
    TemperatureScheduler,
    SharpnessScheduler,
    ResolutionScheduler,
    maskedNL2NormLoss,
    maskedNMSELoss,
    fab_penalty_ls_curve,
    fab_penalty_ls_gap,
    AspectRatioLoss,
    MaxwellResidualLoss,
    GradientLoss,
    SParamLoss,
)

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def collate_fn_keep_spatial_res(batch):
    data, targets = zip(*batch)
    new_size = []
    for item in data:
        if (
            item["device_type"].item() == 1 or item["device_type"].item() == 5
        ):  # which means it is a mmi and resolution = 20
            newSize = (
                int(round(item["Ez"].shape[-2] * item["scaling_factor"].item() * 0.75)),
                int(round(item["Ez"].shape[-1] * item["scaling_factor"].item() * 0.75)),
            )
        else:
            newSize = (
                int(round(item["Ez"].shape[-2] * item["scaling_factor"].item())),
                int(round(item["Ez"].shape[-1] * item["scaling_factor"].item())),
            )
        new_size.append(newSize)
    Hight = [
        int(round(item["Ez"].shape[-2] * item["scaling_factor"].item()))
        for item in data
    ]
    Width = [
        int(round(item["Ez"].shape[-1] * item["scaling_factor"].item()))
        for item in data
    ]
    maxHight = max(Hight)
    maxWidth = max(Width)
    if maxWidth % 2 == 1:
        maxWidth += 1  ## make sure the width is even so that there won't be any mismatch between fourier and inverse fourier
    # Pad all items to the max length and max width using zero padding
    # eps should use background value to padding
    # fields should use zero padding
    # weight masks should use zero padding
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.interpolate(
            item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)  # dummy batch dim and then remove it
        item["eps"] = torch.nn.functional.interpolate(
            item["eps"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["source"] = torch.nn.functional.interpolate(
            item["source"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["mseWeight"] = torch.nn.functional.interpolate(
            item["mseWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["src_mask"] = torch.nn.functional.interpolate(
            item["src_mask"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["epsWeight"] = torch.nn.functional.interpolate(
            item["epsWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["padding_mask"] = torch.ones_like(item["eps"], device=item["eps"].device)

    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.interpolate(
            item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)

    hightPatchSize_bot = [(maxHight - item["Ez"].shape[-2]) // 2 for item in data]
    hightPatchSize_top = [
        maxHight - item["Ez"].shape[-2] - (maxHight - item["Ez"].shape[-2]) // 2
        for item in data
    ]
    widthPatchSize_left = [(maxWidth - item["Ez"].shape[-1]) // 2 for item in data]
    widthPatchSize_right = [
        maxWidth - item["Ez"].shape[-1] - (maxWidth - item["Ez"].shape[-1]) // 2
        for item in data
    ]
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.pad(
            item["Ez"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["eps"] = torch.nn.functional.pad(
            item["eps"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=item["eps_bg"].item(),
        )
        item["padding_mask"] = torch.nn.functional.pad(
            item["padding_mask"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["source"] = torch.nn.functional.pad(
            item["source"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["mseWeight"] = torch.nn.functional.pad(
            item["mseWeight"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["src_mask"] = torch.nn.functional.pad(
            item["src_mask"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["epsWeight"] = torch.nn.functional.pad(
            item["epsWeight"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )

    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.pad(
            item["Ez"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )

    Ez_data = torch.stack([item["Ez"] for item in data], dim=0)
    source_data = torch.stack([item["source"] for item in data], dim=0)
    eps_data = torch.stack([item["eps"] for item in data], dim=0)
    padding_mask_data = torch.stack([item["padding_mask"] for item in data], dim=0)
    mseWeight_data = torch.stack([item["mseWeight"] for item in data], dim=0)
    src_mask_data = torch.stack([item["src_mask"] for item in data], dim=0)
    epsWeight_data = torch.stack([item["epsWeight"] for item in data], dim=0)
    device_type_data = torch.stack([item["device_type"] for item in data], dim=0)
    eps_bg_data = torch.stack([item["eps_bg"] for item in data], dim=0)
    grid_step_data = torch.stack([item["grid_step"] for item in data], dim=0)

    Ez_target = torch.stack([item["Ez"] for item in targets], dim=0)
    std_target = torch.stack([item["std"] for item in targets], dim=0)
    mean_target = torch.stack([item["mean"] for item in targets], dim=0)
    stdDacayRate = torch.stack([item["stdDacayRate"] for item in targets], dim=0)

    raw_data = {
        "Ez": Ez_data,
        "source": source_data,
        "eps": eps_data,
        "padding_mask": padding_mask_data,
        "mseWeight": mseWeight_data,
        "src_mask": src_mask_data,
        "epsWeight": epsWeight_data,
        "device_type": device_type_data,
        "eps_bg": eps_bg_data,
        "grid_step": grid_step_data,
    }
    raw_targets = {
        "Ez": Ez_target,
        "std": std_target,
        "mean": mean_target,
        "stdDacayRate": stdDacayRate,
    }

    return raw_data, raw_targets


def collate_fn_keep_spatial_res_pad_to_256(batch):
    # Extract all items for each key and compute the max length
    data, targets = zip(*batch)
    new_size = []
    for item in data:
        if (
            item["device_type"].item() == 1 or item["device_type"].item == 5
        ):  # train seperately for metaline, so no need to consider the resolution mismatch
            newSize = (
                int(round(item["Ez"].shape[-2] * item["scaling_factor"].item() * 0.75)),
                int(round(item["Ez"].shape[-1] * item["scaling_factor"].item() * 0.75)),
            )
        else:
            newSize = (
                int(round(item["Ez"].shape[-2] * item["scaling_factor"].item())),
                int(round(item["Ez"].shape[-1] * item["scaling_factor"].item())),
            )
        new_size.append(newSize)
    maxHight = 256
    maxWidth = 256
    if item["device_type"].item() == 5:  # which means it is a metaline
        maxHight = 168
        maxWidth = 168
    # Pad all items to the max length and max width using zero padding
    # eps should use background value to padding
    # fields should use zero padding
    # weight masks should use zero padding
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.interpolate(
            item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)  # dummy batch dim and then remove it
        item["eps"] = torch.nn.functional.interpolate(
            item["eps"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["source"] = torch.nn.functional.interpolate(
            item["source"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["mseWeight"] = torch.nn.functional.interpolate(
            item["mseWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["src_mask"] = torch.nn.functional.interpolate(
            item["src_mask"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["epsWeight"] = torch.nn.functional.interpolate(
            item["epsWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)
        item["padding_mask"] = torch.ones_like(item["eps"], device=item["eps"].device)

    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.interpolate(
            item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear"
        ).squeeze(0)

    hightPatchSize_bot = [(maxHight - item["Ez"].shape[-2]) // 2 for item in data]
    hightPatchSize_top = [
        maxHight - item["Ez"].shape[-2] - (maxHight - item["Ez"].shape[-2]) // 2
        for item in data
    ]
    widthPatchSize_left = [(maxWidth - item["Ez"].shape[-1]) // 2 for item in data]
    widthPatchSize_right = [
        maxWidth - item["Ez"].shape[-1] - (maxWidth - item["Ez"].shape[-1]) // 2
        for item in data
    ]
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.pad(
            item["Ez"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["eps"] = torch.nn.functional.pad(
            item["eps"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=item["eps_bg"].item(),
        )
        item["padding_mask"] = torch.nn.functional.pad(
            item["padding_mask"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["source"] = torch.nn.functional.pad(
            item["source"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["mseWeight"] = torch.nn.functional.pad(
            item["mseWeight"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["src_mask"] = torch.nn.functional.pad(
            item["src_mask"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )
        item["epsWeight"] = torch.nn.functional.pad(
            item["epsWeight"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )

    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.pad(
            item["Ez"],
            (
                widthPatchSize_left[idx],
                widthPatchSize_right[idx],
                hightPatchSize_bot[idx],
                hightPatchSize_top[idx],
            ),
            mode="constant",
            value=0,
        )

    Ez_data = torch.stack([item["Ez"] for item in data], dim=0)
    source_data = torch.stack([item["source"] for item in data], dim=0)
    eps_data = torch.stack([item["eps"] for item in data], dim=0)
    padding_mask_data = torch.stack([item["padding_mask"] for item in data], dim=0)
    mseWeight_data = torch.stack([item["mseWeight"] for item in data], dim=0)
    src_mask_data = torch.stack([item["src_mask"] for item in data], dim=0)
    epsWeight_data = torch.stack([item["epsWeight"] for item in data], dim=0)
    device_type_data = torch.stack([item["device_type"] for item in data], dim=0)
    eps_bg_data = torch.stack([item["eps_bg"] for item in data], dim=0)
    grid_step_data = torch.stack([item["grid_step"] for item in data], dim=0)

    Ez_target = torch.stack([item["Ez"] for item in targets], dim=0)
    std_target = torch.stack([item["std"] for item in targets], dim=0)
    mean_target = torch.stack([item["mean"] for item in targets], dim=0)
    stdDacayRate = torch.stack([item["stdDacayRate"] for item in targets], dim=0)

    raw_data = {
        "Ez": Ez_data,
        "source": source_data,
        "eps": eps_data,
        "padding_mask": padding_mask_data,
        "mseWeight": mseWeight_data,
        "src_mask": src_mask_data,
        "epsWeight": epsWeight_data,
        "device_type": device_type_data,
        "eps_bg": eps_bg_data,
        "grid_step": grid_step_data,
    }
    raw_targets = {
        "Ez": Ez_target,
        "std": std_target,
        "mean": mean_target,
        "stdDacayRate": stdDacayRate,
    }

    return raw_data, raw_targets


class MySampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.indices = sorted(
            range(len(data_source)), key=lambda x: data_source[x][0]["area"].item()
        )
        self.shuffle = shuffle
        # self.indices is a list of indices sorted by the area of the devices

    def __iter__(self):
        if self.shuffle:
            group_size = 2500
            group_num = len(self.indices) // group_size
            for i in range(group_num + 1):
                group_indices = (
                    self.indices[i * group_size : (i + 1) * group_size]
                    if i != group_num
                    else self.indices[i * group_size :]
                )
                random.shuffle(group_indices)
                if i != group_num:
                    self.indices[i * group_size : (i + 1) * group_size] = group_indices
                else:
                    self.indices[i * group_size :] = group_indices
        else:
            pass
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def make_dataloader(
    name: str = None,
    splits=["train", "valid", "test"],
    train_noise_cfg=None,
    out_frames=None,
) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "fdfd":
        train_dataset, validation_dataset, test_dataset = (
            (
                FDFDDataset(
                    device_type=configs.dataset.device_type,
                    root=configs.dataset.root,
                    split=split,
                    test_ratio=configs.dataset.test_ratio,
                    train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                    processed_dir=configs.dataset.processed_dir,
                )
                if split in splits
                else None
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=configs.run.batch_size,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        # collate_fn=collate_fn_keep_spatial_res,
        # sampler=MySampler(train_dataset, shuffle=int(configs.dataset.shuffle)),
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=configs.run.batch_size,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        # collate_fn=collate_fn_keep_spatial_res,
        # sampler=MySampler(validation_dataset, shuffle=int(configs.dataset.shuffle)),
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.run.batch_size,
        pin_memory=True,
        num_workers=configs.dataset.num_workers,
        prefetch_factor=8,
        persistent_workers=True,
        # collate_fn=collate_fn_keep_spatial_res,
        # sampler=MySampler(test_dataset, shuffle=int(configs.dataset.shuffle)),
    )

    return train_loader, validation_loader, test_loader

def make_device(device: Device):
    device_to_opt = eval(configs.model.device_type)(
        sim_cfg=configs.model.sim_cfg,
        device=device,
    )
    return device_to_opt

def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    if (
        "repara_phc_1x1" in configs.model.name.lower()
        and "eff_vg" not in configs.model.name.lower()
    ):
        model = eval(configs.model.name)(
            device_cfg=configs.model.device_cfg,
            sim_cfg=configs.model.sim_cfg,
            perturbation=configs.model.perturbation,
            num_rows_perside=configs.model.num_rows_perside,
            num_cols=configs.model.num_cols,
            adjoint_mode=configs.model.adjoint_mode,
            learnable_bdry=configs.model.learnable_bdry,
            df=configs.model.df,
            nf=configs.model.nf,
        )
    elif (
        "repara_phc_1x1" in configs.model.name.lower()
        and "eff_vg" in configs.model.name.lower()
    ):
        model = eval(configs.model.name)(
            coupling_region_cfg=configs.model.coupling_region_cfg,
            sim_cfg=configs.model.sim_cfg,
            superlattice_cfg=configs.model.superlattice_cfg,
            port_width=configs.model.port_width,
            port_len=configs.model.port_len,
            taper_width=configs.model.taper_width,
            taper_len=configs.model.taper_len,
            sy_coupling=configs.model.sy_coupling,
            adjoint_mode=configs.model.adjoint_mode,
            eps_bg=configs.model.eps_bg,
            eps_r=configs.model.eps_r,
            df=configs.model.df,
            nf=configs.model.nf,
            a=configs.model.a,
            r=configs.model.r,
            mfs=configs.model.mfs,
            binary_projection_threshold=configs.sharp_scheduler.sharp_threshold,
            binary_projection_method=configs.model.binary_projection_method,
            coupling_init=configs.model.coupling_init,
            opt_coupling_method=configs.model.opt_coupling_method,
            grad_mode=configs.model.grad_mode,
            cal_bd_mode=configs.model.cal_bd_mode,
            aux_out=True
            if configs.aux_criterion.curl_loss.weight > 0
            or configs.aux_criterion.gap_loss.weight > 0
            else False,
            device=device,
        ).to(device)
    elif "metalens" in configs.model.name.lower():
        model = eval(configs.model.name)(
            ridge_height_max=configs.model.ridge_height_max,
            sub_height=configs.model.sub_height,
            aperture=configs.model.aperture,
            f_min=configs.model.f_min,
            f_max=configs.model.f_max,
            eps_r=configs.model.eps_r,
            eps_bg=configs.model.eps_bg,
            sim_cfg=configs.model.sim_cfg,
            ls_cfg=configs.model.ls_cfg,
            mfs=configs.model.mfs,
            binary_projection_threshold=configs.model.binary_projection_threshold,
            build_method=configs.model.build_method,
            center_ridge=configs.model.center_ridge,
            max_num_ridges_single_side=configs.model.max_num_ridges_single_side,
            operation_device=device,
            aspect_ratio=configs.model.aspect_ratio,
            initial_point=configs.model.initial_point,
            if_constant_period=configs.model.if_constant_period,
            focal_constant=configs.model.focal_constant,
        ).to(device)
    elif configs.model.name.lower() == 'metacoupleroptimization':
        model = eval(configs.model.name)(
            device=kwargs["optDevice"],
            sim_cfg=configs.model.sim_cfg,
        ).to(device)
    elif "simplecnn" in configs.model.name.lower():
        model = eval(configs.model.name)().to(device)
    elif "neurolight" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            device=device,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
            **kwargs,
        ).to(device)
    elif "fno3d" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            mode1=configs.model.mode1,
            mode2=configs.model.mode2,
        ).to(device)
    elif "ffno2d" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            with_cp=False,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
        ).to(device)
    else:
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "dadaptadam":
        optimizer = DAdaptAdam(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            params,
            lr=configs.lr,  # for now, only the lr is tunable, others arguments just use the default value
            line_search_fn=configs.line_search_fn,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(
    optimizer: Optimizer, name: str = None, config_file: dict = {}
) -> Scheduler:
    name = (name or config_file.name).lower()
    if (
        name == "temperature"
    ):  # this temperature scheduler is a cosine annealing scheduler
        scheduler = TemperatureScheduler(
            initial_T=float(configs.temp_scheduler.lr),
            final_T=float(configs.temp_scheduler.lr_min),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "resolution":
        scheduler = ResolutionScheduler(
            initial_res=int(configs.res_scheduler.init_res),
            final_res=int(configs.res_scheduler.final_res),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "sharpness":
        scheduler = SharpnessScheduler(
            initial_sharp=float(configs.sharp_scheduler.init_sharp),
            final_sharp=float(configs.sharp_scheduler.final_sharp),
            total_steps=int(configs.run.n_epochs),
        )
    elif name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.lr_scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "mse":
        criterion = nn.MSELoss()
    elif name == "nmse":
        criterion = NormalizedMSELoss()
    elif name == "curl_loss":
        criterion = fab_penalty_ls_curve(alpha=cfg.weight, min_feature_size=0.02)
    elif name == "gap_loss":
        criterion = fab_penalty_ls_gap(beta=1, min_feature_size=0.02)
    elif name == "nl2norm":
        criterion = NL2NormLoss()
    elif name == "masknl2norm":
        criterion = maskedNL2NormLoss(
            weighted_frames=cfg.weighted_frames,
            weight=cfg.weight,
            if_spatial_mask=cfg.if_spatial_mask,
        )
    elif name == "masknmse":
        criterion = maskedNMSELoss(
            weighted_frames=cfg.weighted_frames,
            weight=cfg.weight,
            if_spatial_mask=cfg.if_spatial_mask,
        )
    elif name == "distanceloss":
        criterion = DistanceLoss(min_distance=cfg.min_distance)
    elif name == "aspect_ratio_loss":
        criterion = AspectRatioLoss(
            aspect_ratio=cfg.aspect_ratio,
        )
    elif name == "maxwell_residual_loss":
        criterion = MaxwellResidualLoss(
            wl_cen=cfg.wl_cen,
            wl_width=cfg.wl_width,
            n_wl=cfg.n_wl,
        )
    elif name == "grad_loss":
        criterion = GradientLoss()
    elif name == "s_param_loss":
        criterion = SParamLoss()
    else:
        raise NotImplementedError(name)
    return criterion
