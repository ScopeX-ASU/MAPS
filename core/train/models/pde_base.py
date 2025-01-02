"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-26 00:36:10
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-03-02 19:24:43
"""

import inspect
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn.bricks import build_activation_layer, build_norm_layer, build_conv_layer
from mmengine.registry import MODELS
from pyutils.general import TimerCtx, logger
from pyutils.torch_train import set_torch_deterministic
from torch import Tensor, nn
from torch.functional import Tensor
from torch.types import Device, _size
from torch.utils.checkpoint import checkpoint
from core.utils import print_stat
import math

__all__ = ["PDE_NN_BASE"]


# def build_linear_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
#     if cfg is None:
#         cfg_ = dict(type="Linear")
#     else:
#         if not isinstance(cfg, dict):
#             raise TypeError("cfg must be a dict")
#         if "type" not in cfg:
#             raise KeyError('the cfg dict must contain the key "type"')
#         cfg_ = cfg.copy()

#     layer_type = cfg_.pop("type")
#     if inspect.isclass(layer_type):
#         return layer_type(*args, **kwargs, **cfg_)  # type: ignore
#     # Switch registry to the target scope. If `linear_layer` cannot be found
#     # in the registry, fallback to search `linear_layer` in the
#     # mmengine.MODELS.
#     with MODELS.switch_scope_and_registry(None) as registry:
#         linear_layer = registry.get(layer_type)
#     if linear_layer is None:
#         raise KeyError(
#             f"Cannot find {linear_layer} in registry under scope "
#             f"name {registry.scope}"
#         )
#     layer = linear_layer(*args, **kwargs, **cfg_)

#     return layer

# class LinearBlock(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         bias: bool = False,
#         linear_cfg: dict = dict(type="TeMPOBlockLinear"),
#         norm_cfg: dict | None = None,
#         act_cfg: dict | None = dict(type="ReLU", inplace=True),
#         dropout: float = 0.0,
#         device: Device = torch.device("cuda:0"),
#     ) -> None:
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
#         if linear_cfg["type"] not in {"Linear", None}:
#             linear_cfg.update({"device": device})
#         self.linear = build_linear_layer(
#             linear_cfg,
#             in_features=in_features,
#             out_features=out_features,
#             bias=bias,
#         )
#         if norm_cfg is not None:
#             self.norm = build_norm_layer(norm_cfg, out_features)
#         else:
#             self.norm = None

#         if act_cfg is not None:
#             self.activation = build_activation_layer(act_cfg)
#         else:
#             self.activation = None

#     def forward(self, x: Tensor) -> Tensor:
#         if self.dropout is not None:
#             x = self.dropout(x)
#         x = self.linear(x)
#         if self.norm is not None:
#             x = self.norm(x)
#         if self.activation is not None:
#             x = self.activation(x)
#         return x


class PDE_NN_BASE(nn.Module):
    def __init__(
        self, *args, encoder_cfg={}, backbone_cfg={}, decoder_cfg={}, **kwargs
    ):
        super().__init__(*args, **kwargs)
        with MODELS.switch_scope_and_registry(None) as registry:
            self._conv = tuple(
                set(
                    [
                        registry.get(encoder_cfg.conv_cfg["type"]),
                        registry.get(backbone_cfg.conv_cfg["type"]),
                        registry.get(decoder_cfg.conv_cfg["type"]),
                    ]
                )
            )

    def reset_parameters(self, random_state: Optional[int] = None):
        for name, m in self.named_modules():
            if isinstance(m, self._conv) and hasattr(m, "reset_parameters"):
                if random_state is not None:
                    # deterministic seed, but different for different layer, and controllable by random_state
                    set_torch_deterministic(random_state + sum(map(ord, name)))
                m.reset_parameters()
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    @lru_cache(maxsize=8)
    def _get_linear_pos_enc(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = torch.arange(0, size_x, device=device)
        gridy = torch.arange(0, size_y, device=device)
        gridx, gridy = torch.meshgrid(gridx, gridy)
        mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
        return mesh

    def get_grid(
        self,
        shape,
        device,
        mode="linear",
        epsilon=None,
        wavelength=None,
        grid_step=None,
    ):
        # epsilon must be real permittivity without normalization
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        if mode == "linear":
            gridx = torch.linspace(0, 1, size_x, device=device)
            gridy = torch.linspace(0, 1, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            return (
                torch.stack([gridy, gridx], dim=0)
                .unsqueeze(0)
                .expand(batchsize, -1, -1, -1)
            )
        elif mode in {"exp", "exp_noeps"}:  # exp in the complex domain
            # gridx = torch.arange(0, size_x, device=device)
            # gridy = torch.arange(0, size_y, device=device)
            # gridx, gridy = torch.meshgrid(gridx, gridy)
            # mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = self._get_linear_pos_enc(shape, device)
            # mesh = torch.view_as_real(
            #     torch.exp(
            #         mesh.mul(grid_step[..., None, None]).mul(
            #             1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
            #         )
            #     )
            # )  # [bs, 2, h, w, 2] real
            # mesh [1 ,2 ,h, w] real
            # grid_step [bs, 2, 1, 1] real
            # wavelength [bs, 1, 1, 1] real
            # epsilon [bs, 1, h, w] complex
            mesh = torch.view_as_real(
                torch.exp(
                    mesh.mul(
                        grid_step.div(wavelength).mul(1j * 2 * np.pi)[..., None, None]
                    ).mul(epsilon.data.sqrt())
                )
            )  # [bs, 2, h, w, 2] real
            return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)
        elif mode == "exp3":  # exp in the complex domain
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = torch.view_as_real(
                torch.cat([mesh, mesh[:, 0:1].add(mesh[:, 1:])], dim=1)
            )  # [bs, 3, h, w, 2] real
            return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)
        elif mode == "exp4":  # exp in the complex domain
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = torch.view_as_real(
                torch.cat(
                    [
                        mesh,
                        mesh[:, 0:1].mul(mesh[:, 1:]),
                        mesh[:, 0:1].div(mesh[:, 1:]),
                    ],
                    dim=1,
                )
            )  # [bs, 4, h, w, 2] real
            return mesh.permute(0, 1, 4, 2, 3).flatten(1, 2)
        elif mode == "exp_full":
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = (
                torch.view_as_real(mesh).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, 2, h, w, 2] real -> [bs, 4, h, w] real
            wavelength_map = wavelength[..., None, None].expand(
                mesh.shape[0], 1, mesh.shape[2], mesh.shape[3]
            )  # [bs, 1, h, w] real
            grid_step_mesh = (
                grid_step[..., None, None].expand(
                    mesh.shape[0], 2, mesh.shape[2], mesh.shape[3]
                )
                * 10
            )  # 0.05 um -> 0.5 for statistical stability # [bs, 2, h, w] real
            return torch.cat(
                [mesh, wavelength_map, grid_step_mesh], dim=1
            )  # [bs, 7, h, w] real
        elif mode == "exp_full_r":
            gridx = torch.arange(0, size_x, device=device)
            gridy = torch.arange(0, size_y, device=device)
            gridx, gridy = torch.meshgrid(gridx, gridy)
            mesh = torch.stack([gridy, gridx], dim=0).unsqueeze(0)  # [1, 2, h, w] real
            mesh = torch.exp(
                mesh.mul(grid_step[..., None, None]).mul(
                    1j * 2 * np.pi / wavelength[..., None, None] * epsilon.data.sqrt()
                )
            )  # [bs, 2, h, w] complex
            mesh = (
                torch.view_as_real(mesh).permute(0, 1, 4, 2, 3).flatten(1, 2)
            )  # [bs, 2, h, w, 2] real -> [bs, 4, h, w] real
            wavelength_map = (1 / wavelength)[..., None, None].expand(
                mesh.shape[0], 1, mesh.shape[2], mesh.shape[3]
            )  # [bs, 1, h, w] real
            grid_step_mesh = (
                grid_step[..., None, None].expand(
                    mesh.shape[0], 2, mesh.shape[2], mesh.shape[3]
                )
                * 10
            )  # 0.05 um -> 0.5 for statistical stability # [bs, 2, h, w] real
            return torch.cat(
                [mesh, wavelength_map, grid_step_mesh], dim=1
            )  # [bs, 7, h, w] real
        elif mode == "raw":
            wavelength_map = wavelength[..., None, None].expand(
                batchsize, 1, size_x, size_y
            )  # [bs, 1, h, w] real
            grid_step_mesh = (
                grid_step[..., None, None].expand(batchsize, 2, size_x, size_y) * 10
            )  # 0.05 um -> 0.5 for statistical stability # [bs, 2, h, w] real
            return torch.cat(
                [wavelength_map, grid_step_mesh], dim=1
            )  # [bs, 3, h, w] real

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def reset_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def set_linear_probing_mode(self, mode: bool = True):
        self.linear_probing_mode = mode
