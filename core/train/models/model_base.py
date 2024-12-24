import inspect
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.registry import MODELS
from torch import Tensor
from torch.types import Device, _size
from pyutils.torch_train import set_torch_deterministic
from traitlets import default
import copy
__all__ = [
    "LinearBlock",
    "ConvBlock",
    "LayerBlock",
    "ModelBase",
]

MODELS.register_module(name="Linear", module=nn.Linear)


def build_linear_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    if cfg is None:
        cfg_ = dict(type="Linear")
    else:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if "type" not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop("type")
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `linear_layer` cannot be found
    # in the registry, fallback to search `linear_layer` in the
    # mmengine.MODELS.
    with MODELS.switch_scope_and_registry(None) as registry:
        linear_layer = registry.get(layer_type)
    if linear_layer is None:
        raise KeyError(
            f"Cannot find {linear_layer} in registry under scope "
            f"name {registry.scope}"
        )
    layer = linear_layer(*args, **kwargs, **cfg_)

    return layer


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        linear_cfg: dict = dict(type="Linear"),
        norm_cfg: dict | None = None,
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
        dropout: float = 0.0,
        device: Device = torch.device("cuda:0"),
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        if linear_cfg["type"] not in {"Linear", None}:
            linear_cfg.update({"device": device})
        self.linear = build_linear_layer(
            linear_cfg,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        if norm_cfg is not None:
            normalization_cfg = copy.deepcopy(norm_cfg)
            normalization_cfg["dim"] = 1
            _, self.norm = build_norm_layer(normalization_cfg, out_features)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

    def forward(self, x: Tensor) -> Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: Union[int, _size] = 1,
        padding: Union[int, _size] = 0,
        dilation: Union[int, _size] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_cfg: dict = dict(type="Conv2d", padding_mode="replicate"),
        norm_cfg: dict | None = dict(
            type="LayerNorm", eps=1e-6, data_format="channels_first"
        ),
        act_cfg: dict | None = dict(type="GELU"),
        skip: bool = False,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        conv_cfg = conv_cfg.copy()
        if conv_cfg["type"] not in {"Conv2d", None}:
            conv_cfg.update({"device": device})
        self.conv = build_conv_layer(
            conv_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

        self.skip = skip

    def forward(self, x: Tensor) -> Tensor:
        y = x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.skip:
            x += y
        return x


class LayerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *args,
        layer_cfg: dict = dict(type="Conv2d", padding_mode="replicate"),
        norm_cfg: dict = dict(
            type="MyLayerNorm", eps=1e-6, data_format="channels_first"
        ),
        act_cfg: dict = dict(type="GELU"),
        skip: bool = False,
        device: Device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ) -> None:
        super().__init__()
        self.layer = build_conv_layer(
            layer_cfg,
            in_channels=in_channels,
            out_channels=out_channels,
            *args,
        )
        if norm_cfg is not None:
            _, self.norm = build_norm_layer(norm_cfg, out_channels)
        else:
            self.norm = None

        if act_cfg is not None:
            self.activation = build_activation_layer(act_cfg)
        else:
            self.activation = None

        self.skip = skip

    def forward(self, x: Tensor) -> Tensor:
        y = x = self.layer(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)

        if self.skip:
            x += y
        return x


class ModelBase(nn.Module):
    # default_cfgs = dict(
    #     conv_cfg=dict(type="Conv2d", padding_mode="replicate"),
    #     layer_cfg=dict(type="Conv2d", padding_mode="replicate"),
    #     linear_cfg=dict(type="Linear"),
    #     norm_cfg=dict(type="BN", affine=True),
    #     act_cfg=dict(type="ReLU", inplace=True),
    #     device=torch.device("cpu"),
    # )
    default_cfgs = dict()

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        # self.load_cfgs(**kwargs)

    def load_cfgs(self, **cfgs):
        # Start with default configurations
        self.__dict__.update(self.default_cfgs)
        # Update with provided configurations
        self.__dict__.update(cfgs)

    def reset_parameters(self, *args, random_state: int = None, **kwargs):
        for name, m in self.named_modules():
            if random_state is not None:
                # deterministic seed, but different for different layer, and controllable by random_state
                set_torch_deterministic(random_state + sum(map(ord, name)))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
