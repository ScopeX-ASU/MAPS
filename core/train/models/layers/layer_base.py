import inspect
from typing import Dict, Optional, Union

import torch
from mmcv.cnn.bricks import build_activation_layer, build_conv_layer, build_norm_layer
from mmengine.registry import MODELS
from torch import Tensor, nn
from torch.types import Device, _size

__all__ = [
    "LinearBlock",
    "ConvBlock",
    "build_linear_layer",
    "build_conv_layer",
    "build_norm_layer",
    "build_activation_layer",
]


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
            f"Cannot find {linear_layer} in registry under scope name {registry.scope}"
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
            self.norm = build_norm_layer(norm_cfg, out_features)
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
        bias: bool = False,
        conv_cfg: dict = dict(type="Conv2d"),
        norm_cfg: dict | None = dict(type="BN", affine=True),
        act_cfg: dict | None = dict(type="ReLU", inplace=True),
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
