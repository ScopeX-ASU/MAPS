import torch
import torch.nn.functional as F
from torch import nn
from mmengine.registry import MODELS

__all__ = ["MyLayerNorm", "preNorm", "postNorm"]


@MODELS.register_module()
class MyLayerNorm(nn.Module):
    r"""LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(
        self,
        normalized_shape,
        eps=1e-6,  # TODO use small dataset to find if -6 is a good order of magnitude
        data_format="channels_last",
        reshape_last_to_first=False,
        interpolate=False,
        is_linear: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        if isinstance(eps, str):
            self.eps = eval(eps)
        else:
            self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first
        self.interpolate = interpolate

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            # u = x.mean(1, keepdim=True)
            # s = (x - u).pow(2).mean(1, keepdim=True)
            s, u = torch.var_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - u) / torch.sqrt(s + self.eps)
            if self.interpolate:
                weight = self.weight.unsqueeze(0).unsqueeze(0)
                weight = F.interpolate(
                    weight, size=x.shape[1], mode="linear", align_corners=True
                )
                weight = weight.squeeze()
                bias = self.bias.unsqueeze(0).unsqueeze(0)
                bias = F.interpolate(
                    bias, size=x.shape[1], mode="linear", align_corners=True
                )
                bias = bias.squeeze()
            else:
                weight = self.weight
                bias = self.bias
            if len(x.shape) == 4:
                x = weight[:, None, None] * x + bias[:, None, None]
            elif len(x.shape) == 5:
                x = weight[:, None, None, None] * x + bias[:, None, None, None]
            elif len(x.shape) == 2:
                x = weight[:] * x + bias[:]
            return x

    def extra_repr(self) -> str:
        s = "eps=" + str(self.eps)
        s += ", data_format=" + str(self.data_format)
        return s


@MODELS.register_module()
class preNorm(nn.Module):
    def __init__(
        self,
        eps=1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.1]))
        self.eps = eps

    def forward(self, x):
        """
        suppose the input std = 0.8 I hope it could output a distribution with std = 0.7. 0.8 --> 0.7
        I calculate std/Weight, and then use this std/Weight to normalize the input (x-u)
        the normalized R.V. will always have std = W lets say = 0.4 so in this iteration, the scaling factor = 2
        the network output will be different from x for example 0.34
        the loss function should be mse(0.34, target/scaling_factor = 0.35)
        Use 2 to put output back to the video
        ground truth std = 0.7 and we want our model to predict a distribution with std = 0.7 while now the prediciton std is 0.68, 0.8 --> 0.68

        the predicted result with std ~ 0.68, put it back to the input of the model, the GT output std should be 0.7*7/8 = 0.6125. 0.72 --> 0.6125
        the model will first normalized the input to dist (0.0, 0.4)
        the model will out put a value around 0.34
        in this iteration, the scaling factor = 0.68/0.4 = 1.7
        mse ask the output to mactch mse(0.34, 0.6125/1.7 = 0.3603)
        the ground T target should be 0.7*7/8 = 0.6125 while the model output is 0.34 * 1.7 = 0.578. 0.68 --> 0.578

        """
        s, u = torch.std_mean(x, dim=(1, 2, 3), keepdim=True)
        s = s + self.eps
        normalized_field = (x - u) / s
        weight = self.weight
        # x = weight[:, None, None] * x + bias[:, None, None]
        normalized_field = weight[:, None, None] * normalized_field
        return normalized_field, s / weight[:, None, None]


@MODELS.register_module()
class postNorm(nn.Module):
    def __init__(
        self,
        eps=1e-6,
    ):
        super().__init__()
        # self.weight = nn.Parameter(torch.tensor([0.1]))
        self.eps = eps

    def forward(self, x):
        """
        the postNorm is used to normalized the output pattern using the std
        after this postNorm, the model output will always have the same std
        considering that the input is also normalized to a std
        the model is now mapping a distribution with std_in to a distribution with std_out
        std_in --> std_out

        the postNorm uses (x/std_out) * weight_out to normalize the output
        we will calculate the mse(x/std_out*weight_out, target/std_target*weight_out) to make the model to learn the pattern,
        beacuse now the two distribution have the same std ~ weight_out
        suppose that we can learn an accurate std_pred~std_target, then we can scale the output to the actual video by *std_pred/weight_out

        if std_pred is slightly different with std_target, the model should be more robust compared to former model
        the error that flows to the next iteration is mainly from the pattern error and the std_pred error
        """
        s, u = torch.std_mean(x, dim=(1, 2, 3), keepdim=True)
        s = s + self.eps
        normalized_field = (x - u) / (
            15 * s
        )  ## TODO not quite sure about if I could substrct the mean.
        return normalized_field
