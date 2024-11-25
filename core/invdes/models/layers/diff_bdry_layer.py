import torch
from pyutils.general import print_stat
from torch import Tensor, nn

from .utils import differentiable_boundary

__all__ = ["DiffBdry", "multi_diff_bdry", "SmoothRidge"]


def check_for_nan(tensor, label):
    print_stat(tensor)
    if torch.isnan(tensor).any():
        print(f"{label} contains NaN values")


class DiffBdry(nn.Module):
    def __init__(self, total_length: float) -> None:
        super().__init__()
        self.total_length = total_length

    def forward(self, axis_tensor, w, temp) -> Tensor:
        axis_multiplier = differentiable_boundary.apply(
            axis_tensor, w, self.total_length, temp
        )
        return axis_multiplier

    def extra_repr(self) -> str:
        return f"total length={self.total_length}"


class SmoothRidge(nn.Module):
    def __init__(self, total_length: float) -> None:
        super().__init__()
        self.total_length = total_length

    def forward(self, x, center, width, sharpness, region):
        w_left = self.total_length + center - width / 2
        w_right = -center - width / 2
        width = width
        if region == "left":
            output = 1 / (
                torch.exp(
                    torch.clamp(
                        -(((x + self.total_length) ** 2 - (w_left) ** 2) * sharpness)
                        * (self.total_length / (3 * w_left)) ** 2,
                        max=10,
                    )
                )
                + 1
            )
        elif region == "middle":
            output = 1 / (
                torch.exp(
                    (((x - center) ** 2 - (width / 2) ** 2) * sharpness)
                    * (self.total_length / (3 * width)) ** 2
                )
                + 1
            )
        elif region == "right":
            output = 1 / (
                torch.exp(
                    torch.clamp(
                        -((x**2 - (w_right) ** 2) * sharpness)
                        * (self.total_length / (3 * w_right)) ** 2,
                        max=10,
                    )
                )
                + 1
            )
        # print("this is the w_left: ", w_left)
        # print("this is the w_right: ", w_right)
        # print("this is the width: ", width)
        # check_for_nan(x, "input")
        # check_for_nan(torch.exp(-(((x+self.total_length)**2 - (w_left)**2) * sharpness) * (self.total_length / (3*w_left))**2), "left")
        # check_for_nan(torch.exp((((x-center)**2 - (width/2)**2) * sharpness) * (self.total_length / (3*width)) ** 2), "middle")
        # check_for_nan(torch.exp(-((x**2 - (w_right)**2) * sharpness) * (self.total_length / (3*w_right))**2), "right")
        # output = torch.where(
        #     x < -width/2 + center,
        #     1/(torch.clamp(torch.exp(-(((x+self.total_length)**2 - (w_left)**2) * sharpness) * (self.total_length / (3*w_left))**2), max=10000) + 1),
        #     torch.where(
        #         x < width/2 + center,
        #         1/(torch.clamp(torch.exp((((x-center)**2 - (width/2)**2) * sharpness) * (self.total_length / (3*width)) ** 2), max=10000) + 1),
        #         1/(torch.clamp(torch.exp(-((x**2 - (w_right)**2) * sharpness) * (self.total_length / (3*w_right))**2), max=10000) + 1),
        #     )
        # )
        return output


class multi_diff_bdry(nn.Module):
    def __init__(self, total_height: float) -> None:
        super(multi_diff_bdry, self).__init__()
        self.total_height = total_height

    def forward(self, x, segment_widths, segment_types, Temp):
        # 确保输入 x 是一个 torch.Tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        # 初始化结果张量
        result = torch.zeros_like(x)

        # 计算每个段的分界点
        segment_boundaries = torch.cumsum(segment_widths, dim=0)
        print("this is the segment_boundaries: ", segment_boundaries)

        # 迭代处理每个分段
        current_start = torch.tensor(0.0)  # 起始边界
        for i, (boundary, segment_type) in enumerate(
            zip(segment_boundaries, segment_types)
        ):
            in_segment = (x >= current_start) & (x < boundary)
            middle_point = (current_start + boundary) / 2

            if i == 0:
                # 根据 segment_type 选择不同的函数
                if segment_type == 0:
                    result = torch.where(
                        in_segment,
                        1
                        / (
                            1
                            + torch.exp(
                                -((x**2 - segment_widths[i] ** 2) / Temp)
                                * (self.total_height / segment_widths[i]) ** 2
                            )
                        ),
                        result,
                    )  # type 0: air
                elif segment_type == 1:
                    result = torch.where(
                        in_segment,
                        1
                        / (
                            1
                            + torch.exp(
                                ((x**2 - segment_widths[i] ** 2) / Temp)
                                * (self.total_height / segment_widths[i]) ** 2
                            )
                        ),
                        result,
                    )  # type 0: media
            else:
                # 根据 segment_type 选择不同的函数
                if segment_type == 0:
                    result = torch.where(
                        in_segment,
                        1
                        / (
                            1
                            + torch.exp(
                                -(
                                    (
                                        (x - middle_point) ** 2
                                        - (segment_widths[i] / 2) ** 2
                                    )
                                    / Temp
                                )
                                * (self.total_height / segment_widths[i]) ** 2
                            )
                        ),
                        result,
                    )  # type 0: air
                elif segment_type == 1:
                    result = torch.where(
                        in_segment,
                        1
                        / (
                            1
                            + torch.exp(
                                +(
                                    (
                                        (x - middle_point) ** 2
                                        - (segment_widths[i] / 2) ** 2
                                    )
                                    / Temp
                                )
                                * (self.total_height / segment_widths[i]) ** 2
                            )
                        ),
                        result,
                    )  # type 0: media

            # 更新起始边界
            current_start = boundary

        # 处理最后一个区间右端点
        result = torch.where(
            x >= current_start, torch.sin(x) if segment_type == 1 else x**2, result
        )

        return result
