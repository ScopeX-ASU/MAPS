import torch
from pyutils.general import print_stat
from torch import Tensor, nn

from .utils import differentiable_boundary

__all__ = ["DiffBdry", "multi_diff_bdry", "SmoothRidge"]


def check_for_nan(tensor, label):
    """Print tensor statistics and report whether NaN values are present.

    Args:
        tensor (torch.Tensor): Tensor to inspect. Shape and dtype are unchanged.
        label (str): Human-readable label used in the diagnostic message.
    """
    print_stat(tensor)
    if torch.isnan(tensor).any():
        print(f"{label} contains NaN values")


class DiffBdry(nn.Module):
    """Apply a differentiable boundary multiplier along one spatial axis.

    Args:
        total_length (float): Physical extent used by the underlying
            ``differentiable_boundary`` operator.
    """

    def __init__(self, total_length: float) -> None:
        """Initialize the differentiable-boundary module.

        Args:
            total_length (float): Physical axis extent passed to the boundary operator.
        """
        super().__init__()
        self.total_length = total_length

    def forward(self, axis_tensor, w, temp) -> Tensor:
        """Evaluate the differentiable boundary multiplier.

        Args:
            axis_tensor (torch.Tensor): Coordinate tensor along the boundary axis.
            w (float | Tensor): Boundary-width or shape parameter.
            temp (float | Tensor): Smoothness/temperature parameter.

        Returns:
            Tensor: Differentiable boundary multiplier with shape broadcast-compatible
            with ``axis_tensor``.
        """
        axis_multiplier = differentiable_boundary.apply(
            axis_tensor, w, self.total_length, temp
        )
        return axis_multiplier

    def extra_repr(self) -> str:
        return f"total length={self.total_length}"


class SmoothRidge(nn.Module):
    """Apply a smooth logistic ridge mask for left, middle, or right regions.

    Args:
        total_length (float): Physical coordinate span used to normalize the
            transition expressions.
    """

    def __init__(self, total_length: float) -> None:
        """Initialize the smooth ridge module.

        Args:
            total_length (float): Physical coordinate span used by the formulas.
        """
        super().__init__()
        self.total_length = total_length

    def forward(self, x, center, width, sharpness, region):
        """Evaluate the selected smooth ridge expression.

        Args:
            x (torch.Tensor): Coordinate tensor. The output has the same shape.
            center (float | Tensor): Center coordinate of the middle ridge region.
            width (float | Tensor): Middle-region width in the same physical units as ``x``.
            sharpness (float | Tensor): Transition steepness parameter.
            region (str): Region selector. Supported values are ``"left"``, ``"middle"``,
                and ``"right"``.

        Returns:
            Tensor: Smooth mask tensor produced by the selected logistic expression.
        """
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
    """Apply differentiable logistic boundary masks across ordered segments.

    Args:
        total_height (float): Total physical extent used to scale segment transitions.
    """

    def __init__(self, total_height: float) -> None:
        """Initialize the multi-segment differentiable-boundary module.

        Args:
            total_height (float): Physical extent used in the transition scaling formulas.
        """
        super(multi_diff_bdry, self).__init__()
        self.total_height = total_height

    def forward(self, x, segment_widths, segment_types, Temp):
        """Evaluate piecewise smooth boundary masks across multiple segments.

        Args:
            x (torch.Tensor | ArrayLike): Coordinate tensor; non-tensor inputs are
                converted with ``torch.tensor``.
            segment_widths (torch.Tensor): One-dimensional ordered physical widths.
            segment_types (Sequence[int]): Segment-type sequence aligned with
                ``segment_widths``; ``0`` selects the air expression and ``1`` selects
                the media expression.
            Temp (float | Tensor): Smoothness/temperature parameter used in the logistic exponents.

        Returns:
            Tensor: Piecewise smooth boundary mask with the same shape as ``x``.

        Notes:
            The implementation prints the computed cumulative segment boundaries and
            uses a final fallback expression for values at or beyond the last boundary.
        """
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
