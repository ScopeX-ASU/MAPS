import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Device
import torch.utils.checkpoint as checkpoint


class Kernel:
    def __init__(self):
        self.optKernels = self.getOptKernels()
        self.comboKernels = self.getComboKernels()

    def getOptKernels(self):
        kernel_head = np.load("./core/inv_litho/tcc/optKernel.npy")

        nku = 24
        kernel_head = kernel_head[:, :nku]

        kernel_scale = np.load("./core/inv_litho/tcc/optKernel_scale.npy")
        kernel_scale = kernel_scale[:, :nku]
        a, b = kernel_scale.shape
        kernel_scale = kernel_scale.reshape(a, b, 1, 1)
        return {"kernel_head": kernel_head, "kernel_scale": kernel_scale}

    def getComboKernels(self):
        kernel_head = np.load("./core/inv_litho/tcc/comboOptKernel.npy")
        nku = 9
        kernel_head = kernel_head[:, nku - 1 : nku]
        kernel_scale = np.array([[1], [1], [1], [1]])
        return {"kernel_head": kernel_head, "kernel_scale": kernel_scale}


def get_kernel():
    litho_kernel = Kernel()
    kernels = litho_kernel.optKernels
    kernels_focus = {
        "kernel_head": kernels["kernel_head"][0],
        "kernel_scale": kernels["kernel_scale"][0],
    }
    kernels_fft_focus = kernels_focus["kernel_head"]  # .get()
    kernels_scale_focus = kernels_focus["kernel_scale"]  # .get()

    kernels_defocus = {
        "kernel_head": kernels["kernel_head"][1],
        "kernel_scale": kernels["kernel_scale"][1],
    }
    kernels_fft_defocus = kernels_defocus["kernel_head"]  # .get()
    kernels_scale_defocus = kernels_defocus["kernel_scale"]  # .get()
    # print(kernels_fft_focus.shape, kernels_fft_defocus.shape)

    return (
        kernels_fft_focus,
        kernels_fft_defocus,
        kernels_scale_focus,
        kernels_scale_defocus,
    )


def get_binary(im, th=0.5):
    im[im >= 0.5] = 1.0
    im[im < 0.5] = 0.0

    return im


def l2_loss(x, y):
    return torch.sum(torch.pow((x - y), 2))


class litho_model(nn.Module):
    """
    This lithography model is reimplemented from the following paper:
    https://arxiv.org/abs/2411.07311
    """

    def __init__(
        self,
        target_img_shape,
        mask_steepness=5,
        resist_th=0.225,
        resist_steepness=20,
        mask_shift=0.5,
        max_dose=1.02,
        min_dose=0.98,
        avepool_kernel=3,
        device: Device = torch.device("cuda:0"),
    ):
        super(litho_model, self).__init__()

        self.device = device

        self.mask_dim1, self.mask_dim2 = target_img_shape
        self.fo, self.defo, self.fo_scale, self.defo_scale = get_kernel()
        self.kernel_focus = torch.tensor(self.fo).to(device)
        self.kernel_focus_scale = torch.tensor(self.fo_scale).to(device)
        self.kernel_defocus = torch.tensor(self.defo).to(device)
        self.kernel_defocus_scale = torch.tensor(self.defo_scale).to(device)
        self.kernel_num, self.kernel_dim1, self.kernel_dim2 = self.fo.shape  # 24 35 35
        self.offset = self.mask_dim1 // 2 - self.kernel_dim1 // 2
        self.max_dose = max_dose
        self.min_dose = min_dose
        self.resist_steepness = resist_steepness
        self.mask_steepness = mask_steepness
        self.resist_th = resist_th
        self.mask_shift = mask_shift
        self.mask_dim1_s = self.mask_dim1
        self.mask_dim2_s = self.mask_dim2

        self.avepool_lres = nn.AvgPool2d(
            kernel_size=avepool_kernel, stride=1, padding=avepool_kernel // 2
        )
        self._relu_lres = nn.LeakyReLU()

        self.ambit = 155
        self.ambit_s = self.ambit
        # for large tiles
        self.base_dim = 620  # 2um
        self.base_offset = self.base_dim // 2 - self.kernel_dim1 // 2
        self.base_dim_s = self.base_dim
        self.base_offset_s = self.base_dim_s // 2 - self.kernel_dim1 // 2
        self.num_of_tiles = (self.mask_dim1 - self.base_dim) * 2 // self.base_dim + 1

        self.iter = 0

    def forward_base(self, x):
        mask = x
        # print(mask.shape)
        n, _, _, _ = mask.shape  # [1, 1, 620, 620]
        mask = self.avepool_lres(mask)  # ----> line 11 in alg.1 DAC'23

        mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask))[
            :,
            :,
            self.base_offset_s : self.base_offset_s + self.kernel_dim1,
            self.base_offset_s : self.base_offset_s + self.kernel_dim2,
        ]  # [1, 1, 620, 620] -> [1, 1, 35, 35]

        self.i_mask_fft = mask_fft

        x_out_ifft = (
            torch.fft.ifft2(
                mask_fft * self.kernel_focus, s=(self.base_dim_s, self.base_dim_s)
            )
            .abs()
            .square()
        )

        x_out = torch.sigmoid(
            F.conv2d(
                x_out_ifft,
                self.resist_steepness * self.kernel_focus_scale.unsqueeze(0),
                bias=torch.tensor(
                    [
                        -self.resist_steepness * self.resist_th,
                    ],
                    device=self.device,
                ),
                stride=1,
                padding=0,
            )
        )

        # [1, 24 ,620, 620] dot [1, 24, 1, 1] -> [1, 1, 620, 620]
        x_out_max = torch.sigmoid(
            F.conv2d(
                x_out_ifft,
                self.resist_steepness
                * self.max_dose**2
                * self.kernel_focus_scale.unsqueeze(0),
                bias=torch.tensor(
                    [
                        -self.resist_steepness * self.resist_th,
                    ],
                    device=self.device,
                ),
                stride=1,
                padding=0,
            )
        )

        x_out_min = (
            torch.fft.ifft2(
                mask_fft * self.min_dose * self.kernel_defocus,
                s=(self.base_dim_s, self.base_dim_s),
            )
            .abs()
            .square()
        )

        x_out_min = torch.sigmoid(
            F.conv2d(
                x_out_min,
                self.resist_steepness * self.kernel_defocus_scale.unsqueeze(0),
                bias=torch.tensor(
                    [
                        -self.resist_steepness * self.resist_th,
                    ],
                    device=self.device,
                ),
                stride=1,
                padding=0,
            )
        )
        # x_out_min = x_out_min * self.kernel_defocus_scale
        # x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
        # x_out_min = torch.sigmoid(self.resist_steepness * (x_out_min - self.resist_th))

        return x_out, x_out_max, x_out_min

    def tile2batch(self, x):
        return self._tile2batch(x)

    def batch2tile(self, x):
        return self._batch2tile(x)

    def forward_batch(self, batch_size=1, target_img=None):
        target_image_s = target_img.view(1, 1, self.mask_dim1, self.mask_dim2).to(
            self.device
        )

        mask_s = target_image_s

        mask_batch = self._tile2batch(mask_s)
        all_size, c, h, w = mask_batch.shape
        # print(all_size,c,h,w)
        x_out_batch_list = []
        x_out_max_batch_list = []
        x_out_min_batch_list = []
        for b in range(math.ceil(1.0 * all_size / batch_size)):
            # print("Processing Batch %g:  %g--->%g"%(b, b*batch_size, min((b+1)*batch_size, all_size)))

            batch = mask_batch[b * batch_size : min((b + 1) * batch_size, all_size)]

            # Wrap the forward_base call with checkpointing
            def forward_base_wrapped(batch):
                return self.forward_base(batch)

            # Use checkpointing to save memory
            x_out_batch, x_out_max_batch, x_out_min_batch = checkpoint.checkpoint(
                forward_base_wrapped, batch
            )
            x_out_batch_list.append(x_out_batch)
            x_out_max_batch_list.append(x_out_max_batch)
            x_out_min_batch_list.append(x_out_min_batch)
        x_out_batch = torch.cat(x_out_batch_list, dim=0)
        x_out_max_batch = torch.cat(x_out_max_batch_list, dim=0)
        x_out_min_batch = torch.cat(x_out_min_batch_list, dim=0)

        x_out = self._batch2tile(x_out_batch)
        x_out_max = self._batch2tile(x_out_max_batch)
        x_out_min = self._batch2tile(x_out_min_batch)

        return x_out, x_out_max, x_out_min

    def forward_batch_test(self, batch_size=1):
        mask = self.mask_s.data

        mask_batch = self._tile2batch(mask)
        all_size, c, h, w = mask_batch.shape
        for b in range(math.ceil(1.0 * all_size / batch_size)):
            batch = mask_batch[b * batch_size : min((b + 1) * batch_size, all_size)]
            if b == 0:
                x_out_batch, x_out_max_batch, x_out_min_batch = self.forward_base(batch)
            else:
                t_x_out_batch, t_x_out_max_batch, t_x_out_min_batch = self.forward_base(
                    batch
                )
                x_out_batch = torch.cat((x_out_batch, t_x_out_batch), dim=0)
                x_out_max_batch = torch.cat((x_out_max_batch, t_x_out_max_batch), dim=0)
                x_out_min_batch = torch.cat((x_out_min_batch, t_x_out_min_batch), dim=0)

        x_out = self._batch2tile(x_out_batch)
        x_out_max = self._batch2tile(x_out_max_batch)
        x_out_min = self._batch2tile(x_out_min_batch)

        return mask, x_out, x_out_max, x_out_min

    def forward_base_test(self, x):
        mask = x
        n, _, _, _ = mask.shape

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask))
        mask_fft = torch.repeat_interleave(mask_fft, self.kernel_num, 1)
        mask_fft_max = mask_fft * self.max_dose
        mask_fft_min = mask_fft * self.min_dose
        self.i_mask_fft = mask_fft
        x_out = torch.view_as_complex(
            torch.zeros(
                (n, self.kernel_num, self.base_dim_s, self.base_dim_s, 2),
                dtype=torch.float32,
            ).cuda()
        )
        x_out_max = torch.view_as_complex(
            torch.zeros(
                (n, self.kernel_num, self.base_dim_s, self.base_dim_s, 2),
                dtype=torch.float32,
            ).cuda()
        )
        x_out_min = torch.view_as_complex(
            torch.zeros(
                (n, self.kernel_num, self.base_dim_s, self.base_dim_s, 2),
                dtype=torch.float32,
            ).cuda()
        )

        x_out[
            :,
            :,
            self.base_offset_s : self.base_offset_s + self.kernel_dim1,
            self.base_offset_s : self.base_offset_s + self.kernel_dim2,
        ] = (
            mask_fft[
                :,
                :,
                self.base_offset_s : self.base_offset_s + self.kernel_dim1,
                self.base_offset_s : self.base_offset_s + self.kernel_dim2,
            ]
            * self.kernel_focus
        )
        x_out = torch.fft.ifft2(x_out)
        x_out = x_out.real * x_out.real + x_out.imag * x_out.imag
        x_out = x_out * self.kernel_focus_scale
        x_out = torch.sum(x_out, axis=1, keepdims=True)
        # x_out = torch.sigmoid(self.resist_steepness*(x_out-self.resist_th))

        x_out_max[
            :,
            :,
            self.base_offset_s : self.base_offset_s + self.kernel_dim1,
            self.base_offset_s : self.base_offset_s + self.kernel_dim2,
        ] = (
            mask_fft_max[
                :,
                :,
                self.base_offset_s : self.base_offset_s + self.kernel_dim1,
                self.base_offset_s : self.base_offset_s + self.kernel_dim2,
            ]
            * self.kernel_focus
        )
        x_out_max = torch.fft.ifft2(x_out_max)
        x_out_max = x_out_max.real * x_out_max.real + x_out_max.imag * x_out_max.imag
        x_out_max = x_out_max * self.kernel_focus_scale
        x_out_max = torch.sum(x_out_max, axis=1, keepdims=True)
        # x_out_max = torch.sigmoid(self.resist_steepness*(x_out_max-self.resist_th))

        x_out_min[
            :,
            :,
            self.base_offset_s : self.base_offset_s + self.kernel_dim1,
            self.base_offset_s : self.base_offset_s + self.kernel_dim2,
        ] = (
            mask_fft_min[
                :,
                :,
                self.base_offset_s : self.base_offset_s + self.kernel_dim1,
                self.base_offset_s : self.base_offset_s + self.kernel_dim2,
            ]
            * self.kernel_defocus
        )
        x_out_min = torch.fft.ifft2(x_out_min)
        x_out_min = x_out_min.real * x_out_min.real + x_out_min.imag * x_out_min.imag
        x_out_min = x_out_min * self.kernel_defocus_scale
        x_out_min = torch.sum(x_out_min, axis=1, keepdims=True)
        # x_out_min = torch.sigmoid(self.resist_steepness*(x_out_min-self.resist_th))

        return x_out, x_out_max, x_out_min

    def _tile2batch(self, x):
        x = F.unfold(x, kernel_size=self.base_dim_s, stride=self.base_dim_s // 2)
        return x.view(1, self.base_dim_s, self.base_dim_s, -1).moveaxis(3, 0)

    def _batch2tile(self, x):
        # y = x
        n, c, h, w = x.shape
        x = x[
            ..., self.ambit_s : -self.ambit_s, self.ambit_s : -self.ambit_s
        ]  # [batch, 1, 310, 310]
        ## we need to feed [1, h*w, n]
        x = F.fold(
            x.permute(1, 2, 3, 0).flatten(1, 2),  # [1, h*w, n]
            kernel_size=x.shape[-1],
            stride=x.shape[-1],
            output_size=(
                self.mask_dim1_s - 2 * self.ambit_s,
                self.mask_dim2_s - 2 * self.ambit_s,
            ),
        )
        x = F.pad(
            x, (self.ambit_s, self.ambit_s, self.ambit_s, self.ambit_s), "constant"
        )

        return x
