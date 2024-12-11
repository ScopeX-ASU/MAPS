#############################
# NVIDIA  All Rights Reserved
# Haoyu Yang
# Design Automation Research
# Last Update: March 29 2024
#############################


import math

import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.morphology import closing, opening
from torch.types import Device
import torch.utils.checkpoint as checkpoint

__all__ = ["My_nvilt2"]
class Kernel:
    def __init__(self):
        self.optKernels = self.getOptKernels()
        self.comboKernels = self.getComboKernels()

    def getOptKernels(self):
        kernel_head = np.load("./tcc/optKernel.npy")

        nku = 24
        kernel_head = kernel_head[:, :nku]

        kernel_scale = np.load("./tcc/optKernel_scale.npy")
        kernel_scale = kernel_scale[:, :nku]
        a, b = kernel_scale.shape
        kernel_scale = kernel_scale.reshape(a, b, 1, 1)
        return {"kernel_head": kernel_head, "kernel_scale": kernel_scale}

    def getComboKernels(self):
        kernel_head = np.load("./tcc/comboOptKernel.npy")
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


def target_smoothing(im, iteration=2, kernel_size=40):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    out = cv2.dilate(im, kernel)
    out = cv2.erode(out, kernel, iterations=iteration)
    out = cv2.dilate(out, kernel)

    return out


def get_binary(im, th=0.5):
    im[im >= 0.5] = 1.0
    im[im < 0.5] = 0.0

    return im


def _rfft2_to_fft2_pytorch(rfft):  # H X W -> H X H
    h, w = rfft.shape

    try:
        w % 2 == 0
    except:
        raise ("only odd dims are allowed")

    result = torch.view_as_complex(torch.zeros((h, w, 2)).cuda())

    result[:, :w] = rfft

    top = rfft[0, 1:]

    result[0, w:] = torch.flip(top, dims=(0,)).conj()
    mid = rfft[1:, 1:]
    mid = torch.flip(mid, dims=(0, 1)).conj()

    result[1:, w:] = mid

    return result


def _rfft2_to_fft2(im_shape, rfft):
    fcols = im_shape[-1]
    fft_cols = rfft.shape[-1]

    result = np.zeros(im_shape, dtype=rfft.dtype)

    result[:, :fft_cols] = rfft

    top = rfft[0, 1:]

    if fcols % 2 == 0:
        result[0, fft_cols - 1 :] = top[::-1].conj()
        mid = rfft[1:, 1:]
        mid = np.hstack((mid, mid[::-1, ::-1][:, 1:].conj()))
    else:
        result[0, fft_cols:] = top[::-1].conj()
        mid = rfft[1:, 1:]
        mid = np.hstack((mid, mid[::-1, ::-1].conj()))

    result[1:, 1:] = mid

    return result


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        pseudo_y_norms = (
            torch.ones(num_examples).type("torch.cuda.FloatTensor") * 250000.0
        )
        y_norms = torch.where(y_norms == 0, pseudo_y_norms, y_norms)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class maskpooling(nn.Module):
    def __init__(self, kernel=3):
        super(maskpooling, self).__init__()
        self.kernel = kernel
        self.max_kernel = 4
        self.avepool_lres = nn.AvgPool2d(
            kernel_size=self.kernel, stride=1, padding=self.kernel // 2
        )
        self.maxpool_lres = nn.MaxPool2d(
            kernel_size=self.max_kernel, stride=self.max_kernel
        )
        self.test_pool = nn.AvgPool2d(
            kernel_size=self.max_kernel, stride=self.max_kernel
        )
        self.gaussian = T.GaussianBlur(kernel, sigma=0.1)

    def forward(self, x):
        # x= self.test_pool(x)
        # x= nn.functional.interpolate(input=x, scale_factor=self.max_kernel, mode='bilinear')
        # x= self.gaussian(x)
        x = self.avepool_lres(x)
        return x


"""
def sample_mask_freq_1(mask_fft, pixel_size=1): 
    #mask_fft: non shifted fft of the mask image
    #pixel_size: spatial
    #return: interleaved fre, dc shifted to center
    new_mask_fft = torch.zeros_like(mask_fft)
    _,_,h,w=new_mask_fft.shape
    sampled_mask_fft = pixel_size*pixel_size*torch.fft.fftshift(mask_fft)[:,:,::pixel_size,::pixel_size]
    _,_,hh,ww=sampled_mask_fft.shape
    new_mask_fft[:,:,(h-hh)//2:(h-hh)//2+hh,(w-ww)//2:(w-ww)//2+ww]=sampled_mask_fft
    return sampled_mask_fft






def sample_mask_freq_2(mask_e_field, pixel_size=1): 

    sampled_mask_fft = mask_fft[:,:,::pixel_size,::pixel_size]
"""


class nvilt2(nn.Module):
    def __init__(
        self,
        target_path,
        mask_steepness=4,
        resist_th=0.225,
        resist_steepness=50,
        mask_shift=0.5,
        pvb_coefficient=0,
        max_dose=1.02,
        min_dose=0.98,
        avepool_kernel=3,
        morph=0,
        scale_factor=1,
        device: Device = torch.device("cuda:0"),
    ):
        super(nvilt2, self).__init__()
        self.target_image = torch.tensor(cv2.imread(target_path, -1)) / 255.0

        print("this is the target image shape", self.target_image.shape, flush=True)

        self.mask_dim1, self.mask_dim2 = self.target_image.shape
        self.fo, self.defo, self.fo_scale, self.defo_scale = get_kernel()
        # self.mask = nn.Parameter(self.target_image)
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
        self.morph = morph
        # members of DAC'23 FDU
        self.scale_factor = scale_factor
        self.mask_dim1_s = self.mask_dim1 // self.scale_factor
        self.mask_dim2_s = self.mask_dim2 // self.scale_factor
        # self.target_image_s = nn.functional.avg_pool2d(torch.tensor(cv2.imread(target_path, -1)).view(1,1,self.mask_dim1,self.mask_dim2).cuda()/255.0, self.scale_factor)
        self.target_image_s = nn.functional.avg_pool2d(
            self.target_image.view(1, 1, self.mask_dim1, self.mask_dim2).cuda(),
            self.scale_factor,
        )

        self.mask_s = nn.Parameter(self.target_image_s)

        self.avepool_lres = maskpooling(kernel=avepool_kernel)
        # self._maxpool_lres = nn.MaxPool2d(kernel_size=3, stride=1, padding = self.avepool_kernel//2)
        self._relu_lres = nn.LeakyReLU()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=self.scale_factor)

        self.ambit = 155
        self.ambit_s = self.ambit // self.scale_factor
        # for large tiles
        self.base_dim = 620  # 2um
        self.base_offset = self.base_dim // 2 - self.kernel_dim1 // 2
        self.base_dim_s = self.base_dim // self.scale_factor
        self.base_offset_s = self.base_dim_s // 2 - self.kernel_dim1 // 2
        self.num_of_tiles = (self.mask_dim1 - self.base_dim) * 2 // self.base_dim + 1

        if self.morph > 0:
            self.morph_kernel_opt_opening = torch.tensor(
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph)).astype(
                    np.float32
                )
            ).cuda()
            self.morph_kernel_opt_closing = torch.tensor(
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (morph + 2, morph + 2)
                ).astype(np.float32)
            ).cuda()
            self.morph_kernel_opening = torch.tensor(
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, ((morph + 1), (morph + 2))
                ).astype(np.float32)
            ).cuda()
            self.morph_kernel_closing = torch.tensor(
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, ((morph + 3), (morph + 2))
                ).astype(np.float32)
            ).cuda()
            # self.dilate = Dilation2d(in_channels=1,out_channels=1,kernel_size=morph)
            # self.erode  = Erosion2d(in_channels=1,out_channels=1,kernel_size=morph)

        self.iter = 0

        # self.kernel_focus = torch.repeat_interleave(self.kernel_focus, self.num_of_tiles*self.num_of_tiles, 0)
        # self.kernel_defocus = torch.repeat_interleave(self.kernel_defocus, self.num_of_tiles*self.num_of_tiles, 0)

        # self.kernel_focus_scale = torch.repeat_interleave(self.kernel_focus_scale, self.num_of_tiles*self.num_of_tiles, 0)
        # self.kernel_defocus_scale = torch.repeat_interleave(self.kernel_defocus_scale, self.num_of_tiles*self.num_of_tiles, 0)

    def forward_base(self, x):
        mask = x
        # print(mask.shape)
        n, _, _, _ = mask.shape
        mask = self.avepool_lres(mask)  # ----> line 11 in alg.1 DAC'23

        mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))

        if self.morph > 0 and self.iter % 20 == 0 and self.iter > 0:
            mask = opening(mask, self.morph_kernel_opt_opening)
            mask = closing(mask, self.morph_kernel_opt_closing)

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask))
        mask_fft = torch.repeat_interleave(mask_fft, self.kernel_num, 1)
        # print(mask_fft.shape)
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

        # print(x_out.shape, self.kernel_focus.shape, mask_fft.shape)
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
        x_out = torch.sigmoid(self.resist_steepness * (x_out - self.resist_th))

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
        x_out_max = torch.sigmoid(self.resist_steepness * (x_out_max - self.resist_th))

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
        x_out_min = torch.sigmoid(self.resist_steepness * (x_out_min - self.resist_th))

        return x_out, x_out_max, x_out_min

    def tile2batch(self, x):
        return self._tile2batch(x)

    def batch2tile(self, x):
        return self._batch2tile(x)

    def forward_batch(self, batch_size=1):
        mask_batch = self._tile2batch(self.mask_s)
        all_size, c, h, w = mask_batch.shape
        # print(all_size,c,h,w)
        for b in range(math.ceil(1.0 * all_size / batch_size)):
            # print("Processing Batch %g:  %g--->%g"%(b, b*batch_size, min((b+1)*batch_size, all_size)))

            batch = mask_batch[b * batch_size : min((b + 1) * batch_size, all_size)]
            # print(batch.shape)
            if b == 0:
                x_out_batch, x_out_max_batch, x_out_min_batch = self.forward_base(batch)
            else:
                t_x_out_batch, t_x_out_max_batch, t_x_out_min_batch = self.forward_base(
                    batch
                )
                x_out_batch = torch.cat((x_out_batch, t_x_out_batch), dim=0)
                x_out_max_batch = torch.cat((x_out_max_batch, t_x_out_max_batch), dim=0)
                x_out_min_batch = torch.cat((x_out_min_batch, t_x_out_min_batch), dim=0)

        if False:
            id = 0
            debug_mask = mask_batch[id][0, :, :]
            debug_img = x_out_batch[id][0, :, :]

            debug = (
                torch.cat((debug_mask, debug_img), dim=1).cpu().detach().numpy() * 255
            )
            cv2.imwrite("./benchmarks/test_20um/im_via/via1.debug.png", debug)

        x_out = self._batch2tile(x_out_batch)
        x_out_max = self._batch2tile(x_out_max_batch)
        x_out_min = self._batch2tile(x_out_min_batch)

        return self.mask_s, x_out, x_out_max, x_out_min

    def forward_batch_test(self, use_morph=True, batch_size=1):
        mask = self.mask_s.data
        # cmask = self.mask.data

        # mask[self.mask_s.data>=0.5]=1.0
        # mask[self.mask_s.data<0.5]=0.0
        # if self.morph>0 and use_morph:
        #    mask = opening(mask, self.morph_kernel_opt_opening)
        #    mask = closing(mask, self.morph_kernel_opt_closing)
        #    mask = opening(mask, self.morph_kernel_opt_opening)
        # mask[mask>=0.5]=1.0
        # mask[mask<0.5]=0.0

        mask_batch = self._tile2batch(mask)
        all_size, c, h, w = mask_batch.shape
        # print(all_size,c,h,w)
        for b in range(math.ceil(1.0 * all_size / batch_size)):
            # print("Processing Batch %g:  %g--->%g"%(b, b*batch_size, min((b+1)*batch_size, all_size)))

            batch = mask_batch[b * batch_size : min((b + 1) * batch_size, all_size)]
            # print(batch.shape)
            if b == 0:
                x_out_batch, x_out_max_batch, x_out_min_batch = self.forward_base(batch)
            else:
                t_x_out_batch, t_x_out_max_batch, t_x_out_min_batch = self.forward_base(
                    batch
                )
                x_out_batch = torch.cat((x_out_batch, t_x_out_batch), dim=0)
                x_out_max_batch = torch.cat((x_out_max_batch, t_x_out_max_batch), dim=0)
                x_out_min_batch = torch.cat((x_out_min_batch, t_x_out_min_batch), dim=0)

        if False:
            id = 0
            debug_mask = mask_batch[id][0, :, :]
            debug_img = x_out_batch[id][0, :, :]

            debug = (
                torch.cat((debug_mask, debug_img), dim=1).cpu().detach().numpy() * 255
            )
            cv2.imwrite("./benchmarks/test_20um/im_via/via1.debug.png", debug)

        x_out = self._batch2tile(x_out_batch)
        x_out_max = self._batch2tile(x_out_max_batch)
        x_out_min = self._batch2tile(x_out_min_batch)

        return mask, x_out, x_out_max, x_out_min

    # def forward_serial(self,):
    def forward_base_test(self, x):
        mask = x
        # print(mask.shape)
        n, _, _, _ = mask.shape
        # mask = self.avepool_lres(mask) #----> line 11 in alg.1 DAC'23

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask))
        mask_fft = torch.repeat_interleave(mask_fft, self.kernel_num, 1)
        # print(mask_fft.shape)
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

        # print(x_out.shape, self.kernel_focus.shape, mask_fft.shape)
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
        return (
            x.view(1, 1, self.base_dim_s, self.base_dim_s, -1)
            .permute(0, 4, 1, 2, 3)
            .reshape(-1, 1, self.base_dim_s, self.base_dim_s)
        )

    def _batch2tile(self, x):
        y = x
        n, c, h, w = x.shape
        turn_off = torch.zeros_like(x)
        turn_off[:, :, self.ambit_s : -self.ambit_s, self.ambit_s : -self.ambit_s] = 1.0
        x = x * turn_off
        x = x.reshape(1, n, 1, self.base_dim_s, self.base_dim_s)
        x = x.reshape(1, n, -1)
        x = x.permute(0, 2, 1)
        x = F.fold(
            x,
            kernel_size=self.base_dim_s,
            stride=self.base_dim_s // 2,
            output_size=(self.mask_dim1_s, self.mask_dim2_s),
        )

        return x


def l2_loss(x, y):
    return torch.sum(torch.pow((x - y), 2))


class nvilt_engine_2:
    def __init__(
        self,
        image_path,
        avepool_kernel=5,
        morph=0,
        scale_factor=1,
        device: Device = torch.device("cuda:0"),
    ):
        self.image_path = image_path
        self.nvilt = nvilt2(
            target_path=image_path,
            avepool_kernel=avepool_kernel,
            morph=morph,
            scale_factor=scale_factor,
            device=device,
            mask_steepness=5,
            resist_steepness=20,
        ).to(device)
        self.optimizer = torch.optim.SGD(self.nvilt.parameters(), lr=1)
        self.iteration = 0
        self.target = torch.tensor(cv2.imread(image_path, -1)) / 255.0
        self.target_s = torch.tensor(cv2.imread(image_path, -1)) / 255.0
        self.mask_dim1, self.mask_dim2 = self.target.shape
        self.target = self.target.view(1, 1, self.mask_dim1, self.mask_dim2).cuda()
        self.mask_dim1_s = self.mask_dim1 // self.nvilt.scale_factor
        self.mask_dim2_s = self.mask_dim2 // self.nvilt.scale_factor
        self.target_s = nn.functional.avg_pool2d(
            self.target_s.view(1, 1, self.mask_dim1, self.mask_dim2).cuda(),
            self.nvilt.scale_factor,
        )

        self.loss = 0
        self.loss_l2 = 0
        self.loss_pvb = 0
        self.loss_pvb_i = 0
        self.loss_pvb_o = 0
        self.loss_pvb = 0
        self.loss_lowpass_reg = 0

        self.forward_type = (
            0  # 0: mask<->fft<->image  #1: mask->fft<->image (_deprecated)
        )

        self.lowpass_reg_lambda = 1e-3

    def backward_s(self):
        self.loss_l2 = l2_loss(self.outer, self.target_s)
        self.loss_pvb = l2_loss(self.inner, self.outer)
        # self.loss_lowpass_reg = torch.norm(torch.abs(self.i_mask_fft*self.lowpass_mask))
        # self.loss_pvb_o = l2_loss(self.outer, self.target_s)
        # self.real_pvb = l2_loss(self.inner, self.outer)
        self.loss = (
            self.loss_l2 + self.loss_pvb
        )  # + self.lowpass_reg_lambda*self.loss_lowpass_reg

        self.loss.backward()

    def forward_lres(self):
        self.mask = self.nvilt.mask_s.data
        print("this is the mask shape", self.mask.shape)

        _, self.nominal, self.outer, self.inner = self.nvilt.forward_batch()
        print("this is the nominal shape", self.nominal.shape)
        # self.i_mask_fft = self.nvilt.i_mask_fft
        # if self.iteration%10==0:
        #    self.all_image = torch.cat((self.mask, self.nominal, self.outer, self.inner), dim=3).cpu().detach().numpy()[0,0,:,:]*255
        #    cv2.imwrite(self.image_path+".iter%g.png"%self.iteration, self.all_image)
        self.iteration = self.iteration + 1

    def optimize_s(self):
        self.forward_lres()
        self.optimizer.zero_grad()
        self.backward_s()
        self.optimizer.step()


class My_nvilt2(nn.Module):
    def __init__(
        self,
        target_img_shape,
        mask_steepness=5,
        resist_th=0.225,
        resist_steepness=20,
        mask_shift=0.5,
        pvb_coefficient=0,
        max_dose=1.02,
        min_dose=0.98,
        avepool_kernel=3,
        morph=0,
        scale_factor=1,
        device: Device = torch.device("cuda:0"),
    ):
        super(My_nvilt2, self).__init__()
        # self.target_image = torch.tensor(cv2.imread(target_path, -1))/255.0

        self.device = device

        self.mask_dim1, self.mask_dim2 = target_img_shape
        self.fo, self.defo, self.fo_scale, self.defo_scale = get_kernel()
        # self.mask = nn.Parameter(self.target_image)
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
        self.morph = morph
        # members of DAC'23 FDU
        self.scale_factor = scale_factor
        self.mask_dim1_s = self.mask_dim1 // self.scale_factor
        self.mask_dim2_s = self.mask_dim2 // self.scale_factor

        self.avepool_lres = maskpooling(kernel=avepool_kernel)
        # self._maxpool_lres = nn.MaxPool2d(kernel_size=3, stride=1, padding = self.avepool_kernel//2)
        self._relu_lres = nn.LeakyReLU()
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=self.scale_factor)

        self.ambit = 155
        self.ambit_s = self.ambit // self.scale_factor
        # for large tiles
        self.base_dim = 620  # 2um
        self.base_offset = self.base_dim // 2 - self.kernel_dim1 // 2
        self.base_dim_s = self.base_dim // self.scale_factor
        self.base_offset_s = self.base_dim_s // 2 - self.kernel_dim1 // 2
        self.num_of_tiles = (self.mask_dim1 - self.base_dim) * 2 // self.base_dim + 1

        if self.morph > 0:
            self.morph_kernel_opt_opening = torch.tensor(
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph)).astype(
                    np.float32
                )
            ).cuda()
            self.morph_kernel_opt_closing = torch.tensor(
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (morph + 2, morph + 2)
                ).astype(np.float32)
            ).cuda()
            self.morph_kernel_opening = torch.tensor(
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, ((morph + 1), (morph + 2))
                ).astype(np.float32)
            ).cuda()
            self.morph_kernel_closing = torch.tensor(
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, ((morph + 3), (morph + 2))
                ).astype(np.float32)
            ).cuda()
            # self.dilate = Dilation2d(in_channels=1,out_channels=1,kernel_size=morph)
            # self.erode  = Erosion2d(in_channels=1,out_channels=1,kernel_size=morph)

        self.iter = 0

        # self.kernel_focus = torch.repeat_interleave(self.kernel_focus, self.num_of_tiles*self.num_of_tiles, 0)
        # self.kernel_defocus = torch.repeat_interleave(self.kernel_defocus, self.num_of_tiles*self.num_of_tiles, 0)

        # self.kernel_focus_scale = torch.repeat_interleave(self.kernel_focus_scale, self.num_of_tiles*self.num_of_tiles, 0)
        # self.kernel_defocus_scale = torch.repeat_interleave(self.kernel_defocus_scale, self.num_of_tiles*self.num_of_tiles, 0)

    def forward_base(self, x):
        mask = x
        # print(mask.shape)
        n, _, _, _ = mask.shape  # [1, 1, 620, 620]
        mask = self.avepool_lres(mask)  # ----> line 11 in alg.1 DAC'23

        mask = torch.sigmoid(self.mask_steepness * (mask - self.mask_shift))

        if self.morph > 0 and self.iter % 20 == 0 and self.iter > 0:
            mask = opening(mask, self.morph_kernel_opt_opening)
            mask = closing(mask, self.morph_kernel_opt_closing)

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
        target_image_s = nn.functional.avg_pool2d(
            target_img.view(1, 1, self.mask_dim1, self.mask_dim2).to(self.device),
            self.scale_factor,
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
            # x_out_batch, x_out_max_batch, x_out_min_batch = self.forward_base(batch)
            x_out_batch_list.append(x_out_batch)
            x_out_max_batch_list.append(x_out_max_batch)
            x_out_min_batch_list.append(x_out_min_batch)
        x_out_batch = torch.cat(x_out_batch_list, dim=0)
        x_out_max_batch = torch.cat(x_out_max_batch_list, dim=0)
        x_out_min_batch = torch.cat(x_out_min_batch_list, dim=0)
        # print(batch.shape)
        # if b==0:
        #     x_out_batch, x_out_max_batch, x_out_min_batch = self.forward_base(batch)
        # else:
        #     t_x_out_batch, t_x_out_max_batch, t_x_out_min_batch = self.forward_base(batch)
        #     x_out_batch = torch.cat((x_out_batch, t_x_out_batch), dim=0)
        #     x_out_max_batch = torch.cat((x_out_max_batch, t_x_out_max_batch), dim=0)
        #     x_out_min_batch = torch.cat((x_out_min_batch, t_x_out_min_batch), dim=0)

        if False:
            id = 0
            debug_mask = mask_batch[id][0, :, :]
            debug_img = x_out_batch[id][0, :, :]

            debug = (
                torch.cat((debug_mask, debug_img), dim=1).cpu().detach().numpy() * 255
            )
            cv2.imwrite("./benchmarks/test_20um/im_via/via1.debug.png", debug)

        x_out = self._batch2tile(x_out_batch)
        x_out_max = self._batch2tile(x_out_max_batch)
        x_out_min = self._batch2tile(x_out_min_batch)

        return x_out, x_out_max, x_out_min

    def forward_batch_test(self, use_morph=True, batch_size=1):
        mask = self.mask_s.data
        # cmask = self.mask.data

        # mask[self.mask_s.data>=0.5]=1.0
        # mask[self.mask_s.data<0.5]=0.0
        # if self.morph>0 and use_morph:
        #    mask = opening(mask, self.morph_kernel_opt_opening)
        #    mask = closing(mask, self.morph_kernel_opt_closing)
        #    mask = opening(mask, self.morph_kernel_opt_opening)
        # mask[mask>=0.5]=1.0
        # mask[mask<0.5]=0.0

        mask_batch = self._tile2batch(mask)
        all_size, c, h, w = mask_batch.shape
        # print(all_size,c,h,w)
        for b in range(math.ceil(1.0 * all_size / batch_size)):
            # print("Processing Batch %g:  %g--->%g"%(b, b*batch_size, min((b+1)*batch_size, all_size)))

            batch = mask_batch[b * batch_size : min((b + 1) * batch_size, all_size)]
            # print(batch.shape)
            if b == 0:
                x_out_batch, x_out_max_batch, x_out_min_batch = self.forward_base(batch)
            else:
                t_x_out_batch, t_x_out_max_batch, t_x_out_min_batch = self.forward_base(
                    batch
                )
                x_out_batch = torch.cat((x_out_batch, t_x_out_batch), dim=0)
                x_out_max_batch = torch.cat((x_out_max_batch, t_x_out_max_batch), dim=0)
                x_out_min_batch = torch.cat((x_out_min_batch, t_x_out_min_batch), dim=0)

        if False:
            id = 0
            debug_mask = mask_batch[id][0, :, :]
            debug_img = x_out_batch[id][0, :, :]

            debug = (
                torch.cat((debug_mask, debug_img), dim=1).cpu().detach().numpy() * 255
            )
            cv2.imwrite("./benchmarks/test_20um/im_via/via1.debug.png", debug)

        x_out = self._batch2tile(x_out_batch)
        x_out_max = self._batch2tile(x_out_max_batch)
        x_out_min = self._batch2tile(x_out_min_batch)

        return mask, x_out, x_out_max, x_out_min

    # def forward_serial(self,):
    def forward_base_test(self, x):
        mask = x
        # print(mask.shape)
        n, _, _, _ = mask.shape
        # mask = self.avepool_lres(mask) #----> line 11 in alg.1 DAC'23

        mask_fft = torch.fft.fftshift(torch.fft.fft2(mask))
        mask_fft = torch.repeat_interleave(mask_fft, self.kernel_num, 1)
        # print(mask_fft.shape)
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

        # print(x_out.shape, self.kernel_focus.shape, mask_fft.shape)
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
        # return (
        #     x.view(1, 1, self.base_dim_s, self.base_dim_s, -1)
        #     .permute(0, 4, 1, 2, 3)
        #     .reshape(-1, 1, self.base_dim_s, self.base_dim_s)
        # )

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

        # turn_off = torch.zeros_like(x)
        # turn_off[:, :, self.ambit_s : -self.ambit_s, self.ambit_s : -self.ambit_s] = 1.0
        # x = x * turn_off
        # x = x.reshape(1, n, 1, self.base_dim_s, self.base_dim_s)
        # x = x.reshape(1, n, -1)
        # x = x.permute(0, 2, 1)
        # x = F.fold(
        #     x,
        #     kernel_size=self.base_dim_s,
        #     stride=self.base_dim_s // 2,
        #     output_size=(self.mask_dim1_s, self.mask_dim2_s),
        # )

        return x


class evaluation:
    def __init__(self, mask, target, nominal, inner, outer):
        self.mask = mask
        self.target = target
        self.nominal = nominal
        self.inner = inner
        self.outer = outer
        # self.epe_check_point=epe_check_point

    def get_l2(self):
        return torch.sum(torch.abs(self.nominal - self.target)).cpu().numpy()

    def get_pvb(self):
        pvb = torch.zeros_like(self.outer).cuda()
        pvb[self.outer == 1.0] = 1
        pvb[self.inner == 1.0] = 0
        pvb = torch.sum(pvb)

        return pvb.cpu().numpy()

    # this function will not be called
    # def get_epe(self):
    #     #modified from https://github.com/cuhk-eda/neural-ilt
    #     target_numpy = self.target.detach().cpu().numpy().astype('uint8')[0,0,:,:]*255
    #     nominal_numpy = self.nominal.detach().cpu().numpy().astype('uint8')[0,0,:,:]*255

    #     epe, _ = epe_eval(nominal_numpy, target_numpy)

    #     return epe


def corner_smooth(im, kernel=45):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    b = cv2.dilate(im, k)
    b = cv2.erode(b, k, iterations=2)
    b = cv2.dilate(b, k)
    return b


def morph_close_cv2(im, kernel=10):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    b = cv2.dilate(im, k)
    b = cv2.erode(b, k)
    return b


def morph_open_cv2(im, kernel=20):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    b = cv2.erode(im, k)
    b = cv2.dilate(b, k)
    return b
