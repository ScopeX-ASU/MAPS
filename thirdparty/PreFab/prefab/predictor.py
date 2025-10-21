import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torch.nn.functional as F
from prefab.processor import *
from torch import nn


class PreFabTorchModel(nn.Module):
    """
    Keras:
      [Conv2D (relu)] -> AvgPool2D(stride=2, same) -> BN
      (repeat 4 times with channels 1,2,4,8)
      -> Flatten -> Dense(16384, activation=sigmoid) -> Reshape(1,128,128)
    Input:  (N,1,128,128)  Output: (N,1,128,128)
    """

    def __init__(self):
        super().__init__()
        bn = dict(
            eps=1e-3, momentum=0.01, affine=True, track_running_stats=True
        )  # Keras 0.99 -> torch 0.01

        self.conv1 = nn.Conv2d(1, 1, 3, padding=1, bias=True)
        self.pool1 = nn.AvgPool2d(2)
        self.bn1 = nn.BatchNorm2d(1, **bn)

        self.conv2 = nn.Conv2d(1, 2, 3, padding=1, bias=True)
        self.pool2 = nn.AvgPool2d(2)
        self.bn2 = nn.BatchNorm2d(2, **bn)

        self.conv3 = nn.Conv2d(2, 4, 3, padding=1, bias=True)
        self.pool3 = nn.AvgPool2d(2)
        self.bn3 = nn.BatchNorm2d(4, **bn)

        self.conv4 = nn.Conv2d(4, 8, 3, padding=1, bias=True)
        self.pool4 = nn.AvgPool2d(2)
        self.bn4 = nn.BatchNorm2d(8, **bn)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8 * 8 * 8, 16384)  # 512 -> 16384
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # NCHW
        x = self.conv1(x)
        x = F.relu(x, inplace=False)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x, inplace=False)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.relu(x, inplace=False)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = F.relu(x, inplace=False)
        x = self.pool4(x)
        x = self.bn4(x)
        # x = x.permute(0, 2, 3, 1).contiguous()   # (N, H, W, C)
        x = x.view(x.size(0), -1)  # (N, 512)
        x = self.fc(x)
        x = self.sigmoid(x)  # Dense activation in Keras
        return x.view(x.size(0), 1, 128, 128)


class Predictor:
    def __init__(self, type, fab, process, version, model_nums):
        self.type = type
        self.fab = fab
        self.process = process
        self.version = version
        self.model_nums = model_nums
        self.models = []
        for j in self.model_nums:
            # self.models.append(models.load_model("../models/" + type + "_" + fab + "_"
            #     + process + "_" + str(version) + "_" + str(j) + ".pb"))
            # self.models.append(k3.models.load_model("../models/" + type + "_" + fab + "_"
            #     + process + "_" + str(version) + "_" + str(j) + ".pb"))
            self.models.append(PreFabTorchModel())
            self.models[-1].load_state_dict(
                torch.load(
                    "../models/"
                    + type
                    + "_"
                    + fab
                    + "_"
                    + process
                    + "_"
                    + str(version)
                    + "_"
                    + str(j)
                    + ".pb"
                    + "/prefab_pytorch_model.pt"
                )
            )
            self.models[-1]

        # self.slice_length = int(np.sqrt(self.models[0].weights[-1].shape)[0])
        self.slice_length = 128
        self.slice_size = (self.slice_length, self.slice_length)

    def make_patches_torch(self, device, slice_size, step_length):
        # ensure torch tensor, shape -> (N=1, C=1, H, W)
        x = torch.as_tensor(device, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif x.ndim == 3:
            x = x.unsqueeze(0)  # (1,1,H,W) if (1,H,W)
        else:
            raise ValueError("Expected (H,W) or (1,H,W) input.")

        kh, kw = slice_size
        H, W = x.shape[-2:]

        # extract sliding windows with stride
        cols = F.unfold(x, kernel_size=(kh, kw), stride=step_length)  # (1, C*kh*kw, L)
        L = cols.shape[-1]  # number of patches

        # reshape to (L, 1, kh, kw) == NCHW per patch
        patches_nchw = cols.transpose(1, 2).reshape(L, 1, kh, kw)

        # (L, 1, kh, kw)
        n_rows = (H - kh) // step_length + 1

        n_cols = (W - kw) // step_length + 1
        return patches_nchw, (n_rows, n_cols, kh, kw)

    def predict(self, device, step_length, binary=False):
        # slice image up
        # x_slices_4D = np.lib.stride_tricks.sliding_window_view(device, self.slice_size)[::step_length, ::step_length]
        # x_slices = x_slices_4D.reshape(-1, *self.slice_size)
        # x_slices = tf.reshape(x_slices, [len(x_slices), self.slice_length, self.slice_length, 1])
        x_slices, x_slices_4D_shape = self.make_patches_torch(
            device, self.slice_size, step_length
        )  # (N,1,128,128)

        # make predictions
        y_sum = 0
        for model in self.models:
            y = model(x_slices)
            y = y.double()
            y_sum += y
        # y_slices = y_sum / len(self.models)
        # y_slices = torch.squeeze(y_slices).reshape(x_slices_4D_shape)

        # # stitch slices back together (needs a better method)
        # y = torch.zeros_like(device)
        # avg_mtx = torch.zeros_like(device)
        # for k in range(0, device.shape[0]-self.slice_length+1, step_length):
        #     for j in range(0, device.shape[1]-self.slice_length+1, step_length):
        #         y[k:k+self.slice_length, j:j+self.slice_length] += y_slices[k//step_length,j//step_length]
        #         avg_mtx[k:k+self.slice_length, j:j+self.slice_length] += np.ones(self.slice_size)
        # prediction = y/avg_mtx

        H, W = device.shape
        kh, kw = self.slice_size
        stride = step_length
        y_patches = y_sum / len(self.models)  # (L,1,kh,kw)

        # --- fold (stitch) back to full image ---
        L = y_patches.shape[0]
        # F.fold expects shape (N, C*kh*kw, L). We'll use N=1, C=1.
        cols = y_patches.reshape(L, 1 * kh * kw).T.unsqueeze(0)  # -> (1, kh*kw, L)

        y_full = F.fold(
            cols, output_size=(H, W), kernel_size=(kh, kw), stride=stride
        )  # (1,1,H,W)

        # --- build overlap counts to average properly ---
        ones_cols = torch.ones(1, kh * kw, L, device=device.device)
        counts = F.fold(
            ones_cols, output_size=(H, W), kernel_size=(kh, kw), stride=stride
        )  # (1,1,H,W)

        prediction = (
            (y_full / counts.clamp_min(1e-12)).squeeze(0).squeeze(0)
        )  # -> (H,W)

        # binarize or leave as raw (showing uncertainty)
        if binary == True:
            prediction = binarize(prediction)

        return prediction
