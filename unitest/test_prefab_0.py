import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_keras as k3
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.keras.utils import plot_model


class KerasPortNet(nn.Module):
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


def _tf_weight_dict(tf_model):
    d = {}
    for v in tf_model.weights + tf_model.non_trainable_variables:
        d[v.name] = v.numpy()
    return d


def port_tf_to_torch(tf_model, torch_model):
    td = _tf_weight_dict(tf_model)

    # Conv1
    W = td["conv2d/kernel:0"]  # (3,3,1,1)
    b = td["conv2d/bias:0"]  # (1,)
    torch_model.conv1.weight.data.copy_(torch.from_numpy(np.transpose(W, (3, 2, 0, 1))))
    torch_model.conv1.bias.data.copy_(torch.from_numpy(b))
    # BN1
    torch_model.bn1.weight.data.copy_(
        torch.from_numpy(td["batch_normalization/gamma:0"])
    )
    torch_model.bn1.bias.data.copy_(torch.from_numpy(td["batch_normalization/beta:0"]))
    torch_model.bn1.running_mean.data.copy_(
        torch.from_numpy(td["batch_normalization/moving_mean:0"])
    )
    torch_model.bn1.running_var.data.copy_(
        torch.from_numpy(td["batch_normalization/moving_variance:0"])
    )

    # Conv2
    W = td["conv2d_1/kernel:0"]  # (3,3,1,2)
    b = td["conv2d_1/bias:0"]  # (2,)
    torch_model.conv2.weight.data.copy_(torch.from_numpy(np.transpose(W, (3, 2, 0, 1))))
    torch_model.conv2.bias.data.copy_(torch.from_numpy(b))
    # BN2
    torch_model.bn2.weight.data.copy_(
        torch.from_numpy(td["batch_normalization_1/gamma:0"])
    )
    torch_model.bn2.bias.data.copy_(
        torch.from_numpy(td["batch_normalization_1/beta:0"])
    )
    torch_model.bn2.running_mean.data.copy_(
        torch.from_numpy(td["batch_normalization_1/moving_mean:0"])
    )
    torch_model.bn2.running_var.data.copy_(
        torch.from_numpy(td["batch_normalization_1/moving_variance:0"])
    )

    # Conv3
    W = td["conv2d_2/kernel:0"]  # (3,3,2,4)
    b = td["conv2d_2/bias:0"]  # (4,)
    torch_model.conv3.weight.data.copy_(torch.from_numpy(np.transpose(W, (3, 2, 0, 1))))
    torch_model.conv3.bias.data.copy_(torch.from_numpy(b))
    # BN3
    torch_model.bn3.weight.data.copy_(
        torch.from_numpy(td["batch_normalization_2/gamma:0"])
    )
    torch_model.bn3.bias.data.copy_(
        torch.from_numpy(td["batch_normalization_2/beta:0"])
    )
    torch_model.bn3.running_mean.data.copy_(
        torch.from_numpy(td["batch_normalization_2/moving_mean:0"])
    )
    torch_model.bn3.running_var.data.copy_(
        torch.from_numpy(td["batch_normalization_2/moving_variance:0"])
    )

    # Conv4
    W = td["conv2d_3/kernel:0"]  # (3,3,4,8)
    b = td["conv2d_3/bias:0"]  # (8,)
    torch_model.conv4.weight.data.copy_(torch.from_numpy(np.transpose(W, (3, 2, 0, 1))))
    torch_model.conv4.bias.data.copy_(torch.from_numpy(b))
    # BN4
    torch_model.bn4.weight.data.copy_(
        torch.from_numpy(td["batch_normalization_3/gamma:0"])
    )
    torch_model.bn4.bias.data.copy_(
        torch.from_numpy(td["batch_normalization_3/beta:0"])
    )
    torch_model.bn4.running_mean.data.copy_(
        torch.from_numpy(td["batch_normalization_3/moving_mean:0"])
    )
    torch_model.bn4.running_var.data.copy_(
        torch.from_numpy(td["batch_normalization_3/moving_variance:0"])
    )

    # Dense
    W = td["dense/kernel:0"]  # (512, 16384)
    b = td["dense/bias:0"]  # (16384,)

    W = torch.from_numpy(W).t().reshape(128 * 128, 8, 8, 8)  # [:, H, W, C]
    W = W.permute(0, 3, 1, 2).contiguous().flatten(1)  # [:, C, H, W]

    torch_model.fc.weight.data.copy_(W)  # (16384,512)
    torch_model.fc.bias.data.copy_(torch.from_numpy(b))

    return torch_model


# from tensorflow.keras import Model,Input
import visualkeras

path = ("./thirdparty/PreFab/models/p_ANT_NanoSOI_v0.1_0.pb",)
model = k3.models.load_model(path, compile=False)
# visualkeras.layered_view(model, legend = True,
#     draw_volume=False).show()
# visualkeras.graph_view(model).show()
# class MyModel(k3.Model):
#     def __init__(self, model, dim=(128,128,1)):
#         super(MyModel, self).__init__()
#         self.model = model
#         self.dim = dim

#     def call(self, x):
#         return self.model(x)

#     def build_graph(self):
#         x = k3.Input(shape=(self.dim))
#         return k3.Model(inputs=[x], outputs=self.call(x))

# model = MyModel(model)
# model.build((None, *(128, 128, 1)))
# model.build_graph().summary()
# plot_model(model.build_graph(), to_file='model_architecture.png', expand_nested=True, show_shapes=True, show_layer_names=True)
# # exit(0)
print(model)
model.summary()
print("=== TRAINABLE VARIABLES ===")
for v in model.trainable_variables:
    print(f"{v.name:<60} shape={tuple(v.shape)}")

# Non-trainable (e.g., BN moving_mean/variance)
print("\n=== NON-TRAINABLE VARIABLES ===")
for v in model.non_trainable_variables:
    print(f"{v.name:<60} shape={tuple(v.shape)}")

print(json.dumps(model.get_config(), indent=2))

## put a square in the middle as binary input
device = np.zeros((128, 128), dtype=np.float32)
device[48:80, 48:80] = 1.0
x_nchw = torch.from_numpy(device).unsqueeze(0).unsqueeze(0)  # to NCHW for Torch
x_nhwc = np.transpose(x_nchw.numpy(), (0, 2, 3, 1))  # to NHWC for Keras

pt_model = KerasPortNet()
pt_model = port_tf_to_torch(model, pt_model)
pt_model.eval()


with torch.no_grad():
    y_pt = pt_model(x_nchw).numpy()  # (N,1,128,128)

y_tf = model.predict(x_nhwc, verbose=0)  # (N,128,128,1)
y_tf_nchw = np.transpose(y_tf, (0, 3, 1, 2))  # -> (N,1,128,128)

## plot
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
## showcolorbar
axes[0, 0].imshow(x_nchw[0, 0].numpy(), cmap="gray")
plt.colorbar(axes[0, 0].imshow(x_nchw[0, 0].numpy(), cmap="gray"), ax=axes[0, 0])
axes[0, 0].set_title("Input (Torch)")
axes[0, 1].imshow(y_pt[0, 0], cmap="gray")
plt.colorbar(axes[0, 1].imshow(y_pt[0, 0], cmap="gray"), ax=axes[0, 1])
axes[0, 1].set_title("Output (Torch)")
axes[0, 2].imshow(y_tf_nchw[0, 0], cmap="gray")
plt.colorbar(axes[0, 2].imshow(y_tf_nchw[0, 0], cmap="gray"), ax=axes[0, 2])
axes[0, 2].set_title("Output (Keras)")
plt.tight_layout()
plt.savefig("output_comparison.png", dpi=300)

print("max |diff| =", np.max(np.abs(y_pt - y_tf_nchw)))

## save pytorch checkpoint
torch.save(pt_model.state_dict(), path + "/prefab_pytorch_model.pt")
