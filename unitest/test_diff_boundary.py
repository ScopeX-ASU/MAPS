import matplotlib.pyplot as plt
import torch

from core.models.layers.diff_bdry_layer import DiffBdry
from core.utils import print_stat


def get_meshgrid(box_size, resolution):
    x = torch.linspace(
        -box_size[0] / 2,
        box_size[0] / 2,
        int(box_size[0] * resolution) + 1,
    )
    y = torch.linspace(
        -box_size[1] / 2,
        box_size[1] / 2,
        int(box_size[1] * resolution) + 1,
    )
    X, Y = torch.meshgrid(x, y)
    return X, Y


if __name__ == "__main__":
    diff_boundary = DiffBdry(10)

    X, Y = get_meshgrid([10, 10], 20)
    x_tensor = torch.tensor(X, dtype=torch.float32)  # Using x values from mesh grid
    y_tensor = torch.tensor(Y, dtype=torch.float32)

    w_x = torch.nn.Parameter(torch.tensor(8, dtype=torch.float32, requires_grad=True))
    w_y = torch.nn.Parameter(torch.tensor(8, dtype=torch.float32, requires_grad=True))

    output_x = diff_boundary(x_tensor, w_x, 1)
    output_y = diff_boundary(y_tensor, w_y, 1)

    final_output = output_x * output_y

    print_stat(final_output)

    loss = torch.sum(final_output)
    loss.backward()
    print("w_x.grad", w_x.grad)
    print("w_y.grad", w_y.grad)

    # plot the final_output with matplotlib
    plt.imshow(final_output.detach().numpy())
    plt.savefig("unitest/test_diff_boundary.png")
