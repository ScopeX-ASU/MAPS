import torch
import numpy as np
from pyutils.compute import interp1d, polynomial
from pyutils.general import TimerCtx
import numpy as np


def test_interp1d():
    # problem dimensions
    D = 1
    Dnew = 10
    N = 8000
    P = 8000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yq_gpu = None
    yq_cpu = None
    x = torch.rand(D, N, device=device) * 10000
    x = x.sort(dim=1)[0]

    y = torch.linspace(0, 1000, D*N, device=device).view(D, -1)
    y = y - y[:, 0, None]

    xnew = torch.rand(Dnew, P, device=device)*10000 + 3000
    # x = torch.tensor([[0, 1, 2, 3]], device=device).float()
    # y = torch.tensor([[0, 1, 2, 3]], device=device).float()
    # xnew = torch.tensor([[0.5, 1.5, 2.5, 3.5]], device=device)

    with TimerCtx() as t:
        for i in range(100):
            yq_gpu = interp1d(x, y, xnew, yq_gpu)
    print(t.interval / 100 / Dnew)
    print(yq_gpu)
    print(yq_gpu.shape)
    xnew = xnew.cpu().numpy()
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    y_out = []
    with TimerCtx() as t:
        for d in range(Dnew):
            for i in range(100):
                y_numpy = np.interp(xnew[d], x[0], y[0], left=0, right=0)
            y_out.append(y_numpy)
    y_numpy = np.array(y_out)
    print(t.interval / 100 / Dnew)
    print(y_numpy)
    # assert np.allclose(y_numpy, yq_gpu.cpu().numpy())
    print(np.sum(np.abs(y_numpy - yq_gpu.cpu().numpy())) / np.sum(np.abs(y_numpy)))

def test_polynomial():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(100, 100, device=device)
    coeffs = torch.randn(6, device=device)
    y = polynomial(x, coeffs)
    for i in range(100):
        y = polynomial(x, coeffs)
    torch.cuda.synchronize()
    with TimerCtx() as t:
        for i in range(100):
            y = polynomial(x, coeffs)
        torch.cuda.synchronize()
    print(t.interval / 100)

if __name__ == "__main__":
    # test_interp1d()
    test_polynomial()