import torch
import torch.nn as nn
import numpy as np
import h5py
import ceviche

def verify_adjoint_sources(filename):
    with h5py.File(filename, 'r') as f:
        adj_srcs = f['adj_src'][()]
        print("this is the shape of the adjoint sources: ", adj_srcs.shape)
        print("this is the type of the adjoint sources: ", type(adj_srcs))
        print("this is the adjoint sources: ", adj_srcs)
        # adjoint = f['adjoint'][()]
        # adjoint_sources = f['adjoint_sources'][()]
        # adjoint_sources = torch.tensor(adjoint_sources)
        # adjoint_sources = adjoint_sources.permute(0, 3, 1, 2)
        # data = torch.tensor(data)
        # adjoint = torch.tensor(adjoint)
        # adjoint = adjoint.permute(0, 3, 1, 2)
        # adjoint_sources = adjoint_sources.permute(0, 3, 1, 2)
        # print(data.shape)
        # print(adjoint.shape)
        # print(adjoint_sources.shape)
        # print(torch.sum(data * adjoint_sources).item())
        # print(torch.sum(adjoint * data).item())
        # print(torch.sum(data * adjoint_sources).item() - torch.sum(adjoint * data).item())
        # print(torch.sum(data * adjoint_sources).item() - torch.sum(adjoint * data).item() < 1e-6)
        # assert torch.sum(data * adjoint_sources).item() - torch.sum(adjoint * data).item() < 1e-6

if __name__ == '__main__':
    verify_adjoint_sources("./data/fdfd/metacoupler/metacoupler_opt_step_0.h5")