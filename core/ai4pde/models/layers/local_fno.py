import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
import torch

from torch import nn
import torch.nn.functional as F
import torch

from torch import nn
import torch
import itertools

import tensorly as tl
from tensorly.plugins import use_opt_einsum
from .layer_norm import MyLayerNorm
tl.set_backend('pytorch')

use_opt_einsum('optimal')

from tltorch.factorized_tensors.core import FactorizedTensor

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


class Lifting(nn.Module):
    def __init__(self, in_channels, out_channels, n_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc = Conv(in_channels, out_channels, 1)

    def forward(self, x):
        return self.fc(x)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, n_dim=2, non_linearity=F.gelu):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        return x

def _contract_dense(x, weight, separable=False):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:]) # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0] 

    eq= ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    return torch.einsum(eq, x, weight)

def _contract_dense_separable(x, weight, separable=True):
    if separable == False:
        raise ValueError('This function is only for separable=True')
    return x*weight

def _contract_cp(x, cp_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)
    if separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym,out_sym+rank_sym] #in, out
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...
    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors)
 

def _contract_tucker(x, tucker_weight, separable=False):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order+1:2*order]
        # factor_syms = [einsum_symbols[1]+core_syms[0]] #in only
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...
    
    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors)


def _contract_tt(x, tt_weight, separable=False):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:]) # no batch-size
    if not separable:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    rank_syms = list(einsum_symbols[order+1:])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)

    return tl.einsum(eq, x, *tt_weight.factors)


def get_contract_fun(weight, implementation='reconstructed', separable=False):
    """Generic ND implementation of Fourier Spectral Conv contraction
    
    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)
    
    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == 'reconstructed':
        if separable:
            print('SEPARABLE')
            return _contract_dense_separable
        else:
            return _contract_dense
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            return _contract_dense
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower().endswith('dense'):
                return _contract_dense
            elif weight.name.lower().endswith('tucker'):
                return _contract_tucker
            elif weight.name.lower().endswith('tt'):
                return _contract_tt
            elif weight.name.lower().endswith('cp'):
                return _contract_cp
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')


class FactorizedSpectralConv(nn.Module):
    """Generic N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels
    out_channels : int, optional
        Number of output channels
    n_modes : int tuple
        total number of modes to keep in Fourier Layer, along each dim
    separable : bool, default is True
    init_std : float or 'auto', default is 'auto'
        std to use for the init
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    factorization : str, {'tucker', 'cp', 'tt'}, optional
        Tensor factorization of the parameters weight to use, by default 'tucker'
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    fft_norm : str, optional
        by default 'forward'
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    """
    def __init__(self, in_channels, out_channels, n_modes, incremental_n_modes=None, bias=True,
                 n_layers=1, separable=False, output_scaling_factor=None,
                 rank=0.5, factorization='cp', implementation='reconstructed', 
                 fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs=dict(),
                 init_std='auto', fft_norm='backward'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_n_modes is half of that except in the last mode, correponding to the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.order = len(n_modes)

        half_total_n_modes = [m//2 for m in n_modes]
        self.half_total_n_modes = half_total_n_modes

        # WE use half_total_n_modes to build the full weights
        # During training we can adjust incremental_n_modes which will also
        # update half_n_modes 
        # So that we can train on a smaller part of the Fourier modes and total weights
        self.incremental_n_modes = incremental_n_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        if output_scaling_factor is not None:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [float(output_scaling_factor)]*len(self.n_modes)
        self.output_scaling_factor = output_scaling_factor

        if init_std == 'auto':
            init_std = (1 / (in_channels * out_channels))
        else:
            init_std = 0.02

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes=[0]
            else:
                fixed_rank_modes=None
        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = 'Dense' # No factorization
        if not factorization.lower().startswith('complex'):
            factorization = f'Complex{factorization}'

        if separable:
            if in_channels != out_channels:
                raise ValueError('To use separable Fourier Conv, in_channels must be equal to out_channels, ',
                                 f'but got {in_channels=} and {out_channels=}')
            weight_shape = (in_channels, *half_total_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *half_total_n_modes)
        self.separable = separable

        self.n_weights_per_layer = 2**(self.order-1)
        if joint_factorization:
            self.weight = FactorizedTensor.new((self.n_weights_per_layer*n_layers, *weight_shape),
                                                rank=self.rank, factorization=factorization, 
                                                fixed_rank_modes=fixed_rank_modes,
                                                **decomposition_kwargs)
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList([
                 FactorizedTensor.new(
                    weight_shape,
                    rank=self.rank, factorization=factorization, 
                    fixed_rank_modes=fixed_rank_modes,
                    **decomposition_kwargs
                    ) for _ in range(self.n_weights_per_layer*n_layers)]
                )
            for w in self.weight:
                w.normal_(0, init_std)
        self._contract = get_contract_fun(self.weight[0], implementation=implementation, separable=separable)

        if bias:
            self.bias = nn.Parameter(init_std * torch.randn(*((n_layers, self.out_channels) + (1, )*self.order)))
        else:
            self.bias = None

    def _get_weight(self, index):
        if self.incremental_n_modes is not None:
            return self.weight[index][self.weight_slices]
        else:
            return self.weight[index]

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        if incremental_n_modes is None:
            self._incremental_n_modes = None
            self.half_n_modes = [m//2 for m in self.n_modes]

        else:
            if isinstance(incremental_n_modes, int):
                self._incremental_n_modes = [incremental_n_modes]*len(self.n_modes)
            else:
                if len(incremental_n_modes) == len(self.n_modes):
                    self._incremental_n_modes = incremental_n_modes
                else:
                    raise ValueError(f'Provided {incremental_n_modes} for actual n_modes={self.n_modes}.')
            self.weight_slices = [slice(None)]*2 + [slice(None, n//2) for n in self._incremental_n_modes]
            self.half_n_modes = [m//2 for m in self._incremental_n_modes]

    def forward(self, x, indices=0):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1]//2 + 1 # Redundant last coefficient
        
        #Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))
        x = torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=fft_dims)

        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size], device=x.device, dtype=torch.cfloat)
        
        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_n_modes[:-1]] + [((None, self.half_n_modes[-1]), )]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            out_fft[idx_tuple] = self._contract(x[idx_tuple], self._get_weight(self.n_weights_per_layer*indices + i), separable=self.separable)

        if self.output_scaling_factor is not None:
            mode_sizes = tuple([int(round(s*r)) for (s, r) in zip(mode_sizes, self.output_scaling_factor)])

        x = torch.fft.irfftn(out_fft, s=(mode_sizes), norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single convolution is parametrized, directly use the main class.')
        
        return SubConv(self, indices)
    
    def __getitem__(self, indices):
        return self.get_conv(indices)



class SubConv(nn.Module):
    """Class representing one of the convolutions from the mother joint factorized convolution

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data, 
    which is shared.
    """
    def __init__(self, main_conv, indices):
        super().__init__()
        self.main_conv = main_conv
        self.indices = indices
    
    def forward(self, x):
        return self.main_conv.forward(x, self.indices)
    
from torch.nn import functional as F
from torch import nn

class DomainPadding(nn.Module):
    """Applies domain padding scaled automatically to the input's resolution

    Parameters
    ----------
    domain_padding : float
        typically, between zero and one, percentage of padding to use
    padding_mode : {'symmetric', 'one-sided'}, optional
        whether to pad on both sides, by default 'one-sided'

    Notes
    -----
    This class works for any input resolution, as long as it is in the form
    `(batch-size, channels, d1, ...., dN)`
    """
    def __init__(self, domain_padding, padding_mode='one-sided', output_scale_factor = None):
        super().__init__()
        self.domain_padding = domain_padding
        self.padding_mode = padding_mode.lower()
        self.output_scale_factor = output_scale_factor
        
        # dict(f'{resolution}'=padding) such that padded = F.pad(x, indices)
        self._padding = dict()
        
        # dict(f'{resolution}'=indices_to_unpad) such that unpadded = x[indices]
        self._unpad_indices = dict()

    def forward(self, x):
        """forward pass: pad the input"""
        self.pad(x)
    
    def pad(self, x):
        """Take an input and pad it by the desired fraction
        
        The amount of padding will be automatically scaled with the resolution
        """
        resolution = x.shape[2:]

        if isinstance(self.domain_padding, (float, int)):
            self.domain_padding = [float(self.domain_padding)]*len(resolution)
        if self.output_scale_factor is None:
            self.output_scale_factor = [1]*len(resolution)
        elif isinstance(self.output_scale_factor, (float, int)):
            self.output_scale_factor = [float(self.output_scale_factor)]*len(resolution)

        try:
            padding = self._padding[f'{resolution}']
            return F.pad(x, padding, mode='constant')

        except KeyError:
            padding = [int(round(p*r)) for (p, r) in zip(self.domain_padding, resolution)]
            
            print(f'Padding inputs of {resolution=} with {padding=}, {self.padding_mode}')

            output_pad = padding

            for scale_factor in self.output_scale_factor:
                if isinstance(scale_factor, (float, int)):
                    scale_factor = [scale_factor]*len(resolution)
                output_pad = [int(round(i*j)) for (i,j) in zip(scale_factor,output_pad)]

            if self.padding_mode == 'symmetric':
                # Pad both sides
                unpad_indices = (Ellipsis, ) + tuple([slice(p, -p, None) for p in output_pad ])
                padding = [i for p in padding for i in (p, p)]

            elif self.padding_mode == 'one-sided':
                # One-side padding
                unpad_indices = (Ellipsis, ) + tuple([slice(None, -p, None) for p in output_pad])
                padding = [i for p in padding for i in (0, p)]
            else:
                raise ValueError(f'Got {self.padding_mode=}')
            
            self._padding[f'{resolution}'] = padding

            padded = F.pad(x, padding, mode='constant')

            out_put_shape = padded.shape[2:]
            for scale_factor in self.output_scale_factor:
                if isinstance(scale_factor, (float, int)):
                    scale_factor = [scale_factor]*len(resolution)
                out_put_shape = [int(round(i*j)) for (i,j) in zip(scale_factor,out_put_shape)]
            
            self._unpad_indices[f'{[i for i in out_put_shape]}'] = unpad_indices

            return padded

    def unpad(self, x):
        """Remove the padding from padding inputs
        """
        unpad_indices = self._unpad_indices[f'{list(x.shape[2:])}']
        return x[unpad_indices]


class FNOBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes,
                 output_scaling_factor=None,
                 n_layers=1,
                 incremental_n_modes=None,
                 use_mlp=False, mlp=None,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 fno_skip='linear',
                 mlp_skip='soft-gating',
                 separable=False,
                 factorization=None,
                 rank=1.0,
                 SpectralConv=FactorizedSpectralConv,
                 joint_factorization=False, 
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 fft_norm='forward',
                 **kwargs):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.n_dim = len(n_modes)

        if output_scaling_factor is not None:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [float(output_scaling_factor)]*len(self.n_modes)
        self.output_scaling_factor = output_scaling_factor

        self._incremental_n_modes = incremental_n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip,
        self.mlp_skip = mlp_skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation

        self.convs = SpectralConv(
                self.in_channels, self.out_channels, self.n_modes, 
                output_scaling_factor=output_scaling_factor,
                incremental_n_modes=incremental_n_modes,
                rank=rank,
                fft_norm=fft_norm,
                fixed_rank_modes=fixed_rank_modes, 
                implementation=implementation,
                separable=separable,
                factorization=factorization,
                decomposition_kwargs=decomposition_kwargs,
                joint_factorization=joint_factorization,
                n_layers=n_layers,
            )

        self.fno_skips = nn.ModuleList([skip_connection(self.in_channels, self.out_channels, type=fno_skip, n_dim=self.n_dim) for _ in range(n_layers)])

        if use_mlp:
            self.mlp = nn.ModuleList(
                [MLP(in_channels=self.out_channels, 
                     hidden_channels=int(round(self.out_channels*mlp['expansion'])),
                     dropout=mlp['dropout'], n_dim=self.n_dim) for _ in range(n_layers)]
            )
            self.mlp_skips = nn.ModuleList([skip_connection(self.in_channels, self.out_channels, type=mlp_skip, n_dim=self.n_dim) for _ in range(n_layers)])
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == 'instance_norm':
            self.norm = nn.ModuleList([getattr(nn, f'InstanceNorm{self.n_dim}d')(num_features=self.out_channels) for _ in range(n_layers*self.n_norms)])
        elif norm == 'group_norm':
            self.norm = nn.ModuleList([nn.GroupNorm(num_groups=1, num_channels=self.out_channels) for _ in range(n_layers*self.n_norms)])
        elif norm == 'layer_norm':
            self.norm = nn.ModuleList([nn.LayerNorm(elementwise_affine=False) for _ in range(n_layers*self.n_norms)])
        elif norm == "my_layernorm":
            self.norm = nn.ModuleList([MyLayerNorm(out_channels, data_format="channels_first") for _ in range(n_layers*self.n_norms)])
        else:
            raise ValueError(f'Got {norm=} but expected None or one of [instance_norm, group_norm, layer_norm]')

    def forward(self, x, index=0):
        
        if self.preactivation:
            x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms*index](x)
    
        x_skip_fno = self.fno_skips[index](x)
        if self.convs.output_scaling_factor is not None:
            x_skip_fno = resample(x_skip_fno, self.convs.output_scaling_factor, list(range(-len(self.convs.output_scaling_factor), 0)))

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            if self.convs.output_scaling_factor is not None:
                x_skip_mlp = resample(x_skip_mlp, self.convs.output_scaling_factor, list(range(-len(self.convs.output_scaling_factor), 0)))

        x_fno = self.convs(x, index)

        if not self.preactivation and self.norm is not None:
            x_fno = self.norm[self.n_norms*index](x_fno)
    
        x = x_fno + x_skip_fno

        if not self.preactivation and (self.mlp is not None) or (index < (self.n_layers - index)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            # x_skip = self.mlp_skips[index](x)

            if self.preactivation:
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)

                if self.norm is not None:
                    x = self.norm[self.n_norms*index+1](x)

            x = self.mlp[index](x) + x_skip_mlp

            if not self.preactivation and self.norm is not None:
                x = self.norm[self.n_norms*index+1](x)

            if not self.preactivation:
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)
        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.convs.incremental_n_modes = incremental_n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError('A single layer is parametrized, directly use the main class.')
        
        return SubModule(self, indices)
    
    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules, they all point to the same data, 
    which is shared.
    """
    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices
    
    def forward(self, x):
        return self.main_module.forward(x, self.indices)
    
class MLP(nn.Module):
    """A Multi-Layer Perceptron, with arbitrary number of layers
    
    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """
    def __init__(self, in_channels, out_channels=None, hidden_channels=None, 
                 n_layers=2, n_dim=2, non_linearity=F.gelu, dropout=0., **kwargs):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)]) if dropout > 0. else None
        
        Conv = getattr(nn, f'Conv{n_dim}d')
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.fcs.append(Conv(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(Conv(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(Conv(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout(x)

        return x
    
def skip_connection(in_features, out_features, n_dim=2, bias=False, type="soft-gating"):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        number of input features
    out_features : int
        number of output features
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D. 
    bias : bool, optional
        whether to use a bias, by default False
    type : {'identity', 'linear', soft-gating'}
        kind of skip connection to use, by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if type.lower() == 'soft-gating':
        return SoftGating(in_features=in_features, out_features=out_features, bias=bias, n_dim=n_dim)
    elif type.lower() == 'linear':
        return getattr(nn, f'Conv{n_dim}d')(in_channels=in_features, out_channels=out_features, kernel_size=1, bias=bias)
    elif type.lower() == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Got skip-connection {type=}, expected one of {'soft-gating', 'linear', 'id'}.")
    
class SoftGating(nn.Module):
    """Applies soft-gating by weighting the channels of the given input

    Given an input x of size `(batch-size, channels, height, width)`,
    this returns `x * w `
    where w is of shape `(1, channels, 1, 1)`

    Parameters
    ----------
    in_features : int
    out_features : None
        this is provided for API compatibility with nn.Linear only
    n_dim : int, default is 2
        Dimensionality of the input (excluding batch-size and channels).
        ``n_dim=2`` corresponds to having Module2D. 
    bias : bool, default is False
    """
    def __init__(self, in_features, out_features=None, n_dim=2, bias=False):
        super().__init__()
        if out_features is not None and in_features != out_features:
            raise ValueError(f"Got {in_features=} and {out_features=}"
                             "but these two must be the same for soft-gating")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.in_features, *(1,)*n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations
        """
        if self.bias is not None:
            return self.weight*x + self.bias
        else:
            return self.weight*x
    
def resample(x, res_scale, axis):
    """
    A module for generic n-dimentional interpolation (Fourier resampling).

    Parameters
    ----------
    x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic 
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed. 
    """
    """
    A module for generic n-dimentional interpolation (Fourier resampling).

    Parameters
    ----------
    x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
    res_scale: int or tuple
            Scaling factor along each of the dimensions in 'axis' parameter. If res_scale is scaler, then isotropic 
            scaling is performed
    axis: axis or dimensions along which interpolation will be performed. 
    """

    if isinstance(res_scale, (float, int)):
        if axis is None:
            axis = list(range(2, x.ndim))
            res_scale = [res_scale]*len(axis)
        elif isinstance(axis, int):
            axis = [axis]
            res_scale = [res_scale]
        else:
              res_scale = [res_scale]*len(axis)
    else:
        assert len(res_scale) == len(axis), "leght of res_scale and axis are not same"

    old_size = x.shape[-len(axis):]
    new_size = tuple([int(round(s*r)) for (s, r) in zip(old_size, res_scale)])

    if len(axis) == 1:
        return F.interpolate(x, size = new_size[0], mode = 'linear', align_corners = True)
    if len(axis) == 2:
        return F.interpolate(x, size = new_size, mode = 'bicubic', align_corners = True, antialias = True)

    X = torch.fft.rfftn(x.float(), norm='forward', dim=axis)
    
    new_fft_size = list(new_size)
    new_fft_size[-1] = new_fft_size[-1]//2 + 1 # Redundant last coefficient
    new_fft_size_c = [min(i,j) for (i,j) in zip(new_fft_size, X.shape[-len(axis):])]
    out_fft = torch.zeros([x.shape[0], x.shape[1], *new_fft_size], device=x.device, dtype=torch.cfloat)

    mode_indexing = [((None, m//2), (-m//2, None)) for m in new_fft_size_c[:-1]] + [((None, new_fft_size_c[-1]), )]
    for i, boundaries in enumerate(itertools.product(*mode_indexing)):

        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

        out_fft[idx_tuple] = X[idx_tuple]
    y = torch.fft.irfftn(out_fft, s =  new_size ,norm='forward', dim = axis)

    return y

class FNO(nn.Module):
    """N-Dimensional Fourier Neural Operator

    Parameters
    ----------
    n_modes : int tuple
        number of modes to keep in Fourier Layer, along each dimension
        The dimensionality of the TFNO is inferred from ``len(n_modes)``
    hidden_channels : int
        width of the FNO (i.e. number of channels)
    in_channels : int, optional
        Number of input channels, by default 3
    out_channels : int, optional
        Number of output channels, by default 1
    lifting_channels : int, optional
        number of hidden channels of the lifting block of the FNO, by default 256
    projection_channels : int, optional
        number of hidden channels of the projection block of the FNO, by default 256
    n_layers : int, optional
        Number of Fourier Layers, by default 4
    incremental_n_modes : None or int tuple, default is None
        * If not None, this allows to incrementally increase the number of modes in Fourier domain 
          during training. Has to verify n <= N for (n, m) in zip(incremental_n_modes, n_modes).
        
        * If None, all the n_modes are used.

        This can be updated dynamically during training.
    use_mlp : bool, optional
        Whether to use an MLP layer after each FNO block, by default False
    mlp : dict, optional
        Parameters of the MLP, by default None
        {'expansion': float, 'dropout': float}
    non_linearity : nn.Module, optional
        Non-Linearity module to use, by default F.gelu
    norm : F.module, optional
        Normalization layer to use, by default None
    preactivation : bool, default is False
        if True, use resnet-style preactivation
    skip : {'linear', 'identity', 'soft-gating'}, optional
        Type of skip connection to use, by default 'soft-gating'
    separable : bool, default is False
        if True, use a depthwise separable spectral convolution
    factorization : str or None, {'tucker', 'cp', 'tt'}
        Tensor factorization of the parameters weight to use, by default None.
        * If None, a dense tensor parametrizes the Spectral convolutions
        * Otherwise, the specified tensor factorization is used.
    joint_factorization : bool, optional
        Whether all the Fourier Layers should be parametrized by a single tensor (vs one per layer), by default False
    rank : float or rank, optional
        Rank of the tensor factorization of the Fourier weights, by default 1.0
    fixed_rank_modes : bool, optional
        Modes to not factorize, by default False
    implementation : {'factorized', 'reconstructed'}, optional, default is 'factorized'
        If factorization is not None, forward mode to use::
        * `reconstructed` : the full weight tensor is reconstructed from the factorization and used for the forward pass
        * `factorized` : the input is directly contracted with the factors of the decomposition
    decomposition_kwargs : dict, optional, default is {}
        Optionaly additional parameters to pass to the tensor decomposition
    domain_padding : None or float, optional
        If not None, percentage of padding to use, by default None
    domain_padding_mode : {'symmetric', 'one-sided'}, optional
        How to perform domain padding, by default 'one-sided'
    fft_norm : str, optional
        by default 'forward'
    """
    def __init__(self, n_modes, hidden_channels,
                 in_channels=3, 
                 out_channels=1,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 incremental_n_modes=None,
                 use_mlp=False, mlp=None,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
                 fno_skip='linear',
                 mlp_skip='soft-gating',
                 separable=False,
                 factorization=None,
                 rank=1.0,
                 joint_factorization=False, 
                 fixed_rank_modes=False,
                 implementation='factorized',
                 decomposition_kwargs=dict(),
                 domain_padding=None,
                 domain_padding_mode='one-sided',
                 fft_norm='forward',
                 SpectralConv=FactorizedSpectralConv,
                 **kwargs):
        super().__init__()
        self.n_dim = len(n_modes)
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip,
        self.mlp_skip = mlp_skip,
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._incremental_n_modes = incremental_n_modes

        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels, 
            n_modes=self.n_modes,
            use_mlp=use_mlp, mlp=mlp,
            non_linearity=non_linearity,
            norm=norm, preactivation=preactivation,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes, 
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            SpectralConv=SpectralConv,
            n_layers=n_layers)

        self.lifting = Lifting(in_channels=in_channels, out_channels=self.hidden_channels, n_dim=self.n_dim)
        self.projection = Projection(in_channels=self.hidden_channels, out_channels=out_channels, hidden_channels=projection_channels,
                                     non_linearity=non_linearity, n_dim=self.n_dim)

    def forward(self, x):
        """TFNO's forward pass
        """
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.fno_blocks.incremental_n_modes = incremental_n_modes