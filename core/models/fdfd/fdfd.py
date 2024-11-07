import numpy as np
import torch
from ceviche import fdfd_ez as fdfd_ez_ceviche
from ceviche.constants import *
from torch import Tensor, nn
from torch_sparse import spmm
# from torch import Tensor
# from torch_scatter import scatter_add


# def spmm(index: Tensor, value: Tensor, m: int, n: int,
#          matrix: Tensor) -> Tensor:
#     """Matrix product of sparse matrix with dense matrix.

#     Args:
#         index (:class:`LongTensor`): The index tensor of sparse matrix.
#         value (:class:`Tensor`): The value tensor of sparse matrix, either of
#             floating-point or integer type. Does not work for boolean and
#             complex number data types.
#         m (int): The first dimension of sparse matrix.
#         n (int): The second dimension of sparse matrix.
#         matrix (:class:`Tensor`): The dense matrix of same type as
#             :obj:`value`.

#     :rtype: :class:`Tensor`
#     """

#     assert n == matrix.size(-2)

#     row, col = index[0], index[1]
#     matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)
#     # print("this is the shape of the matrix: ", matrix.shape, flush=True)
#     # print("Indices in 'col':", col)
#     # print("Minimum index:", col.min().item())
#     # print("Maximum index:", col.max().item())
#     out = matrix.index_select(-2, col)
#     out = out * value.unsqueeze(-1)
#     out = scatter_add(out, row, dim=-2, dim_size=m)

#     return out


from .derivatives import compute_derivative_matrices
from .solver import sparse_solve_torch, SparseSolveTorch

# notataion is similar to that used in: http://www.jpier.org/PIERB/pierb36/11.11092006.pdf
from .utils import sparse_mm, sparse_mv
from core.models.layers.utils import (
    # Si_eps,
    # SiO2_eps,
    Slice,
    # apply_regions_gpu,
    get_eigenmode_coefficients,
    get_flux,
    # get_grid,
    # insert_mode,
    # plot_eps_field,
)
from core.utils import print_stat

__all__ = ["fdfd", "fdfd_ez", "My_fdfd_ez"]


class fdfd(nn.Module):
    """Base class for FDFD simulation"""

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None, device="cpu"):
        """initialize with a given structure and source
        omega: angular frequency (rad/s)
        dL: grid cell size (m)
        eps_r: array containing relative permittivity
        npml: list of number of PML grid cells in [x, y]
        bloch_{x,y} phase difference across {x,y} boundaries for bloch periodic boundary conditions (default = 0 = periodic)
        """
        super().__init__()
        self.omega = omega
        self.dL = dL
        self.npml = npml
        self.device = device

        self._setup_bloch_phases(bloch_phases)

        self.eps_r = eps_r

        self._setup_derivatives()

    """ what happens when you reassign the permittivity of the fdfd object """

    @property
    def eps_r(self):
        """Returns the relative permittivity grid"""
        return self._eps_r

    @eps_r.setter
    def eps_r(self, new_eps):
        """Defines some attributes when eps_r is set."""
        self._save_shape(new_eps)
        self._eps_r = new_eps

    """ classes inherited from fdfd() must implement their own versions of these functions for `fdfd.solve()` to work """

    def _make_A(self, eps_r):
        """This method constucts the entries and indices into the system matrix"""
        raise NotImplementedError("need to make a _make_A() method")

    def _solve_fn(self, entries_a, indices_a, source_vec):
        """This method takes the system matrix and source and returns the x, y, and z field components"""
        raise NotImplementedError(
            "need to implement function to solve for field components"
        )

    """ You call this to function to solve for the electromagnetic fields """

    def solve(self, source_z):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""

        # flatten the permittivity and source grid
        if isinstance(source_z, np.ndarray):
            source_z = torch.from_numpy(source_z).to(self.device)

        source_vec = self._grid_to_vec(source_z)
        eps_vec = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        # entries_a, indices_a = self._make_A(eps_vec)
        A = self._make_A(eps_vec)

        # solve field componets usng A and the source
        # Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
        #     eps_vec, entries_a, indices_a, source_vec
        # )

        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(eps_vec, A, source_vec)

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = self._vec_to_grid(Fx_vec)
        Fy = self._vec_to_grid(Fy_vec)
        Fz = self._vec_to_grid(Fz_vec)

        return Fx, Fy, Fz

    """ Utility functions for FDFD object """

    def _setup_derivatives(self):
        """Makes the sparse derivative matrices and does some processing for ease of use"""

        # Creates all of the operators needed for later
        ## returned coo sparse matrix
        derivs = compute_derivative_matrices(
            self.omega,
            self.shape,
            self.npml,
            self.dL,
            bloch_x=self.bloch_x,
            bloch_y=self.bloch_y,
            device=self.device,
        )

        # stores the raw sparse matrices
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = derivs

        # store the entries and elements
        # self.entries_Dxf, self.indices_Dxf = get_entries_indices(self.Dxf)
        # self.entries_Dxb, self.indices_Dxb = get_entries_indices(self.Dxb)
        # self.entries_Dyf, self.indices_Dyf = get_entries_indices(self.Dyf)
        # self.entries_Dyb, self.indices_Dyb = get_entries_indices(self.Dyb)

        # stores some convenience functions for multiplying derivative matrices by a vector `vec`
        # self.sp_mult_Dxf = lambda vec: sp_mult(self.entries_Dxf, self.indices_Dxf, vec)
        # self.sp_mult_Dxb = lambda vec: sp_mult(self.entries_Dxb, self.indices_Dxb, vec)
        # self.sp_mult_Dyf = lambda vec: sp_mult(self.entries_Dyf, self.indices_Dyf, vec)
        # self.sp_mult_Dyb = lambda vec: sp_mult(self.entries_Dyb, self.indices_Dyb, vec)

        self.sp_mult_Dxf = lambda vec: sparse_mv(self.Dxf, vec)
        self.sp_mult_Dxb = lambda vec: sparse_mv(self.Dxb, vec)
        self.sp_mult_Dyf = lambda vec: sparse_mv(self.Dyf, vec)
        self.sp_mult_Dyb = lambda vec: sparse_mv(self.Dyb, vec)

    def _setup_bloch_phases(self, bloch_phases):
        """Saves the x y and z bloch phases based on list of them 'bloch_phases'"""

        self.bloch_x = 0.0
        self.bloch_y = 0.0
        self.bloch_z = 0.0
        if bloch_phases is not None:
            self.bloch_x = bloch_phases[0]
            if len(bloch_phases) > 1:
                self.bloch_y = bloch_phases[1]
            if len(bloch_phases) > 2:
                self.bloch_z = bloch_phases[2]

    def _vec_to_grid(self, vec):
        """converts a vector quantity into an array of the shape of the FDFD simulation"""
        return vec.reshape(self.shape)

    def _grid_to_vec(self, grid):
        """converts a grid of the shape of the FDFD simulation to a flat vector"""
        return grid.flatten()

    def _save_shape(self, grid):
        """Sores the shape and size of `grid` array to the FDFD object"""
        self.shape = grid.shape
        self.Nx, self.Ny = self.shape
        self.N = self.Nx * self.Ny

    @staticmethod
    def _default_val(val, default_val=None):
        # not used yet
        return val if val is not None else default_val

    """ Field conversion functions for 2D.  Function names are self explanatory """

    def _Ex_Ey_to_Hz(self, Ex_vec, Ey_vec):
        return (
            1
            / 1j
            / self.omega
            / MU_0
            * (self.sp_mult_Dxb(Ey_vec) - self.sp_mult_Dyb(Ex_vec))
        )

    def _Ez_to_Hx(self, Ez_vec):
        return -1 / 1j / self.omega / MU_0 * self.sp_mult_Dyb(Ez_vec)

    def _Ez_to_Hy(self, Ez_vec):
        return 1 / 1j / self.omega / MU_0 * self.sp_mult_Dxb(Ez_vec)

    def _Ez_to_Hx_Hy(self, Ez_vec):
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec

    # addition of 1e-5 is for numerical stability when tracking gradients of eps_xx, and eps_yy -> 0
    def _Hz_to_Ex(self, Hz_vec, eps_vec_xx):
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_xx + 1e-5)
            * self.sp_mult_Dyf(Hz_vec)
        )

    def _Hz_to_Ey(self, Hz_vec, eps_vec_yy):
        return (
            -1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_yy + 1e-5)
            * self.sp_mult_Dxf(Hz_vec)
        )

    def _Hx_Hy_to_Ez(self, Hx_vec, Hy_vec, eps_vec_zz):
        return (
            1
            / 1j
            / self.omega
            / EPSILON_0
            / (eps_vec_zz + 1e-5)
            * (self.sp_mult_Dxf(Hy_vec) - self.sp_mult_Dyf(Hx_vec))
        )

    def _Hz_to_Ex_Ey(self, Hz_vec, eps_vec_xx, eps_vec_yy):
        Ex_vec = self._Hz_to_Ex(Hz_vec, eps_vec_xx)
        Ey_vec = self._Hz_to_Ey(Hz_vec, eps_vec_yy)
        return Ex_vec, Ey_vec


""" These are the fdfd classes that you'll actually want to use """


class fdfd_ez_torch(fdfd):
    """deprecated"""

    def __init__(self, omega, dL, eps_r, npml, bloch_phases=None, device="cpu"):
        if isinstance(eps_r, np.ndarray):
            eps_r = torch.from_numpy(eps_r).to(device)

        super().__init__(
            omega, dL, eps_r, npml, bloch_phases=bloch_phases, device=device
        )

    @torch.inference_mode()
    def _make_A(self, eps_vec):
        C = -1 / MU_0 * (sparse_mm(self.Dxf, self.Dxb) + sparse_mm(self.Dyf, self.Dyb))

        # indices into the diagonal of a sparse matrix
        entries_diag = -EPSILON_0 * self.omega**2 * eps_vec
        A = C + torch.sparse.spdiags(
            entries_diag[None, :].cpu(), torch.tensor([0]), (self.N, self.N)
        ).to(self.device)
        return A

    def _solve_fn(self, eps_vec, A, Jz_vec):
        b_vec = 1j * self.omega * Jz_vec
        A = A.coalesce()
        # Ez_vec = sp_solve(A, b_vec)
        Ez_vec = sparse_solve_torch(A, self.eps_r, b_vec)

        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec


class fdfd_ez(fdfd_ez_ceviche):
    def __init__(self, omega, dL, eps_r, npml, power=1e-8, bloch_phases=None):
        self.solver = SparseSolveTorch()
        self.power = power
        self.A = None
        if isinstance(eps_r, np.ndarray):
            eps_r = torch.from_numpy(eps_r)
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

    def _make_A(self, eps_vec: torch.Tensor):
        return super()._make_A(eps_vec.detach().cpu().numpy())

    def _Ez_to_Hx(self, Ez_vec: Tensor) -> Tensor:
        # device = Ez_vec.device
        # return torch.from_numpy(
        #     -1
        #     / 1j
        #     / self.omega
        #     / MU_0
        #     * self.sp_mult_Dyb(Ez_vec.data.cpu().numpy())
        # ).to(device)

        # print(self.indices_Dyb)
        return (
            -1
            / 1j
            / self.omega
            / MU_0
            * spmm(
                torch.from_numpy(self.indices_Dyb).to(Ez_vec.device).long(),
                torch.from_numpy(self.entries_Dyb).to(Ez_vec.device),
                m=self.N,
                n=self.N,
                matrix=Ez_vec[:, None],
            )[:, 0]
        )

    def _Ez_to_Hy(self, Ez_vec: Tensor) -> Tensor:
        # device = Ez_vec.device
        # return torch.from_numpy(
        #     1
        #     / 1j
        #     / self.omega
        #     / MU_0
        #     * self.sp_mult_Dxb(Ez_vec.data.cpu().numpy())
        # ).to(device)
        return (
            1
            / 1j
            / self.omega
            / MU_0
            * spmm(
                torch.from_numpy(self.indices_Dxb).to(Ez_vec.device).long(),
                torch.from_numpy(self.entries_Dxb).to(Ez_vec.device),
                m=self.N,
                n=self.N,
                matrix=Ez_vec[:, None],
            )[:, 0]
        )

    def _Ez_to_Hx_Hy(self, Ez_vec):
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec
    
    def norm_adj_power(self):
        Nx = self.eps_r.shape[0]
        Ny = self.eps_r.shape[1]
        x_slices = [
            Slice(
                x=np.array(self.npml[0] + 5),
                y=np.arange(
                    0 + self.npml[1] + 5,
                    Ny - self.npml[1] - 5,
                ),
            ), 
            Slice(
                x=np.array(Nx - self.npml[0] - 5),
                y=np.arange(
                    0 + self.npml[1] + 5,
                    Ny - self.npml[1] - 5,
                ),
            ), 
        ]
        y_slices = [
            Slice(
                x=np.arange(
                    0 + self.npml[0] + 5,
                    Nx - self.npml[0] - 5,
                ),
                y=np.array(self.npml[1] + 5),
            ),
            Slice(
                x=np.arange(
                    0 + self.npml[0] + 5,
                    Nx - self.npml[0] - 5,
                ),
                y=np.array(Ny - self.npml[1] - 5),
            ),
        ]
        ez_adj_dict = {}
        hx_adj_dict = {}
        hy_adj_dict = {}
        normalization_factor = {}
        with torch.no_grad():
            for key in self.solver.adj_src:
                J_adj = self.solver.adj_src[key] / 1j / self.omega # b_adj --> J_adj
                # print("this is the state of the J_adj")
                # print_stat(torch.abs(J_adj))
                hx_adj, hy_adj, ez_adj = self.solve(J_adj, "adj", "adj") # J_adj --> Hx_adj, Hy_adj, Ez_adj
                total_flux = torch.tensor([0.0,], device=J_adj.device, dtype=torch.float64) # Hx_adj, Hy_adj, Ez_adj --> 2 * total_flux
                for frame_slice in x_slices:
                    # print("this is the increment of the total flux: ", torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "x")))
                    total_flux = total_flux + torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "x")) # absolute to ensure positive flux
                for frame_slice in y_slices:
                    # print("this is the increment of the total flux: ", torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "y")))
                    total_flux = total_flux + torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "y")) # in case that opposite direction cancel each other
                total_flux = total_flux / 2 # 2 * total_flux --> total_flux
                # print(f"this is the total flux: {total_flux}")
                scale_factor = (self.power / total_flux)**0.5
                # print(f"this is the scale factor: {scale_factor}")
                normalization_factor[key] = scale_factor
                # print("ez_adj before scaling")
                # print_stat(torch.abs(ez_adj))
                ez_adj_dict[key] = ez_adj * scale_factor
                # print("ez_adj after scaling")
                # print_stat(torch.abs(ez_adj_dict[key]))
                # print("hx_adj before scaling")
                # print_stat(torch.abs(hx_adj))
                hx_adj_dict[key] = hx_adj * scale_factor
                # print("hx_adj after scaling")
                # print_stat(torch.abs(hx_adj_dict[key]))
                # print("hy_adj before scaling")
                # print_stat(torch.abs(hy_adj))
                hy_adj_dict[key] = hy_adj * scale_factor
                # print("hy_adj after scaling")
                # print_stat(torch.abs(hy_adj_dict[key]))
                self.solver.adj_src[key] = J_adj * scale_factor * 1j * self.omega # J_adj --> b_adj
        return ez_adj_dict, hx_adj_dict, hy_adj_dict, normalization_factor


    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec, port_name=None, mode=None):
        assert port_name is not None, "port_name must be provided"
        assert mode is not None, "mode must be provided"
        b_vec = 1j * self.omega * Jz_vec
        eps_diag = -EPSILON_0 * self.omega**2 * eps_vec
        # Ez_vec = sparse_solve_torch(entries_a, indices_a, eps_diag, b_vec)
        Ez_vec = self.solver(entries_a, indices_a, eps_diag, b_vec, port_name, mode)

        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

    def solve(self, source_z, port_name=None, mode=None):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec: torch.Tensor = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec)
        self.A = (entries_a, indices_a) # record the A matrix for later storage
        if port_name == "adj" and mode == "adj":
            indices_a = np.flip(indices_a, axis=0) # need to flip the indices for adjoint source
        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
            eps_vec, entries_a, indices_a, source_vec, port_name, mode
        )

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = Fx_vec.reshape(self.shape)
        Fy = Fy_vec.reshape(self.shape)
        Fz = Fz_vec.reshape(self.shape)

        return Fx, Fy, Fz

class My_fdfd_ez(fdfd):
    def __init__(self, omega, dL, eps_r, npml, power=1e-8, bloch_phases=None, device="cpu"):
        self.power = power
        if isinstance(eps_r, np.ndarray):
            eps_r = torch.from_numpy(eps_r)
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases, device=device)
        self.solver = SparseSolveTorch()

    def _setup_derivatives(self):
        """Makes the sparse derivative matrices and does some processing for ease of use"""

        # Creates all of the operators needed for later
        ## returned coo sparse matrix
        derivs = compute_derivative_matrices(
            self.omega,
            self.shape,
            self.npml,
            self.dL,
            bloch_x=self.bloch_x,
            bloch_y=self.bloch_y,
            device=self.device,
        )

        # stores the raw sparse matrices
        self.Dxf, self.Dxb, self.Dyf, self.Dyb = derivs

        # store the entries and elements
        # self.entries_Dxf, self.indices_Dxf = get_entries_indices(self.Dxf)
        # self.entries_Dxb, self.indices_Dxb = get_entries_indices(self.Dxb)
        # self.entries_Dyf, self.indices_Dyf = get_entries_indices(self.Dyf)
        # self.entries_Dyb, self.indices_Dyb = get_entries_indices(self.Dyb)

        self.sp_mult_Dxf = lambda vec: sparse_mv(self.Dxf, vec)
        self.sp_mult_Dxb = lambda vec: sparse_mv(self.Dxb, vec)
        self.sp_mult_Dyf = lambda vec: sparse_mv(self.Dyf, vec)
        self.sp_mult_Dyb = lambda vec: sparse_mv(self.Dyb, vec)
    
    # def _make_A(self, eps_vec: torch.Tensor):

    #     C = - 1 / MU_0 * (self.Dxf @ self.Dxb) \
    #         - 1 / MU_0 * (self.Dyf @ self.Dyb)
    #     entries_c, indices_c = get_entries_indices(C)

    #     # Indices into the diagonal of a sparse matrix
    #     entries_diag = - EPSILON_0 * self.omega**2 * eps_vec

    #     # Create diagonal indices for a sparse matrix
    #     indices_diag = torch.vstack((torch.arange(self.N), torch.arange(self.N)))

    #     # Concatenate entries and indices along the appropriate axis
    #     entries_a = torch.hstack((entries_diag, entries_c))
    #     indices_a = torch.hstack((indices_diag, indices_c))

    #     return entries_a, indices_a

    @torch.inference_mode()
    def _make_A(self, eps_vec):
        C = -1 / MU_0 * (sparse_mm(self.Dxf, self.Dxb) + sparse_mm(self.Dyf, self.Dyb))
        C = C.to(self.device)
        # indices into the diagonal of a sparse matrix
        entries_diag = -EPSILON_0 * self.omega**2 * eps_vec
        entries_diag = entries_diag.to(self.device)
        A = C + torch.sparse.spdiags(
            entries_diag[None, :], torch.tensor([0], device=self.device), (self.N, self.N)
        )
        # A = C + torch.sparse.spdiags(
        #     entries_diag[None, :].cpu(), torch.tensor([0]), (self.N, self.N)
        # ).to(self.device)
        entries_a, indices_a = A._values(), A._indices()
        return entries_a, indices_a

    def _Ez_to_Hx(self, Ez_vec: Tensor) -> Tensor:
        # device = Ez_vec.device
        # return torch.from_numpy(
        #     -1
        #     / 1j
        #     / self.omega
        #     / MU_0
        #     * self.sp_mult_Dyb(Ez_vec.data.cpu().numpy())
        # ).to(device)

        # print(self.indices_Dyb)
        return (
            -1
            / 1j
            / self.omega
            / MU_0
            * spmm(
                torch.from_numpy(self.indices_Dyb).to(Ez_vec.device).long(),
                torch.from_numpy(self.entries_Dyb).to(Ez_vec.device),
                m=self.N,
                n=self.N,
                matrix=Ez_vec[:, None],
            )[:, 0]
        )

    def _Ez_to_Hy(self, Ez_vec: Tensor) -> Tensor:
        # device = Ez_vec.device
        # return torch.from_numpy(
        #     1
        #     / 1j
        #     / self.omega
        #     / MU_0
        #     * self.sp_mult_Dxb(Ez_vec.data.cpu().numpy())
        # ).to(device)
        return (
            1
            / 1j
            / self.omega
            / MU_0
            * spmm(
                torch.from_numpy(self.indices_Dxb).to(Ez_vec.device).long(),
                torch.from_numpy(self.entries_Dxb).to(Ez_vec.device),
                m=self.N,
                n=self.N,
                matrix=Ez_vec[:, None],
            )[:, 0]
        )

    def _Ez_to_Hx_Hy(self, Ez_vec):
        Hx_vec = self._Ez_to_Hx(Ez_vec)
        Hy_vec = self._Ez_to_Hy(Ez_vec)
        return Hx_vec, Hy_vec
    
    def norm_adj_power(self):
        Nx = self.eps_r.shape[0]
        Ny = self.eps_r.shape[1]
        x_slices = [
            Slice(
                x=np.array(self.npml[0] + 5),
                y=np.arange(
                    0 + self.npml[1] + 5,
                    Ny - self.npml[1] - 5,
                ),
            ), 
            Slice(
                x=np.array(Nx - self.npml[0] - 5),
                y=np.arange(
                    0 + self.npml[1] + 5,
                    Ny - self.npml[1] - 5,
                ),
            ), 
        ]
        y_slices = [
            Slice(
                x=np.arange(
                    0 + self.npml[0] + 5,
                    Nx - self.npml[0] - 5,
                ),
                y=np.array(self.npml[1] + 5),
            ),
            Slice(
                x=np.arange(
                    0 + self.npml[0] + 5,
                    Nx - self.npml[0] - 5,
                ),
                y=np.array(Ny - self.npml[1] - 5),
            ),
        ]
        ez_adj_dict = {}
        hx_adj_dict = {}
        hy_adj_dict = {}
        normalization_factor = {}
        with torch.no_grad():
            for key in self.solver.adj_src:
                J_adj = self.solver.adj_src[key] / 1j / self.omega # b_adj --> J_adj
                # print("this is the state of the J_adj")
                # print_stat(torch.abs(J_adj))
                hx_adj, hy_adj, ez_adj = self.solve(J_adj, "adj", "adj") # J_adj --> Hx_adj, Hy_adj, Ez_adj
                total_flux = torch.tensor([0.0,], device=J_adj.device, dtype=torch.float64) # Hx_adj, Hy_adj, Ez_adj --> 2 * total_flux
                for frame_slice in x_slices:
                    # print("this is the increment of the total flux: ", torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "x")))
                    total_flux = total_flux + torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "x")) # absolute to ensure positive flux
                for frame_slice in y_slices:
                    # print("this is the increment of the total flux: ", torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "y")))
                    total_flux = total_flux + torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "y")) # in case that opposite direction cancel each other
                total_flux = total_flux / 2 # 2 * total_flux --> total_flux
                # print(f"this is the total flux: {total_flux}")
                scale_factor = (self.power / total_flux)**0.5
                # print(f"this is the scale factor: {scale_factor}")
                normalization_factor[key] = scale_factor
                # print("ez_adj before scaling")
                # print_stat(torch.abs(ez_adj))
                ez_adj_dict[key] = ez_adj * scale_factor
                # print("ez_adj after scaling")
                # print_stat(torch.abs(ez_adj_dict[key]))
                # print("hx_adj before scaling")
                # print_stat(torch.abs(hx_adj))
                hx_adj_dict[key] = hx_adj * scale_factor
                # print("hx_adj after scaling")
                # print_stat(torch.abs(hx_adj_dict[key]))
                # print("hy_adj before scaling")
                # print_stat(torch.abs(hy_adj))
                hy_adj_dict[key] = hy_adj * scale_factor
                # print("hy_adj after scaling")
                # print_stat(torch.abs(hy_adj_dict[key]))
                self.solver.adj_src[key] = J_adj * scale_factor * 1j * self.omega # J_adj --> b_adj
        return ez_adj_dict, hx_adj_dict, hy_adj_dict, normalization_factor


    def _solve_fn(self, eps_vec, entries_a, indices_a, Jz_vec, port_name=None, mode=None):
        assert port_name is not None, "port_name must be provided"
        assert mode is not None, "mode must be provided"
        b_vec = 1j * self.omega * Jz_vec
        eps_diag = -EPSILON_0 * self.omega**2 * eps_vec
        # Ez_vec = sparse_solve_torch(entries_a, indices_a, eps_diag, b_vec)
        Ez_vec = self.solver(entries_a, indices_a, eps_diag, b_vec, port_name, mode)

        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

    def solve(self, source_z, port_name=None, mode=None):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec: torch.Tensor = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec)
        if port_name == "adj" and mode == "adj":
            indices_a = torch.flip(indices_a, [0])

        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
            eps_vec, entries_a, indices_a, source_vec, port_name, mode
        )

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = Fx_vec.reshape(self.shape)
        Fy = Fy_vec.reshape(self.shape)
        Fz = Fz_vec.reshape(self.shape)

        return Fx, Fy, Fz