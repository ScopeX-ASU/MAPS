import numpy as np
import torch
from thirdparty.ceviche.ceviche import fdfd_ez as fdfd_ez_ceviche
from thirdparty.ceviche.ceviche.constants import *
from thirdparty.ceviche.ceviche.derivatives import create_sfactor
from torch import Tensor, nn
from torch_sparse import spmm

from core.utils import (
    Slice,
    get_flux,
)

from .derivatives import compute_derivative_matrices
from .preconditioner import create_symmetrizer
from .solver import SparseSolveTorch, sparse_solve_torch

# notataion is similar to that used in: http://www.jpier.org/PIERB/pierb36/11.11092006.pdf
from .utils import sparse_mm, sparse_mv

__all__ = ["fdfd", "fdfd_ez"]


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
    def __init__(
        self,
        omega,
        dL,
        eps_r,
        npml,
        power=1e-8,
        bloch_phases=None,
        neural_solver=None,
        numerical_solver="solve_direct",
        use_autodiff: bool = False,
        sym_precond: bool = False,
    ):
        self.power = power
        self.A = None
        self.neural_solver = neural_solver
        self.numerical_solver = numerical_solver
        if self.numerical_solver == "solve_direct":
            assert (
                self.neural_solver is None
            ), "neural_solver is useless if numerical_solver is solve_direct"
        self.solver = SparseSolveTorch(
            shape=eps_r.shape,
            neural_solver=self.neural_solver, 
            numerical_solver=self.numerical_solver,
            use_autodiff=use_autodiff,
        )
        if isinstance(eps_r, np.ndarray):
            eps_r = torch.from_numpy(eps_r)
        super().__init__(omega, dL, eps_r, npml, bloch_phases=bloch_phases)

        self.Pl = self.Pr = None
        # if run this function, will enable symmetric precondictioner
        if sym_precond:
            self._make_precond()
    
    def _save_shape(self, grid):
        """ 
        Sores the shape and size of `grid` array to the FDFD object 
        override the parent class method
        """
        self.shape = grid.shape
        self.Nx, self.Ny = self.shape
        self.N = self.Nx * self.Ny
        if hasattr(self.solver, "set_shape"):
            self.solver.set_shape(self.shape)

    def switch_solver(self, neural_solver=None, numerical_solver="solve_direct", use_autodiff=False):
        assert hasattr(self, "shape"), "shape must be set before switching solver"
        self.neural_solver = neural_solver
        self.numerical_solver = numerical_solver
        self.solver = SparseSolveTorch(
            shape=self.shape,
            neural_solver=self.neural_solver, 
            numerical_solver=self.numerical_solver,
            use_autodiff=use_autodiff,
        ).set_shape(self.shape)

    def _make_precond(self):
        Nx, Ny = self.shape
        Nx_pml, Ny_pml = self.npml

        # Create the sfactor in each direction and for 'f' and 'b'
        sxb = create_sfactor("b", self.omega, self.dL, Nx, Nx_pml)
        syb = create_sfactor("b", self.omega, self.dL, Ny, Ny_pml)
   
        [Sxb, Syb] = np.meshgrid(sxb, syb, indexing="ij")
        self.Pl, self.Pr = create_symmetrizer(Sxb, Syb)

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
        if not hasattr(self, "indices_Dyb_torch"):
            self.indices_Dyb_torch = torch.from_numpy(self.indices_Dyb).to(Ez_vec.device).long()
            self.entries_Dyb_torch = torch.from_numpy(self.entries_Dyb).to(Ez_vec.device)
        
        if len(Ez_vec.shape) == 1:
            Ez = Ez_vec[:, None]
        else:
            ## can have many batch dimension[..., n]
            Ez = Ez.flatten(0, -2).t()
        Hx = (
            -1
            / 1j
            / self.omega
            / MU_0
            * spmm(
                self.indices_Dyb_torch,
                self.entries_Dyb_torch,
                m=self.N,
                n=self.N,
                matrix=Ez,
            )
        )

        return Hx.t().reshape(Ez_vec.shape)
        


    def _Ez_to_Hy(self, Ez_vec: Tensor) -> Tensor:
        # device = Ez_vec.device
        # return torch.from_numpy(
        #     1
        #     / 1j
        #     / self.omega
        #     / MU_0
        #     * self.sp_mult_Dxb(Ez_vec.data.cpu().numpy())
        # ).to(device)
        if not hasattr(self, "indices_Dxb_torch"):
            self.indices_Dxb_torch = torch.from_numpy(self.indices_Dxb).to(Ez_vec.device).long()
            self.entries_Dxb_torch = torch.from_numpy(self.entries_Dxb).to(Ez_vec.device)

        if len(Ez_vec.shape) == 1:
            Ez = Ez_vec[:, None]
        else:
            ## can have many batch dimension[..., n]
            Ez = Ez.flatten(0, -2).t()

        Hy = (
            1
            / 1j
            / self.omega
            / MU_0
            * spmm(
                self.indices_Dxb_torch,
                self.entries_Dxb_torch,
                m=self.N,
                n=self.N,
                matrix=Ez,
            )
        )
        return Hy.t().reshape(Ez_vec.shape)

    def _Ez_to_Hx_Hy(self, Ez_vec):
        ## Ez_vec: [..., Nx*Ny] can support arbitrary batch or just 1-D vector
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
                J_adj = self.solver.adj_src[key] / 1j / self.omega  # b_adj --> J_adj
                # print("this is the state of the J_adj")
                # print_stat(torch.abs(J_adj))
                hx_adj, hy_adj, ez_adj = self.solve(
                    J_adj, "adj", "adj"
                )  # J_adj --> Hx_adj, Hy_adj, Ez_adj
                total_flux = torch.tensor(
                    [
                        0.0,
                    ],
                    device=J_adj.device,
                    dtype=torch.float64,
                )  # Hx_adj, Hy_adj, Ez_adj --> 2 * total_flux
                for frame_slice in x_slices:
                    # print("this is the increment of the total flux: ", torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "x")))
                    total_flux = total_flux + torch.abs(
                        get_flux(
                            hx_adj, hy_adj, ez_adj, frame_slice, self.dL / 1e-6, "x"
                        )
                    )  # absolute to ensure positive flux
                for frame_slice in y_slices:
                    # print("this is the increment of the total flux: ", torch.abs(get_flux(hx_adj, hy_adj, ez_adj, frame_slice, self.dL/1e-6, "y")))
                    total_flux = total_flux + torch.abs(
                        get_flux(
                            hx_adj, hy_adj, ez_adj, frame_slice, self.dL / 1e-6, "y"
                        )
                    )  # in case that opposite direction cancel each other
                total_flux = total_flux / 2  # 2 * total_flux --> total_flux
                # print(f"this is the total flux: {total_flux}")
                scale_factor = (self.power / total_flux) ** 0.5
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
                self.solver.adj_src[key] = (
                    J_adj * scale_factor * 1j * self.omega
                )  # J_adj --> b_adj
        return ez_adj_dict, hx_adj_dict, hy_adj_dict, normalization_factor

    def _solve_fn(
        self, eps_vec, entries_a, indices_a, Jz_vec, port_name=None, mode=None
    ):
        assert port_name is not None, "port_name must be provided"
        assert mode is not None, "mode must be provided"
        b_vec = 1j * self.omega * Jz_vec
        eps_diag = -EPSILON_0 * self.omega**2 * eps_vec
        # Ez_vec = sparse_solve_torch(entries_a, indices_a, eps_diag, b_vec)
        Ez_vec = self.solver(
            entries_a, indices_a, eps_diag, self.omega, b_vec, port_name, mode, self.Pl, self.Pr
        )
        Hx_vec, Hy_vec = self._Ez_to_Hx_Hy(Ez_vec)
        return Hx_vec, Hy_vec, Ez_vec

    def solve(self, source_z, port_name=None, mode=None):
        """Outward facing function (what gets called by user) that takes a source grid and returns the field components"""

        # flatten the permittivity and source grid
        source_vec = self._grid_to_vec(source_z)
        eps_vec: torch.Tensor = self._grid_to_vec(self.eps_r)

        # create the A matrix for this polarization
        entries_a, indices_a = self._make_A(eps_vec)
        self.A = (entries_a, indices_a)  # record the A matrix for later storage
        if port_name == "adj" and mode == "adj":
            indices_a = np.flip(
                indices_a, axis=0
            )  # need to flip the indices for adjoint source
        # solve field componets usng A and the source
        Fx_vec, Fy_vec, Fz_vec = self._solve_fn(
            eps_vec, entries_a, indices_a, source_vec, port_name, mode
        )

        # put all field components into a tuple, convert to grid shape and return them all
        Fx = Fx_vec.reshape(self.shape)
        Fy = Fy_vec.reshape(self.shape)
        Fz = Fz_vec.reshape(self.shape)

        return Fx, Fy, Fz