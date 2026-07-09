from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
mpl.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class MetalHeaterPhaseShifterCrossSection:
    """2D x-z cross section matching the Tidy3D metal-heater phase-shifter tutorial."""

    width_um: float = 16.0
    wafer_thickness_um: float = 0.5
    box_thickness_um: float = 2.0
    clad_thickness_um: float = 2.8
    core_width_um: float = 0.5
    core_thickness_um: float = 0.22
    heater_width_um: float = 2.0
    heater_thickness_um: float = 0.14
    heater_offset_um: float = 2.0
    sigma_heater_s_per_um: float = 2.3
    ambient_temperature_k: float = 300.0
    spacing_um: tuple[float, float] = (0.05, 0.05)

    si_k: float = 148e-6
    sio2_k: float = 1.38e-6
    tin_k: float = 28e-6
    si_capacity: float = 710.0
    sio2_capacity: float = 709.0
    tin_capacity: float = 598.0
    si_dn_dT: float = 1.86e-4
    sio2_dn_dT: float = 1.0e-5
    tin_dn_dT: float = 0.0

    def __post_init__(self) -> None:
        self.dx_um = float(self.spacing_um[0])
        self.dz_um = float(self.spacing_um[1])
        self.x_min_um = -self.width_um / 2
        self.x_max_um = self.width_um / 2
        self.z_min_um = -self.wafer_thickness_um
        self.z_max_um = self.box_thickness_um + self.clad_thickness_um

        self.x_centers_um = self._centers(self.x_min_um, self.x_max_um, self.dx_um)
        self.z_centers_um = self._centers(self.z_min_um, self.z_max_um, self.dz_um)
        self.xx_um, self.zz_um = np.meshgrid(
            self.x_centers_um,
            self.z_centers_um,
            indexing="ij",
        )

        self.core_z_min_um = self.box_thickness_um
        self.core_z_max_um = self.box_thickness_um + self.core_thickness_um
        self.heater_z_min_um = self.core_z_max_um + self.heater_offset_um
        self.heater_z_max_um = self.heater_z_min_um + self.heater_thickness_um

        self.wafer_mask = self.zz_um < 0.0
        self.core_mask = (
            (np.abs(self.xx_um) <= self.core_width_um / 2)
            & (self.zz_um >= self.core_z_min_um)
            & (self.zz_um <= self.core_z_max_um)
        )
        self.heater_mask = (
            (np.abs(self.xx_um) <= self.heater_width_um / 2)
            & (self.zz_um >= self.heater_z_min_um)
            & (self.zz_um <= self.heater_z_max_um)
        )
        self.clad_mask = ~(self.wafer_mask | self.core_mask | self.heater_mask)

    @staticmethod
    def _centers(vmin: float, vmax: float, step: float) -> np.ndarray:
        count = int(round((vmax - vmin) / step))
        if count <= 0:
            raise ValueError("Invalid domain or spacing.")
        return vmin + (np.arange(count, dtype=np.float64) + 0.5) * step

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.x_centers_um.size, self.z_centers_um.size)

    def build_conductivity_map(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        k_map = np.full(self.grid_shape, self.sio2_k, dtype=np.float64)
        k_map[self.wafer_mask] = self.si_k
        k_map[self.core_mask] = self.si_k
        k_map[self.heater_mask] = self.tin_k
        return torch.as_tensor(k_map, device=device, dtype=dtype)

    def build_heat_capacity_map(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        c_map = np.full(self.grid_shape, self.sio2_capacity, dtype=np.float64)
        c_map[self.wafer_mask] = self.si_capacity
        c_map[self.core_mask] = self.si_capacity
        c_map[self.heater_mask] = self.tin_capacity
        return torch.as_tensor(c_map, device=device, dtype=dtype)

    def build_thermo_optic_map(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        dn_map = np.full(self.grid_shape, self.sio2_dn_dT, dtype=np.float64)
        dn_map[self.wafer_mask] = self.si_dn_dT
        dn_map[self.core_mask] = self.si_dn_dT
        dn_map[self.heater_mask] = self.tin_dn_dT
        return torch.as_tensor(dn_map, device=device, dtype=dtype)

    def build_heat_source_from_current(
        self,
        current_a: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        q_map = np.zeros(self.grid_shape, dtype=np.float64)
        heat_rate = (current_a / self.heater_thickness_um / self.heater_width_um) ** 2
        heat_rate /= self.sigma_heater_s_per_um
        q_map[self.heater_mask] = heat_rate
        return torch.as_tensor(q_map, device=device, dtype=dtype)

    def summarize_temperature(
        self,
        temperature_map_k: torch.Tensor,
    ) -> dict[str, float]:
        temperature_np = temperature_map_k.detach().cpu().numpy()
        core_values = temperature_np[self.core_mask]
        heater_values = temperature_np[self.heater_mask]
        return {
            "temperature_min_k": float(temperature_np.min()),
            "temperature_max_k": float(temperature_np.max()),
            "core_mean_k": float(core_values.mean()),
            "core_max_k": float(core_values.max()),
            "heater_mean_k": float(heater_values.mean()),
            "heater_max_k": float(heater_values.max()),
        }

    def plot_temperature_map(
        self,
        temperature_map_k: torch.Tensor,
        *,
        current_a: float,
        output_path: str | Path,
        show_delta_t: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temperature_np = temperature_map_k.detach().cpu().numpy()
        plot_values = (
            temperature_np - self.ambient_temperature_k
            if show_delta_t
            else temperature_np
        )
        label = "Delta T (K)" if show_delta_t else "Temperature (K)"

        with mpl.rc_context({"text.usetex": False}):
            fig, ax = plt.subplots(figsize=(8, 4))
            image = ax.imshow(
                plot_values.T,
                origin="lower",
                extent=(
                    self.x_min_um,
                    self.x_max_um,
                    self.z_min_um,
                    self.z_max_um,
                ),
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
                cmap="RdBu_r",
            )
            self._add_geometry_overlays(ax)
            ax.set_xlabel("x (um)")
            ax.set_ylabel("z (um)")
            ax.set_title(f"Metal heater phase shifter, I = {1e3 * current_a:.2f} mA")
            cbar = fig.colorbar(image, ax=ax)
            cbar.set_label(label)
            fig.tight_layout()
            fig.savefig(output_path, dpi=180)
            plt.close(fig)
        return output_path

    def _add_geometry_overlays(self, ax) -> None:
        ax.add_patch(
            plt.Rectangle(
                (-self.core_width_um / 2, self.core_z_min_um),
                self.core_width_um,
                self.core_thickness_um,
                fill=False,
                edgecolor="cyan",
                linewidth=1.2,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (-self.heater_width_um / 2, self.heater_z_min_um),
                self.heater_width_um,
                self.heater_thickness_um,
                fill=False,
                edgecolor="lime",
                linewidth=1.2,
            )
        )
