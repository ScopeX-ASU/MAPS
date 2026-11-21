from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

"""Run a heater-on-waveguide thermal sanity check inspired by the Tidy3D tutorial.
https://www.flexcompute.com/tidy3d/examples/notebooks/MetalHeaterPhaseShifter/
"""
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.heat import HeatSolver, MetalHeaterPhaseShifterCrossSection
from core.heat.problem import HAS_PYDISO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a heater-on-waveguide thermal sanity check inspired by the Tidy3D tutorial."
    )
    parser.add_argument(
        "--currents-ma",
        nargs="+",
        type=float,
        default=[0.0, 3.7, 7.4, 10.0],
        help="Heater currents in mA.",
    )
    parser.add_argument(
        "--spacing-um",
        nargs=2,
        type=float,
        default=[0.05, 0.05],
        metavar=("DX", "DZ"),
        help="Structured grid spacing in um.",
    )
    parser.add_argument(
        "--backend",
        choices=["jax", "pydiso", "scipy"],
        default="pydiso" if HAS_PYDISO else "jax",
        help="Linear solver backend.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figs/heat/metal_heater_phase_shifter"),
        help="Directory for plots.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use.",
    )
    return parser.parse_args()


def build_solver(
    device_spec: MetalHeaterPhaseShifterCrossSection, backend: str
) -> HeatSolver:
    return HeatSolver(
        spacing=device_spec.spacing_um,
        backend=backend,
        dirichlet_bc={"ymin": device_spec.ambient_temperature_k},
        neumann_bc={"xmin": 0.0, "xmax": 0.0, "ymax": 0.0},
    )


def main() -> None:
    args = parse_args()
    torch_device = torch.device(args.device)
    dtype = torch.float64

    device_spec = MetalHeaterPhaseShifterCrossSection(
        spacing_um=(float(args.spacing_um[0]), float(args.spacing_um[1]))
    )
    solver = build_solver(device_spec, args.backend)

    k_map = device_spec.build_conductivity_map(device=torch_device, dtype=dtype)
    capacity_map = device_spec.build_heat_capacity_map(device=torch_device, dtype=dtype)
    thermo_optic_map = device_spec.build_thermo_optic_map(
        device=torch_device, dtype=dtype
    )

    print("Metal heater phase-shifter thermal sanity check")
    print(
        f"grid_shape={device_spec.grid_shape}, spacing_um={device_spec.spacing_um}, backend={args.backend}"
    )
    print(
        "material ranges: "
        f"k=[{float(k_map.min()):.3e}, {float(k_map.max()):.3e}] W/(um*K), "
        f"capacity=[{float(capacity_map.min()):.1f}, {float(capacity_map.max()):.1f}] J/(kg*K), "
        f"dn/dT=[{float(thermo_optic_map.min()):.3e}, {float(thermo_optic_map.max()):.3e}] 1/K"
    )

    results = []
    for current_ma in args.currents_ma:
        current_a = float(current_ma) * 1e-3
        q_map = device_spec.build_heat_source_from_current(
            current_a,
            device=torch_device,
            dtype=dtype,
        )
        temperature = solver(k_map, q_map)
        summary = device_spec.summarize_temperature(temperature)
        summary["current_ma"] = float(current_ma)
        summary["heat_rate_max_w_per_um3"] = float(q_map.max().item())
        results.append(summary)

        plot_path = device_spec.plot_temperature_map(
            temperature,
            current_a=current_a,
            output_path=args.output_dir / f"temperature_{current_ma:.2f}mA.png",
            show_delta_t=False,
            vmin=300,
            vmax=340,
        )

        print(
            f"I={current_ma:5.2f} mA | "
            f"q_max={summary['heat_rate_max_w_per_um3']:.3e} W/um^3 | "
            f"Tmax={summary['temperature_max_k']:.3f} K | "
            f"Tcore_mean={summary['core_mean_k']:.3f} K | "
            f"Theater_mean={summary['heater_mean_k']:.3f} K | "
            f"plot={plot_path}"
        )

    if len(results) >= 2:
        reference = next((item for item in results if item["current_ma"] > 0), None)
        if reference is not None:
            ref_scale = reference["current_ma"] ** 2
            print("quadratic scaling sanity check against first nonzero current:")
            for item in results:
                if item["current_ma"] <= 0:
                    continue
                ratio = (item["core_mean_k"] - device_spec.ambient_temperature_k) / max(
                    item["current_ma"] ** 2, 1e-30
                )
                ref_ratio = (
                    reference["core_mean_k"] - device_spec.ambient_temperature_k
                ) / max(ref_scale, 1e-30)
                print(
                    f"  I={item['current_ma']:5.2f} mA | "
                    f"DeltaT_core/I^2={ratio:.6f} | "
                    f"normalized_vs_ref={ratio / ref_ratio:.4f}"
                )


if __name__ == "__main__":
    main()
