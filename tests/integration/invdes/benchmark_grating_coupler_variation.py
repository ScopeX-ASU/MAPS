"""GPU regression fixtures and profiling for variation-aware bilayer gratings.

Run this test utility from the repository root for nominal/fixed-seed/random-seed
comparisons. Short simulation durations establish execution, not convergence.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from core.invdes.models import GratingCouplerOptimization
from core.invdes.models.base_optimization import DefaultSimulationConfig
from core.invdes.models.layers import GratingCoupler


def build_model(operation_device, resolution=20, max_time=1e-13, optical_grid=None):
    """Construct the small bilayer device used only for GPU regression/profiling.

    Args:
        operation_device: Explicit PyTorch device for material/design tensors.
        resolution: Uniform samples per micrometer.
        max_time: FDTD simulation duration in seconds.
        optical_grid: Optional native rectilinear grid configuration.

    Returns:
        Initialized grating optimization model with native source normalization.
    """
    sim_cfg = DefaultSimulationConfig()
    sim_cfg.update(
        dict(
            solver="fdtdx",
            subpixel=True,
            resolution=resolution,
            max_time=max_time,
            symmetry=(0, 0, 0),
            border_width=[0.8, 0.0, 0.3, 0.3, 0.0, 0.0],
            PML=[0.3, 0.3, 0.3],
            wl_cen=1.31,
            wl_width=0.0,
            n_wl=1,
            plot_root="/tmp/maps-variation",
        )
    )
    if optical_grid is not None:
        sim_cfg["optical_grid"] = optical_grid
    device = GratingCoupler(
        material_r=3.5**2,
        material_layers=[3.5**2, 2.0**2],
        material_buried=1.44**2,
        material_cladding=1.44**2,
        thickness_layers=[0.22, 0.4],
        etch_depth_layers=[0.16, 0.3],
        gap_layers=[0.0, 0.1],
        sim_cfg=sim_cfg,
        design_region_size=(1.2, 1.2),
        port_len=0.8,
        port_width=0.4,
        port_height=0.22,
        top_port_len=1.5,
        substrate_thickness=0.3,
        substrate_gap=0.6,
        waist_radius=0.6,
        waist_distance=0.0,
        is_3d=True,
        in_mode="Hz2",
        out_mode="Hz1",
        device=operation_device,
        verbose=False,
    )
    hr_device = device.copy(resolution=resolution)
    params = {
        name: dict(
            method="levelset",
            rho_resolution=[5, 5],
            transform=[],
            init_method="random",
            denorm_mode="linear_eps",
            interpolation="bilinear",
            binary_projection=dict(fw_threshold=100, bw_threshold=100, mode="regular"),
        )
        for name in device.design_region_cfgs
    }
    objectives = dict(
        override=True,
        fwd_trans=dict(
            weight=1.0,
            in_slice_name="in_slice_1",
            out_slice_name="out_slice_1",
            in_mode="Hz2",
            wl=[1.31],
            temp=[300],
            out_modes=("Hz1",),
            type="eigenmode",
            direction="x+",
        ),
    )
    return GratingCouplerOptimization(
        device=device,
        hr_device=hr_device,
        design_region_param_cfgs=params,
        sim_cfg=sim_cfg,
        obj_cfgs=objectives,
        operation_device=operation_device,
    )


def variation_config(seed=42, num_samples=2):
    """Return fabrication/source deviation distributions for this bilayer.

    Args:
        seed: Integer deterministic seed, or None for a recorded entropy seed.
        num_samples: Sequential scenarios evaluated per optimization step.

    Returns:
        Variation configuration using independent SciPy distributions. Etch
        depth samples are additive deviations from the nominal depth in um.
    """
    return dict(
        enabled=True,
        seed=seed,
        num_samples=num_samples,
        resample_every=1,
        strategy="iid",
        aggregation=dict(mode="mean"),
        distributions=[
            dict(
                params=["grating_layer_1.dx", "grating_layer_1.dy"],
                type="norm",
                loc=0.0,
                scale=0.01,
                min=-0.03,
                max=0.03,
            ),
            dict(
                params=["grating_layer_2.dx", "grating_layer_2.dy"],
                type="norm",
                loc=0.0,
                scale=0.02,
                min=-0.06,
                max=0.06,
            ),
            dict(
                params=["grating_layer_1.etch_depth"],
                type="uniform",
                min=-0.01,
                max=0.01,
            ),
            dict(
                params=["source.in_slice_1.dx", "source.in_slice_1.dy"],
                type="norm",
                loc=0.0,
                scale=0.02,
                min=-0.06,
                max=0.06,
            ),
        ],
    )


def benchmark(model, steps=1, variation=None, learning_rate=0.01):
    """Run shared-design optimizer steps and record synchronized timings.

    Args:
        model: Initialized optimization model; parameters are updated in-place.
        steps: Number of logical optimization steps.
        variation: Optional enabled/disabled variation configuration.
        learning_rate: Adam learning rate.

    Returns:
        List of objective, timing, and peak Torch GPU-memory records. JAX memory
        is not included in Torch's allocator measurement.
    """
    if variation is not None:
        model.configure_variation(variation)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    cuda = next(model.parameters()).is_cuda
    records = []
    for step in range(steps):
        if getattr(model, "variation", None) is not None:
            model.variation.step = step
        optimizer.zero_grad()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        results = model(sharpness=2.0)
        if cuda:
            torch.cuda.synchronize()
        forward_end = time.perf_counter()
        (-results["obj"]).backward()
        if cuda:
            torch.cuda.synchronize()
        backward_end = time.perf_counter()
        optimizer.step()
        if cuda:
            torch.cuda.synchronize()
        record = dict(
            step=step,
            objective=float(results["obj"].detach()),
            forward_seconds=forward_end - start,
            outer_backward_seconds=backward_end - forward_end,
            total_seconds=time.perf_counter() - start,
            peak_torch_gpu_bytes=torch.cuda.max_memory_allocated() if cuda else None,
            gradient_norm=float(
                torch.stack(
                    [p.grad.norm() for p in model.parameters() if p.grad is not None]
                ).norm()
            ),
            variation_host_timings=results.get("variation_timings"),
        )
        if "scenario_objectives" in results:
            record["scenario_objectives"] = results["scenario_objectives"].tolist()
        records.append(record)
        print(json.dumps(record), flush=True)
    return records


def main():
    """Run the selected benchmark and optionally save machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["nominal", "fixed", "random", "compare"], default="compare"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--max-time", type=float, default=1e-13)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(541)
    model = build_model(torch.device(args.device), args.resolution, args.max_time)
    nominal_state = {k: v.clone() for k, v in model.state_dict().items()}
    output = {}
    modes = ["nominal", "fixed", "random"] if args.mode == "compare" else [args.mode]
    for mode in modes:
        if getattr(model, "variation", None) is not None:
            model.configure_variation({"enabled": False})
        model.load_state_dict(nominal_state)
        config = (
            None
            if mode == "nominal"
            else variation_config(None if mode == "random" else 42, args.samples)
        )
        if config is not None:
            model.configure_variation(config)
        output[mode] = dict(
            seed=model.variation.sampler.seed if config is not None else None,
            steps=benchmark(model, args.steps),
        )
    if args.output:
        args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
