"""Compare bounded envelopes with full-domain recording in the same phasor path."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gradient,
    pytest.mark.fdtdx,
    pytest.mark.gpu,
    pytest.mark.slow,
]


def _record_result(output, grid_mode="uniform", recording="envelope"):
    """Run identical shared bilayer corners in the process's recording mode.

    Args:
        output: JSON destination for objectives and flattened design gradients.
        grid_mode: Uniform or native nonuniform rectilinear discretization.
        recording: Envelope or full-domain design-region detector coverage.
    """
    import torch
    from benchmark_grating_coupler_variation import build_model

    from core.invdes.variation import MapsGeometryTransform

    torch.manual_seed(541)
    optical_grid = None
    if grid_mode == "nonuniform":
        optical_grid = dict(
            mode="fdtdx_rectilinear_nonuniform",
            wavelength=1.31,
            **{
                f"grid_{axis}": dict(type="auto", min_steps_per_wvl=15)
                for axis in "xyz"
            },
        )
    model = build_model(
        torch.device("cuda:0"), resolution=20, max_time=3e-14, optical_grid=optical_grid
    )

    class FullDomainTransform(MapsGeometryTransform):
        """Use full-domain recording as a reference for spatial coverage only."""

        def recording_bounds(self, support):
            """Validate support, then record every native cell.

            Args:
                support: Geometry deviation intervals.

            Returns:
                Full-domain native bounds for each design region.
            """
            validated = super().recording_bounds(support)
            full = tuple(
                (0, len(axis))
                for axis in model.device.fdtdx_native_grid_metadata["coords"]
            )
            return {name: full for name in validated}

    materializer = (
        FullDomainTransform(model)
        if recording == "full"
        else MapsGeometryTransform(model)
    )
    model.configure_variation(
        dict(
            enabled=True,
            seed=42,
            scenarios=[
                {},
                {"grating_layer_1.dx": 0.13, "grating_layer_1.dy": -0.11},
                {"grating_layer_2.dx": -0.12, "grating_layer_2.dy": 0.10},
                {
                    "grating_layer_1.etch_depth": 0.025,
                    "grating_layer_1.thickness": 0.015,
                },
            ],
        ),
        materializer=materializer,
    )
    result = model(sharpness=2.0)
    (-result["obj"]).backward()
    gradient = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
    Path(output).write_text(
        json.dumps(
            dict(
                objective=float(result["obj"].detach()),
                scenarios=result["scenario_objectives"].cpu().tolist(),
                gradient=gradient.cpu().tolist(),
            )
        )
    )


@pytest.mark.parametrize("grid_mode", ["uniform", "nonuniform"])
def test_sparse_variation_matches_full_domain_objectives_and_design_gradients(
    require_cuda, tmp_path, grid_mode
):
    """Compare spatial coverage without changing the field/adjoint convention."""
    results = []
    for recording in ["full", "envelope"]:
        output = tmp_path / f"recording-{recording}.json"
        log = tmp_path / f"recording-{recording}.log"
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="0",
            MAPS_SPARSE_PHASOR="1",
            OMP_NUM_THREADS="1",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            MPLCONFIGDIR="/tmp/maps-variation-mpl",
        )
        with log.open("w") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    str(output),
                    grid_mode,
                    recording,
                ],
                cwd=Path(__file__).resolve().parents[3],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=360,
            )
        assert process.returncode == 0, log.read_text()[-10000:]
        results.append(json.loads(output.read_text()))
    dense, sparse = results
    # Coverage changes preserve float32 phasors. A 1e-12 absolute floor avoids
    # making comparisons vacuous for the ~1e-8 objectives of this short pulse.
    np.testing.assert_allclose(
        sparse["scenarios"], dense["scenarios"], rtol=1e-4, atol=1e-12
    )
    np.testing.assert_allclose(
        sparse["objective"], dense["objective"], rtol=1e-4, atol=1e-12
    )
    assert np.linalg.norm(dense["gradient"]) > 0
    np.testing.assert_allclose(
        sparse["gradient"], dense["gradient"], rtol=1e-4, atol=1e-12
    )


if __name__ == "__main__":
    _record_result(sys.argv[1], sys.argv[2], sys.argv[3])
