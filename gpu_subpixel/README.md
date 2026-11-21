# GPU subpixel rasterizer

`gpu_subpixel_rasterize()` uses CuPy `RawKernel` CUDA code to compute scalar
geometric volume averages on an existing Tidy3D rectilinear grid. It supports
`td.Box`, cylinders along any Cartesian axis, and z-extruded `td.PolySlab`,
uniform and nonuniform boundaries,
multiple materials, and priority-based composition. The output is a CuPy
array shaped `(Nx, Ny, Nz)`.

```python
import cupy as cp
import tidy3d as td
from gpu_subpixel import gpu_subpixel_rasterize

eps = gpu_subpixel_rasterize(
    structures=sim.structures,
    grid=sim.grid,
    background_eps=1.0,
    dtype=cp.float32,
)
```

This is area/volume averaging, not Tidy3D's full anisotropic subpixel model.
The grid is never regenerated. CPU preprocessing extracts flat geometry and
cell-index bounding ranges; the CUDA kernel uses one thread per output cell,
analytical box and circle overlap, and Sutherland-Hodgman polygon clipping for
boundary cells. Kernel compilation is cached by CuPy. `GPUSubpixelRasterizer(grid)` can
be reused across iterations to retain the grid and compiled kernel.

Run focused tests with:

```bash
pytest gpu_subpixel/tests/test_subpixel.py
python -m gpu_subpixel.benchmark --size 512
```

The supersampled CPU checker in `validation.py` is intentionally slow and is
provided for correctness experiments rather than production rasterization.
