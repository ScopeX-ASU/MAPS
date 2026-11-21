"""Command-line microbenchmark for the reusable GPU rasterizer."""

from __future__ import annotations

import argparse
import time

import cupy as cp
import tidy3d as td

from .subpixel import GPUSubpixelRasterizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    axis = tuple(__import__("numpy").linspace(-5.0, 5.0, args.size + 1))
    grid = td.Grid(boundaries=td.Coords(x=axis, y=axis, z=(-0.5, 0.5)))
    structure = td.Structure(
        geometry=td.Box(center=(0, 0, 0), size=(4, 2, 1)),
        medium=td.Medium(permittivity=12.0),
    )
    rasterizer = GPUSubpixelRasterizer(grid)
    rasterizer.rasterize([structure])
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        rasterizer.rasterize([structure])
    cp.cuda.Stream.null.synchronize()
    elapsed = (time.perf_counter() - start) / args.iterations
    cells = args.size * args.size
    print(f"{cells / elapsed / 1e6:.2f} million cells/sec")
    print(f"average kernel + transfer time: {elapsed * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
