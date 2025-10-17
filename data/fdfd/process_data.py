#!/usr/bin/env python3
"""Convert an epsilon map stored in an HDF5 file into a PNG image."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required for process_date.py.\n"
        "Install it with `pip install pillow` and re-run the script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the 'eps_map' dataset from an HDF5 file to a PNG image."
    )
    parser.add_argument(
        "--input",
        # default="/home/hzhou144/projects/MAPS_local/data/fdfd/mdm/raw_opt_traj_ptb/mdm_id-0_opt_step_63-in_slice_1-1.55-Ez2-300.h5",
        default="/home/hzhou144/projects/MAPS_local/data/fdfd/tdm/raw_opt_traj_ptb/tdm_id-0_opt_step_62-in_slice_1-1.55-Ez1-360.h5",
        type=Path,
        help="Path to the input HDF5 file containing the dataset.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional path for the output PNG file. Defaults to the input name with a .png suffix.",
    )
    parser.add_argument(
        "--dataset",
        default="eps_map",
        help="Name of the dataset to read (default: eps_map).",
    )
    return parser.parse_args()


def load_eps_map(path: Path, dataset: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"Dataset '{dataset}' not found in {path}.")
        # data = np.flipud(np.array(handle[dataset]).T)
        data = np.array(handle[dataset])
    return data


def normalize_to_uint8(eps_map: np.ndarray) -> np.ndarray:
    array = np.asarray(eps_map)
    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D array for the epsilon map, received shape {array.shape}."
        )
    if np.iscomplexobj(array):
        array = np.abs(array)
    array = array.astype(np.float32, copy=False)
    min_val = float(array.min())
    max_val = float(array.max())
    if max_val == min_val:
        return np.zeros_like(array, dtype=np.uint8)
    threshold = 0.5 * (min_val + max_val)
    binary = np.where(array > threshold, 255, 0)
    return binary.astype(np.uint8)


def save_png(data: np.ndarray, output_path: Path) -> None:
    image = Image.fromarray(data, mode="L")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")
    eps_map = load_eps_map(input_path, args.dataset)
    unique_count = np.unique(eps_map).size
    normalized = normalize_to_uint8(eps_map)
    normalized_count = np.unique(normalized).size
    output_path = args.output or input_path.with_suffix(".png")
    save_png(normalized, output_path)
    print(
        f"Saved {args.dataset} from {input_path} to {output_path} "
        f"(unique elements: {unique_count})"
        f"(unique elements: {normalized_count})"
    )


if __name__ == "__main__":
    main()
