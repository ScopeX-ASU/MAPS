import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def build_effective_psf(
    eta: float = 0.55,
    sig_ax: float = 25.0,
    sig_ay: float = 25.0,
    sig_bx: float = 70.0,
    sig_by: float = 70.0,
    step: float = 2.0,
    radius: float = 500.0,
) -> torch.Tensor:
    """Construct the effective PSF kernel using torch tensors."""
    w1 = 1.0 / (1.0 + eta)
    w2 = eta / (1.0 + eta)

    xs = torch.arange(-radius, radius + step, step, dtype=torch.float32)
    ys = torch.arange(-radius, radius + step, step, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")

    def gauss2d(x: torch.Tensor, y: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
        coeff = 1.0 / (2.0 * math.pi * sx * sy)
        exponent = -(x.square() / (2.0 * sx * sx) + y.square() / (2.0 * sy * sy))
        return coeff * torch.exp(exponent)

    kernel = w1 * gauss2d(xx, yy, sig_ax, sig_ay) + w2 * gauss2d(xx, yy, sig_bx, sig_by)
    kernel /= kernel.sum()
    return kernel


def load_image(path: Path) -> torch.Tensor:
    """Load an image and return a tensor in (C, H, W) with values in [0, 1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def synthesize_demo_image(
    image_size: tuple[int, int] = (1000, 1000),
    strip_size: tuple[int, int] = (100, 500),
    spacing: int = 50,
    strip_value: float = 1.0,
    background_value: float = 0.0,
) -> torch.Tensor:
    """Create a synthetic RGB image with two bright strips centered in the frame."""
    height, width = image_size
    strip_height, strip_width = strip_size
    if strip_height > height or strip_width > width:
        raise ValueError("Strip size must fit within the image dimensions.")
    if spacing < 0:
        raise ValueError("Spacing must be non-negative.")

    image = torch.full((1, height, width), background_value, dtype=torch.float32)
    left = (width - strip_width) // 2

    total_strip_height = strip_height * 2 + spacing
    if total_strip_height > height:
        raise ValueError("Combined strip height and spacing exceed image height.")
    top_start = (height - total_strip_height) // 2

    for i in range(2):
        top = top_start + i * (strip_height + spacing)
        image[:, top : top + strip_height, left : left + strip_width] = strip_value
    return image.repeat(3, 1, 1)


def save_tensor_image(image: torch.Tensor, path: Path) -> None:
    """Save a tensor image (C, H, W) to disk as an 8-bit PNG."""
    arr = (image.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


def apply_kernel(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply depthwise convolution with the given kernel to an image tensor."""
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape (C, H, W)")

    c, _, _ = image.shape
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    weight = kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1, 1)
    padded = F.pad(image.unsqueeze(0), (pad_w, pad_w, pad_h, pad_h), mode="reflect")
    filtered = F.conv2d(padded, weight, groups=c)
    return filtered.squeeze(0).clamp(0.0, 1.0)


def apply_etching_threshold(
    image: torch.Tensor, threshold: float, background_value: float = 0.0
) -> torch.Tensor:
    """Return a binary mask where low-intensity regions map to the background value."""
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("Threshold must be within [0, 1].")
    intensity = image.mean(dim=0, keepdim=True)
    mask = intensity >= threshold
    background = torch.full_like(image, background_value)
    highlight = torch.ones_like(image)
    return torch.where(mask, highlight, background)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo of applying PSF kernel with PyTorch."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional path to an image. If omitted, synthetic strip images are generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fdfd/synthesis_data"),
        help="Directory to save demo outputs.",
    )
    parser.add_argument(
        "--spacings",
        type=int,
        nargs="+",
        default=[40, 20],
        help="Vertical spacings (pixels) between strips for synthetic demos.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional etching threshold (0-1). Values below become background.",
    )
    args = parser.parse_args()

    kernel = build_effective_psf()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold = args.threshold

    if args.input:
        image = load_image(args.input)
        filtered = apply_kernel(image, kernel)
        thresholded = (
            apply_etching_threshold(filtered, threshold)
            if threshold is not None
            else None
        )

        cols = 2 if thresholded is None else 3
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
        axes = np.atleast_1d(axes)

        axes[0].imshow(image.permute(1, 2, 0).numpy())
        axes[0].set_title(args.input.name)
        axes[0].axis("off")
        axes[1].imshow(filtered.permute(1, 2, 0).numpy())
        axes[1].set_title("Filtered with PSF")
        axes[1].axis("off")

        if thresholded is not None:
            axes[2].imshow(thresholded.permute(1, 2, 0).numpy())
            axes[2].set_title(f"Thresholded (>{threshold:g})")
            axes[2].axis("off")

        plt.tight_layout()
        output_path = output_dir / f"{args.input.stem}_psf_filtered.png"
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"Saved filtered visualization to {output_path}")

        if thresholded is not None:
            threshold_path = output_dir / f"{args.input.stem}_psf_thresholded.png"
            save_tensor_image(thresholded, threshold_path)
            print(f"Saved thresholded image to {threshold_path}")
    else:
        spacings = args.spacings
        rows = len(spacings)
        cols = 3 if threshold is None else 4
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axes = np.atleast_2d(axes)

        comparison_path = output_dir / "pytorch_psf_demo_spacings_comparison.png"

        for row, spacing in enumerate(spacings):
            image = synthesize_demo_image(spacing=spacing)
            filtered = apply_kernel(image, kernel)
            thresholded = (
                apply_etching_threshold(filtered, threshold)
                if threshold is not None
                else None
            )
            diff_source = thresholded if thresholded is not None else filtered
            diff = (diff_source - image).abs()

            input_save_path = (
                output_dir / f"pytorch_psf_demo_spacing_{spacing}_input.png"
            )
            filtered_save_path = (
                output_dir / f"pytorch_psf_demo_spacing_{spacing}_filtered.png"
            )
            diff_save_path = (
                output_dir / f"pytorch_psf_demo_spacing_{spacing}_difference.png"
            )

            save_tensor_image(image, input_save_path)
            save_tensor_image(filtered, filtered_save_path)
            max_diff = diff.max()
            if max_diff <= 1e-8:
                diff_norm = torch.zeros_like(diff)
            else:
                diff_norm = diff / max_diff
            save_tensor_image(diff_norm, diff_save_path)

            axes[row, 0].imshow(image.permute(1, 2, 0).numpy())
            axes[row, 0].set_title(f"Spacing {spacing}px (input)")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(filtered.permute(1, 2, 0).numpy())
            axes[row, 1].set_title("Filtered (PSF)")
            axes[row, 1].axis("off")

            col_idx = 2
            if thresholded is not None:
                threshold_save_path = (
                    output_dir / f"pytorch_psf_demo_spacing_{spacing}_thresholded.png"
                )
                save_tensor_image(thresholded, threshold_save_path)
                axes[row, col_idx].imshow(thresholded.permute(1, 2, 0).numpy())
                axes[row, col_idx].set_title(f"Thresholded (>{threshold:g})")
                axes[row, col_idx].axis("off")
                print(f"Saved thresholded image to {threshold_save_path}")
                col_idx += 1

            axes[row, col_idx].imshow(
                diff_norm.permute(1, 2, 0).numpy()[:, :, 0], cmap="magma"
            )
            axes[row, col_idx].set_title("Abs difference")
            axes[row, col_idx].axis("off")

            print(f"Saved synthetic input to {input_save_path}")
            print(f"Saved filtered visualization to {filtered_save_path}")
            print(f"Saved difference map to {diff_save_path}")

        plt.tight_layout()
        fig.savefig(comparison_path, dpi=200)
        plt.close(fig)
        print(f"Saved comparison figure to {comparison_path}")


if __name__ == "__main__":
    main()
