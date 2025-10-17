import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from psf_model.calibrate_psf_model import (
    PSFCalibrationModel,
    _ensure_grayscale,
    _round_up_to_odd,
    resolve_device,
)


Image.MAX_IMAGE_PIXELS = None


def _load_grayscale_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = _ensure_grayscale(img)
        array = np.array(img, dtype=np.uint8)
    return torch.from_numpy(array.copy()).float() / 255.0


def _compute_positions(length: int, window: int, stride: int) -> List[int]:
    if length <= window:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def sliding_window_predict(
    model: PSFCalibrationModel,
    image: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    sharpness: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    _, _, height, width = image.shape
    output = torch.zeros((1, 1, height, width), device=device)
    counts = torch.zeros((1, 1, height, width), device=device)

    positions_y = _compute_positions(height, patch_size, stride)
    positions_x = _compute_positions(width, patch_size, stride)

    with torch.no_grad():
        for top in positions_y:
            for left in positions_x:
                bottom = top + patch_size
                right = left + patch_size
                patch = image[:, :, top:bottom, left:right]
                pred, _, _, _ = model(patch, sharpness=sharpness)
                output[:, :, top:bottom, left:right] += pred
                counts[:, :, top:bottom, left:right] += 1.0

    output /= torch.clamp_min(counts, 1.0)
    return output.clamp_(0.0, 1.0)


def load_checkpoint_model(checkpoint_path: Path, device: torch.device) -> tuple[PSFCalibrationModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "args" not in checkpoint:
        raise KeyError("Checkpoint missing 'args' entry. Please provide a checkpoint saved by calibrate_psf_model.py.")

    ckpt_args = checkpoint["args"]
    namespace = SimpleNamespace(**ckpt_args)
    kernel_size = _round_up_to_odd(getattr(namespace, "kernel_size", 501))
    threshold_kernel_size = ckpt_args.get("threshold_kernel_size", 3)
    model = PSFCalibrationModel(
        kernel_size=kernel_size,
        threshold_kernel_size=threshold_kernel_size,
        eta=getattr(namespace, "eta", 0.55),
        sig_ax=getattr(namespace, "sig_ax", 25.0),
        sig_ay=getattr(namespace, "sig_ay", 25.0),
        sig_bx=getattr(namespace, "sig_bx", 70.0),
        sig_by=getattr(namespace, "sig_by", 70.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, ckpt_args


def collect_image_pairs(
    data_dir: Path,
    design_glob: str,
    fabricated_glob: str,
) -> Sequence[tuple[Path, Path]]:
    design_paths = sorted(data_dir.glob(design_glob))
    fabricated_paths = sorted(data_dir.glob(fabricated_glob))
    if not design_paths or not fabricated_paths:
        raise FileNotFoundError(f"No images found in {data_dir} using globs {design_glob} / {fabricated_glob}.")
    if len(design_paths) != len(fabricated_paths):
        raise ValueError("Mismatch between number of design and fabricated images.")
    return list(zip(design_paths, fabricated_paths))


def save_prediction(prediction: torch.Tensor, output_path: Path) -> None:
    array = (prediction.squeeze().cpu().numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, mode="L").save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate calibrated PSF model on design/fabricated data.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint.")
    parser.add_argument("--data-dir", type=Path, default=Path("psf_model/data"), help="Directory containing x*.png/y*.png.")
    parser.add_argument("--design-glob", type=str, default="x*.png", help="Glob for design (input) images.")
    parser.add_argument("--fabricated-glob", type=str, default="y*.png", help="Glob for fabricated (target) images.")
    parser.add_argument("--use-sliding-window", action="store_true", help="Enable sliding-window evaluation instead of full-image inference.")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size used during evaluation when --use-sliding-window is set.")
    parser.add_argument("--stride", type=int, default=256, help="Stride for sliding-window inference when enabled.")
    parser.add_argument("--sharpness", type=float, default=None, help="Override sigmoid sharpness used during evaluation.")
    parser.add_argument("--device", type=str, default=None, help="Device for inference. Defaults to cuda if available.")
    parser.add_argument("--save-dir", type=Path, default=Path("psf_model/eval_outputs"), help="Directory to store predictions.")
    parser.add_argument("--save-predictions", action="store_true", help="If set, save predicted fabricated images.")
    return parser.parse_args()


def evaluate_image(
    model: PSFCalibrationModel,
    design_tensor: torch.Tensor,
    fabricated_tensor: torch.Tensor,
    *,
    use_sliding_window: bool,
    patch_size: int,
    stride: int,
    sharpness: float,
    device: torch.device,
) -> dict:
    design_tensor = design_tensor.unsqueeze(0).unsqueeze(0).to(device)
    fabricated_tensor = fabricated_tensor.unsqueeze(0).unsqueeze(0).to(device)
    if use_sliding_window:
        prediction = sliding_window_predict(
            model,
            design_tensor,
            patch_size=patch_size,
            stride=stride,
            sharpness=sharpness,
            device=device,
        )
    else:
        with torch.no_grad():
            prediction, _, _, _ = model(design_tensor, sharpness=sharpness)

    mse = F.mse_loss(prediction, fabricated_tensor).item()
    bce = F.binary_cross_entropy(prediction, fabricated_tensor).item()
    return {
        "prediction": prediction,
        "mse": mse,
        "bce": bce,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Evaluating on device: {device}")

    model, ckpt_args = load_checkpoint_model(args.checkpoint, device)
    sharpness = args.sharpness if args.sharpness is not None else ckpt_args.get("max_sharpness", 50.0)
    print(f"Using sigmoid sharpness: {sharpness}")

    pairs = collect_image_pairs(args.data_dir, args.design_glob, args.fabricated_glob)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    total_mse = 0.0
    total_bce = 0.0

    for idx, (design_path, fabricated_path) in enumerate(pairs):
        print(f"Processing pair {idx + 1}/{len(pairs)}: {design_path.name} vs {fabricated_path.name}")
        design_tensor = _load_grayscale_tensor(design_path)
        fabricated_tensor = _load_grayscale_tensor(fabricated_path)

        result = evaluate_image(
            model,
            design_tensor,
            fabricated_tensor,
            use_sliding_window=args.use_sliding_window,
            patch_size=args.patch_size,
            stride=args.stride,
            sharpness=sharpness,
            device=device,
        )

        print(f"  MSE={result['mse']:.6f} | BCE={result['bce']:.6f}")

        total_mse += result["mse"]
        total_bce += result["bce"]

        if args.save_predictions:
            output_path = args.save_dir / f"prediction_{design_path.stem}.png"
            save_prediction(result["prediction"], output_path)
            print(f"  Saved prediction to {output_path}")

    num_pairs = len(pairs)
    if num_pairs > 0:
        print(f"\nAverage metrics across {num_pairs} pairs:")
        print(f"  MSE={total_mse / num_pairs:.6f}")
        print(f"  BCE={total_bce / num_pairs:.6f}")


if __name__ == "__main__":
    main()
