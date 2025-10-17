import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from psf_model.model.psf_model import build_effective_psf

Image.MAX_IMAGE_PIXELS = None

def _ensure_grayscale(image: Image.Image) -> Image.Image:
    if image.mode != "L":
        return image.convert("L")
    return image


def _load_grayscale_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = _ensure_grayscale(img)
        array = np.array(img, dtype=np.uint8)
    return torch.from_numpy(array.copy())


class PatchDataset(Dataset):
    def __init__(
        self,
        design_paths: Sequence[Path],
        fabricated_paths: Sequence[Path],
        patch_size: int,
        patches_per_image: int,
        *,
        seed: int,
        augment: bool = False,
    ) -> None:
        if len(design_paths) != len(fabricated_paths):
            raise ValueError("Design and fabricated image counts do not match.")
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive.")
        self.patch_size = patch_size
        self.augment = augment
        self.design_tensors = [_load_grayscale_tensor(p) for p in design_paths]
        self.fab_tensors = [_load_grayscale_tensor(p) for p in fabricated_paths]

        rng = np.random.default_rng(seed)
        self.samples: List[Tuple[int, int, int]] = []
        for idx, (design, fabricated) in enumerate(zip(self.design_tensors, self.fab_tensors)):
            if design.shape != fabricated.shape:
                raise ValueError(f"Shape mismatch for image pair {design_paths[idx]} and {fabricated_paths[idx]}.")
            height, width = design.shape
            if height < patch_size or width < patch_size:
                raise ValueError(
                    f"Patch size {patch_size} exceeds image dimensions {(height, width)} for {design_paths[idx]}."
                )
            max_top = max(height - patch_size, 0)
            max_left = max(width - patch_size, 0)
            for _ in range(patches_per_image):
                top = int(rng.integers(0, max_top + 1)) if max_top > 0 else 0
                left = int(rng.integers(0, max_left + 1)) if max_left > 0 else 0
                self.samples.append((idx, top, left))
        random.Random(seed).shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx, top, left = self.samples[index]
        bottom = top + self.patch_size
        right = left + self.patch_size
        design_patch = self.design_tensors[img_idx][top:bottom, left:right].to(torch.float32) / 255.0
        fabricated_patch = self.fab_tensors[img_idx][top:bottom, left:right].to(torch.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                design_patch = torch.flip(design_patch, dims=[1])
                fabricated_patch = torch.flip(fabricated_patch, dims=[1])
            if random.random() < 0.5:
                design_patch = torch.flip(design_patch, dims=[0])
                fabricated_patch = torch.flip(fabricated_patch, dims=[0])

        return design_patch.unsqueeze(0), fabricated_patch.unsqueeze(0)


class FullImageDataset(Dataset):
    def __init__(
        self,
        design_paths: Sequence[Path],
        fabricated_paths: Sequence[Path],
        *,
        augment: bool = False,
    ) -> None:
        if len(design_paths) != len(fabricated_paths):
            raise ValueError("Design and fabricated image counts do not match.")
        self.augment = augment
        self.design_tensors = [_load_grayscale_tensor(p) for p in design_paths]
        self.fab_tensors = [_load_grayscale_tensor(p) for p in fabricated_paths]
        for idx, (design, fabricated) in enumerate(zip(self.design_tensors, self.fab_tensors)):
            if design.shape != fabricated.shape:
                raise ValueError(f"Shape mismatch for image pair {design_paths[idx]} and {fabricated_paths[idx]}.")

    def __len__(self) -> int:
        return len(self.design_tensors)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        design = self.design_tensors[index].to(torch.float32) / 255.0
        fabricated = self.fab_tensors[index].to(torch.float32) / 255.0

        if self.augment:
            if random.random() < 0.5:
                design = torch.flip(design, dims=[1])
                fabricated = torch.flip(fabricated, dims=[1])
            if random.random() < 0.5:
                design = torch.flip(design, dims=[0])
                fabricated = torch.flip(fabricated, dims=[0])

        return design.unsqueeze(0), fabricated.unsqueeze(0)


class PSFConvolution(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        *,
        eta: float,
        sig_ax: float,
        sig_ay: float,
        sig_bx: float,
        sig_by: float,
        step: float = 1.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("PSF kernel size must be odd.")

        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32) * step
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)

        self.eta = nn.Parameter(torch.tensor(eta, dtype=torch.float32))
        self.sig_ax = nn.Parameter(torch.tensor(sig_ax, dtype=torch.float32))
        self.sig_ay = nn.Parameter(torch.tensor(sig_ay, dtype=torch.float32))
        self.sig_bx = nn.Parameter(torch.tensor(sig_bx, dtype=torch.float32))
        self.sig_by = nn.Parameter(torch.tensor(sig_by, dtype=torch.float32))

        self.padding = half

    def _gaussian(self, sigma_x: torch.Tensor, sigma_y: torch.Tensor) -> torch.Tensor:
        sigma_x = torch.clamp(sigma_x, min=1e-6)
        sigma_y = torch.clamp(sigma_y, min=1e-6)

        coeff = 1.0 / (2.0 * math.pi * sigma_x * sigma_y)
        norm_x = self.xx / sigma_x
        norm_y = self.yy / sigma_y
        exponent = -0.5 * (norm_x.square() + norm_y.square())
        return coeff * torch.exp(exponent)

    def _build_kernel(self, dtype: torch.dtype) -> torch.Tensor:
        eta = torch.clamp(self.eta, min=1e-6)
        w2 = eta / (1.0 + eta)
        w1 = 1.0 - w2

        g1 = self._gaussian(self.sig_ax, self.sig_ay)
        g2 = self._gaussian(self.sig_bx, self.sig_by)
        kernel = w1 * g1 + w2 * g2
        kernel = kernel / kernel.sum()
        return kernel.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self._build_kernel(dtype=x.dtype)
        weight = kernel.unsqueeze(0).unsqueeze(0)
        return F.conv2d(x, weight, padding=self.padding)


class PSFCalibrationModel(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        threshold_kernel_size: int = 3,
        *,
        eta: float,
        sig_ax: float,
        sig_ay: float,
        sig_bx: float,
        sig_by: float,
    ) -> None:
        super().__init__()
        if threshold_kernel_size % 2 == 0:
            raise ValueError("threshold_kernel_size must be odd.")
        self.psf = PSFConvolution(
            kernel_size,
            eta=eta,
            sig_ax=sig_ax,
            sig_ay=sig_ay,
            sig_bx=sig_bx,
            sig_by=sig_by,
        )
        padding = threshold_kernel_size // 2
        self.threshold_head = nn.Conv2d(1, 1, kernel_size=threshold_kernel_size, padding=padding)
        nn.init.zeros_(self.threshold_head.weight)
        nn.init.zeros_(self.threshold_head.bias)

    def forward(self, x: torch.Tensor, sharpness: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        psf_out = self.psf(x)
        threshold = torch.sigmoid(self.threshold_head(psf_out))
        logits = sharpness * (psf_out - threshold)
        binary = torch.sigmoid(logits)
        return binary, psf_out, threshold, logits


@dataclass
class LossWeights:
    mse: float
    bce: float


@dataclass
class EpochMetrics:
    loss: float = 0.0
    mse: float = 0.0
    bce: float = 0.0

    def update(self, loss: torch.Tensor, mse: torch.Tensor, bce: torch.Tensor, batch_size: int) -> None:
        self.loss += float(loss.detach()) * batch_size
        self.mse += float(mse.detach()) * batch_size
        self.bce += float(bce.detach()) * batch_size

    def normalize(self, total_samples: int) -> None:
        if total_samples == 0:
            return
        inv = 1.0 / total_samples
        self.loss *= inv
        self.mse *= inv
        self.bce *= inv


def _round_up_to_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def build_psf_kernel(args: argparse.Namespace) -> torch.Tensor:
    kernel_size = _round_up_to_odd(args.kernel_size)
    radius = (kernel_size - 1) / 2.0
    kernel = build_effective_psf(
        eta=args.eta,
        sig_ax=args.sig_ax,
        sig_ay=args.sig_ay,
        sig_bx=args.sig_bx,
        sig_by=args.sig_by,
        step=1.0,
        radius=radius,
    )
    # Adjust any numerical off-by-one coming from floating point meshgrid construction.
    if kernel.shape[-1] != kernel_size:
        center = kernel.shape[-1] // 2
        half = kernel_size // 2
        start = center - half
        end = start + kernel_size
        kernel = kernel[start:end, start:end]
    return kernel


def compute_sharpness(args: argparse.Namespace, epoch: int, step: int, steps_per_epoch: int) -> float:
    if args.sharpness_ramp_epochs <= 0:
        return args.max_sharpness
    progress = (epoch + step / max(steps_per_epoch, 1)) / max(args.sharpness_ramp_epochs, 1)
    progress = max(0.0, min(1.0, progress))
    return args.min_sharpness + progress * (args.max_sharpness - args.min_sharpness)


def run_epoch(
    model: PSFCalibrationModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_weights: LossWeights,
    args: argparse.Namespace,
    epoch: int,
    train: bool,
) -> EpochMetrics:
    metrics = EpochMetrics()
    total_samples = 0
    if train:
        model.train()
    else:
        model.eval()
    steps_per_epoch = len(loader)

    for step, (design, fabricated) in enumerate(loader):
        design = design.to(device)
        fabricated = fabricated.to(device)
        sharpness = compute_sharpness(args, epoch, step, steps_per_epoch)

        with torch.set_grad_enabled(train):
            pred, psf_out, threshold, logits = model(design, sharpness=sharpness)
            mse_loss = (
                F.mse_loss(pred, fabricated) if loss_weights.mse > 0.0 else pred.new_tensor(0.0)
            )
            bce_loss = (
                F.binary_cross_entropy(pred, fabricated) if loss_weights.bce > 0.0 else pred.new_tensor(0.0)
            )
            total_loss = loss_weights.mse * mse_loss + loss_weights.bce * bce_loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

        batch_size = design.size(0)
        total_samples += batch_size
        metrics.update(total_loss, mse_loss, bce_loss, batch_size)

    metrics.normalize(total_samples)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate PSF model against fabrication data.")
    parser.add_argument("--data-dir", type=Path, default=Path("psf_model/data"), help="Directory with x*.png/y*.png pairs.")
    parser.add_argument("--design-glob", type=str, default="x*.png", help="Glob for design images.")
    parser.add_argument("--fabricated-glob", type=str, default="y*.png", help="Glob for fabricated images.")
    parser.add_argument("--kernel-size", type=int, default=501, help="Effective PSF kernel size in nanometers (pixels).")
    # parser.add_argument("--use-patches", action="store_true", help="Train on random patches instead of full images.")
    parser.add_argument("--use-patches", type=bool, default=False, help="Train on random patches instead of full images.")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size used for training when --use-patches is set.")
    parser.add_argument("--train-patches-per-image", type=int, default=512, help="Number of training patches per image (requires --use-patches).")
    parser.add_argument("--val-patches-per-image", type=int, default=0, help="Validation samples per image; when --use-patches is not set, full images are used and this acts as a boolean flag.")
    parser.add_argument("--batch-size", type=int, default=1, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--min-sharpness", type=float, default=5.0, help="Initial sigmoid sharpness.")
    parser.add_argument("--max-sharpness", type=float, default=50.0, help="Final sigmoid sharpness.")
    parser.add_argument("--sharpness-ramp-epochs", type=int, default=20, help="Epochs to ramp sharpness.")
    parser.add_argument(
        "--loss",
        choices=["mse", "bce", "combined"],
        # default="combined",
        default="mse",
        help="Loss function configuration.",
    )
    parser.add_argument("--mse-weight", type=float, default=0.5, help="Weight for MSE loss when using combined loss.")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE loss when using combined loss.")
    parser.add_argument("--threshold-kernel-size", type=int, default=3, help="Kernel size for threshold prediction conv.")
    parser.add_argument("--eta", type=float, default=0.55, help="PSF eta parameter.")
    parser.add_argument("--sig-ax", type=float, default=25.0, help="Sigma ax for PSF.")
    parser.add_argument("--sig-ay", type=float, default=25.0, help="Sigma ay for PSF.")
    parser.add_argument("--sig-bx", type=float, default=70.0, help="Sigma bx for PSF.")
    parser.add_argument("--sig-by", type=float, default=70.0, help="Sigma by for PSF.")
    parser.add_argument("--device", type=str, default=None, help="Computation device. Defaults to cuda if available.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("psf_model/checkpoints"), help="Directory to save checkpoints.")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint interval in epochs.")
    parser.add_argument("--no-augment", action="store_true", help="Disable random flip augmentation during training.")
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def prepare_dataloaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader | None]:
    design_paths = sorted(args.data_dir.glob(args.design_glob))
    fabricated_paths = sorted(args.data_dir.glob(args.fabricated_glob))
    if not design_paths or not fabricated_paths:
        raise FileNotFoundError(f"No design/fabricated images found in {args.data_dir}.")
    if len(design_paths) != len(fabricated_paths):
        raise ValueError("Mismatch between number of design and fabricated images.")

    augment = not args.no_augment

    if args.use_patches:
        if args.patch_size <= 0:
            raise ValueError("patch_size must be positive when using patches.")
        if args.train_patches_per_image <= 0:
            raise ValueError("train_patches_per_image must be positive when using patches.")
        train_dataset = PatchDataset(
            design_paths,
            fabricated_paths,
            args.patch_size,
            args.train_patches_per_image,
            seed=args.seed,
            augment=augment,
        )

        val_loader = None
        if args.val_patches_per_image > 0:
            val_dataset = PatchDataset(
                design_paths,
                fabricated_paths,
                args.patch_size,
                args.val_patches_per_image,
                seed=args.seed + 1,
                augment=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
    else:
        train_dataset = FullImageDataset(
            design_paths,
            fabricated_paths,
            augment=augment,
        )
        val_loader = None
        if args.val_patches_per_image > 0:
            val_dataset = FullImageDataset(
                design_paths,
                fabricated_paths,
                augment=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def derive_loss_weights(args: argparse.Namespace) -> LossWeights:
    if args.loss == "mse":
        return LossWeights(mse=1.0, bce=0.0)
    if args.loss == "bce":
        return LossWeights(mse=0.0, bce=1.0)
    total = args.mse_weight + args.bce_weight
    if total <= 0.0:
        raise ValueError("Combined loss requires positive weights.")
    return LossWeights(mse=args.mse_weight / total, bce=args.bce_weight / total)


def save_checkpoint(
    args: argparse.Namespace,
    model: PSFCalibrationModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint_dir / f"psf_calibration_epoch_{epoch:04d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    kernel = build_psf_kernel(args)
    kernel_size = int(kernel.shape[-1])
    print(f"Constructed PSF kernel with shape {tuple(kernel.shape)}.")

    train_loader, val_loader = prepare_dataloaders(args)

    model = PSFCalibrationModel(
        kernel_size=kernel_size,
        threshold_kernel_size=args.threshold_kernel_size,
        eta=args.eta,
        sig_ax=args.sig_ax,
        sig_ay=args.sig_ay,
        sig_bx=args.sig_bx,
        sig_by=args.sig_by,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_weights = derive_loss_weights(args)

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            loss_weights=loss_weights,
            args=args,
            epoch=epoch,
            train=True,
        )

        message = (
            f"Epoch {epoch + 1:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_mse={train_metrics.mse:.4f} "
            f"train_bce={train_metrics.bce:.4f}"
        )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer,
                    loss_weights=loss_weights,
                    args=args,
                    epoch=epoch,
                    train=False,
                )
            message += (
                f" | val_loss={val_metrics.loss:.4f} "
                f"val_mse={val_metrics.mse:.4f} "
                f"val_bce={val_metrics.bce:.4f}"
            )

        print(message)

        should_save = (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs
        if should_save:
            save_checkpoint(args, model, optimizer, epoch + 1)


if __name__ == "__main__":
    main()
