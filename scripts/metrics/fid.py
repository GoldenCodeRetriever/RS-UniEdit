#!/usr/bin/env python3
"""Compute FID between generated and target image sets."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    from torchmetrics.image.fid import FID as FrechetInceptionDistance


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FID_SUPPORTS_NORMALIZE = "normalize" in inspect.signature(FrechetInceptionDistance).parameters


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def collect_images(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    images = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise RuntimeError(f"No images found in: {path}")
    return images


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, image_size: int = 299):
        self.paths = collect_images(root)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.image_size:
                image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            array = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


@torch.inference_mode()
def update_metric(
    metric: FrechetInceptionDistance,
    loader: DataLoader,
    device: torch.device,
    real: bool,
    label: str,
) -> None:
    for batch in tqdm(loader, desc=label, ncols=100):
        batch = batch.to(device, non_blocking=True)
        if not FID_SUPPORTS_NORMALIZE:
            batch = (batch * 255.0).clamp(0, 255).to(torch.uint8)
        metric.update(batch, real=real)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute FID for generated and target images.")
    parser.add_argument("--result", required=True, type=Path, help="Generated result image or directory.")
    parser.add_argument("--target", required=True, type=Path, help="Ground-truth target image or directory.")
    parser.add_argument("--source", required=True, type=Path, help="Original source image or directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=299)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Path does not exist: {args.source}")

    device = choose_device(args.device)
    result_data = ImagePathDataset(args.result, image_size=args.image_size)
    target_data = ImagePathDataset(args.target, image_size=args.image_size)

    result_loader = DataLoader(
        result_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    target_loader = DataLoader(
        target_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    if FID_SUPPORTS_NORMALIZE:
        metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    else:
        metric = FrechetInceptionDistance(feature=2048).to(device)

    update_metric(metric, target_loader, device, real=True, label="Target")
    update_metric(metric, result_loader, device, real=False, label="Result")
    fid = float(metric.compute().item())

    print(f"FID: {fid:.6f}")
    print(f"Result images: {len(result_data)}")
    print(f"Target images: {len(target_data)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
