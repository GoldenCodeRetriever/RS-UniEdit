#!/usr/bin/env python3
"""Compute LPIPS and DISTS for one RS-UniEdit result."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
RESAMPLE_BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image, got shape {array.shape}")
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_pair(result_path: Path, target_path: Path, metric_size: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    result = read_rgb(result_path)
    target = read_rgb(target_path)

    if metric_size:
        size = (metric_size, metric_size)
        result = result.resize(size, RESAMPLE_BICUBIC)
        target = target.resize(size, RESAMPLE_BICUBIC)
    elif result.size != target.size:
        result = result.resize(target.size, RESAMPLE_BICUBIC)

    return image_to_tensor(result), image_to_tensor(target)


def load_lpips(net: str, device: torch.device) -> torch.nn.Module:
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("Missing package: lpips. Install it with `pip install lpips`.") from exc

    model = lpips.LPIPS(net=net)
    model.to(device)
    model.eval()
    return model


def load_dists(device: torch.device) -> torch.nn.Module:
    try:
        from DISTS_pytorch import DISTS

        model = DISTS()
    except ImportError:
        try:
            import pyiqa
        except ImportError as exc:
            raise RuntimeError("Missing DISTS backend. Install `DISTS-pytorch` or `pyiqa`.") from exc
        model = pyiqa.create_metric("dists", device=device, as_loss=False)

    model.to(device)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LPIPS and DISTS for a generated image.")
    parser.add_argument("--result", required=True, type=Path, help="Generated result image.")
    parser.add_argument("--target", required=True, type=Path, help="Ground-truth target image.")
    parser.add_argument("--source", required=True, type=Path, help="Original source image.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--metric-size", type=int, default=None, help="Resize both images to this square size.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Image does not exist: {args.source}")

    device = choose_device(args.device)
    result, target = load_pair(args.result, args.target, args.metric_size)
    result = result.to(device)
    target = target.to(device)

    lpips_model = load_lpips(args.lpips_net, device)
    dists_model = load_dists(device)

    with torch.inference_mode():
        lpips_value = float(lpips_model(result * 2.0 - 1.0, target * 2.0 - 1.0).reshape(-1)[0].item())
        dists_value = float(dists_model(result, target).reshape(-1)[0].item())

    print(f"LPIPS: {lpips_value:.6f}")
    print(f"DISTS: {dists_value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
