#!/usr/bin/env python3
"""Compute CLIP image similarity for one RS-UniEdit result."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor


ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_CLIP_MODEL = os.environ.get("RS_UNIEDIT_CLIP_MODEL", "openai/clip-vit-large-patch14")


def choose_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def read_rgb(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def encode_image(
    image_path: Path,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    image = read_rgb(image_path)
    inputs = processor(images=image, return_tensors="pt").to(device)
    features = model.get_image_features(**inputs)
    return F.normalize(features, dim=-1)


def cosine_percent(a: torch.Tensor, b: torch.Tensor) -> float:
    return (100.0 * (a * b).sum(dim=-1)).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CLIP-I for a generated image.")
    parser.add_argument("--result", required=True, type=Path, help="Generated result image.")
    parser.add_argument("--target", required=True, type=Path, help="Ground-truth target image.")
    parser.add_argument("--source", required=True, type=Path, help="Original source image.")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = choose_device(args.device)

    model = CLIPModel.from_pretrained(
        args.clip_model,
        local_files_only=args.local_files_only,
    ).to(device)
    processor = CLIPProcessor.from_pretrained(
        args.clip_model,
        local_files_only=args.local_files_only,
    )
    model.eval()

    with torch.inference_mode():
        result_features = encode_image(args.result, model, processor, device)
        target_features = encode_image(args.target, model, processor, device)
        source_features = encode_image(args.source, model, processor, device)

        clip_i = cosine_percent(result_features, target_features)
        target_direction = F.normalize(target_features - source_features, dim=-1)
        result_direction = F.normalize(result_features - source_features, dim=-1)
        clip_direction = (target_direction * result_direction).sum(dim=-1).item()

    print(f"CLIP-I: {clip_i:.6f}")
    print(f"CLIP-Directional: {clip_direction:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
