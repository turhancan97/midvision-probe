from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def build_image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def build_mask_transform(image_size: int):
    if isinstance(image_size, int):
        size = (image_size, image_size)
    else:
        size = image_size
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )


def load_masks(
    image_path: Path,
    metadata_dir: str,
    pattern: str,
    groups: Sequence[str],
    image_size: int,
    include_background: bool,
    background_name: str,
) -> Dict[str, torch.Tensor]:
    meta_path = image_path.parent / metadata_dir
    transform = build_mask_transform(image_size)
    zero = torch.zeros((image_size, image_size), dtype=torch.float32)
    masks: Dict[str, torch.Tensor] = {}
    stack = []
    for group in groups:
        mask_file = meta_path / pattern.format(name=group)
        if mask_file.exists():
            with Image.open(mask_file) as m:
                tensor = transform(m.convert("L")).squeeze(0)
                tensor = (tensor > 0.5).float()
        else:
            tensor = zero.clone()
        masks[group] = tensor
        stack.append(tensor)
    if include_background:
        if stack:
            union = torch.stack(stack, dim=0).max(dim=0).values
            background = torch.clamp(1.0 - union, min=0.0, max=1.0)
        else:
            background = torch.ones_like(zero)
        masks[background_name] = background
    return masks


def mask_to_patch_grid(mask: torch.Tensor, patch_size: int, group: str) -> torch.Tensor:
    mask = mask.unsqueeze(0).unsqueeze(0)
    pooled = F.avg_pool2d(mask, kernel_size=patch_size, stride=patch_size)
    # if group == "Background":
    #     # make values 0 if any value smaller than 1
    #     pooled = 1 - (pooled < 1).float()
    # else:
    #     pooled = (pooled > 0).float()
    return pooled.squeeze(0).squeeze(0)


def visualize_image(
    image_path: Path,
    masks: Dict[str, torch.Tensor],
    patch_size: int,
    image_size: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    img_transform = build_image_transform(image_size)
    with Image.open(image_path) as img:
        img_tensor = img_transform(img.convert("RGB"))
    img_np = np.clip(img_tensor.permute(1, 2, 0).numpy(), 0, 1)
    patch_hw = image_size // patch_size
    groups = list(masks.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, group in zip(axes, groups):
        grid = mask_to_patch_grid(masks[group], patch_size, group)
        grid = F.interpolate(
            grid.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="nearest",
        ).squeeze().numpy()
        ax.imshow(img_np)
        ax.imshow(grid, cmap="viridis", alpha=0.4, interpolation="nearest")
        ax.set_title(group)
        ax.axis("off")
    fig.suptitle(image_path.name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{image_path.stem}_patch_overlay.png", dpi=200)
    plt.close(fig)


def collect_images(root: Path, limit: int) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        [
            p
            for p in root.rglob("*")
            if p.suffix.lower() in supported and "metadata" not in p.parts
        ]
    )
    if limit > 0:
        images = images[:limit]
    return images


def main():
    parser = argparse.ArgumentParser(description="Visualize patch-level masks to verify extraction.")
    parser.add_argument("--image-dir", type=str, default="analysis/images")
    parser.add_argument("--metadata-dir", type=str, default="metadata")
    parser.add_argument("--mask-pattern", type=str, default="mask_{name}.png")
    parser.add_argument("--groups", nargs="+", default=["Human", "Tree", "Truck"])
    parser.add_argument("--include-background", action="store_true", default=True)
    parser.add_argument("--background-name", type=str, default="Background")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="analysis/outputs/patch_debug")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    images = collect_images(image_dir, args.max_images)
    if not images:
        raise FileNotFoundError(f"No images found under {image_dir}")

    for img_path in images:
        masks = load_masks(
            img_path,
            args.metadata_dir,
            args.mask_pattern,
            args.groups,
            args.image_size,
            args.include_background,
            args.background_name,
        )
        visualize_image(img_path, masks, args.patch_size, args.image_size, output_dir)
        print(f"Saved overlay for {img_path}")


if __name__ == "__main__":
    main()
