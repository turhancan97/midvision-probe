from __future__ import annotations

import argparse

import torch

from evals.models.dinov3_timm import DINOV3TIMM


def run_validation(model_name: str, image_size: int, device: str, layer: int) -> None:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    batch_size = 2
    x = torch.randn(batch_size, 3, image_size, image_size, device=dev)

    print(f"Validating timm DINOv3 model: {model_name}")
    print(f"Device: {dev}")
    print(f"Input shape: {tuple(x.shape)}")

    # CLS shortcut mode.
    model_cls = DINOV3TIMM(
        model_name=model_name,
        output="dense",
        layer=layer,
        add_norm=True,
        return_cls=True,
        mean_pool=False,
        efficient_probe=False,
        pretrained=True,
    ).to(dev)
    with torch.no_grad():
        y_cls = model_cls(x)
    if y_cls.dim() != 2:
        raise RuntimeError(f"CLS mode expected rank-2 output, got shape {tuple(y_cls.shape)}")
    print(f"CLS mode output: {tuple(y_cls.shape)}")

    # Mean-pool shortcut mode.
    model_mean = DINOV3TIMM(
        model_name=model_name,
        output="dense",
        layer=layer,
        add_norm=True,
        return_cls=False,
        mean_pool=True,
        efficient_probe=False,
        pretrained=True,
    ).to(dev)
    with torch.no_grad():
        y_mean = model_mean(x)
    if y_mean.dim() != 2:
        raise RuntimeError(f"Mean mode expected rank-2 output, got shape {tuple(y_mean.shape)}")
    print(f"Mean mode output: {tuple(y_mean.shape)}")

    # Efficient probe mode (patch tokens + cls token).
    model_patch = DINOV3TIMM(
        model_name=model_name,
        output="dense",
        layer=layer,
        add_norm=True,
        return_cls=False,
        mean_pool=False,
        efficient_probe=True,
        pretrained=True,
    ).to(dev)
    with torch.no_grad():
        y_patch = model_patch(x)
    if y_patch.dim() != 3:
        raise RuntimeError(
            f"Efficient probe mode expected rank-3 output, got shape {tuple(y_patch.shape)}"
        )
    if y_patch.shape[1] <= 1:
        raise RuntimeError(
            f"Efficient probe output should include cls + patch tokens, got shape {tuple(y_patch.shape)}"
        )
    print(f"Efficient probe mode output: {tuple(y_patch.shape)}")

    print("Validation passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate DINOV3TIMM output shapes for cls/mean/patch modes."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_base_patch16_dinov3.lvd1689m",
        help="timm DINOv3 model name (without hf_hub prefix).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Square image size for synthetic validation input.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Backbone layer index to evaluate (-1 means last default layer).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_validation(
        model_name=args.model_name,
        image_size=args.image_size,
        device=args.device,
        layer=args.layer,
    )


if __name__ == "__main__":
    main()

